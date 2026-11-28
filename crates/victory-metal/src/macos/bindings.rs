//! Metal Shader Converter root-signature ABI bindings (ADR-006 scope).
//!
//! Owns the mapping from a translated DXIL root signature to the argument-buffer
//! layout the converter emits (top-level argument buffer, root constants, CBV /
//! SRV / UAV / Sampler tables) and the encoders that bind it on an
//! `MTLRenderCommandEncoder` / `MTLComputeCommandEncoder`.
//!
//! # What the converter actually emits
//!
//! The layout below is not guessed from D3D12 semantics: it is parsed from
//! Apple's own reflection (`IRShaderReflectionCopyJSONString`, reached through
//! [`crate::macos::irconverter::convert_dxil_to_metallib_with_reflection`]).
//! For the converter shipped with D3DMetal 1.9 / Metal toolchain 32023 the
//! converted MSL always receives three implicit buffers:
//!
//! ```text
//! buffer(0) = resource descriptor heap argument buffer   (unused unless a *heap* is indexed)
//! buffer(1) = sampler descriptor heap argument buffer    (unused unless a *heap* is indexed)
//! buffer(2) = top-level global argument buffer           (always present)
//! ```
//!
//! Every root parameter is flattened into the top-level argument buffer in root
//! signature order. Descriptor parameters occupy one field containing a 64-bit
//! device address; the converter may report that field with an 8-byte or a
//! 24-byte `Size` (`_Desc { address, 0, 0 }`, of which only the first 8 bytes are
//! used). Descriptor *tables* occupy one field containing the device address of
//! an array of descriptors in GPU-visible memory:
//!
//! ```text
//! root CBV / SRV / UAV : field = address of the resource data
//! table of CBV/SRV/UAV : field = address of [_Desc { addr, 0, 0 }; count]  (stride 24)
//! table of textures    : field = address of [_Desc { tex, tex, id }; count]  (stride 24)
//! table of samplers    : field = address of [_Desc { smp, id, id }; count]    (stride 24)
//! root constants       : inline bytes at the reflected offset (see below)
//! ```
//!
//! A sampler table is reflected as `{"Type":"Sampler",…}` (not `"Table"`), and
//! an argument buffer with no parameters may be an absent or empty
//! `TopLevelArgumentBuffer`; both are valid layouts.
//!
//! Verified against a converted compute shader with
//! `RootConstants(b0), CBV(b1), SRV(t0), UAV(u0), DescriptorTable(SRV(t1)),
//! DescriptorTable(Sampler(s0))`: reflection reports fields at offsets
//! `0, 8, 16, 24, 32, 40`, and the converted MSL loads the root constants
//! directly from the top-level argument buffer at offset 0.
//!
//! ## Root constants
//!
//! For this converter version D3D12 root constants are **inline in the
//! top-level argument buffer**, not a `[[buffer(n)]]` `setBytes` slot. The
//! reflection entry is `{"Type":"Constant","EltOffset":…,"Size":…}` and the
//! generated AIR reads the fields through the `top_level_global_ab` pointer.
//! [`ArgumentBufferBuilder::set_root_constants`] therefore writes them at that
//! offset. A layout produced by a converter that instead reports a dedicated
//! Metal constant-buffer index can opt into the `setBytes` form with
//! [`RootSignatureLayout::with_root_constants_metal_index`];
//! [`ArgumentBufferEncoder::set_root_constants`] then issues
//! `setBytes:length:atIndex:` for that index and refuses to write the bytes
//! into the argument buffer at the same time.
//!
//! # Validation
//!
//! Layout construction refuses overlapping, misaligned, oversized or
//! unknown-typed reflection entries. Binding refuses zero handles, zero-length
//! or misaligned ranges, out-of-bounds ranges (the range end must stay inside
//! the registered backing length), and descriptor tables whose converted extent
//! (`count * stride`) exceeds the bound table range. Generation validation
//! stays in the registration layer, exactly as in [`crate::macos::msc`]; the
//! encoder re-checks the handle identity and the byte range it was handed.
//!
//! # Example
//!
//! ```no_run
//! use victory_metal::macos::bindings::{
//!     ArgumentBufferEncoder, BufferBinding, DescriptorTableBinding, DescriptorTableKind,
//!     EncoderKind, ResourceUsage, RootSignatureLayout,
//! };
//!
//! # fn example(layout: &RootSignatureLayout, device: *mut std::ffi::c_void,
//! #            encoder: *mut std::ffi::c_void) -> Result<(), victory_metal::MetalError> {
//! let mut binder = unsafe {
//!     ArgumentBufferEncoder::new(device, encoder, EncoderKind::Compute, layout)?
//! };
//! # Ok(())
//! # }
//! ```

use std::ffi::c_void;
use std::fmt;

use crate::abi::VmBufferId;
use crate::macos::objc;
use crate::transport::MetalError;
use crate::wire;

/// Metal buffer index of the converted top-level global argument buffer.
///
/// Verified with `arc64-dxil.sh` and with every reflected program in this
/// module's hardware tests: the converted function takes
/// `air.indirect_buffer … air.location_index 2`.
pub const TOP_LEVEL_ARGUMENT_BUFFER_INDEX: usize = 2;

/// Metal buffer index of the converted resource descriptor heap argument buffer.
pub const RESOURCE_DESCRIPTOR_HEAP_BUFFER_INDEX: usize = 0;

/// Metal buffer index of the converted sampler descriptor heap argument buffer.
pub const SAMPLER_DESCRIPTOR_HEAP_BUFFER_INDEX: usize = 1;

/// Size of one encoded GPU device address (`MTLGPUAddress`).
pub const DEVICE_ADDRESS_BYTES: usize = 8;

/// Stride of one texture descriptor in a converted descriptor table.
///
/// Observed AIR: `_Desc { texture*, texture*, i64 }`; the sampler reads element
/// 1 (the second pointer) and the runtime writes both pointers.
pub const TEXTURE_DESCRIPTOR_BYTES: usize = 24;

/// Stride of one sampler descriptor in a converted descriptor table.
///
/// Observed AIR: `_Desc { sampler*, i64, i64 }`; the texture sample reads
/// element 0.
pub const SAMPLER_DESCRIPTOR_BYTES: usize = 24;

/// Upper bound for a converted top-level argument buffer (64 KiB).
pub const MAX_ARGUMENT_BUFFER_BYTES: usize = 64 * 1024;

/// Upper bound for one root-constants block (D3D12 allows 64 dwords; 4 KiB is a
/// defensive ceiling that still accepts a converter-generated struct).
pub const MAX_ROOT_CONSTANT_BYTES: usize = 4096;

/// Largest descriptor count accepted in one table binding.
pub const MAX_DESCRIPTOR_TABLE_ENTRIES: u32 = 65_536;

/// Exclusive upper bound for a Metal buffer argument index (`0..=30`).
///
/// Index 31 and above is never a valid `setBuffer:offset:atIndex:` target.
pub const MAX_METAL_BUFFER_ARGUMENT_INDEX: usize = 31;

/// Shader stage reported by the converter reflection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectedStage {
    /// `"Compute"`; binds through a compute encoder.
    Compute,
    /// `"Vertex"`; binds through a render encoder.
    Vertex,
    /// `"Fragment"`; binds through a render encoder.
    Fragment,
    /// Any other stage (geometry, mesh, ray tracing…). Accepted by the layout
    /// but not yet encodable by this module.
    Other,
}

impl ReflectedStage {
    fn from_json(text: &str) -> Self {
        match text {
            "Compute" => Self::Compute,
            "Vertex" => Self::Vertex,
            "Fragment" => Self::Fragment,
            _ => Self::Other,
        }
    }
}

/// Kind of descriptor table stored in one flattened argument-buffer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorTableKind {
    /// Table of constant buffer views: 24-byte `{device_address, 0, 0}`.
    ConstantBufferView,
    /// Table of buffer shader resource views: 24-byte `{device_address, 0, 0}`.
    ShaderResourceView,
    /// Table of unordered access views: 24-byte `{device_address, 0, 0}`.
    UnorderedAccessView,
    /// Table of `Texture2D`-style descriptors: 24-byte `{texture, texture, 0}`.
    Texture,
    /// Table of sampler states: 24-byte `{sampler, 0, 0}`.
    Sampler,
    /// `"Table"` in reflection without a caller-supplied range type.
    ///
    /// The converter reflection does not say which descriptor range a table
    /// declares. Callers attach the kind in root-signature order with
    /// [`RootSignatureLayout::with_descriptor_table_kinds`]; until then the
    /// table can still be bound as the converter's 24-byte resource descriptor.
    Unspecified,
}

impl DescriptorTableKind {
    /// Stable name used by counters, refusals and the telemetry census.
    pub const fn name(self) -> &'static str {
        match self {
            Self::ConstantBufferView => "ConstantBufferView",
            Self::ShaderResourceView => "ShaderResourceView",
            Self::UnorderedAccessView => "UnorderedAccessView",
            Self::Texture => "Texture",
            Self::Sampler => "Sampler",
            Self::Unspecified => "Unspecified",
        }
    }

    /// Default stride of one descriptor of this kind in converted memory.
    ///
    /// Every converted table element is a 24-byte descriptor. Address tables
    /// hold `{device_address, 0, 0}`, texture tables `{id, id, 0}` and sampler
    /// tables `{id, 0, 0}`; the host writes all three fields.
    pub fn descriptor_bytes(self) -> usize {
        TEXTURE_DESCRIPTOR_BYTES
    }

    fn is_address_table(self) -> bool {
        matches!(
            self,
            Self::ConstantBufferView
                | Self::ShaderResourceView
                | Self::UnorderedAccessView
                | Self::Unspecified
        )
    }
}

/// Kind of one root parameter after conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootParameterKind {
    /// D3D12 root constants. Stored inline in the top-level argument buffer for
    /// this converter version; `size` is the reflected byte size.
    Constants,
    /// Root constant buffer view: one 8-byte device address.
    ConstantBufferView,
    /// Root shader resource view: one 8-byte device address.
    ShaderResourceView,
    /// Root unordered access view: one 8-byte device address.
    UnorderedAccessView,
    /// Descriptor table: one 8-byte device address of a descriptor array.
    DescriptorTable(DescriptorTableKind),
}

/// One flattened root parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootParameter {
    kind: RootParameterKind,
    argument_offset: usize,
    size: usize,
    register: u32,
    space: u32,
}

impl RootParameter {
    /// Parameter kind.
    pub fn kind(&self) -> RootParameterKind {
        self.kind
    }

    /// Byte offset inside the top-level argument buffer.
    pub fn argument_offset(&self) -> usize {
        self.argument_offset
    }

    /// Reflected size in bytes. `8` for descriptor/table slots, the declared
    /// byte size for root constants.
    pub fn size(&self) -> usize {
        self.size
    }

    /// HLSL register index (`bN`/`tN`/`uN`/`sN`); `0` for tables.
    pub fn register(&self) -> u32 {
        self.register
    }

    /// HLSL register space.
    pub fn space(&self) -> u32 {
        self.space
    }

    fn with_table_kind(self, kind: DescriptorTableKind) -> Self {
        Self {
            kind: RootParameterKind::DescriptorTable(kind),
            ..self
        }
    }
}

/// One descriptor table a stage layout declares, as registration recorded it.
///
/// ADR-066: converter reflection types a flattened descriptor-table field by
/// the range kind it guessed (`"Table"`, `"Sampler"`, or a view family) and
/// reports the field's descriptor extent in `Size`. Both are carried per stage
/// so the host binding path can decide, without re-parsing reflection, which
/// declared table a wire set addresses and whether that field is the
/// converter's 24-byte descriptor array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeclaredTable {
    /// Flattened argument-buffer parameter index of the table's field.
    pub parameter: usize,
    /// Reflected range kind of the field ([`DescriptorTableKind::Unspecified`]
    /// for the converter's kind-less `"Table"`).
    pub kind: DescriptorTableKind,
    /// Reflected extent of the field in bytes (`RootParameter::size`).
    pub field_bytes: usize,
    /// Descriptor slots the reflected field spans: `field_bytes` divided by
    /// [`TEXTURE_DESCRIPTOR_BYTES`], at least one. This is the count the wire
    /// set is packed against.
    pub slots: u32,
}

/// Declaration used by [`RootSignatureLayout::from_parameters`].
///
/// This is the converter-independent form for callers that already hold a
/// verified root-signature description (for example the translated root
/// signature blob of a title). Offsets are assigned in declaration order with
/// 8-byte alignment, matching the observed converter flattening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootParameterSpec {
    /// Parameter kind.
    pub kind: RootParameterKind,
    /// Reflected/declared byte size (constants only; `8` for descriptors).
    pub size: usize,
    /// HLSL register index.
    pub register: u32,
    /// HLSL register space.
    pub space: u32,
}

/// Flattened root-signature layout of one translated program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootSignatureLayout {
    stage: ReflectedStage,
    entry_point: String,
    argument_buffer_index: usize,
    argument_buffer_len: usize,
    parameters: Vec<RootParameter>,
    root_constants_metal_index: Option<usize>,
}

/// A bounded, content-free reason one converter reflection cannot become a
/// root-signature layout.
///
/// The reflection JSON is never reproduced: the error carries only the
/// zero-based entry index (when the fault is specific to one entry), the
/// reflected field name and a fixed explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReflectionLayoutError {
    entry: Option<usize>,
    field: &'static str,
    reason: &'static str,
}

impl ReflectionLayoutError {
    fn document(reason: &'static str) -> Self {
        Self {
            entry: None,
            field: "reflection document",
            reason,
        }
    }

    fn top_level(field: &'static str, reason: &'static str) -> Self {
        Self {
            entry: None,
            field,
            reason,
        }
    }

    fn entry(index: usize, field: &'static str, reason: &'static str) -> Self {
        Self {
            entry: Some(index),
            field,
            reason,
        }
    }

    /// Zero-based `TopLevelArgumentBuffer` entry that failed, when the fault is
    /// specific to one entry.
    pub fn entry_index(&self) -> Option<usize> {
        self.entry
    }

    /// Reflected field that failed.
    pub fn field(&self) -> &'static str {
        self.field
    }

    /// Fixed, content-free explanation.
    pub fn reason(&self) -> &'static str {
        self.reason
    }
}

impl fmt::Display for ReflectionLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.entry {
            Some(index) => write!(
                formatter,
                "entry {index} field `{}`: {}",
                self.field, self.reason
            ),
            None => write!(formatter, "field `{}`: {}", self.field, self.reason),
        }
    }
}

impl From<ReflectionLayoutError> for MetalError {
    fn from(error: ReflectionLayoutError) -> Self {
        // `MetalError::InvalidArg` carries a `&'static str`; callers that need
        // the entry index use the detailed variant instead.
        MetalError::InvalidArg(error.reason)
    }
}

/// One way a reflected root parameter is not representable in the argument
/// buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParameterFault {
    ConstantsSize,
    ConstantsSizeAlignment,
    ConstantsOffsetAlignment,
    DescriptorTooSmall,
    DescriptorAlignment,
    Overlap,
    ExtentOverflow,
    ArgumentBufferTooLarge,
}

impl ParameterFault {
    fn field(self) -> &'static str {
        match self {
            Self::ConstantsSize
            | Self::ConstantsSizeAlignment
            | Self::DescriptorTooSmall
            | Self::ExtentOverflow => "Size",
            Self::ConstantsOffsetAlignment
            | Self::DescriptorAlignment
            | Self::Overlap
            | Self::ArgumentBufferTooLarge => "EltOffset",
        }
    }

    fn reason(self) -> &'static str {
        match self {
            Self::ConstantsSize => "root constants size is outside 1..=4096 bytes",
            Self::ConstantsSizeAlignment | Self::ConstantsOffsetAlignment => {
                "root constants must be 4-byte aligned and sized"
            }
            Self::DescriptorTooSmall => {
                "descriptor argument-buffer slot is smaller than one address"
            }
            Self::DescriptorAlignment => "descriptor argument-buffer offset must be 8-byte aligned",
            Self::Overlap => "root parameters overlap in the argument buffer",
            Self::ExtentOverflow => "root parameter extent overflows",
            Self::ArgumentBufferTooLarge => "root parameters exceed the 64 KiB argument buffer",
        }
    }
}

/// Validates parameters already sorted by argument-buffer offset.
///
/// `sources` is aligned with `parameters` and carries the original reflection
/// entry index so a refusal can name it.
fn validate_parameters<E>(
    parameters: &[RootParameter],
    sources: &[Option<usize>],
    mut report: impl FnMut(Option<usize>, ParameterFault) -> E,
) -> Result<(), E> {
    let mut end = 0usize;
    for (position, parameter) in parameters.iter().enumerate() {
        let source = sources.get(position).copied().flatten();
        if parameter.kind == RootParameterKind::Constants {
            if parameter.size == 0 || parameter.size > MAX_ROOT_CONSTANT_BYTES {
                return Err(report(source, ParameterFault::ConstantsSize));
            }
            if !parameter.size.is_multiple_of(4) {
                return Err(report(source, ParameterFault::ConstantsSizeAlignment));
            }
            if !parameter.argument_offset.is_multiple_of(4) {
                return Err(report(source, ParameterFault::ConstantsOffsetAlignment));
            }
        } else {
            if parameter.size < DEVICE_ADDRESS_BYTES {
                return Err(report(source, ParameterFault::DescriptorTooSmall));
            }
            if !parameter
                .argument_offset
                .is_multiple_of(DEVICE_ADDRESS_BYTES)
            {
                return Err(report(source, ParameterFault::DescriptorAlignment));
            }
        }
        if parameter.argument_offset < end {
            return Err(report(source, ParameterFault::Overlap));
        }
        end = parameter
            .argument_offset
            .checked_add(parameter.size)
            .ok_or_else(|| report(source, ParameterFault::ExtentOverflow))?;
        if end > MAX_ARGUMENT_BUFFER_BYTES {
            return Err(report(source, ParameterFault::ArgumentBufferTooLarge));
        }
    }
    Ok(())
}

impl RootSignatureLayout {
    /// Parses Apple's `IRShaderReflectionCopyJSONString` output.
    ///
    /// `TopLevelArgumentBuffer` is optional: an absent, `null` or empty array is
    /// a valid layout with no parameters. When present, every entry must have
    /// `EltOffset`, `Size`, `Slot`, `Space` and a `Type` of `Constant`, `CBV`,
    /// `SRV`, `UAV`, `Table` or `Sampler`; descriptor fields must be 8-byte
    /// aligned and at least 8 bytes, root constants must be 4-byte aligned and
    /// 4-byte sized, and no two parameters may overlap.
    ///
    /// Use [`Self::from_converter_reflection_json_detailed`] when the refusal
    /// reason must name the failing reflection entry.
    pub fn from_converter_reflection_json(json: &str) -> Result<Self, MetalError> {
        Self::from_converter_reflection_json_detailed(json).map_err(MetalError::from)
    }

    /// Parses Apple's `IRShaderReflectionCopyJSONString` output, naming the
    /// failing field on refusal.
    ///
    /// The accepted shapes are the same as
    /// [`Self::from_converter_reflection_json`]. Reflection field names are
    /// matched case-insensitively because the converter's JSON key casing is not
    /// part of its contract. A refusal never reproduces reflected names or
    /// values: it carries only the zero-based entry index, the reflected field
    /// name and a fixed explanation.
    pub fn from_converter_reflection_json_detailed(
        json: &str,
    ) -> Result<Self, ReflectionLayoutError> {
        let root = parse_json(json)
            .map_err(|error| ReflectionLayoutError::document(metal_error_reason(error)))?;
        let entry_point = json_string_field(&root, "EntryPoint")
            .unwrap_or_default()
            .to_string();
        let stage = json_string_field(&root, "ShaderType")
            .map_or(ReflectedStage::Other, ReflectedStage::from_json);
        // A converter that emits no resources omits the array (observed empty
        // arrays too); both mean "no root parameters", not an unusable layout.
        let entries: &[Json] = match json_field(&root, &["TopLevelArgumentBuffer"]) {
            None | Some(Json::Null) => &[],
            Some(value) => value.as_array().ok_or(ReflectionLayoutError::top_level(
                "TopLevelArgumentBuffer",
                "TopLevelArgumentBuffer is not an array",
            ))?,
        };
        if entries.len() > 64 {
            return Err(ReflectionLayoutError::top_level(
                "TopLevelArgumentBuffer",
                "converter reflection has more than 64 root parameters",
            ));
        }

        let mut parameters = Vec::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            let object = entry.as_object().ok_or(ReflectionLayoutError::entry(
                index,
                "TopLevelArgumentBuffer",
                "TopLevelArgumentBuffer entry is not an object",
            ))?;
            let offset = reflection_u64(object, index, "EltOffset")?;
            let size = reflection_u64(object, index, "Size")?;
            let slot = reflection_u64(object, index, "Slot")?;
            let space = reflection_u64(object, index, "Space")?;
            let type_name = json_object_field(object, &["Type"])
                .ok_or(ReflectionLayoutError::entry(
                    index,
                    "Type",
                    missing_field_reason("Type"),
                ))?
                .as_str()
                .ok_or(ReflectionLayoutError::entry(
                    index,
                    "Type",
                    "Type is not a string",
                ))?;
            let kind = root_parameter_kind(type_name).ok_or(ReflectionLayoutError::entry(
                index,
                "Type",
                "converter reflection has an unknown root parameter type",
            ))?;

            let (Ok(argument_offset), Ok(size)) = (usize::try_from(offset), usize::try_from(size))
            else {
                return Err(ReflectionLayoutError::entry(
                    index,
                    "EltOffset",
                    "reflected offset or size does not fit usize",
                ));
            };
            parameters.push((
                index,
                RootParameter {
                    kind,
                    argument_offset,
                    size,
                    register: u32::try_from(slot).unwrap_or(u32::MAX),
                    space: u32::try_from(space).unwrap_or(u32::MAX),
                },
            ));
        }

        parameters.sort_by_key(|(_, parameter)| parameter.argument_offset);
        let sources: Vec<Option<usize>> = parameters
            .iter()
            .map(|(index, _)| Some(*index))
            .collect();
        let flattened: Vec<RootParameter> = parameters
            .into_iter()
            .map(|(_, parameter)| parameter)
            .collect();
        validate_parameters(&flattened, &sources, |source, fault| {
            ReflectionLayoutError::entry(
                source.unwrap_or_default(),
                fault.field(),
                fault.reason(),
            )
        })?;

        Self::assemble_sorted(
            stage,
            entry_point,
            TOP_LEVEL_ARGUMENT_BUFFER_INDEX,
            flattened,
            None,
        )
        .map_err(|error| ReflectionLayoutError::document(metal_error_reason(error)))
    }

    /// Builds a layout from an explicit parameter declaration list.
    ///
    /// Offsets are assigned in order, 8-byte aligned; constants keep their
    /// declared byte size. Use this only with a root signature whose flattening
    /// has been verified against converter reflection.
    pub fn from_parameters(
        stage: ReflectedStage,
        parameters: &[RootParameterSpec],
    ) -> Result<Self, MetalError> {
        if parameters.len() > 64 {
            return Err(MetalError::InvalidArg(
                "root signature has more than 64 parameters",
            ));
        }
        let mut cursor = 0usize;
        let mut flattened = Vec::with_capacity(parameters.len());
        for spec in parameters {
            let offset = align_up(cursor, DEVICE_ADDRESS_BYTES)?;
            let size = match spec.kind {
                RootParameterKind::Constants => spec.size,
                _ => {
                    if spec.size != 0 && spec.size != DEVICE_ADDRESS_BYTES {
                        return Err(MetalError::InvalidArg(
                            "descriptor parameter size must be 8 bytes",
                        ));
                    }
                    DEVICE_ADDRESS_BYTES
                }
            };
            cursor = offset
                .checked_add(size)
                .ok_or(MetalError::InvalidArg("root parameter offset overflows"))?;
            flattened.push(RootParameter {
                kind: spec.kind,
                argument_offset: offset,
                size,
                register: spec.register,
                space: spec.space,
            });
        }
        Self::assemble(
            stage,
            String::new(),
            TOP_LEVEL_ARGUMENT_BUFFER_INDEX,
            flattened,
            None,
        )
    }

    fn assemble(
        stage: ReflectedStage,
        entry_point: String,
        argument_buffer_index: usize,
        mut parameters: Vec<RootParameter>,
        root_constants_metal_index: Option<usize>,
    ) -> Result<Self, MetalError> {
        parameters.sort_by_key(|parameter| parameter.argument_offset);
        Self::assemble_sorted(
            stage,
            entry_point,
            argument_buffer_index,
            parameters,
            root_constants_metal_index,
        )
    }

    /// Builds a layout from parameters already sorted by argument offset.
    fn assemble_sorted(
        stage: ReflectedStage,
        entry_point: String,
        argument_buffer_index: usize,
        parameters: Vec<RootParameter>,
        root_constants_metal_index: Option<usize>,
    ) -> Result<Self, MetalError> {
        if argument_buffer_index >= MAX_METAL_BUFFER_ARGUMENT_INDEX {
            return Err(MetalError::InvalidArg(
                "argument buffer index is not a valid Metal buffer index",
            ));
        }

        let sources = vec![None; parameters.len()];
        validate_parameters(&parameters, &sources, |_, fault| {
            MetalError::InvalidArg(fault.reason())
        })?;
        let constants_count = parameters
            .iter()
            .filter(|parameter| parameter.kind == RootParameterKind::Constants)
            .count();
        if let Some(index) = root_constants_metal_index {
            if index >= MAX_METAL_BUFFER_ARGUMENT_INDEX {
                return Err(MetalError::InvalidArg(
                    "root constants Metal index is not a valid buffer index",
                ));
            }
            if constants_count != 1 {
                return Err(MetalError::InvalidArg(
                    "setBytes root constants require exactly one constants parameter",
                ));
            }
        }

        let end = parameters
            .last()
            .map_or(0, |parameter| parameter.argument_offset + parameter.size);
        Ok(Self {
            stage,
            entry_point,
            argument_buffer_index,
            argument_buffer_len: align_up(end.max(DEVICE_ADDRESS_BYTES), DEVICE_ADDRESS_BYTES)?,
            parameters,
            root_constants_metal_index,
        })
    }

    /// Attaches descriptor-range kinds to `"Table"` entries in declaration
    /// order.
    ///
    /// Reflection reports only `"Table"`, so this is the one piece of the
    /// layout the caller must supply from its root signature description. The
    /// number of kinds must equal the number of table parameters.
    pub fn with_descriptor_table_kinds(
        mut self,
        kinds: &[DescriptorTableKind],
    ) -> Result<Self, MetalError> {
        let table_count = self
            .parameters
            .iter()
            .filter(|parameter| {
                matches!(parameter.kind, RootParameterKind::DescriptorTable(_))
            })
            .count();
        if kinds.len() != table_count {
            return Err(MetalError::InvalidArg(
                "descriptor table kind count does not match the reflected tables",
            ));
        }
        let mut kinds = kinds.iter();
        for parameter in &mut self.parameters {
            if matches!(parameter.kind, RootParameterKind::DescriptorTable(_)) {
                let Some(kind) = kinds.next() else {
                    return Err(MetalError::InvalidArg(
                        "descriptor table kind count does not match the reflected tables",
                    ));
                };
                *parameter = parameter.with_table_kind(*kind);
            }
        }
        Ok(self)
    }

    /// Opts into the `setBytes` form of root constants at Metal buffer
    /// `index`.
    ///
    /// Only for converter output that reports a dedicated constant-buffer
    /// index. The observed D3DMetal converter keeps root constants inline in
    /// the top-level argument buffer and does not set this.
    pub fn with_root_constants_metal_index(mut self, index: usize) -> Result<Self, MetalError> {
        if index >= MAX_METAL_BUFFER_ARGUMENT_INDEX {
            return Err(MetalError::InvalidArg(
                "root constants Metal index is not a valid buffer index",
            ));
        }
        let constants = self
            .parameters
            .iter()
            .filter(|parameter| parameter.kind == RootParameterKind::Constants)
            .count();
        if constants != 1 {
            return Err(MetalError::InvalidArg(
                "setBytes root constants require exactly one constants parameter",
            ));
        }
        self.root_constants_metal_index = Some(index);
        Ok(self)
    }

    /// Reflected shader stage.
    pub fn stage(&self) -> ReflectedStage {
        self.stage
    }

    /// Descriptor tables this stage layout declares, in flattened parameter
    /// order, with the reflected kind and descriptor slot count registration
    /// carried (ADR-066).
    ///
    /// The count is the reflection's own: a field with an 8-byte or 24-byte
    /// `Size` spans one converter descriptor, so `slots` is `1` for the shapes
    /// this converter emits. It exists so the binding path never has to guess
    /// the count from the wire alone.
    pub fn declared_tables(&self) -> Vec<DeclaredTable> {
        self.parameters
            .iter()
            .enumerate()
            .filter_map(|(parameter, entry)| {
                let RootParameterKind::DescriptorTable(kind) = entry.kind else {
                    return None;
                };
                let slots = entry.size.div_ceil(TEXTURE_DESCRIPTOR_BYTES).max(1);
                Some(DeclaredTable {
                    parameter,
                    kind,
                    field_bytes: entry.size,
                    slots: u32::try_from(slots).unwrap_or(u32::MAX),
                })
            })
            .collect()
    }

    /// Reflected entry-point name (empty for `from_parameters`).
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Metal buffer index the top-level argument buffer is bound to.
    pub fn argument_buffer_index(&self) -> usize {
        self.argument_buffer_index
    }

    /// Byte length of the top-level argument buffer (8-byte aligned, minimum 8).
    pub fn argument_buffer_len(&self) -> usize {
        self.argument_buffer_len
    }

    /// Flattened parameters in argument-buffer order.
    pub fn parameters(&self) -> &[RootParameter] {
        &self.parameters
    }

    /// One parameter by flattened index.
    pub fn parameter(&self, index: usize) -> Result<&RootParameter, MetalError> {
        self.parameters.get(index).ok_or(MetalError::InvalidArg(
            "root parameter index is out of range",
        ))
    }

    /// Flattened index of the parameter the translated root signature names by
    /// `(space, register, kind)`.
    ///
    /// The wire protocol addresses a root argument by its root-parameter slot,
    /// while the converter flattens parameters into the argument buffer in its
    /// own order. This lookup is the bridge: it requires an exact `space` and
    /// `register` match. A converted layout may legitimately declare the same
    /// `(space, register, kind)` more than once (two stages sharing one
    /// parameter, or a reflection that lists a table twice); the first
    /// declaration in flattened order wins deterministically, which is the
    /// parameter the converter itself addresses first. Refusing outright would
    /// drop a valid pipeline, as the Synthetic pre-boot gate proved.
    ///
    /// `Ok(None)` means the layout does not declare the argument. A table kind
    /// of [`DescriptorTableKind::Unspecified`] matches any concrete table kind,
    /// because converter reflection reports every table as `"Table"` and the
    /// root signature is the only source of the range kind.
    pub fn parameter_by(
        &self,
        space: u32,
        register: u32,
        kind: RootParameterKind,
    ) -> Result<Option<usize>, MetalError> {
        for (index, parameter) in self.parameters.iter().enumerate() {
            // Converter reflection reports `Slot`/`Space` as `u32::MAX` for a
            // descriptor table; the root signature is the only source of that
            // table's register and space, so an unspecified reflected field
            // still matches.
            let table = matches!(parameter.kind, RootParameterKind::DescriptorTable(_));
            let space_matches = parameter.space == space || (table && parameter.space == u32::MAX);
            let register_matches =
                parameter.register == register || (table && parameter.register == u32::MAX);
            if space_matches && register_matches && root_kind_matches(parameter.kind, kind) {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    /// Metal buffer index used by `setBytes` root constants, when opted in.
    pub fn root_constants_metal_index(&self) -> Option<usize> {
        self.root_constants_metal_index
    }

    /// True when the layout can be bound on this encoder family.
    pub fn supports(&self, kind: EncoderKind) -> bool {
        match (kind, self.stage) {
            (EncoderKind::Compute, ReflectedStage::Compute) => true,
            (EncoderKind::Render, ReflectedStage::Vertex | ReflectedStage::Fragment) => true,
            (_, ReflectedStage::Other) => true,
            _ => false,
        }
    }
}

/// Whether a reflected parameter kind satisfies a requested kind.
///
/// Only a descriptor table's range kind may be unspecified: converter
/// reflection reports `"Table"` for every table, so the root signature alone
/// supplies the concrete range kind. Every other kind must match exactly.
fn root_kind_matches(declared: RootParameterKind, requested: RootParameterKind) -> bool {
    match (declared, requested) {
        (
            RootParameterKind::DescriptorTable(declared),
            RootParameterKind::DescriptorTable(requested),
        ) => {
            declared == requested
                || declared == DescriptorTableKind::Unspecified
                || requested == DescriptorTableKind::Unspecified
        }
        _ => declared == requested,
    }
}

/// One root-parameter slot of the translated root signature.
///
/// The slot is the root-parameter index the wire protocol names on
/// `SetRootConstants`, `SetRootBufferView` and `SetDescriptorTable`; `space`,
/// `register`, `kind` and `size` are what the translated signature declares
/// there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootParameterSlot {
    space: u32,
    register: u32,
    kind: RootParameterKind,
    size: usize,
}

impl RootParameterSlot {
    /// Builds one slot declaration.
    pub fn new(space: u32, register: u32, kind: RootParameterKind, size: usize) -> Self {
        Self {
            space,
            register,
            kind,
            size,
        }
    }

    /// HLSL register space.
    pub fn space(&self) -> u32 {
        self.space
    }

    /// HLSL register index.
    pub fn register(&self) -> u32 {
        self.register
    }

    /// Parameter kind.
    pub fn kind(&self) -> RootParameterKind {
        self.kind
    }

    /// Declared byte size (root constants: the serialized value count times
    /// four; descriptors and tables: eight).
    pub fn size(&self) -> usize {
        self.size
    }
}

/// D3D12's limit on root parameters.
const ROOT_SIGNATURE_PARAMETER_LIMIT: usize = 64;
/// Header of a serialized versioned root signature.
const ROOT_SIGNATURE_HEADER_BYTES: usize = 24;
/// One parameter-array entry.
const ROOT_SIGNATURE_PARAMETER_BYTES: usize = 12;
/// One descriptor-table range.
const ROOT_SIGNATURE_RANGE_BYTES: usize = 24;

/// Authoritative slot -> `(space, register, kind)` map of a translated root
/// signature.
///
/// This is the host's resolution table for wire root arguments. It is built
/// from `PipelineRequestV3`'s translated root-signature bytes, never from a
/// guest claim about the layout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RootParameterMap {
    slots: Vec<RootParameterSlot>,
}

impl RootParameterMap {
    /// Builds a map from declarations in root-parameter slot order.
    pub fn from_slots(slots: Vec<RootParameterSlot>) -> Self {
        Self { slots }
    }

    /// The declaration at root-parameter `slot`.
    pub fn slot(&self, slot: usize) -> Option<&RootParameterSlot> {
        self.slots.get(slot)
    }

    /// Number of declared slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the signature declares no root parameter.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Declarations in slot order.
    pub fn slots(&self) -> &[RootParameterSlot] {
        &self.slots
    }

    /// Parses the `D3D12SerializeVersionedRootSignature` bytes of a translated
    /// root signature into a slot map.
    ///
    /// Accepts the documented version 1.0/1.1 layout: a 24-byte header, a
    /// 12-byte parameter array, then each parameter's 12-byte record and, for a
    /// descriptor table, its 24-byte ranges. Static samplers follow and are not
    /// needed here. A truncated, misaligned or unknown declaration is
    /// `InvalidArg` with a fixed reason; no bytes of the signature are
    /// reproduced.
    pub fn from_versioned_blob(blob: &[u8]) -> Result<Self, MetalError> {
        let word = |at: usize| -> Result<u32, MetalError> {
            let bytes = blob
                .get(at..at + 4)
                .ok_or(MetalError::InvalidArg(
                    "translated root signature is truncated",
                ))?;
            Ok(u32::from_le_bytes(
                bytes.try_into().expect("a four-byte slice"),
            ))
        };
        if blob.len() < ROOT_SIGNATURE_HEADER_BYTES {
            return Err(MetalError::InvalidArg(
                "translated root signature is shorter than its header",
            ));
        }
        let version = word(0)?;
        if version != 1 && version != 2 {
            return Err(MetalError::InvalidArg(
                "translated root signature has an unknown version",
            ));
        }
        let count = word(4)? as usize;
        if count > ROOT_SIGNATURE_PARAMETER_LIMIT {
            return Err(MetalError::InvalidArg(
                "translated root signature has too many parameters",
            ));
        }
        let array = word(8)? as usize;
        let array_end = array
            .checked_add(count * ROOT_SIGNATURE_PARAMETER_BYTES)
            .ok_or(MetalError::InvalidArg(
                "translated root signature parameter array overflows",
            ))?;
        if array < ROOT_SIGNATURE_HEADER_BYTES || array_end > blob.len() {
            return Err(MetalError::InvalidArg(
                "translated root signature parameter array is out of bounds",
            ));
        }

        let mut slots = Vec::with_capacity(count);
        for index in 0..count {
            let at = array + index * ROOT_SIGNATURE_PARAMETER_BYTES;
            let parameter_type = word(at)?;
            let record = word(at + 8)? as usize;
            if record + 12 > blob.len() {
                return Err(MetalError::InvalidArg(
                    "translated root signature parameter record is out of bounds",
                ));
            }
            let (kind, size) = match parameter_type {
                1 => {
                    let values = word(record + 8)? as usize;
                    let size = values
                        .checked_mul(4)
                        .ok_or(MetalError::InvalidArg(
                            "translated root signature constant size overflows",
                        ))?;
                    (RootParameterKind::Constants, size)
                }
                2 => (RootParameterKind::ConstantBufferView, DEVICE_ADDRESS_BYTES),
                3 => (RootParameterKind::ShaderResourceView, DEVICE_ADDRESS_BYTES),
                4 => (RootParameterKind::UnorderedAccessView, DEVICE_ADDRESS_BYTES),
                0 => {
                    let ranges = word(record)? as usize;
                    if ranges == 0 {
                        return Err(MetalError::InvalidArg(
                            "translated root signature table has no ranges",
                        ));
                    }
                    let ranges_at = word(record + 4)? as usize;
                    let ranges_end = ranges_at
                        .checked_add(ranges * ROOT_SIGNATURE_RANGE_BYTES)
                        .ok_or(MetalError::InvalidArg(
                            "translated root signature ranges overflow",
                        ))?;
                    if ranges_at < ROOT_SIGNATURE_HEADER_BYTES || ranges_end > blob.len() {
                        return Err(MetalError::InvalidArg(
                            "translated root signature ranges are out of bounds",
                        ));
                    }
                    let range_kind = |kind| match kind {
                        0 => Some(DescriptorTableKind::ShaderResourceView),
                        1 => Some(DescriptorTableKind::UnorderedAccessView),
                        2 => Some(DescriptorTableKind::ConstantBufferView),
                        3 => Some(DescriptorTableKind::Sampler),
                        _ => None,
                    };
                    let mut table_kind = None;
                    for range in 0..ranges {
                        let at = ranges_at + range * ROOT_SIGNATURE_RANGE_BYTES;
                        let declared = range_kind(word(at)?).ok_or(MetalError::InvalidArg(
                            "translated root signature table range has an unknown kind",
                        ))?;
                        table_kind = match table_kind {
                            None => Some(declared),
                            Some(previous) if previous == declared => Some(previous),
                            Some(_) => Some(DescriptorTableKind::Unspecified),
                        };
                    }
                    // A table's register/space is its first range's.
                    let register = word(ranges_at + 8)?;
                    let space = word(ranges_at + 12)?;
                    slots.push(RootParameterSlot::new(
                        space,
                        register,
                        RootParameterKind::DescriptorTable(
                            table_kind.unwrap_or(DescriptorTableKind::Unspecified),
                        ),
                        DEVICE_ADDRESS_BYTES,
                    ));
                    continue;
                }
                _ => {
                    return Err(MetalError::InvalidArg(
                        "translated root signature has an unknown parameter type",
                    ));
                }
            };
            let register = word(record)?;
            let space = word(record + 4)?;
            slots.push(RootParameterSlot::new(space, register, kind, size));
        }
        Ok(Self { slots })
    }
}

/// CPU encoder for the converted top-level argument buffer.
///
/// Writes checked 64-bit device addresses, table addresses and inline root
/// constants into a zero-initialized byte buffer. No Metal object is created
/// here; [`ArgumentBufferEncoder::bind_argument_buffer`] uploads the result.
#[derive(Debug, Clone)]
pub struct ArgumentBufferBuilder<'a> {
    layout: &'a RootSignatureLayout,
    bytes: Vec<u8>,
    encoded: Vec<bool>,
    /// `(start, end)` byte range the last write for each parameter covered.
    written: Vec<Option<(usize, usize)>>,
}

impl<'a> ArgumentBufferBuilder<'a> {
    /// Fresh zero-filled builder for `layout`.
    pub fn new(layout: &'a RootSignatureLayout) -> Self {
        Self {
            layout,
            bytes: vec![0; layout.argument_buffer_len()],
            encoded: vec![false; layout.parameters().len()],
            written: vec![None; layout.parameters().len()],
        }
    }

    /// Layout this builder encodes.
    pub fn layout(&self) -> &'a RootSignatureLayout {
        self.layout
    }

    /// Encoded argument-buffer bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// True when `parameter` was written at least once.
    pub fn is_encoded(&self, parameter: usize) -> bool {
        self.encoded.get(parameter).copied().unwrap_or(false)
    }

    /// Number of parameters written.
    pub fn encoded_parameters(&self) -> usize {
        self.encoded.iter().filter(|written| **written).count()
    }

    /// Byte range the last write for `parameter` covered, when it was written.
    ///
    /// Root constants cover their whole reflected block (the partial tail is
    /// zeroed); a descriptor or table covers only the 8-byte address field, not
    /// the reflected descriptor size.
    pub fn written_extent(&self, parameter: usize) -> Option<(usize, usize)> {
        self.written.get(parameter).copied().flatten()
    }

    /// `(first, last)` byte offset any parameter write touched, when any was
    /// written. Bounded to the argument-buffer layout; no payload is exposed.
    pub fn written_span(&self) -> Option<(usize, usize)> {
        self.written
            .iter()
            .flatten()
            .copied()
            .fold(None, |span, (start, end)| match span {
                None => Some((start, end)),
                Some((first, last)) => Some((first.min(start), end.max(last))),
            })
    }

    /// Total bytes the parameter writes covered (overlapping ranges are not a
    /// legal layout, so this is an exact sum).
    pub fn written_bytes(&self) -> usize {
        self.written
            .iter()
            .flatten()
            .map(|(start, end)| end.saturating_sub(*start))
            .sum()
    }

    /// Writes inline root constants at the reflected offset.
    ///
    /// `bytes` must be non-empty, 4-byte sized, and fit the reflected constant
    /// block. The parameter range is zeroed first, so a partial update zeroes
    /// the remaining dwords deterministically.
    pub fn set_root_constants(
        &mut self,
        parameter: usize,
        bytes: &[u8],
    ) -> Result<(), MetalError> {
        let root = *self.layout.parameter(parameter)?;
        if root.kind != RootParameterKind::Constants {
            return Err(MetalError::InvalidArg(
                "root parameter is not a root-constants block",
            ));
        }
        if bytes.is_empty()
            || !bytes.len().is_multiple_of(4)
            || bytes.len() > root.size
        {
            return Err(MetalError::InvalidArg(
                "root constants must be 1..=reflected bytes, in 4-byte units",
            ));
        }
        let start = root.argument_offset;
        let end = start
            .checked_add(root.size)
            .ok_or(MetalError::InvalidArg("root constants extent overflows"))?;
        let destination = self
            .bytes
            .get_mut(start..end)
            .ok_or(MetalError::InvalidArg(
                "root constants do not fit the argument buffer",
            ))?;
        destination.fill(0);
        destination[..bytes.len()].copy_from_slice(bytes);
        self.encoded[parameter] = true;
        self.written[parameter] = Some((start, end));
        Ok(())
    }

    /// Writes one root CBV/SRV/UAV device address.
    ///
    /// `address` must be nonzero and 8-byte aligned.
    pub fn set_buffer_address(
        &mut self,
        parameter: usize,
        address: u64,
    ) -> Result<(), MetalError> {
        let root = *self.layout.parameter(parameter)?;
        match root.kind {
            RootParameterKind::ConstantBufferView
            | RootParameterKind::ShaderResourceView
            | RootParameterKind::UnorderedAccessView => {}
            _ => {
                return Err(MetalError::InvalidArg(
                    "root parameter is not a root buffer view",
                ));
            }
        }
        self.write_address(parameter, root, address)
    }

    /// Writes the device address of one descriptor-table array.
    ///
    /// `address` must be nonzero and 8-byte aligned. Range/extent validation
    /// happens in [`ArgumentBufferEncoder::set_descriptor_table`]; this
    /// low-level builder only owns the argument-buffer field.
    pub fn set_table_address(
        &mut self,
        parameter: usize,
        address: u64,
    ) -> Result<(), MetalError> {
        let root = *self.layout.parameter(parameter)?;
        if !matches!(root.kind, RootParameterKind::DescriptorTable(_)) {
            return Err(MetalError::InvalidArg(
                "root parameter is not a descriptor table",
            ));
        }
        self.write_address(parameter, root, address)
    }

    fn write_address(
        &mut self,
        parameter: usize,
        root: RootParameter,
        address: u64,
    ) -> Result<(), MetalError> {
        if address == 0 {
            return Err(MetalError::InvalidArg(
                "argument-buffer device address is null",
            ));
        }
        if !address.is_multiple_of(DEVICE_ADDRESS_BYTES as u64) {
            return Err(MetalError::InvalidArg(
                "argument-buffer device address must be 8-byte aligned",
            ));
        }
        let start = root.argument_offset;
        let end = start
            .checked_add(DEVICE_ADDRESS_BYTES)
            .ok_or(MetalError::InvalidArg("argument-buffer slot overflows"))?;
        let slot = self
            .bytes
            .get_mut(start..end)
            .ok_or(MetalError::InvalidArg(
                "argument-buffer slot does not fit the buffer",
            ))?;
        slot.copy_from_slice(&address.to_le_bytes());
        self.encoded[parameter] = true;
        self.written[parameter] = Some((start, end));
        Ok(())
    }
}

/// Range of one registered buffer view, resolved from an allocation handle.
///
/// `buffer`/`generation` name the guest allocation; `address` and
/// `registered_length` are the base GPU address and full length the
/// registration layer resolved for that exact handle. Generation revalidation
/// against the allocation table stays in that layer; this encoder refuses a
/// zero handle and any range outside `registered_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferBinding {
    /// Registered buffer handle.
    pub buffer: VmBufferId,
    /// Guest allocation generation the resolver used.
    pub generation: u32,
    /// Base GPU address of the registered backing.
    pub address: u64,
    /// Byte offset of the view inside the backing.
    pub offset: u64,
    /// Byte length of the view.
    pub length: u64,
    /// Total byte length of the registered backing.
    pub registered_length: u64,
}

impl BufferBinding {
    /// Validates the handle and range, returning the 8-byte-aligned view address.
    pub fn resolved_address(&self) -> Result<u64, MetalError> {
        if self.buffer.0 == 0 {
            return Err(MetalError::InvalidArg(
                "root buffer view handle is the null handle",
            ));
        }
        if self.address == 0 {
            return Err(MetalError::InvalidArg("root buffer view has no GPU address"));
        }
        if self.length == 0 {
            return Err(MetalError::InvalidArg(
                "root buffer view length must be nonzero",
            ));
        }
        if self.length > wire::MAX_RESOURCE_LEN {
            return Err(MetalError::InvalidArg(
                "root buffer view exceeds the maximum resource length",
            ));
        }
        if !self.offset.is_multiple_of(4) {
            return Err(MetalError::InvalidArg(
                "root buffer view offset must be 4-byte aligned",
            ));
        }
        let Some(end) = self.offset.checked_add(self.length) else {
            return Err(MetalError::InvalidArg("root buffer view range overflows"));
        };
        if end > self.registered_length {
            return Err(MetalError::InvalidArg(
                "root buffer view exceeds the registered backing length",
            ));
        }
        let Some(address) = self.address.checked_add(self.offset) else {
            return Err(MetalError::InvalidArg("root buffer view address overflows"));
        };
        Ok(address)
    }
}

/// One converted descriptor table range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorTableBinding {
    /// Base GPU address of the descriptor array.
    pub address: u64,
    /// Bytes available at `address`.
    pub length: u64,
    /// Number of descriptors in the table.
    pub count: u32,
    /// Stride of one converted descriptor in bytes (24; see
    /// [`DescriptorTableKind::descriptor_bytes`]).
    pub stride: usize,
}

impl DescriptorTableBinding {
    /// Validates the table and returns its converted byte extent.
    pub fn extent(&self) -> Result<u64, MetalError> {
        if self.address == 0 {
            return Err(MetalError::InvalidArg(
                "descriptor table has no GPU address",
            ));
        }
        if !self
            .address
            .is_multiple_of(DEVICE_ADDRESS_BYTES as u64)
        {
            return Err(MetalError::InvalidArg(
                "descriptor table address must be 8-byte aligned",
            ));
        }
        if self.count == 0 || self.count > MAX_DESCRIPTOR_TABLE_ENTRIES {
            return Err(MetalError::InvalidArg(
                "descriptor table count is outside 1..=65536",
            ));
        }
        if self.stride < DEVICE_ADDRESS_BYTES || !self.stride.is_multiple_of(DEVICE_ADDRESS_BYTES)
        {
            return Err(MetalError::InvalidArg(
                "descriptor table stride must be an 8-byte multiple of at least 8",
            ));
        }
        let extent = (self.count as u64)
            .checked_mul(self.stride as u64)
            .ok_or(MetalError::InvalidArg("descriptor table extent overflows"))?;
        if extent > self.length {
            return Err(MetalError::InvalidArg(
                "descriptor table converted extent exceeds the bound range",
            ));
        }
        Ok(extent)
    }
}

/// CPU builder for the descriptor-array memory a converted table points at.
///
/// Encodes the three descriptor shapes observed in converted AIR. The caller
/// uploads [`Self::bytes`] to GPU-visible memory and passes the resulting
/// address to [`ArgumentBufferEncoder::set_descriptor_table`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorTableBuilder {
    kind: DescriptorTableKind,
    bytes: Vec<u8>,
    count: u32,
}

impl DescriptorTableBuilder {
    /// Builder for `kind` with the converter's descriptor stride.
    pub fn new(kind: DescriptorTableKind) -> Self {
        Self {
            kind,
            bytes: Vec::new(),
            count: 0,
        }
    }

    /// Table kind.
    pub fn kind(&self) -> DescriptorTableKind {
        self.kind
    }

    /// Number of descriptors written.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Encoded descriptor bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Writes one 24-byte resource-address descriptor `{address, 0, 0}`.
    ///
    /// The converted AIR names this type `%struct__desc = { i8*, texture*, i64 }`
    /// and indexes entry `i` at byte offset `i * 24`; the address goes in the
    /// first pointer slot and both other fields are zero.
    pub fn push_address(&mut self, address: u64) -> Result<(), MetalError> {
        if !self.kind.is_address_table() {
            return Err(MetalError::InvalidArg(
                "descriptor table kind does not hold device addresses",
            ));
        }
        if address == 0 || !address.is_multiple_of(DEVICE_ADDRESS_BYTES as u64) {
            return Err(MetalError::InvalidArg(
                "descriptor address must be nonzero and 8-byte aligned",
            ));
        }
        self.bytes.extend_from_slice(&address.to_le_bytes());
        self.bytes.extend_from_slice(&0u64.to_le_bytes());
        self.bytes.extend_from_slice(&0u64.to_le_bytes());
        self.count = self.count.saturating_add(1);
        Ok(())
    }

    /// Writes one 24-byte texture descriptor (`{texture, texture, 0}`).
    ///
    /// Converted MSL loads element 1, but the runtime writes both pointers, so
    /// both fields carry the texture handle. `texture` is an opaque 64-bit
    /// `gpuResourceID`, not a device address: only zero is invalid, and its
    /// low bits carry no alignment guarantee.
    pub fn push_texture(&mut self, texture: u64) -> Result<(), MetalError> {
        if self.kind != DescriptorTableKind::Texture {
            return Err(MetalError::InvalidArg(
                "descriptor table kind is not a texture table",
            ));
        }
        if texture == 0 {
            return Err(MetalError::InvalidArg(
                "texture descriptor resource id must be nonzero",
            ));
        }
        self.bytes.extend_from_slice(&texture.to_le_bytes());
        self.bytes.extend_from_slice(&texture.to_le_bytes());
        self.bytes.extend_from_slice(&0u64.to_le_bytes());
        self.count = self.count.saturating_add(1);
        Ok(())
    }

    /// Writes one 24-byte sampler descriptor (`{sampler, 0, 0}`).
    ///
    /// `sampler` is an opaque 64-bit `gpuResourceID`; only zero is invalid.
    pub fn push_sampler(&mut self, sampler: u64) -> Result<(), MetalError> {
        if self.kind != DescriptorTableKind::Sampler {
            return Err(MetalError::InvalidArg(
                "descriptor table kind is not a sampler table",
            ));
        }
        if sampler == 0 {
            return Err(MetalError::InvalidArg(
                "sampler descriptor resource id must be nonzero",
            ));
        }
        self.bytes.extend_from_slice(&sampler.to_le_bytes());
        self.bytes.extend_from_slice(&0u64.to_le_bytes());
        self.bytes.extend_from_slice(&0u64.to_le_bytes());
        self.count = self.count.saturating_add(1);
        Ok(())
    }

    /// Binding for a descriptor array copied to `address`.
    pub fn binding(&self, address: u64) -> Result<DescriptorTableBinding, MetalError> {
        if self.bytes.is_empty() {
            return Err(MetalError::InvalidArg("descriptor table is empty"));
        }
        Ok(DescriptorTableBinding {
            address,
            length: self.bytes.len() as u64,
            count: self.count,
            stride: self.kind.descriptor_bytes(),
        })
    }
}

/// Encoder family a layout is bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderKind {
    /// `MTLRenderCommandEncoder` (vertex + fragment stages).
    Render,
    /// `MTLComputeCommandEncoder`.
    Compute,
}

/// Residency usage for a resource referenced by the argument buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceUsage {
    /// Read-only (`MTLResourceUsageRead`).
    Read,
    /// Read and write (`MTLResourceUsageRead | MTLResourceUsageWrite`).
    ReadWrite,
}

impl ResourceUsage {
    fn mtl_bits(self) -> usize {
        match self {
            Self::Read => 1,
            Self::ReadWrite => 3,
        }
    }
}

/// Argument-buffer encoder for one render or compute scope.
///
/// All validation happens before the argument buffer is uploaded. The caller
/// keeps its encoder open, calls the setters, calls
/// [`Self::bind_argument_buffer`], then dispatches/draws and ends encoding. The
/// returned [`BoundArgumentBuffer`] must stay alive until the command buffer
/// completes.
pub struct ArgumentBufferEncoder<'a> {
    layout: &'a RootSignatureLayout,
    device: objc::Id,
    encoder: objc::Id,
    kind: EncoderKind,
    builder: ArgumentBufferBuilder<'a>,
    resources: Vec<(objc::Id, usize)>,
}

impl<'a> ArgumentBufferEncoder<'a> {
    /// Wraps a live encoder.
    ///
    /// # Safety
    ///
    /// `device` must be a live `MTLDevice` and `encoder` a live
    /// `MTLComputeCommandEncoder` for [`EncoderKind::Compute`] or
    /// `MTLRenderCommandEncoder` for [`EncoderKind::Render`]. Both must stay
    /// valid until [`Self::bind_argument_buffer`] returns and the caller ends
    /// encoding.
    pub unsafe fn new(
        device: *mut c_void,
        encoder: *mut c_void,
        kind: EncoderKind,
        layout: &'a RootSignatureLayout,
    ) -> Result<Self, MetalError> {
        if device.is_null() || encoder.is_null() {
            return Err(MetalError::InvalidArg(
                "argument-buffer encoder needs a live device and encoder",
            ));
        }
        if !layout.supports(kind) {
            return Err(MetalError::InvalidArg(
                "root signature stage does not match the encoder family",
            ));
        }
        Ok(Self {
            layout,
            device,
            encoder,
            kind,
            builder: ArgumentBufferBuilder::new(layout),
            resources: Vec::new(),
        })
    }

    /// Layout being encoded.
    pub fn layout(&self) -> &'a RootSignatureLayout {
        self.layout
    }

    /// Sets inline root constants.
    ///
    /// In the observed converter ABI this writes the bytes at the reflected
    /// offset of the top-level argument buffer. When the layout was opted into
    /// [`RootSignatureLayout::with_root_constants_metal_index`], the bytes go
    /// to that Metal buffer index through `setBytes:length:atIndex:` instead.
    pub fn set_root_constants(
        &mut self,
        parameter: usize,
        bytes: &[u8],
    ) -> Result<(), MetalError> {
        let root = *self.layout.parameter(parameter)?;
        if root.kind != RootParameterKind::Constants {
            return Err(MetalError::InvalidArg(
                "root parameter is not a root-constants block",
            ));
        }
        if bytes.is_empty() || !bytes.len().is_multiple_of(4) || bytes.len() > root.size {
            return Err(MetalError::InvalidArg(
                "root constants must be 1..=reflected bytes, in 4-byte units",
            ));
        }
        match self.layout.root_constants_metal_index() {
            Some(index) => {
                let selector = match self.kind {
                    EncoderKind::Compute => objc::sel(c"setBytes:length:atIndex:"),
                    EncoderKind::Render => {
                        // A render scope has both stages; bind both so a
                        // vertex- or fragment-stage constants block works.
                        unsafe {
                            send_bytes(
                                self.encoder,
                                objc::sel(c"setVertexBytes:length:atIndex:"),
                                bytes.as_ptr().cast(),
                                bytes.len(),
                                index,
                            );
                            send_bytes(
                                self.encoder,
                                objc::sel(c"setFragmentBytes:length:atIndex:"),
                                bytes.as_ptr().cast(),
                                bytes.len(),
                                index,
                            );
                        }
                        return Ok(());
                    }
                };
                // SAFETY: `self.encoder` is a live compute encoder and Metal
                // copies `bytes` during the call; index was validated.
                unsafe {
                    send_bytes(
                        self.encoder,
                        selector,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        index,
                    );
                }
                Ok(())
            }
            None => self.builder.set_root_constants(parameter, bytes),
        }
    }

    /// Binds one root CBV/SRV/UAV device address and declares residency.
    ///
    /// # Safety
    ///
    /// `resource` must be the live Metal resource whose base address is
    /// `binding.address`, and must stay valid until the command buffer
    /// completes.
    pub unsafe fn set_root_buffer_view(
        &mut self,
        parameter: usize,
        binding: BufferBinding,
        resource: *mut c_void,
        usage: ResourceUsage,
    ) -> Result<(), MetalError> {
        let root = *self.layout.parameter(parameter)?;
        match root.kind {
            RootParameterKind::ConstantBufferView
            | RootParameterKind::ShaderResourceView
            | RootParameterKind::UnorderedAccessView => {}
            _ => {
                return Err(MetalError::InvalidArg(
                    "root parameter is not a root buffer view",
                ));
            }
        }
        if resource.is_null() {
            return Err(MetalError::InvalidArg(
                "root buffer view resource is null",
            ));
        }
        let address = binding.resolved_address()?;
        self.builder.set_buffer_address(parameter, address)?;
        self.resources.push((resource, usage.mtl_bits()));
        Ok(())
    }

    /// Binds one descriptor table and refuses an extent larger than `table`.
    pub fn set_descriptor_table(
        &mut self,
        parameter: usize,
        table: DescriptorTableBinding,
    ) -> Result<(), MetalError> {
        let root = *self.layout.parameter(parameter)?;
        let RootParameterKind::DescriptorTable(kind) = root.kind else {
            return Err(MetalError::InvalidArg(
                "root parameter is not a descriptor table",
            ));
        };
        let _extent = table.extent()?;
        if table.stride < kind.descriptor_bytes() {
            return Err(MetalError::InvalidArg(
                "descriptor table stride is smaller than the converted descriptor",
            ));
        }
        self.builder.set_table_address(parameter, table.address)
    }

    /// Declares one resource resident for this encoder scope.
    ///
    /// Used for textures and samplers referenced through table memory, which
    /// are not covered by [`Self::set_root_buffer_view`].
    ///
    /// # Safety
    ///
    /// `resource` must be a live Metal resource that stays valid until the
    /// command buffer completes.
    pub unsafe fn use_resource(
        &mut self,
        resource: *mut c_void,
        usage: ResourceUsage,
    ) -> Result<(), MetalError> {
        if resource.is_null() {
            return Err(MetalError::InvalidArg("resident resource is null"));
        }
        self.resources.push((resource, usage.mtl_bits()));
        Ok(())
    }

    /// Uploads the argument buffer and binds it at the layout index.
    ///
    /// For a render encoder the buffer is bound for both the vertex and
    /// fragment stages; for a compute encoder through
    /// `setBuffer:offset:atIndex:`. Residency declarations collected by the
    /// setters are issued on the same encoder.
    pub fn bind_argument_buffer(self) -> Result<BoundArgumentBuffer, MetalError> {
        let _pool = objc::AutoreleasePool::new();
        let length = self.builder.bytes().len();
        // SAFETY: `self.device` is a live `MTLDevice`; shared storage is
        // CPU-visible and Metal zero-initializes the buffer.
        let raw = unsafe { objc::msg_send_new_buffer(self.device, length, 0) };
        let buffer = objc::ObjcObject::from_retained(raw).ok_or_else(|| {
            MetalError::InternalError("argument buffer allocation returned null".to_string())
        })?;
        // SAFETY: `buffer` is a shared-storage `MTLBuffer` of `length` bytes.
        let contents = unsafe {
            objc::msg_send_ptr_0(buffer.as_raw(), objc::sel(c"contents"))
        };
        if contents.is_null() {
            return Err(MetalError::InternalError(
                "argument buffer has no CPU-visible contents".to_string(),
            ));
        }
        // SAFETY: The slice covers the whole freshly created shared buffer and
        // is written before any command buffer references it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.builder.bytes().as_ptr(),
                contents.cast::<u8>(),
                length,
            );
        }

        let index = self.layout.argument_buffer_index();
        // SAFETY: `self.encoder` is a live encoder of the declared family,
        // `buffer` is a live `MTLBuffer`, and `index` was validated by layout
        // construction.
        unsafe {
            match self.kind {
                EncoderKind::Compute => {
                    objc::msg_send_set_buffer(self.encoder, buffer.as_raw(), 0, index);
                }
                EncoderKind::Render => {
                    send_buffer(
                        self.encoder,
                        objc::sel(c"setVertexBuffer:offset:atIndex:"),
                        buffer.as_raw(),
                        0,
                        index,
                    );
                    send_buffer(
                        self.encoder,
                        objc::sel(c"setFragmentBuffer:offset:atIndex:"),
                        buffer.as_raw(),
                        0,
                        index,
                    );
                }
            }
            for (resource, usage) in &self.resources {
                objc::msg_send_use_resource(self.encoder, *resource, *usage);
            }
        }
        Ok(BoundArgumentBuffer { buffer })
    }
}

/// Keeps the uploaded argument buffer alive until the command buffer completes.
pub struct BoundArgumentBuffer {
    buffer: objc::ObjcObject,
}

impl BoundArgumentBuffer {
    /// Raw retained `MTLBuffer` handle.
    pub fn as_raw(&self) -> *mut c_void {
        self.buffer.as_raw()
    }
}

type SendBuffer = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, usize, usize);
type SendBytes =
    unsafe extern "C" fn(objc::Id, objc::Sel, *const c_void, usize, usize);

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_msgSend();
}

unsafe fn send_buffer(
    encoder: objc::Id,
    selector: objc::Sel,
    buffer: objc::Id,
    offset: usize,
    index: usize,
) {
    // SAFETY: Caller supplies a live encoder, selector and buffer matching the
    // `set*Buffer:offset:atIndex:` ABI.
    let function: SendBuffer = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: Same contract as above.
    unsafe { function(encoder, selector, buffer, offset, index) };
}

unsafe fn send_bytes(
    encoder: objc::Id,
    selector: objc::Sel,
    bytes: *const c_void,
    length: usize,
    index: usize,
) {
    // SAFETY: Caller supplies a live encoder and a readable `bytes` range.
    let function: SendBytes = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: Same contract as above.
    unsafe { function(encoder, selector, bytes, length, index) };
}

fn align_up(value: usize, alignment: usize) -> Result<usize, MetalError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(MetalError::InvalidArg("invalid alignment"));
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|rounded| rounded & !mask)
        .ok_or(MetalError::InvalidArg("alignment overflows"))
}

// ---------------------------------------------------------------------------
// Minimal JSON reader for the converter reflection payload
// ---------------------------------------------------------------------------

const MAX_REFLECTION_JSON_BYTES: usize = 1024 * 1024;
const MAX_JSON_DEPTH: usize = 32;

#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }

    fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Self::Object(entries) => Some(entries),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(value)
                if value.is_finite()
                    && *value >= 0.0
                    && value.fract() == 0.0
                    && *value <= u64::MAX as f64 =>
            {
                Some(*value as u64)
            }
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Result<Self, MetalError> {
        if input.len() > MAX_REFLECTION_JSON_BYTES {
            return Err(MetalError::InvalidArg(
                "converter reflection JSON exceeds 1 MiB",
            ));
        }
        Ok(Self {
            input: input.as_bytes(),
            position: 0,
        })
    }

    fn parse(mut self) -> Result<Json, MetalError> {
        let value = self.value(0)?;
        self.skip_whitespace();
        if self.position != self.input.len() {
            return Err(MetalError::InvalidArg(
                "trailing bytes after converter reflection JSON",
            ));
        }
        Ok(value)
    }

    fn value(&mut self, depth: usize) -> Result<Json, MetalError> {
        if depth > MAX_JSON_DEPTH {
            return Err(MetalError::InvalidArg("converter reflection JSON is too deep"));
        }
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Json::String),
            Some(b't') => {
                self.literal(b"true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.literal(b"null")?;
                Ok(Json::Null)
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(MetalError::InvalidArg(
                "converter reflection JSON has an invalid value",
            )),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, MetalError> {
        self.advance();
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.advance();
            return Ok(Json::Object(entries));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return Err(MetalError::InvalidArg(
                    "converter reflection JSON object is missing ':'",
                ));
            }
            self.advance();
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.advance(),
                Some(b'}') => {
                    self.advance();
                    return Ok(Json::Object(entries));
                }
                _ => {
                    return Err(MetalError::InvalidArg(
                        "converter reflection JSON object is malformed",
                    ));
                }
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, MetalError> {
        self.advance();
        let mut values = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.advance();
            return Ok(Json::Array(values));
        }
        loop {
            values.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.advance(),
                Some(b']') => {
                    self.advance();
                    return Ok(Json::Array(values));
                }
                _ => {
                    return Err(MetalError::InvalidArg(
                        "converter reflection JSON array is malformed",
                    ));
                }
            }
        }
    }

    fn string(&mut self) -> Result<String, MetalError> {
        if self.peek() != Some(b'"') {
            return Err(MetalError::InvalidArg(
                "converter reflection JSON expected a string",
            ));
        }
        self.advance();
        let mut out = String::new();
        loop {
            let Some(byte) = self.peek() else {
                return Err(MetalError::InvalidArg(
                    "converter reflection JSON string is unterminated",
                ));
            };
            self.advance();
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let Some(escape) = self.peek() else {
                        return Err(MetalError::InvalidArg(
                            "converter reflection JSON string has a trailing escape",
                        ));
                    };
                    self.advance();
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        _ => {
                            return Err(MetalError::InvalidArg(
                                "converter reflection JSON has an unsupported escape",
                            ));
                        }
                    }
                }
                byte if byte < 0x20 => {
                    return Err(MetalError::InvalidArg(
                        "converter reflection JSON string has a control byte",
                    ));
                }
                byte => {
                    // The reflection payload is JSON text; copy UTF-8 bytes
                    // verbatim. A malformed sequence is rejected below.
                    let start = self.position - 1;
                    let width = utf8_width(byte);
                    let end = start + width;
                    let Some(slice) = self.input.get(start..end) else {
                        return Err(MetalError::InvalidArg(
                            "converter reflection JSON has truncated UTF-8",
                        ));
                    };
                    let Ok(text) = std::str::from_utf8(slice) else {
                        return Err(MetalError::InvalidArg(
                            "converter reflection JSON has invalid UTF-8",
                        ));
                    };
                    out.push_str(text);
                    self.position = end;
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, MetalError> {
        let start = self.position;
        if self.peek() == Some(b'-') {
            self.advance();
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.advance();
        }
        if self.peek() == Some(b'.') {
            self.advance();
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.advance();
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.advance();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.advance();
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.advance();
            }
        }
        let Some(text) = self.input.get(start..self.position) else {
            return Err(MetalError::InvalidArg(
                "converter reflection JSON number is truncated",
            ));
        };
        let Ok(text) = std::str::from_utf8(text) else {
            return Err(MetalError::InvalidArg(
                "converter reflection JSON number is not UTF-8",
            ));
        };
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| MetalError::InvalidArg("converter reflection JSON number is invalid"))
    }

    fn literal(&mut self, expected: &[u8]) -> Result<(), MetalError> {
        let Some(slice) = self.input.get(self.position..self.position + expected.len()) else {
            return Err(MetalError::InvalidArg(
                "converter reflection JSON literal is truncated",
            ));
        };
        if slice != expected {
            return Err(MetalError::InvalidArg(
                "converter reflection JSON literal is invalid",
            ));
        }
        self.position += expected.len();
        Ok(())
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.advance();
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }

    fn advance(&mut self) {
        self.position += 1;
    }
}

fn utf8_width(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => 1,
    }
}

fn parse_json(input: &str) -> Result<Json, MetalError> {
    JsonParser::new(input)?.parse()
}

/// Extracts the fixed reason from a refusal that carries no dynamic text.
fn metal_error_reason(error: MetalError) -> &'static str {
    match error {
        MetalError::InvalidArg(reason) => reason,
        _ => "converter reflection cannot be a root layout",
    }
}

/// Looks up one field of a JSON object, case-insensitively.
fn json_field<'a>(value: &'a Json, names: &[&str]) -> Option<&'a Json> {
    let Json::Object(entries) = value else {
        return None;
    };
    json_object_field(entries, names)
}

/// Looks up one field of a parsed JSON object, case-insensitively.
///
/// Key casing is not part of the converter's JSON contract, so `EltOffset` and
/// `eltoffset` name the same field. A field that is truly missing is still a
/// refusal; only the spelling is tolerated.
fn json_object_field<'a>(entries: &'a [(String, Json)], names: &[&str]) -> Option<&'a Json> {
    entries
        .iter()
        .find(|(key, _)| names.iter().any(|name| key.eq_ignore_ascii_case(name)))
        .map(|(_, value)| value)
}

/// Reads one string field of the reflection document.
fn json_string_field<'a>(value: &'a Json, name: &str) -> Option<&'a str> {
    json_field(value, &[name]).and_then(Json::as_str)
}

/// Reads one unsigned integer field of an argument-buffer entry.
fn reflection_u64(
    entries: &[(String, Json)],
    index: usize,
    field: &'static str,
) -> Result<u64, ReflectionLayoutError> {
    let value = json_object_field(entries, &[field])
        .ok_or(ReflectionLayoutError::entry(index, field, missing_field_reason(field)))?;
    value.as_u64().ok_or(ReflectionLayoutError::entry(
        index,
        field,
        non_integer_reason(field),
    ))
}

/// Fixed refusal for a reflected field that is absent.
///
/// The simple [`RootSignatureLayout::from_converter_reflection_json`] path can
/// only carry a `&'static str`, so each reflected field has its own sentence.
fn missing_field_reason(field: &'static str) -> &'static str {
    match field {
        "EltOffset" => "TopLevelArgumentBuffer entry has no EltOffset",
        "Size" => "TopLevelArgumentBuffer entry has no Size",
        "Slot" => "TopLevelArgumentBuffer entry has no Slot",
        "Space" => "TopLevelArgumentBuffer entry has no Space",
        "Type" => "TopLevelArgumentBuffer entry has no Type",
        _ => "TopLevelArgumentBuffer entry is missing a required field",
    }
}

/// Fixed refusal for a reflected field of the wrong type.
fn non_integer_reason(field: &'static str) -> &'static str {
    match field {
        "EltOffset" => "EltOffset is not an unsigned integer",
        "Size" => "Size is not an unsigned integer",
        "Slot" => "Slot is not an unsigned integer",
        "Space" => "Space is not an unsigned integer",
        _ => "field is not an unsigned integer",
    }
}

/// Maps one reflected `Type` to a root parameter kind.
///
/// `Sampler` is emitted by the converter for a sampler descriptor table (the
/// only sampler form D3D12 flattens into the top-level argument buffer), so it
/// is a table with a known kind rather than an unknown type.
fn root_parameter_kind(type_name: &str) -> Option<RootParameterKind> {
    if type_name.eq_ignore_ascii_case("Constant") {
        Some(RootParameterKind::Constants)
    } else if type_name.eq_ignore_ascii_case("CBV") {
        Some(RootParameterKind::ConstantBufferView)
    } else if type_name.eq_ignore_ascii_case("SRV") {
        Some(RootParameterKind::ShaderResourceView)
    } else if type_name.eq_ignore_ascii_case("UAV") {
        Some(RootParameterKind::UnorderedAccessView)
    } else if type_name.eq_ignore_ascii_case("Table") {
        Some(RootParameterKind::DescriptorTable(
            DescriptorTableKind::Unspecified,
        ))
    } else if type_name.eq_ignore_ascii_case("Sampler") {
        Some(RootParameterKind::DescriptorTable(
            DescriptorTableKind::Sampler,
        ))
    } else {
        None
    }
}


#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use super::{
        ArgumentBufferBuilder, ArgumentBufferEncoder, BufferBinding, DescriptorTableBinding,
        DescriptorTableBuilder, DescriptorTableKind, EncoderKind, ResourceUsage,
        MAX_ARGUMENT_BUFFER_BYTES, MAX_ROOT_CONSTANT_BYTES, ReflectedStage,
        ReflectionLayoutError, RootParameterKind, RootParameterSpec, RootSignatureLayout,
        SAMPLER_DESCRIPTOR_BYTES, TEXTURE_DESCRIPTOR_BYTES, TOP_LEVEL_ARGUMENT_BUFFER_INDEX,
    };
    use crate::macos::device::DirectMetalTransport;
    use crate::macos::objc;
    use crate::transport::MetalError;
    use crate::wire;
    use crate::{MetalTransport, VmBufferId};

    /// Reflection captured from the converter for:
    /// `RootConstants(b0, 2), CBV(b1), SRV(t0), UAV(u0),
    ///  DescriptorTable(SRV(t1)), DescriptorTable(Sampler(s0))`.
    const COMPUTE_REFLECTION: &str = r#"{"ConstantBufferIndices":[],"EntryPoint":"main","Features":0,"FunctionConstants":[],"HasYieldLoop":false,"NeedsFunctionConstants":false,"ResourceCount":1,"ResourceUsages":14,"SamplerIndices":[],"ShaderID":"884161606958344559316453121658487405624","ShaderResourceViewIndices":[],"ShaderType":"Compute","TopLevelArgumentBuffer":[{"EltOffset":0,"Name":"","Size":8,"Slot":0,"Space":0,"Type":"Constant"},{"EltOffset":8,"Name":"","Size":8,"Slot":1,"Space":0,"Type":"CBV"},{"EltOffset":16,"Name":"","Size":8,"Slot":0,"Space":0,"Type":"SRV"},{"EltOffset":24,"Name":"","Size":8,"Slot":0,"Space":0,"Type":"UAV"},{"EltOffset":32,"Name":"","Size":8,"Slot":4294967295,"Space":4294967295,"Type":"Table"},{"EltOffset":40,"Name":"","Size":8,"Slot":4294967295,"Space":4294967295,"Type":"Table"}],"UnorderedAccessViewIndices":[],"UsedResources":[],"UsesRayQuery":false,"state":{"tg_size":[64,1,1]}}"#;

    /// Reflection captured from the converter for a vertex shader with
    /// `CBV(b0)` only.
    const VERTEX_REFLECTION: &str = r#"{"EntryPoint":"main","ResourceCount":1,"ResourceUsages":0,"ShaderType":"Vertex","TopLevelArgumentBuffer":[{"EltOffset":0,"Name":"","Size":8,"Slot":0,"Space":0,"Type":"CBV"}],"UsedResources":[]}"#;

    fn compute_layout() -> RootSignatureLayout {
        RootSignatureLayout::from_converter_reflection_json(COMPUTE_REFLECTION)
            .expect("observed compute reflection must parse")
            .with_descriptor_table_kinds(&[
                DescriptorTableKind::Texture,
                DescriptorTableKind::Sampler,
            ])
            .expect("two reflected tables take two range kinds")
    }

    fn reflection_with(entries: &str) -> String {
        format!(
            r#"{{"EntryPoint":"main","ShaderType":"Compute","TopLevelArgumentBuffer":[{entries}]}}"#
        )
    }

    #[test]
    fn converter_reflection_layout_matches_the_observed_abi() {
        let layout = compute_layout();
        assert_eq!(layout.stage(), ReflectedStage::Compute);
        assert_eq!(layout.entry_point(), "main");
        assert_eq!(
            layout.argument_buffer_index(),
            TOP_LEVEL_ARGUMENT_BUFFER_INDEX
        );
        assert_eq!(layout.argument_buffer_len(), 48);
        assert_eq!(layout.parameters().len(), 6);

        let expected = [
            (RootParameterKind::Constants, 0usize, 8usize, 0u32),
            (RootParameterKind::ConstantBufferView, 8, 8, 1),
            (RootParameterKind::ShaderResourceView, 16, 8, 0),
            (RootParameterKind::UnorderedAccessView, 24, 8, 0),
            (
                RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
                32,
                8,
                u32::MAX,
            ),
            (
                RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler),
                40,
                8,
                u32::MAX,
            ),
        ];
        for (index, (kind, offset, size, register)) in expected.into_iter().enumerate() {
            let parameter = layout.parameter(index).expect("parameter exists");
            assert_eq!(parameter.kind(), kind, "parameter {index} kind");
            assert_eq!(parameter.argument_offset(), offset, "parameter {index} offset");
            assert_eq!(parameter.size(), size, "parameter {index} size");
            assert_eq!(parameter.register(), register, "parameter {index} register");
            if register == u32::MAX {
                assert_eq!(parameter.space(), u32::MAX);
            } else {
                assert_eq!(parameter.space(), 0);
            }
        }
    }

    #[test]
    fn vertex_reflection_is_a_render_layout() {
        let layout = RootSignatureLayout::from_converter_reflection_json(VERTEX_REFLECTION)
            .expect("observed vertex reflection must parse");
        assert_eq!(layout.stage(), ReflectedStage::Vertex);
        assert_eq!(layout.argument_buffer_len(), 8);
        assert_eq!(layout.parameters().len(), 1);
        assert_eq!(
            layout.parameters()[0].kind(),
            RootParameterKind::ConstantBufferView
        );
        assert!(layout.supports(EncoderKind::Render));
        assert!(!layout.supports(EncoderKind::Compute));
        assert!(compute_layout().supports(EncoderKind::Compute));
        assert!(!compute_layout().supports(EncoderKind::Render));
    }

    #[test]
    fn unspecified_table_kinds_are_attached_in_declaration_order() {
        let layout = RootSignatureLayout::from_converter_reflection_json(COMPUTE_REFLECTION)
            .expect("parse");
        assert!(matches!(
            layout.parameters()[4].kind(),
            RootParameterKind::DescriptorTable(DescriptorTableKind::Unspecified)
        ));
        let attached = layout
            .with_descriptor_table_kinds(&[
                DescriptorTableKind::ShaderResourceView,
                DescriptorTableKind::Sampler,
            ])
            .expect("attach kinds");
        assert_eq!(
            attached.parameters()[4].kind(),
            RootParameterKind::DescriptorTable(DescriptorTableKind::ShaderResourceView)
        );
        assert_eq!(
            attached.parameters()[5].kind(),
            RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler)
        );
    }

    #[test]
    fn table_kind_count_must_match_the_reflected_tables() {
        let layout = RootSignatureLayout::from_converter_reflection_json(COMPUTE_REFLECTION)
            .expect("parse");
        assert!(matches!(
            layout.clone().with_descriptor_table_kinds(&[DescriptorTableKind::Sampler]),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            layout.with_descriptor_table_kinds(&[
                DescriptorTableKind::Sampler,
                DescriptorTableKind::Sampler,
                DescriptorTableKind::Sampler,
            ]),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn malformed_reflection_is_rejected() {
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json("not json"),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(
                r#"{"EntryPoint":"main","ShaderType":"Compute","TopLevelArgumentBuffer":42}"#
            ),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(
                r#"{"EltOffset":0,"Size":8,"Slot":0,"Space":0,"Type":"Texture"}"#
            )),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(
                r#"{"EltOffset":0,"Size":8,"Slot":0,"Space":0}"#
            )),
            Err(MetalError::InvalidArg(_))
        ));
        // Descriptor slot not 8-byte aligned.
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(
                r#"{"EltOffset":4,"Size":8,"Slot":0,"Space":0,"Type":"CBV"}"#
            )),
            Err(MetalError::InvalidArg(_))
        ));
        // Zero-sized and oversized root constants.
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(
                r#"{"EltOffset":0,"Size":0,"Slot":0,"Space":0,"Type":"Constant"}"#
            )),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(&format!(
                r#"{{"EltOffset":0,"Size":{},"Slot":0,"Space":0,"Type":"Constant"}}"#,
                MAX_ROOT_CONSTANT_BYTES + 4
            ))),
            Err(MetalError::InvalidArg(_))
        ));
        // Root constants not a 4-byte multiple.
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(
                r#"{"EltOffset":0,"Size":6,"Slot":0,"Space":0,"Type":"Constant"}"#
            )),
            Err(MetalError::InvalidArg(_))
        ));
        // Overlapping parameters.
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(
                concat!(
                    r#"{"EltOffset":8,"Size":8,"Slot":0,"Space":0,"Type":"CBV"},"#,
                    r#"{"EltOffset":8,"Size":8,"Slot":1,"Space":0,"Type":"SRV"}"#
                )
            )),
            Err(MetalError::InvalidArg(_))
        ));
        // More than 64 root parameters.
        let entries = (0..65)
            .map(|index| {
                format!(
                    r#"{{"EltOffset":{},"Size":8,"Slot":{index},"Space":0,"Type":"CBV"}}"#,
                    index * 8
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(&entries)),
            Err(MetalError::InvalidArg(_))
        ));
        // Trailing junk after the JSON document.
        let mut trailing = reflection_with(
            r#"{"EltOffset":0,"Size":8,"Slot":0,"Space":0,"Type":"CBV"}"#,
        );
        trailing.push_str(" {}");
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&trailing),
            Err(MetalError::InvalidArg(_))
        ));
        // Argument buffer larger than the 64 KiB ceiling.
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(&format!(
                r#"{{"EltOffset":{},"Size":8,"Slot":0,"Space":0,"Type":"CBV"}}"#,
                MAX_ARGUMENT_BUFFER_BYTES
            ))),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn absent_null_or_empty_argument_buffer_is_an_empty_layout() {
        // The converter omits the array on a resource-free stage. That is a
        // valid layout with no parameters, not a malformed reflection.
        for json in [
            r#"{"EntryPoint":"main","ShaderType":"Fragment"}"#,
            r#"{"EntryPoint":"main","ShaderType":"Fragment","TopLevelArgumentBuffer":null}"#,
            r#"{"EntryPoint":"main","ShaderType":"Fragment","TopLevelArgumentBuffer":[]}"#,
        ] {
            let layout = RootSignatureLayout::from_converter_reflection_json(json)
                .expect("a reflection with no parameters is a valid layout");
            assert_eq!(layout.stage(), ReflectedStage::Fragment);
            assert_eq!(layout.entry_point(), "main");
            assert!(layout.parameters().is_empty());
            assert_eq!(layout.argument_buffer_len(), 8);
            assert!(layout.supports(EncoderKind::Render));
        }
    }

    /// The exact shape the converter emitted for the Cycle 14 pixel shader that
    /// the host refused: a root CBV then a sampler table, both reported with the
    /// converter's 24-byte descriptor extent. Field names and values only; no
    /// source bytes.
    const PIXEL_SAMPLER_REFLECTION: &str = r#"{"EntryPoint":"main","Features":0,"FunctionConstants":[],"HasYieldLoop":false,"NeedsFunctionConstants":false,"ResourceCount":1,"ResourceUsages":0,"SamplerIndices":[],"ShaderType":"Fragment","TopLevelArgumentBuffer":[{"EltOffset":0,"Name":"","Size":24,"Slot":0,"Space":0,"Type":"CBV"},{"EltOffset":24,"Name":"","Size":24,"Slot":0,"Space":0,"Type":"Sampler"}],"UnorderedAccessViewIndices":[],"UsedResources":[],"UsesRayQuery":false}"#;

    #[test]
    fn declared_tables_carry_the_reflected_kind_and_slot_count() {
        // The composition shape: a view field, then a `Sampler` table field
        // with the converter's 24-byte descriptor extent. Both the guessed
        // kind and the descriptor slot count reach the binding path.
        let layout =
            RootSignatureLayout::from_converter_reflection_json(PIXEL_SAMPLER_REFLECTION)
                .expect("the observed pixel reflection must parse");
        let tables = layout.declared_tables();
        assert_eq!(tables.len(), 1, "only the sampler field is a table");
        assert_eq!(tables[0].parameter, 1);
        assert_eq!(tables[0].kind, DescriptorTableKind::Sampler);
        assert_eq!(tables[0].field_bytes, 24);
        assert_eq!(tables[0].slots, 1, "one 24-byte converter descriptor");

        // A plain `Table` keeps the kind-less reflected kind, and an 8-byte
        // field still spans one descriptor slot.
        let unspecified = RootSignatureLayout::from_parameters(
            ReflectedStage::Fragment,
            &[RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Unspecified),
                size: 8,
                register: 0,
                space: 0,
            }],
        )
        .expect("one kind-less table");
        let tables = unspecified.declared_tables();
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].kind, DescriptorTableKind::Unspecified);
        assert_eq!(tables[0].field_bytes, 8);
        assert_eq!(tables[0].slots, 1);
    }

    #[test]
    fn observed_pixel_sampler_reflection_is_a_usable_layout() {
        let layout =
            RootSignatureLayout::from_converter_reflection_json(PIXEL_SAMPLER_REFLECTION)
                .expect("the observed pixel reflection must parse");
        assert_eq!(layout.stage(), ReflectedStage::Fragment);
        assert_eq!(layout.parameters().len(), 2);
        assert_eq!(layout.argument_buffer_len(), 48);
        assert_eq!(
            layout.parameters()[0].kind(),
            RootParameterKind::ConstantBufferView
        );
        assert_eq!(layout.parameters()[0].argument_offset(), 0);
        assert_eq!(layout.parameters()[0].size(), 24);
        assert_eq!(
            layout.parameters()[1].kind(),
            RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler)
        );
        assert_eq!(layout.parameters()[1].argument_offset(), 24);
        assert!(layout.supports(EncoderKind::Render));
    }

    #[test]
    fn reflected_type_and_field_names_are_case_insensitive_but_not_invented() {
        let lower = reflection_with(
            r#"{"eltoffset":0,"name":"","size":8,"slot":0,"space":0,"type":"sampler"}"#,
        );
        let layout = RootSignatureLayout::from_converter_reflection_json(&lower)
            .expect("field and type spelling variants are tolerated");
        assert_eq!(
            layout.parameters()[0].kind(),
            RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler)
        );
        // A type the converter never emits stays a refusal.
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(
                r#"{"EltOffset":0,"Size":8,"Slot":0,"Space":0,"Type":"Texture"}"#
            )),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn detailed_refusal_names_the_entry_and_field() {
        let error = RootSignatureLayout::from_converter_reflection_json_detailed(
            &reflection_with(concat!(
                r#"{"EltOffset":0,"Size":8,"Slot":0,"Space":0,"Type":"CBV"},"#,
                r#"{"EltOffset":8,"Size":8,"Slot":1,"Type":"SRV"}"#
            )),
        )
        .expect_err("a missing Space must be refused");
        assert_eq!(error.entry_index(), Some(1));
        assert_eq!(error.field(), "Space");
        assert_eq!(
            error.reason(),
            "TopLevelArgumentBuffer entry has no Space"
        );
        assert_eq!(
            error.to_string(),
            "entry 1 field `Space`: TopLevelArgumentBuffer entry has no Space"
        );
        // The simple entry point keeps the same bounded reason as InvalidArg.
        assert!(matches!(
            RootSignatureLayout::from_converter_reflection_json(&reflection_with(
                r#"{"EltOffset":0,"Size":8,"Slot":0,"Space":0,"Type":"Bogus"}"#
            )),
            Err(MetalError::InvalidArg(
                "converter reflection has an unknown root parameter type"
            ))
        ));
    }

    #[test]
    fn detailed_refusal_names_the_failing_validation_entry() {
        let error = RootSignatureLayout::from_converter_reflection_json_detailed(
            &reflection_with(concat!(
                r#"{"EltOffset":0,"Size":8,"Slot":0,"Space":0,"Type":"CBV"},"#,
                r#"{"EltOffset":4,"Size":8,"Slot":1,"Space":0,"Type":"SRV"}"#
            )),
        )
        .expect_err("a misaligned descriptor must be refused");
        assert_eq!(error.entry_index(), Some(1));
        assert_eq!(error.field(), "EltOffset");
        assert_eq!(
            error.reason(),
            "descriptor argument-buffer offset must be 8-byte aligned"
        );
        assert_eq!(
            error.to_string(),
            "entry 1 field `EltOffset`: descriptor argument-buffer offset must be 8-byte aligned"
        );
    }

    #[test]
    fn detailed_refusal_names_the_top_level_field() {
        let error = RootSignatureLayout::from_converter_reflection_json_detailed(
            r#"{"EntryPoint":"main","TopLevelArgumentBuffer":7}"#,
        )
        .expect_err("a non-array argument buffer must be refused");
        assert_eq!(error.entry_index(), None);
        assert_eq!(error.field(), "TopLevelArgumentBuffer");
        assert_eq!(error.reason(), "TopLevelArgumentBuffer is not an array");
    }

    #[test]
    fn argument_buffer_builder_encodes_the_observed_fields() {
        let layout = compute_layout();
        let mut builder = ArgumentBufferBuilder::new(&layout);
        assert_eq!(builder.bytes().len(), 48);
        assert_eq!(builder.encoded_parameters(), 0);

        builder
            .set_root_constants(0, &[2.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat())
            .expect("root constants");
        builder.set_buffer_address(1, 0x1000).expect("root CBV");
        builder.set_buffer_address(2, 0x2000).expect("root SRV");
        builder.set_buffer_address(3, 0x3008).expect("root UAV");
        builder.set_table_address(4, 0x4000).expect("texture table");
        builder.set_table_address(5, 0x5000).expect("sampler table");
        assert_eq!(builder.encoded_parameters(), 6);

        let bytes = builder.bytes();
        assert_eq!(&bytes[0..4], &2.0f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &1.0f32.to_le_bytes());
        for (offset, address) in [
            (8, 0x1000u64),
            (16, 0x2000),
            (24, 0x3008),
            (32, 0x4000),
            (40, 0x5000),
        ] {
            assert_eq!(
                &bytes[offset..offset + 8],
                &address.to_le_bytes(),
                "address at offset {offset}"
            );
        }
        assert!(builder.is_encoded(0));
        assert!(builder.is_encoded(5));
        assert!(!builder.is_encoded(6));
        // Per-parameter written extents make the draw trace's argument-buffer
        // byte ranges exact: root constants cover their whole block, a
        // descriptor/table covers only its 8-byte address field.
        assert_eq!(builder.written_extent(0), Some((0, 8)));
        assert_eq!(builder.written_extent(1), Some((8, 16)));
        assert_eq!(builder.written_extent(4), Some((32, 40)));
        assert_eq!(builder.written_extent(5), Some((40, 48)));
        assert_eq!(builder.written_span(), Some((0, 48)));
        assert_eq!(builder.written_bytes(), 48);
    }

    #[test]
    fn argument_buffer_builder_rejects_wrong_kinds_and_bad_addresses() {
        let layout = compute_layout();
        let mut builder = ArgumentBufferBuilder::new(&layout);

        // Root constants: empty, not 4-byte sized, larger than the block.
        assert!(matches!(
            builder.set_root_constants(0, &[]),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            builder.set_root_constants(0, &[1, 2, 3]),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            builder.set_root_constants(0, &[0u8; 12]),
            Err(MetalError::InvalidArg(_))
        ));
        // Partial constants zero the remainder deterministically.
        builder.set_root_constants(0, &7u32.to_le_bytes()).expect("partial");
        assert_eq!(&builder.bytes()[0..4], &7u32.to_le_bytes());
        assert_eq!(&builder.bytes()[4..8], &[0u8; 4]);
        assert!(matches!(
            builder.set_root_constants(1, &[0u8; 8]),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            builder.set_root_constants(99, &[0u8; 8]),
            Err(MetalError::InvalidArg(_))
        ));

        // Null and misaligned addresses are refused.
        assert!(matches!(
            builder.set_buffer_address(1, 0),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            builder.set_buffer_address(1, 0x1004),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            builder.set_buffer_address(0, 0x1000),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            builder.set_buffer_address(4, 0x1000),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            builder.set_table_address(1, 0x1000),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            builder.set_table_address(4, 0),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            builder.set_table_address(4, 0x1002),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn set_bytes_root_constants_are_opt_in_only() {
        let layout = compute_layout();
        assert_eq!(layout.root_constants_metal_index(), None);
        let set_bytes = layout
            .clone()
            .with_root_constants_metal_index(6)
            .expect("one constants block can move to setBytes");
        assert_eq!(set_bytes.root_constants_metal_index(), Some(6));
        assert!(matches!(
            set_bytes.clone().with_root_constants_metal_index(31),
            Err(MetalError::InvalidArg(_))
        ));

        let no_constants = RootSignatureLayout::from_converter_reflection_json(VERTEX_REFLECTION)
            .expect("parse");
        assert!(matches!(
            no_constants.with_root_constants_metal_index(6),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn buffer_binding_validates_handle_offset_and_length() {
        let valid = BufferBinding {
            buffer: VmBufferId(1),
            generation: 3,
            address: 0x1_0000,
            offset: 8,
            length: 16,
            registered_length: 64,
        };
        assert_eq!(valid.resolved_address().expect("valid"), 0x1_0008);

        let invalid = [
            BufferBinding {
                buffer: VmBufferId(0),
                ..valid
            },
            BufferBinding { address: 0, ..valid },
            BufferBinding { length: 0, ..valid },
            BufferBinding {
                length: wire::MAX_RESOURCE_LEN + 1,
                ..valid
            },
            BufferBinding { offset: 2, ..valid },
            BufferBinding {
                offset: u64::MAX - 3,
                ..valid
            },
            BufferBinding {
                registered_length: 16,
                ..valid
            },
            BufferBinding {
                address: u64::MAX - 4,
                ..valid
            },
        ];
        for (index, binding) in invalid.into_iter().enumerate() {
            assert!(
                matches!(binding.resolved_address(), Err(MetalError::InvalidArg(_))),
                "invalid binding {index} must be refused"
            );
        }
    }

    #[test]
    fn descriptor_table_extent_must_fit_the_bound_range() {
        let valid = DescriptorTableBinding {
            address: 0x1000,
            length: 48,
            count: 2,
            stride: TEXTURE_DESCRIPTOR_BYTES,
        };
        assert_eq!(valid.extent().expect("exact fit"), 48);
        assert_eq!(
            DescriptorTableBinding {
                count: 2,
                stride: TEXTURE_DESCRIPTOR_BYTES,
                ..valid
            }
            .extent()
            .expect("address table"),
            48
        );

        let invalid = [
            DescriptorTableBinding { address: 0, ..valid },
            DescriptorTableBinding {
                address: 0x1004,
                ..valid
            },
            DescriptorTableBinding { count: 0, ..valid },
            DescriptorTableBinding { stride: 4, ..valid },
            DescriptorTableBinding { stride: 12, ..valid },
            DescriptorTableBinding { count: 3, ..valid },
            DescriptorTableBinding {
                count: 65_537,
                ..valid
            },
            DescriptorTableBinding {
                count: 65_536,
                stride: 1usize << 62,
                length: u64::MAX,
                ..valid
            },
        ];
        for (index, table) in invalid.into_iter().enumerate() {
            assert!(
                matches!(table.extent(), Err(MetalError::InvalidArg(_))),
                "invalid table {index} must be refused"
            );
        }
    }

    #[test]
    fn descriptor_table_builder_encodes_converter_shapes() {
        let mut addresses = DescriptorTableBuilder::new(DescriptorTableKind::ConstantBufferView);
        addresses.push_address(0x1000).expect("first CBV");
        addresses.push_address(0x2000).expect("second CBV");
        assert_eq!(addresses.count(), 2);
        assert_eq!(addresses.bytes().len(), 48);
        assert_eq!(&addresses.bytes()[0..8], &0x1000u64.to_le_bytes());
        assert_eq!(&addresses.bytes()[8..16], &0u64.to_le_bytes());
        assert_eq!(&addresses.bytes()[16..24], &0u64.to_le_bytes());
        assert_eq!(&addresses.bytes()[24..32], &0x2000u64.to_le_bytes());
        assert_eq!(&addresses.bytes()[32..40], &0u64.to_le_bytes());
        assert_eq!(&addresses.bytes()[40..48], &0u64.to_le_bytes());
        let table = addresses.binding(0x9000).expect("binding");
        assert_eq!(table.count, 2);
        assert_eq!(table.stride, 24);
        assert_eq!(table.extent().expect("extent"), 48);

        let mut textures = DescriptorTableBuilder::new(DescriptorTableKind::Texture);
        textures.push_texture(0x3000).expect("texture");
        assert_eq!(textures.bytes().len(), TEXTURE_DESCRIPTOR_BYTES);
        assert_eq!(&textures.bytes()[0..8], &0x3000u64.to_le_bytes());
        assert_eq!(&textures.bytes()[8..16], &0x3000u64.to_le_bytes());
        assert_eq!(&textures.bytes()[16..24], &0u64.to_le_bytes());

        let mut samplers = DescriptorTableBuilder::new(DescriptorTableKind::Sampler);
        samplers.push_sampler(0x4000).expect("sampler");
        assert_eq!(samplers.bytes().len(), SAMPLER_DESCRIPTOR_BYTES);
        assert_eq!(&samplers.bytes()[0..8], &0x4000u64.to_le_bytes());
        assert_eq!(&samplers.bytes()[8..16], &0u64.to_le_bytes());
        assert_eq!(&samplers.bytes()[16..24], &0u64.to_le_bytes());

        // Wrong kind, empty table and bad handles are refused.
        let mut wrong = DescriptorTableBuilder::new(DescriptorTableKind::Sampler);
        assert!(matches!(
            wrong.push_address(0x1000),
            Err(MetalError::InvalidArg(_))
        ));
        let mut wrong = DescriptorTableBuilder::new(DescriptorTableKind::ConstantBufferView);
        assert!(matches!(
            wrong.push_texture(0x1000),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            wrong.push_sampler(0x1000),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            DescriptorTableBuilder::new(DescriptorTableKind::Texture).push_texture(0),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            DescriptorTableBuilder::new(DescriptorTableKind::Sampler).push_sampler(0),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            DescriptorTableBuilder::new(DescriptorTableKind::Sampler).binding(0x1000),
            Err(MetalError::InvalidArg(_))
        ));
    }

    /// `gpuResourceID` is an opaque 64-bit id, not a device address: its low
    /// bits carry no alignment guarantee, so an unaligned non-zero id is a
    /// valid texture/sampler descriptor.
    #[test]
    fn texture_and_sampler_resource_ids_need_not_be_address_aligned() {
        let mut textures = DescriptorTableBuilder::new(DescriptorTableKind::Texture);
        textures.push_texture(0x3).expect("an unaligned texture id");
        assert_eq!(textures.count(), 1);
        assert_eq!(&textures.bytes()[0..8], &0x3u64.to_le_bytes());

        let mut samplers = DescriptorTableBuilder::new(DescriptorTableKind::Sampler);
        samplers.push_sampler(0x5).expect("an unaligned sampler id");
        assert_eq!(samplers.count(), 1);
        assert_eq!(&samplers.bytes()[0..8], &0x5u64.to_le_bytes());
    }

    #[test]
    fn layout_from_parameters_flattens_in_declaration_order() {
        let layout = RootSignatureLayout::from_parameters(
            ReflectedStage::Compute,
            &[
                RootParameterSpec {
                    kind: RootParameterKind::Constants,
                    size: 16,
                    register: 0,
                    space: 0,
                },
                RootParameterSpec {
                    kind: RootParameterKind::ConstantBufferView,
                    size: 8,
                    register: 1,
                    space: 0,
                },
                RootParameterSpec {
                    kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler),
                    size: 8,
                    register: 0,
                    space: 0,
                },
            ],
        )
        .expect("declared layout");
        assert_eq!(layout.argument_buffer_len(), 32);
        let offsets: Vec<usize> = layout
            .parameters()
            .iter()
            .map(|parameter| parameter.argument_offset())
            .collect();
        assert_eq!(offsets, vec![0, 16, 24]);
        assert_eq!(layout.parameters()[0].size(), 16);
        assert_eq!(layout.parameters()[1].size(), 8);

        assert!(matches!(
            RootSignatureLayout::from_parameters(
                ReflectedStage::Compute,
                &[RootParameterSpec {
                    kind: RootParameterKind::ConstantBufferView,
                    size: 16,
                    register: 0,
                    space: 0,
                }],
            ),
            Err(MetalError::InvalidArg(_))
        ));
        let too_many = vec![
            RootParameterSpec {
                kind: RootParameterKind::ConstantBufferView,
                size: 8,
                register: 0,
                space: 0,
            };
            65
        ];
        assert!(matches!(
            RootSignatureLayout::from_parameters(ReflectedStage::Compute, &too_many),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn empty_layout_uses_the_minimum_argument_buffer() {
        let layout = RootSignatureLayout::from_parameters(ReflectedStage::Other, &[])
            .expect("empty layout");
        assert_eq!(layout.argument_buffer_len(), 8);
        assert!(layout.supports(EncoderKind::Render));
        assert!(layout.supports(EncoderKind::Compute));
        let builder = ArgumentBufferBuilder::new(&layout);
        assert_eq!(builder.bytes().len(), 8);
        assert_eq!(builder.encoded_parameters(), 0);
    }

    /// A root signature layout whose parameters are distinct in `(space,
    /// register, kind)`: user-data constants, a root CBV, a root UAV and the
    /// two table kinds a title memory/table signature declares.
    fn lookup_layout() -> RootSignatureLayout {
        RootSignatureLayout::from_parameters(
            ReflectedStage::Compute,
            &[
                RootParameterSpec {
                    kind: RootParameterKind::Constants,
                    size: 8,
                    register: 0,
                    space: 0,
                },
                RootParameterSpec {
                    kind: RootParameterKind::ConstantBufferView,
                    size: 8,
                    register: 0,
                    space: 1,
                },
                RootParameterSpec {
                    kind: RootParameterKind::UnorderedAccessView,
                    size: 8,
                    register: 0,
                    space: 2,
                },
                RootParameterSpec {
                    kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
                    size: 8,
                    register: 0,
                    space: 0,
                },
                RootParameterSpec {
                    kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler),
                    size: 8,
                    register: 0,
                    space: 0,
                },
            ],
        )
        .expect("lookup layout")
    }

    #[test]
    fn parameter_by_resolves_exact_space_register_and_kind() {
        let layout = lookup_layout();
        assert_eq!(
            layout
                .parameter_by(0, 0, RootParameterKind::Constants)
                .expect("unique constants"),
            Some(0)
        );
        assert_eq!(
            layout
                .parameter_by(1, 0, RootParameterKind::ConstantBufferView)
                .expect("unique CBV"),
            Some(1)
        );
        assert_eq!(
            layout
                .parameter_by(2, 0, RootParameterKind::UnorderedAccessView)
                .expect("unique UAV"),
            Some(2)
        );
        assert_eq!(
            layout
                .parameter_by(
                    0,
                    0,
                    RootParameterKind::DescriptorTable(DescriptorTableKind::Texture)
                )
                .expect("texture table"),
            Some(3)
        );
        assert_eq!(
            layout
                .parameter_by(
                    0,
                    0,
                    RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler)
                )
                .expect("sampler table"),
            Some(4)
        );
        // Converter reflection cannot report a table's range kind: with one
        // table of that space/register the unspecified kind resolves it.
        let one_table = RootSignatureLayout::from_parameters(
            ReflectedStage::Compute,
            &[RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
                size: 8,
                register: 0,
                space: 0,
            }],
        )
        .expect("one table");
        assert_eq!(
            one_table
                .parameter_by(
                    0,
                    0,
                    RootParameterKind::DescriptorTable(DescriptorTableKind::Unspecified)
                )
                .expect("unspecified table kind"),
            Some(0)
        );
        // Two tables at the same space/register with different kinds make an
        // unspecified lookup ambiguous; the first flattened declaration wins
        // deterministically instead of refusing the whole pipeline.
        assert_eq!(
            layout
                .parameter_by(
                    0,
                    0,
                    RootParameterKind::DescriptorTable(DescriptorTableKind::Unspecified)
                )
                .expect("ambiguous table kind resolves deterministically"),
            Some(3)
        );
        // Converter reflection gives a table `u32::MAX` slot/space; the root
        // signature supplies the concrete pair, so the lookup must still match.
        let reflected_tables = compute_layout();
        assert_eq!(
            reflected_tables
                .parameter_by(
                    3,
                    7,
                    RootParameterKind::DescriptorTable(DescriptorTableKind::Texture)
                )
                .expect("reflected texture table"),
            Some(4)
        );
        assert_eq!(
            reflected_tables
                .parameter_by(
                    3,
                    9,
                    RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler)
                )
                .expect("reflected sampler table"),
            Some(5)
        );
    }

    #[test]
    fn parameter_by_reports_a_missing_argument() {
        let layout = lookup_layout();
        assert_eq!(
            layout
                .parameter_by(3, 0, RootParameterKind::Constants)
                .expect("no space 3 parameter"),
            None
        );
        assert_eq!(
            layout
                .parameter_by(0, 7, RootParameterKind::Constants)
                .expect("no b7 constant"),
            None
        );
        assert_eq!(
            layout
                .parameter_by(0, 0, RootParameterKind::ShaderResourceView)
                .expect("no SRV at b0"),
            None
        );
    }

    #[test]
    fn parameter_by_takes_the_first_of_a_repeated_declaration() {
        let layout = RootSignatureLayout::from_parameters(
            ReflectedStage::Compute,
            &[
                RootParameterSpec {
                    kind: RootParameterKind::Constants,
                    size: 4,
                    register: 0,
                    space: 0,
                },
                RootParameterSpec {
                    kind: RootParameterKind::Constants,
                    size: 4,
                    register: 0,
                    space: 0,
                },
            ],
        )
        .expect("the flattening layer allows repeated declarations");
        // Deterministic first match; refusing would drop a valid pipeline (the
        // Synthetic pre-boot gate's multi-entry signature repeats a parameter).
        assert_eq!(
            layout
                .parameter_by(0, 0, RootParameterKind::Constants)
                .expect("a repeated declaration resolves"),
            Some(0)
        );
    }

    #[test]
    fn root_parameter_map_parses_a_versioned_blob() {
        // A minimal version 1.1 blob: header, one parameter array entry, one
        // constants record at `b0, space 2` with two dwords.
        let mut blob = vec![0u8; 24 + 12 + 12];
        let put = |blob: &mut Vec<u8>, at: usize, value: u32| {
            blob[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };
        put(&mut blob, 0, 2); // version 1.1
        put(&mut blob, 4, 1); // one parameter
        put(&mut blob, 8, 24); // array offset
        put(&mut blob, 12, 0); // no static samplers
        put(&mut blob, 16, 36); // sampler offset (unused)
        put(&mut blob, 20, 0); // flags
        put(&mut blob, 24, 1); // root constants
        put(&mut blob, 28, 0); // visibility all
        put(&mut blob, 32, 36); // record offset
        put(&mut blob, 36, 0); // register b0
        put(&mut blob, 40, 2); // space 2
        put(&mut blob, 44, 2); // two 32-bit values
        let map = super::RootParameterMap::from_versioned_blob(&blob).expect("parse");
        assert_eq!(map.len(), 1);
        let slot = map.slot(0).expect("slot 0");
        assert_eq!(slot.space(), 2);
        assert_eq!(slot.register(), 0);
        assert_eq!(slot.kind(), RootParameterKind::Constants);
        assert_eq!(slot.size(), 8);
        assert!(map.slot(1).is_none());
    }

    #[test]
    fn root_parameter_map_refuses_truncated_and_unknown_blobs() {
        assert!(matches!(
            super::RootParameterMap::from_versioned_blob(&[]),
            Err(MetalError::InvalidArg(_))
        ));
        let mut blob = vec![0u8; 24];
        blob[0..4].copy_from_slice(&9u32.to_le_bytes());
        assert!(matches!(
            super::RootParameterMap::from_versioned_blob(&blob),
            Err(MetalError::InvalidArg(_))
        ));
        let mut blob = vec![0u8; 24];
        blob[0..4].copy_from_slice(&2u32.to_le_bytes());
        blob[4..8].copy_from_slice(&1u32.to_le_bytes());
        blob[8..12].copy_from_slice(&24u32.to_le_bytes());
        assert!(matches!(
            super::RootParameterMap::from_versioned_blob(&blob),
            Err(MetalError::InvalidArg(_))
        ));
    }
    // ------------------------------------------------------------------
    // Hardware proof: IRConverter -> Metal argument buffer -> GPU results
    // ------------------------------------------------------------------

    /// Synthetic compute DXIL (no title payload): root constants `b0`,
    /// root `CBV(b1)`, root `SRV(t0)`, root `UAV(u0)`,
    /// `DescriptorTable(SRV(t1))` and `DescriptorTable(Sampler(s0))`.
    /// `gOut[i] = gConsts.scale*gIn[i] + gConsts.offset + gParams.scale + tex.x`
    const COMPUTE_DXIL_HEX: &str = "4458424302fcdc555175b0a57050d48fa1395f2701000000fc11000008000000400000005000000060000000700000005801000030020000280a0000440a000053464930080000000000000000000000495347310800000000000000080000004f53473108000000000000000800000050535630e0000000340000000000000000000000000000000000000000000000ffffffff050000000000000000000000400000000100000001000000010000000600000018000000020000000000000000000000000000000d00000000000000020000000000000001000000010000000d00000000000000010000000000000000000000000000000e00000000000000050000000000000000000000000000000c00000000000000030000000000000001000000010000000200000000000000080000000000000000000000000000000c0000000000000008000000006d61696e0000000000000052545330d000000002000000060000001800000000000000d00000000000000001000000000000006000000002000000000000006c0000000300000000000000780000000400000000000000840000000000000000000000900000000000000000000000b000000000000000000000000200000001000000000000000000000000000000000000000000000000000000000000000000000001000000980000000000000001000000010000000000000000000000ffffffff01000000b80000000300000001000000000000000000000000000000ffffffff53544154f007000060000500fc0100004458494c0001000010000000d80700004243c0de210c0000f30100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628018450242920b42c41032143808184b0a32628848901420434688a500193242e4480e901123c4504151818ce183e58a0431460651180000080000001b8ce0ffffffff074002a80d86f0ffffffff032001d50662f8ffffffff019000491800000300000013826042204c080600000000892000007300000032228809206485041323a484041323e384a19014124c8c8c0b84c44c10a4c1084009000ae608c0a00cc33010310370d370f913f61092bf12d24a4c7e71dba8300cc318e6081042ee192e7fc21e42f243a01916020549618e41510cc33086613068396ab8fc097b08c9e736aa5889c92f6e1b118661188568066520678e2028863220c300515414605086c1300c03a1a914c7a00c035503013375e3c00ee1300ff3e006b3400ff2500fe3400ff5200fe5400ea2500fe6600ee5200f7c600eecf00ee1400f7e800283b06104c298a90dc6811dc2611ee6c10d68a11cf0811eea411eca410e48810feca11cc6811ede411ef8c01cd8e11dc2811ed8000ce8c00fc0c00ff4400fda211de0611e7e811ef2011eca0105c34c62300eec100ef3300f6e400be5800ff4500ff2500e72400a7c600fe5300ef4f00ef2c007e6c00eef100ef4c0066040077e00067e80046d236ea66f1cd8211ce6611edc40166e6116e8411eea611ce8a11ee4a11cc84114eac11ccca11ce4810fcc811dde211ce8c10f506090778d34459430f99cd348938404c3b8c3e09c469a241480145e234d11254c7e0a44002321c130ee305220021809052391374953440993cf02ccb310113b0113818281cc39025000000000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0008000000000000000000863c04100001000000000000000c79162000040000000000000018f2384000080000000000000030e4898000100000000000000060c83301013000000000000000c090a702024000000000000000802c1016000000321e981819114c908c092647c604431a4a6004a0180aa3108aa0240aa20ccaa100038a85d00221680480b01900ca6600a89b01206f0680be1900026700289c0120710680c61900226700a89c01a07306808c19000000000079180000980000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a50604052646cc46268c8ce6266543104c1086638230201b8481d820100485b1b909c2906c180e849820641687b324b709c2a04c1086658330341b124259086260086782d05d24cea8caf02608033341189a0d02316d48062822864122a80dc2534d10be8cc4d9531ddd0401a2362cc48511c4c06459e66c08b4098218703cce86dedce6e8e6260883b30121b88e20060f982094c1c7e30c2a4c2eac6d6e033280411810c3e0011b844f0c260867000624cea6dae0362004199401410c04b02130830d84b58dc1194c10c04063313716c6563641189e09c2006d3010355803820d261a6f66667365741b0cc40dd6a0618369c3e0b5c11b4c10c6a09b20601503b40d0632076b40b0c1344130c2608340d4c186c00e3608c41d6c181a3ac0830982874d10c260db4020dd1a101b048f0f260864e06d20903058036283e0fdc1868648033888033890833cd0833cd8833ef00350982008c00660c3408cc2286c084861c33088422990680b4b739b20a0c1344118a20d432a0cc30682408546153614a2700a001aac020d33b6b730ba3916696e73747313844122425786f7c5f616464684ae0cefcbed4dae8d095d19ded71cdd9b5cd906a5151a577805581862619085a10a1b9b5d9b4b1a59991bdd9420a84286e76257263797f6e63625209a90e1b9d885b1d995c94d098a3a64782e7368616465724d6f64656c5302a40c199e8b5cd9dc5b9ddc58d9dc94e00c2a91e1b9d0e5c19505b9b9bdd185d1a5bdb9cd4d1140a114ea90e1b994b9d1c9e541bda5b9d1cd4d0956a10b199ecbd85b9d1b5d99dcdc94401600791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000001d00000066400d97ef3c7e40150511950e30f8c56d5b41355cbef3f801551444c44e4e44f8c56ddbc0365cbef3f84240150511950e3094840108985fdcb61148c3e53b8f2f4404301121d00c0b6102d570f9cee34b00f32c444954c4e217b76d01d270f9cee34f4744008338f8c86d1b00c100480300000000000000484153481400000000000000f71887a09e6db98fcc3c6feb1118946b4458494cb007000060000500ec0100004458494c0001000010000000980700004243c0de210c0000e30100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628018450242920b42c41032143808184b0a32628848901420434688a500193242e4480e901123c4504151818ce183e58a0431460651180000080000001b8ce0ffffffff074002a80d86f0ffffffff032001d50662f8ffffffff019000491800000300000013826042204c080600000000892000007300000032228809206485041323a484041323e384a19014124c8c8c0b84c44c10a4c1084009000ae608c0a00cc33010310370d370f913f61092bf12d24a4c7e71dba8300cc318e6081042ee192e7fc21e42f243a01916020549618e41510cc33086613068396ab8fc097b08c9e736aa5889c92f6e1b118661188568066520678e2028863220c300515414605086c1300c03a1a914c7a00c035503013375e3c00ee1300ff3e006b3400ff2500fe3400ff5200fe5400ea2500fe6600ee5200f7c600eecf00ee1400f7e800283b06104c298a90dc6811dc2611ee6c10d68a11cf0811eea411eca410e48810feca11cc6811ede411ef8c01cd8e11dc2811ed8000ce8c00fc0c00ff4400fda211de0611e7e811ef2011eca0105c34c62300eec100ef3300f6e400be5800ff4500ff2500e72400a7c600fe5300ef4f00ef2c007e6c00eef100ef4c0066040077e00067e80046d236ea66f1cd8211ce6611edc40166e6116e8411eea611ce8a11ee4a11cc84114eac11ccca11ce4810fcc811dde211ce8c10f506090778d34459430f99cd348938404c3b8c3e09c469a241480145e234d11254c7e0a44002321c130ee305220021809052391374953440993cf02ccb310113b0113818281cc39025000000000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0000000000000000000000863c04100001000000000000000c79162000040000000000000018f2384000080000000000000030e4898000100000000000000060c83301013000000000000000c090a702024000000000000000802c100f000000321e981419114c908c092647c604431a4a6004a0248aa1300aa1080aa20003081a0120b440a89b01a06f0680c419002a6700e89c01a06c0600000000791800004f0000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a50604052646cc46268c8ce6266543104c1086638230201b848198200cc9066130288ccd4d100665c38024c40421bb084c1086658230301b84c1d990100b431043433c13844e9a200ccd068130362443c410c32011d30601a22608df344180a00d0b6131043134d7753d1b026c822006d4046170362084c610c4b001138432a8362043c710c3b0011b04ce9b209c81b50121c08021888100360461b081a8b24f0c26080240a22d2ccd6d828006d1046178360c67300c1b08c20c1c34d8509041190063900655d8d8ecda5cd2c8cadce8a6044115323c17bb32b9b9b437b72901d1840ccfc52e8ccdae4c6e4a60d421c37399430b232b936b7a232b639b122465c8f05ce4cae6deeae4c6cae6a6046250870ccfa5cc8d4e2e0fea2dcd8d6e6e4a90060000791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000001d00000066400d97ef3c7e40150511950e30f8c56d5b41355cbef3f801551444c44e4e44f8c56ddbc0365cbef3f84240150511950e3094840108985fdcb61148c3e53b8f2f4404301121d00c0b6102d570f9cee34b00f32c444954c4e217b76d01d270f9cee34f4744008338f8c86d1b00c1004803000000612000003e0000001304412c100000000900000034945cd9956e40f1cd0014624029061052064530025002648c008c11802008c2df0c000023060900826010898143789e35629000200806d1183c061880c13562900020080611194007188001366290002008065119444a18844136629000200806911948ca188c8136629000200806d1194ccb188cc136626000200806c41a48643062f000200806cb1a48069124085206651029a30901306270002008060e1b58c4198c2604c0880102802018546b4025451a8c26048005837c46138a60c4e00040100c1c39e09636184d08001b12f85850c0c782013e23060d00826030d50147397210545585210000000000000000";

    /// Synthetic vertex DXIL: `CBV(b0)`, colour `(1, 0, gParams.scale, 1)`.
    const VERTEX_DXIL_HEX: &str = "445842436ae585095d9f9c8d12d274b9f7a0bf89010000003c0e00000800000040000000500000008c000000f0000000ac010000e4010000ec070000080800005346493008000000000000000000000049534731340000000100000008000000000000002800000000000000060000000100000000000000010100000000000053565f5665727465784944004f5347315c00000002000000080000000000000048000000000000000100000003000000000000000f000000000000000000000054000000000000000000000003000000010000000f0000000000000053565f506f736974696f6e00434f4c4f5200000050535630b4000000340000000100000000000000000000000000000000000000ffffffff010000000102000102000000000000000000000000000000070000000100000018000000020000000000000000000000000000000d000000000000000c00000000434f4c4f52006d61696e00010000000000000010000000000000000000000001004101010000000000000000000000010044030304000001000000000000000101440003020000030000000000000000000000000000005254533030000000020000000100000018000000000000003000000000000000020000000000000024000000000000000000000000000000535441540006000060000100800100004458494c0001000010000000e80500004243c0de210c0000770100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628014450242920b42a41032143808184b0a32528848901420434688a500193242e4480e909122c4504151818ce183e58a0429460651180000080000001b8ce0ffffffff074002a80d84f0ffffffff03206d3086ffffffff1f0009a800491800000300000013826042204c080600000000892000002200000032224809206485049322a484049322e384a19014124c8a8c0b84a44c105c2300250014e608c0608e0029c6186390416506a018608c410ea1a386cb9fb087907c6ea38a9598fce2b61171ce3944ee192e7fc21e42f243a019160205ab106ab0416d8e2028061b648c477020e01a698a2861f2532002180909e7dc61a4400430120a26d11c18000000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0008000000000000000000863c05100001000000000000000c79102000040000000000000018f23440000c0000000000000030e48180001800000000000000200b04000010000000321e981419114c908c092647c604431a253002500c055106e5501285501e255204444aa10446004aa20c0aa10868ce00109d01a03a034076ac2140060000000079180000820000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a50604052646cc46268c8ce6266543104c1086628230181b8481d820100405bbb909c2706c180e84982054198f33a830b9b0b6b909c2806c400865218881013604cd0602001c6082305d2ce6c6c2d8ca2608433241189409c2b06c30904822268ac69b99d95c19dd046160361888255d13b561602a6c8240611b086491880d02b36d280828d3b80982006c00360cc4f76d08c060c33078613041b0b40dc11890680b4b73e33265f56555264757869744344120a00902116d08880902216d10a461c34298c119a0411aa8c18006841aacc186800db84c597d41bdcda5d1a5bdb94d1088698240501304a29a200ccd064192830d0be1066f0007691007431c106a30072c869e989ea4260884b56119eae00dd4200dec608883410de66083400777b06168033c0036145e19e4c103b048739ba39b9b200c0e893437bab909c2f0d0984b3bfb6223a3319776f63547b701d9033ee8033ff8830114982a6c6c766d2e6964656e745382a00a199e8b5d99dc5cda9bdb9480684286e76217c6665726372528ea90e1b9cca1859195c935bd9195b14d09903264782e7265736f75726365735302a712199e0b5d1e5c59909bdb1b5d185dda9bdbdc14810b833a64782e7669657749645374617465538231a84386e752e646279707f596e646373725c8832e64782e636f756e7465727353025000791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000001400000036b00d97ef3cbe1050454144a5030c25610002e617b76d04d270f9cee30b11014c440834c34258c0345cbef3f88b030c62f350938fdcb60954c3e53b8f2f4d4e44a0d4f450935fdcb601100c803400000000000048415348140000000000000051aa5dc6e0e08e98b053cbc3977fe2cb4458494c2c060000600001008b0100004458494c0001000010000000140600004243c0de210c0000820100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628014450242920b42a41032143808184b0a32528848901420434688a500193242e4480e909122c4504151818ce183e58a0429460651180000080000001b8ce0ffffffff074002a80d84f0ffffffff03206d3086ffffffff1f0009a800491800000300000013826042204c080600000000892000002200000032224809206485049322a484049322e384a19014124c8a8c0b84a44c105c2300250014e608c0608e0029c6186390416506a018608c410ea1a386cb9fb087907c6ea38a9598fce2b61171ce3944ee192e7fc21e42f243a019160205ab106ab0416d8e2028061b648c477020e01a698a2861f2532002180909e7dc61a4400430120a26d11c18000000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0000000000000000000000863c05100001000000000000000c79102000040000000000000018f23440000c0000000000000030e48180001800000000000000200b0400000d000000321e981419114c908c092647c604431a2530025012c5501065501e444aa10446004aa20c0aa108a8ce00901d6b0890010000000079180000560000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a50604052646cc46268c8ce6266543104c1086628230181b848198200cc706613028d8cd4d100664c38024c404a19a084c10866403422c0c410c0db02170361000f00013048bda104413040120d11696e6c665caeacbaa4c8eae0c2f89688240301304a2d910101304c299200cca06411b362c44655d58365c44b66d08382e53565f506f736974696f6e1304e2992010d00481882608c3b241d0c460c342781f186061308401918d018ba127a627a90902216d588632f832cc0c863018b231d82090c1196c183a340036141395061050858dcdaecd258daccc8d6e4a105421c373b12b939b4b7b739b12104dc8f05cecc2d8eccae4a604461d323c9739b430b232b9a637b232b6294152860ccf45ae6ceead4e6eac6c6e4af0d421c373b14b2bbb4b229ba20ba32b9b124475c8f05ccadce8e4f2a0ded2dce8e6a60469000000791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000001400000036b00d97ef3cbe1050454144a5030c25610002e617b76d04d270f9cee30b11014c440834c34258c0345cbef3f88b030c62f350938fdcb60954c3e53b8f2f4d4e44a0d4f450935fdcb601100c8034000061200000380000001304412c100000000a000000344aa1e40a6106a008ca8ede08009d3102100441120cc608401004f15f18011823004110c43f000023060900826000718aa369c58841028020181460a06cdb940c370409184c371c4630dc307c6030dd9020c188c1018020183463c01860309a10002306090082608098c1240662a0152306090082608098c1240662b00d2306090082608098c124066260292306090082608098c124066280252306090082608098c1340662a0252306090082608098c1340662b0292306090082608098c134066260052306090082608098c13406628025080000000000";

    /// Synthetic pass-through pixel DXIL with an empty root signature.
    const PIXEL_DXIL_HEX: &str = "44584243095444da7cbad7a6210ea02ce0dfca4d01000000a80b0000080000004000000050000000b4000000f0000000a0010000c001000094060000b006000053464930080000000000000000000000495347315c00000002000000080000000000000048000000000000000100000003000000000000000f000000000000000000000054000000000000000000000003000000010000000f0f00000000000053565f506f736974696f6e00434f4c4f520000004f5347313400000001000000080000000000000028000000000000004000000003000000000000000f0000000000000053565f54617267657400000050535630a8000000340000000000000000000000000000000000000000000000ffffffff00000000020100020100000000000000000000000000000007000000000000000c00000000434f4c4f52006d61696e000100000000000000100000000000000000000000010044030304000001000000000000000101440003020000000000000000000001004410030000000000000000000000000000000000000001000000020000000400000008000000525453301800000002000000000000001800000000000000180000000000000053544154cc04000060000000330100004458494c0001000010000000b40400004243c0de210c00002a0100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628010450242920b42841032143808184b0a32428848901420434688a500193242e4480e901122c4504151818ce183e58a0421460651180000060000001b8ce0ffffffff074002a80d84f0ffffffff03200100000049180000020000001382604220000000892000000f00000032220809206485041322a484041322e384a19014124c888c0b84844c1030230025008a19803902309823408a31445444560c20a21ac2818054200000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0008000000000000000000863c06100001000000000000000c791020000400000000000000c802010d000000321e981419114c908c092647c60443a2121801288632288f922804aa922883421801288202a11d4b410800008040201000000000791800005e0000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a50604052646cc46268c8ce6266543104c1088618240101b8481d820100405b8b90902516c180e84982008c00660c3402ccb8680d9300c4a334158a00dc143a22d2ccd8dcb94d517d4db5c1a5dda9bdb04a1502608c5b221202608053341289a0d0b214d54650d1671012c869e989ea42608853341208c0902716c10366ec33264d3556983355cdd0601f3984c597d5185c99d95d14d108a67c34280c11406d53558c4d56d08c460c3f08d01b0a1502232000016696e737473130402a1319776f6c546364120121a7369675f73741b0c3338033448033548832a6c6c766d2e6964656e745382a00a199e8b5d99dc5cda9bdb9480684286e76217c6665726372528ea90e1b9cca1859195c935bd9195b14d09904a64782e74797065416e6e6f746174696f6e735382a60e199e8b5d5ad95d12d9145d185dd994e0a94386e752e646279707f596e64637372520832e64782e636f756e7465727353023500000000791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000000b00000016300d97ef3cfee20083d83cd4e417b76d02d570f9cee34b93131128353dd4e417b76d000403200d0000000000000000484153481400000000000000e2bbed3f2ec0c786d9a22420425304524458494cf0040000600000003c0100004458494c0001000010000000d80400004243c0de210c0000330100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628010450242920b42841032143808184b0a32428848901420434688a500193242e4480e901122c4504151818ce183e58a0421460651180000060000001b8ce0ffffffff074002a80d84f0ffffffff03200100000049180000020000001382604220000000892000000f00000032220809206485041322a484041322e384a19014124c888c0b84844c1030230025008a19803902309823408a31445444560c20a21ac2818054200000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0000000000000000000000863c06100001000000000000000c791020000400000000000000c802010c000000321e981419114c908c092647c60443a21218012889622883f2a02a893228841180222810dab11484000000080402010079180000480000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a50604052646cc46268c8ce6266543104c1088618240101b8481982010c506613028c0cd4d100863c38024c40461793604cb04410048b485a5b97199b2fa827a9b4ba34b7b739b2014c904a1503604c404a158260805b361211e2892a661222880c5d013d393d404a1682608c4314120900d02966d58060ba2a46b98064adb20541b9329ab2faa30b9b332ba0942e16c58880ef2246a98084adb107c1b060e0c800d45e3840100546163b36b7349232b73a39b120455c8f05ceccae4e6d2dedca6044413323c17bb3036bb32b9298151870ccf650e2d8cac4caee98dac8c6d4a90d421c373b14b2bbb4b229ba20ba32b9b122c75c8f05ccadce8e4f2a0ded2dce8e6a60461000000791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000000b00000016300d97ef3cfee20083d83cd4e417b76d02d570f9cee34b93131128353dd4e417b76d000403200d00000000612000001e0000001304412c100000000300000044853003500a542500000000230609008260604c440421c388410280201818542145023162900020080646654c52528c182400088281611dd4d4182306090082608058064531c488410280201820964151c530629000200806886550d4228c182400088201621914e504080000000000";

    const HOST_PAGE_BYTES: usize = 16 * 1024;
    const PIPELINE_COLOR_FORMAT_RGBA8_UNORM: usize = 70;
    const PIPELINE_COLOR_FORMAT_RGBA32_FLOAT: usize = 125;

    fn hex_bytes(hex: &str) -> Vec<u8> {
        let digits: Vec<u8> = hex.bytes().filter(|byte| !byte.is_ascii_whitespace()).collect();
        assert!(digits.len() % 2 == 0, "hex literal has an odd length");
        digits
            .chunks_exact(2)
            .map(|pair| {
                let high = char::from(pair[0]).to_digit(16).expect("hex digit");
                let low = char::from(pair[1]).to_digit(16).expect("hex digit");
                ((high << 4) | low) as u8
            })
            .collect()
    }

    fn converter_path() -> Option<std::path::PathBuf> {
        std::env::var_os("VICTORY_METAL_IRCONVERTER")
            .map(std::path::PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(crate::macos::irconverter::converter_library_path)
    }

    struct HostPage(*mut c_void);

    impl HostPage {
        fn new() -> Self {
            unsafe extern "C" {
                fn posix_memalign(ptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
            }
            let mut raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: `posix_memalign` writes one 16 KiB-aligned allocation.
            let rc = unsafe { posix_memalign(&mut raw, HOST_PAGE_BYTES, HOST_PAGE_BYTES) };
            assert_eq!(rc, 0, "posix_memalign failed");
            assert!(!raw.is_null());
            // SAFETY: `raw` is a fresh 16 KiB allocation owned by this guard.
            unsafe { std::ptr::write_bytes(raw.cast::<u8>(), 0, HOST_PAGE_BYTES) };
            Self(raw)
        }

        fn as_ptr(&self) -> *mut u8 {
            self.0.cast()
        }
    }

    impl Drop for HostPage {
        fn drop(&mut self) {
            unsafe extern "C" {
                fn free(ptr: *mut c_void);
            }
            // SAFETY: `self.0` came from `posix_memalign` and was unregistered.
            unsafe { free(self.0) };
        }
    }

    #[repr(C)]
    struct MTLOrigin {
        x: usize,
        y: usize,
        z: usize,
    }

    #[repr(C)]
    struct MTLRegion {
        origin: MTLOrigin,
        size: objc::MTLSize,
    }

    type MsgSend = unsafe extern "C" fn();

    unsafe fn send_fn<T: Copy>() -> T {
        // SAFETY: Caller picks a `T` matching the selector ABI it calls.
        unsafe { std::mem::transmute_copy(&(super::objc_msgSend as MsgSend)) }
    }

    unsafe fn send_id_usize(target: objc::Id, sel: objc::Sel, value: usize) -> objc::Id {
        // SAFETY: `target` is a live Objective-C object and the selector takes
        // one `NSUInteger` and returns an object pointer.
        unsafe {
            let function: unsafe extern "C" fn(objc::Id, objc::Sel, usize) -> objc::Id =
                send_fn();
            function(target, sel, value)
        }
    }

    unsafe fn send_void_usize(target: objc::Id, sel: objc::Sel, value: usize) {
        // SAFETY: `target` is a live Objective-C object and the selector takes
        // one `NSUInteger` and returns void.
        unsafe {
            let function: unsafe extern "C" fn(objc::Id, objc::Sel, usize) = send_fn();
            function(target, sel, value);
        }
    }

    unsafe fn send_class_texture_descriptor(
        class: objc::Class,
        sel: objc::Sel,
        format: usize,
        width: usize,
        height: usize,
    ) -> objc::Id {
        // SAFETY: `class` is `MTLTextureDescriptor` and the selector takes
        // `(pixelFormat, width, height, mipmapped)`.
        unsafe {
            let function: unsafe extern "C" fn(
                objc::Class,
                objc::Sel,
                usize,
                usize,
                usize,
                u8,
            ) -> objc::Id = send_fn();
            function(class, sel, format, width, height, 0)
        }
    }

    unsafe fn send_draw_primitives(
        encoder: objc::Id,
        primitive: usize,
        start: usize,
        count: usize,
    ) {
        // SAFETY: `encoder` is a live render encoder.
        unsafe {
            let function: unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize) =
                send_fn();
            function(
                encoder,
                objc::sel(c"drawPrimitives:vertexStart:vertexCount:"),
                primitive,
                start,
                count,
            );
        }
    }

    unsafe fn send_clear_color(encoder: objc::Id, color: [f64; 4]) {
        #[repr(C)]
        struct MTLClearColor {
            red: f64,
            green: f64,
            blue: f64,
            alpha: f64,
        }
        let value = MTLClearColor {
            red: color[0],
            green: color[1],
            blue: color[2],
            alpha: color[3],
        };
        // SAFETY: `encoder` is a live render-pass attachment descriptor.
        unsafe {
            let function: unsafe extern "C" fn(objc::Id, objc::Sel, MTLClearColor) = send_fn();
            function(encoder, objc::sel(c"setClearColor:"), value);
        }
    }

    unsafe fn send_replace_region(
        texture: objc::Id,
        region: MTLRegion,
        bytes: &[u8],
        bytes_per_row: usize,
    ) {
        // SAFETY: `texture` is a live texture, `region` matches its extent and
        // `bytes` is readable for the whole region.
        unsafe {
            let function: unsafe extern "C" fn(
                objc::Id,
                objc::Sel,
                MTLRegion,
                usize,
                *const c_void,
                usize,
            ) = send_fn();
            function(
                texture,
                objc::sel(c"replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                region,
                0,
                bytes.as_ptr().cast(),
                bytes_per_row,
            );
        }
    }

    unsafe fn send_get_bytes(
        texture: objc::Id,
        region: MTLRegion,
        bytes: &mut [u8],
        bytes_per_row: usize,
    ) {
        // SAFETY: `texture` is a live shared-storage texture and `bytes` is
        // writable for the whole region.
        unsafe {
            let function: unsafe extern "C" fn(
                objc::Id,
                objc::Sel,
                *mut c_void,
                usize,
                MTLRegion,
                usize,
            ) = send_fn();
            function(
                texture,
                objc::sel(c"getBytes:bytesPerRow:fromRegion:mipmapLevel:"),
                bytes.as_mut_ptr().cast(),
                bytes_per_row,
                region,
                0,
            );
        }
    }

    fn gpu_address(object: &objc::ObjcObject) -> u64 {
        // SAFETY: `object` is a live `MTLBuffer`.
        unsafe { objc::msg_send_u64_0(object.as_raw(), objc::sel(c"gpuAddress")) }
    }

    fn resource_id(object: &objc::ObjcObject) -> u64 {
        // SAFETY: `object` is a live `MTLTexture` or `MTLSamplerState`.
        unsafe { objc::msg_send_u64_0(object.as_raw(), objc::sel(c"gpuResourceID")) }
    }

    fn new_shared_buffer(device: objc::Id, length: usize) -> objc::ObjcObject {
        // SAFETY: `device` is a live `MTLDevice`; shared storage is CPU-visible.
        let raw = unsafe { objc::msg_send_new_buffer(device, length, 0) };
        objc::ObjcObject::from_retained(raw).expect("shared MTLBuffer allocation")
    }

    fn write_bytes(buffer: &objc::ObjcObject, offset: usize, bytes: &[u8]) {
        // SAFETY: `buffer` is a shared-storage buffer and the caller stays
        // inside its allocation.
        let contents = unsafe { objc::msg_send_ptr_0(buffer.as_raw(), objc::sel(c"contents")) };
        assert!(!contents.is_null());
        // SAFETY: `contents` is CPU-visible and `offset + bytes.len()` is in bounds.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.cast::<u8>().add(offset), bytes.len())
        };
    }

    fn new_library(device: objc::Id, bytes: &[u8]) -> objc::ObjcObject {
        let data = objc::dispatch_data_copy(bytes).expect("dispatch data");
        let mut error = objc::empty_id();
        // SAFETY: `device` is a live `MTLDevice`, `data` is a retained
        // dispatch data object and `error` receives an `NSError *`.
        let raw = unsafe {
            objc::msg_send_id_err(
                device,
                objc::sel(c"newLibraryWithData:error:"),
                data.as_raw(),
                &mut error,
            )
        };
        objc::ObjcObject::from_retained(raw).unwrap_or_else(|| {
            // SAFETY: Metal populated `error` on failure.
            panic!("load metallib: {}", unsafe { objc::nserror_message(error) })
        })
    }

    fn library_function(library: &objc::ObjcObject, name: &str) -> objc::ObjcObject {
        let name = objc::nsstring(name).expect("entry name string");
        // SAFETY: `library` is a live `MTLLibrary` and `name` a live `NSString`.
        let raw = unsafe {
            objc::msg_send_id_1(
                library.as_raw(),
                objc::sel(c"newFunctionWithName:"),
                name.as_raw(),
            )
        };
        objc::ObjcObject::from_retained(raw).expect("entry point function")
    }

    fn new_compute_pipeline(
        device: objc::Id,
        function: &objc::ObjcObject,
    ) -> objc::ObjcObject {
        let mut error = objc::empty_id();
        // SAFETY: `device` is a live `MTLDevice` and `function` a live `MTLFunction`.
        let raw = unsafe {
            objc::msg_send_id_err(
                device,
                objc::sel(c"newComputePipelineStateWithFunction:error:"),
                function.as_raw(),
                &mut error,
            )
        };
        objc::ObjcObject::from_retained(raw).unwrap_or_else(|| {
            // SAFETY: Metal populated `error` on failure.
            panic!("compute pipeline: {}", unsafe { objc::nserror_message(error) })
        })
    }

    fn new_render_pipeline(
        device: objc::Id,
        vertex: &objc::ObjcObject,
        fragment: &objc::ObjcObject,
        format: usize,
    ) -> objc::ObjcObject {
        let class = objc::class(c"MTLRenderPipelineDescriptor");
        // SAFETY: `class` is `MTLRenderPipelineDescriptor`; `new` returns +1.
        let descriptor = objc::ObjcObject::from_retained(unsafe {
            objc::msg_send_id_0(class, objc::sel(c"new"))
        })
        .expect("render pipeline descriptor");
        // SAFETY: All receivers are live Metal objects and the selectors take
        // the documented object/`NSUInteger` arguments.
        unsafe {
            objc::msg_send_void_1(
                descriptor.as_raw(),
                objc::sel(c"setVertexFunction:"),
                vertex.as_raw(),
            );
            objc::msg_send_void_1(
                descriptor.as_raw(),
                objc::sel(c"setFragmentFunction:"),
                fragment.as_raw(),
            );
            let attachments = objc::msg_send_id_0(
                descriptor.as_raw(),
                objc::sel(c"colorAttachments"),
            );
            assert!(!attachments.is_null());
            let attachment = send_id_usize(
                attachments,
                objc::sel(c"objectAtIndexedSubscript:"),
                0,
            );
            assert!(!attachment.is_null());
            send_void_usize(attachment, objc::sel(c"setPixelFormat:"), format);
        }
        let mut error = objc::empty_id();
        // SAFETY: `device` is a live `MTLDevice` and the descriptor is configured.
        let raw = unsafe {
            objc::msg_send_id_err(
                device,
                objc::sel(c"newRenderPipelineStateWithDescriptor:error:"),
                descriptor.as_raw(),
                &mut error,
            )
        };
        objc::ObjcObject::from_retained(raw).unwrap_or_else(|| {
            // SAFETY: Metal populated `error` on failure.
            panic!("render pipeline: {}", unsafe { objc::nserror_message(error) })
        })
    }

    fn new_texture(
        device: objc::Id,
        format: usize,
        width: usize,
        height: usize,
        usage: usize,
    ) -> objc::ObjcObject {
        let class = objc::class(c"MTLTextureDescriptor");
        // SAFETY: `class` is `MTLTextureDescriptor`; the returned descriptor is
        // autoreleased into the active test pool.
        let descriptor = unsafe {
            send_class_texture_descriptor(
                class,
                objc::sel(c"texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
                format,
                width,
                height,
            )
        };
        assert!(!descriptor.is_null());
        // SAFETY: `descriptor` is a live texture descriptor; shared storage (0)
        // keeps the pixels CPU-visible for readback.
        unsafe {
            send_void_usize(descriptor, objc::sel(c"setStorageMode:"), 0);
            send_void_usize(descriptor, objc::sel(c"setUsage:"), usage);
        }
        let raw = unsafe {
            objc::msg_send_id_1(
                device,
                objc::sel(c"newTextureWithDescriptor:"),
                descriptor,
            )
        };
        objc::ObjcObject::from_retained(raw).expect("texture allocation")
    }

    fn new_sampler(device: objc::Id) -> objc::ObjcObject {
        let class = objc::class(c"MTLSamplerDescriptor");
        // SAFETY: `class` is `MTLSamplerDescriptor`; `new` returns +1.
        let descriptor = objc::ObjcObject::from_retained(unsafe {
            objc::msg_send_id_0(class, objc::sel(c"new"))
        })
        .expect("sampler descriptor");
        // SAFETY: Nearest filtering and clamp-to-edge keep the 1x1 sample exact.
        unsafe {
            send_void_usize(descriptor.as_raw(), objc::sel(c"setMinFilter:"), 0);
            send_void_usize(descriptor.as_raw(), objc::sel(c"setMagFilter:"), 0);
            send_void_usize(descriptor.as_raw(), objc::sel(c"setSAddressMode:"), 0);
            send_void_usize(descriptor.as_raw(), objc::sel(c"setTAddressMode:"), 0);
        }
        let raw = unsafe {
            objc::msg_send_id_1(
                device,
                objc::sel(c"newSamplerStateWithDescriptor:"),
                descriptor.as_raw(),
            )
        };
        objc::ObjcObject::from_retained(raw).expect("sampler state")
    }

    fn registered(
        state: &crate::macos::device::DeviceState,
        id: VmBufferId,
    ) -> (objc::Id, u64, usize) {
        let entry = state
            .buffers
            .get(&id.0)
            .unwrap_or_else(|| panic!("buffer {} is not registered", id.0));
        (
            entry.buffer.as_raw(),
            // SAFETY: `entry.buffer` is a live `MTLBuffer`.
            unsafe { objc::msg_send_u64_0(entry.buffer.as_raw(), objc::sel(c"gpuAddress")) },
            entry.len,
        )
    }

    fn check_completion(command_buffer: objc::Id) {
        // SAFETY: `command_buffer` completed via `waitUntilCompleted`.
        let status = unsafe { objc::msg_send_usize_0(command_buffer, objc::sel(c"status")) };
        if status != 4 {
            // SAFETY: `error` returns null or an autoreleased `NSError *`.
            let error = unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"error")) };
            // SAFETY: `error` is null or a valid `NSError *`.
            let message = unsafe { objc::nserror_message(error) };
            panic!("Metal command buffer failed (status={status}): {message}");
        }
    }

    /// Hardware proof for the general root-signature ABI.
    ///
    /// Translates synthetic DXIL through the x86_64 IRConverter (Rosetta),
    /// builds the layout from the converter reflection, binds a root-constants
    /// block, a root CBV, a root SRV, a root UAV, a texture table and a sampler
    /// table, then asserts every output dword exactly.
    #[test]
    #[ignore = "GPU hardware test; x86_64 IRConverter and a free M4 Max Metal device required"]
    fn root_signature_abi_compute_hardware() {
        let Some(converter) = converter_path() else {
            eprintln!("skipping root-signature compute proof: no IRConverter library found");
            return;
        };
        assert_eq!(
            std::mem::size_of::<MTLRegion>(),
            48,
            "MTLRegion must match the Metal ABI"
        );
        let dxil = hex_bytes(COMPUTE_DXIL_HEX);
        let converted =
            match crate::macos::irconverter::convert_dxil_to_metallib_with_reflection(
                &dxil, "main", &converter,
            ) {
                Ok(converted) => converted,
                Err(MetalError::NotSupported(reason)) => {
                    eprintln!("skipping root-signature compute proof: {reason}");
                    return;
                }
                Err(error) => panic!("IRConverter translation failed: {error}"),
            };
        let layout = RootSignatureLayout::from_converter_reflection_json(
            &converted.reflection_json,
        )
        .expect("converter reflection must describe the compute layout")
        .with_descriptor_table_kinds(&[DescriptorTableKind::Texture, DescriptorTableKind::Sampler])
        .expect("attach the two reflected descriptor tables");
        assert_eq!(layout.argument_buffer_len(), 48);

        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let device = transport.device_raw();

        // Root CBV: Params { scale=0.5, offset=0.25 }.
        let cbv_page = HostPage::new();
        // SAFETY: Fresh 16 KiB page, written before registration.
        unsafe {
            let words = cbv_page.as_ptr().cast::<f32>();
            std::ptr::write(words, 0.5);
            std::ptr::write(words.add(1), 0.25);
        }
        let input_page = HostPage::new();
        // SAFETY: Fresh 16 KiB page, written before registration.
        unsafe {
            let words = input_page.as_ptr().cast::<f32>();
            for index in 0..64 {
                std::ptr::write(words.add(index), 3.0);
            }
        }
        let output_page = HostPage::new();

        // SAFETY: The page is a live, page-aligned 16 KiB allocation.
        let cbv_id = unsafe { transport.register_zerocopy(cbv_page.as_ptr(), HOST_PAGE_BYTES) }
            .expect("register CBV");
        // SAFETY: Same 16 KiB zero-copy contract.
        let input_id = unsafe { transport.register_zerocopy(input_page.as_ptr(), HOST_PAGE_BYTES) }
            .expect("register SRV");
        // SAFETY: Same 16 KiB zero-copy contract.
        let output_id = unsafe { transport.register_zerocopy(output_page.as_ptr(), HOST_PAGE_BYTES) }
            .expect("register UAV");

        let _pool = objc::AutoreleasePool::new();
        let texture = new_texture(device, PIPELINE_COLOR_FORMAT_RGBA32_FLOAT, 1, 1, 1);
        let texel = [0.25f32, 0.0, 0.0, 1.0].iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        // SAFETY: The 1x1 RGBA32Float texture is shared-storage and the region
        // matches its extent.
        unsafe {
            send_replace_region(
                texture.as_raw(),
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc::MTLSize { width: 1, height: 1, depth: 1 },
                },
                &texel,
                16,
            );
        }
        let sampler = new_sampler(device);

        // Converter descriptor arrays: texture at offset 0, sampler at offset
        // 32 in one shared buffer.
        let descriptor_buffer = new_shared_buffer(device, 256);
        let texture_id = resource_id(&texture);
        let sampler_id = resource_id(&sampler);
        write_bytes(&descriptor_buffer, 0, &texture_id.to_le_bytes());
        write_bytes(&descriptor_buffer, 8, &texture_id.to_le_bytes());
        write_bytes(&descriptor_buffer, 16, &0u64.to_le_bytes());
        write_bytes(&descriptor_buffer, 32, &sampler_id.to_le_bytes());
        write_bytes(&descriptor_buffer, 40, &0u64.to_le_bytes());
        write_bytes(&descriptor_buffer, 48, &0u64.to_le_bytes());
        let descriptor_address = gpu_address(&descriptor_buffer);

        let library = new_library(device, &converted.metallib.bytes);
        let function = library_function(&library, "main");
        let pipeline = new_compute_pipeline(device, &function);

        let state = transport.lock_state().expect("allocation registry lock");
        let (cbv_raw, cbv_address, cbv_len) = registered(&state, cbv_id);
        let (input_raw, input_address, input_len) = registered(&state, input_id);
        let (output_raw, output_address, output_len) = registered(&state, output_id);

        // SAFETY: `queue` is a live `MTLCommandQueue`; the command buffer is
        // autoreleased into `_pool` and stays alive through completion.
        let command_buffer = unsafe {
            objc::msg_send_id_0(transport.queue_raw(), objc::sel(c"commandBuffer"))
        };
        assert!(!command_buffer.is_null());
        // SAFETY: The command buffer is uncommitted.
        let encoder = unsafe {
            objc::msg_send_id_0(command_buffer, objc::sel(c"computeCommandEncoder"))
        };
        assert!(!encoder.is_null());
        // SAFETY: `encoder` is a live compute encoder and `pipeline` a live PSO.
        unsafe {
            objc::msg_send_void_1(
                encoder,
                objc::sel(c"setComputePipelineState:"),
                pipeline.as_raw(),
            );
        }

        // SAFETY: The device and encoder are live for this scope.
        let mut binder = unsafe {
            ArgumentBufferEncoder::new(device, encoder, EncoderKind::Compute, &layout)
        }
        .expect("compute argument-buffer encoder");
        binder
            .set_root_constants(0, &[2.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat())
            .expect("root constants");
        // SAFETY: Each registered buffer handle resolves to the live MTLBuffer
        // whose `gpuAddress` and registered length are passed here.
        unsafe {
            binder
                .set_root_buffer_view(
                    1,
                    BufferBinding {
                        buffer: cbv_id,
                        generation: 1,
                        address: cbv_address,
                        offset: 0,
                        length: 8,
                        registered_length: cbv_len as u64,
                    },
                    cbv_raw,
                    ResourceUsage::Read,
                )
                .expect("root CBV");
            binder
                .set_root_buffer_view(
                    2,
                    BufferBinding {
                        buffer: input_id,
                        generation: 1,
                        address: input_address,
                        offset: 0,
                        length: 256,
                        registered_length: input_len as u64,
                    },
                    input_raw,
                    ResourceUsage::Read,
                )
                .expect("root SRV");
            binder
                .set_root_buffer_view(
                    3,
                    BufferBinding {
                        buffer: output_id,
                        generation: 1,
                        address: output_address,
                        offset: 0,
                        length: 256,
                        registered_length: output_len as u64,
                    },
                    output_raw,
                    ResourceUsage::ReadWrite,
                )
                .expect("root UAV");
            binder
                .use_resource(texture.as_raw(), ResourceUsage::Read)
                .expect("texture residency");
        }
        binder
            .set_descriptor_table(
                4,
                DescriptorTableBinding {
                    address: descriptor_address,
                    length: TEXTURE_DESCRIPTOR_BYTES as u64,
                    count: 1,
                    stride: TEXTURE_DESCRIPTOR_BYTES,
                },
            )
            .expect("texture table");
        binder
            .set_descriptor_table(
                5,
                DescriptorTableBinding {
                    address: descriptor_address + 32,
                    length: SAMPLER_DESCRIPTOR_BYTES as u64,
                    count: 1,
                    stride: SAMPLER_DESCRIPTOR_BYTES,
                },
            )
            .expect("sampler table");
        let _argument_buffer = binder
            .bind_argument_buffer()
            .expect("argument buffer must upload");

        // SAFETY: `encoder` is a live compute encoder; 64 threads with 64
        // threads per group matches the translated kernel.
        unsafe {
            objc::msg_send_dispatch_threads(
                encoder,
                objc::MTLSize { width: 64, height: 1, depth: 1 },
                objc::MTLSize { width: 64, height: 1, depth: 1 },
            );
            objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"commit"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"waitUntilCompleted"));
        }
        check_completion(command_buffer);

        drop(state);
        // The command buffer completed, so the registered pages are no longer
        // referenced by the GPU.
        transport.unregister_buffer(cbv_id).expect("unregister CBV");
        transport.unregister_buffer(input_id).expect("unregister SRV");
        transport.unregister_buffer(output_id).expect("unregister UAV");

        let expected = 2.0f32 * 3.0 + 1.0 + 0.5 + 0.25;
        // SAFETY: The output page is a live 16 KiB host allocation.
        let words = unsafe { std::slice::from_raw_parts(output_page.as_ptr().cast::<f32>(), 64) };
        for (index, value) in words.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "dword {index} mismatch: got {value}, expected {expected}"
            );
        }
        eprintln!(
            "root-signature compute proof: {} dwords == {} (layout offsets {:?})",
            words.len(),
            expected,
            layout.parameters().iter().map(|p| p.argument_offset()).collect::<Vec<_>>()
        );
    }

    /// Hardware proof for the render encoder path: converted vertex shader
    /// reads a root CBV and writes an exact colour, read back from a 2x2
    /// RGBA8Unorm render target.
    #[test]
    #[ignore = "GPU hardware test; x86_64 IRConverter and a free M4 Max Metal device required"]
    fn root_signature_abi_graphics_hardware() {
        let Some(converter) = converter_path() else {
            eprintln!("skipping root-signature graphics proof: no IRConverter library found");
            return;
        };
        let vertex_dxil = hex_bytes(VERTEX_DXIL_HEX);
        let pixel_dxil = hex_bytes(PIXEL_DXIL_HEX);
        let vertex = match crate::macos::irconverter::convert_dxil_to_metallib_with_reflection(
            &vertex_dxil, "main", &converter,
        ) {
            Ok(converted) => converted,
            Err(MetalError::NotSupported(reason)) => {
                eprintln!("skipping root-signature graphics proof: {reason}");
                return;
            }
            Err(error) => panic!("IRConverter vertex translation failed: {error}"),
        };
        let pixel = crate::macos::irconverter::convert_dxil_to_metallib(&pixel_dxil, "main", &converter)
            .expect("IRConverter pixel translation");
        let layout = RootSignatureLayout::from_converter_reflection_json(&vertex.reflection_json)
            .expect("vertex reflection must describe the layout");
        assert_eq!(layout.argument_buffer_len(), 8);
        assert!(layout.supports(EncoderKind::Render));

        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let device = transport.device_raw();
        let cbv_page = HostPage::new();
        // SAFETY: Fresh 16 KiB page; Params { scale=0.5, offset=0.0 }.
        unsafe {
            let words = cbv_page.as_ptr().cast::<f32>();
            std::ptr::write(words, 0.5);
            std::ptr::write(words.add(1), 0.0);
        }
        // SAFETY: The page is a live, page-aligned 16 KiB allocation.
        let cbv_id = unsafe { transport.register_zerocopy(cbv_page.as_ptr(), HOST_PAGE_BYTES) }
            .expect("register CBV");

        let target = new_texture(
            device,
            PIPELINE_COLOR_FORMAT_RGBA8_UNORM,
            2,
            2,
            // MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget
            1 | 4,
        );

        let vertex_library = new_library(device, &vertex.metallib.bytes);
        let pixel_library = new_library(device, &pixel.bytes);
        let vertex_function = library_function(&vertex_library, "main");
        let pixel_function = library_function(&pixel_library, "main");
        let pipeline = new_render_pipeline(
            device,
            &vertex_function,
            &pixel_function,
            PIPELINE_COLOR_FORMAT_RGBA8_UNORM,
        );

        let _pool = objc::AutoreleasePool::new();
        let pass = objc::class(c"MTLRenderPassDescriptor");
        // SAFETY: Class method, returns an autoreleased descriptor.
        let pass = unsafe { objc::msg_send_id_0(pass, objc::sel(c"renderPassDescriptor")) };
        assert!(!pass.is_null());
        // SAFETY: The descriptor is autoreleased but alive in `_pool`.
        let attachment = unsafe {
            let attachments = objc::msg_send_id_0(pass, objc::sel(c"colorAttachments"));
            let attachment = send_id_usize(attachments, objc::sel(c"objectAtIndexedSubscript:"), 0);
            objc::msg_send_void_1(attachment, objc::sel(c"setTexture:"), target.as_raw());
            send_void_usize(attachment, objc::sel(c"setLoadAction:"), 2);
            send_void_usize(attachment, objc::sel(c"setStoreAction:"), 1);
            send_clear_color(attachment, [0.0, 0.0, 0.0, 1.0]);
            attachment
        };
        assert!(!attachment.is_null());

        let state = transport.lock_state().expect("allocation registry lock");
        let (cbv_raw, cbv_address, cbv_len) = registered(&state, cbv_id);

        // SAFETY: `queue` is a live queue; objects are autoreleased into `_pool`.
        let command_buffer = unsafe {
            objc::msg_send_id_0(transport.queue_raw(), objc::sel(c"commandBuffer"))
        };
        assert!(!command_buffer.is_null());
        // SAFETY: The command buffer is uncommitted.
        let encoder = unsafe {
            objc::msg_send_id_1(
                command_buffer,
                objc::sel(c"renderCommandEncoderWithDescriptor:"),
                pass,
            )
        };
        assert!(!encoder.is_null());
        // SAFETY: `encoder` is a live render encoder and `pipeline` a live PSO.
        unsafe {
            objc::msg_send_void_1(
                encoder,
                objc::sel(c"setRenderPipelineState:"),
                pipeline.as_raw(),
            );
        }

        // SAFETY: Device and encoder are live for this scope.
        let mut binder = unsafe {
            ArgumentBufferEncoder::new(device, encoder, EncoderKind::Render, &layout)
        }
        .expect("render argument-buffer encoder");
        // SAFETY: The registered handle resolves to `cbv_raw`.
        unsafe {
            binder
                .set_root_buffer_view(
                    0,
                    BufferBinding {
                        buffer: cbv_id,
                        generation: 1,
                        address: cbv_address,
                        offset: 0,
                        length: 8,
                        registered_length: cbv_len as u64,
                    },
                    cbv_raw,
                    ResourceUsage::Read,
                )
                .expect("root CBV");
        }
        let _argument_buffer = binder
            .bind_argument_buffer()
            .expect("argument buffer must upload");

        // SAFETY: `encoder` is a live render encoder; three vertices form the
        // fullscreen triangle the converted vertex shader expects.
        unsafe {
            send_draw_primitives(encoder, 3, 0, 3);
            objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"commit"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"waitUntilCompleted"));
        }
        check_completion(command_buffer);

        drop(state);
        // The command buffer completed before the page is released.
        transport.unregister_buffer(cbv_id).expect("unregister CBV");

        let mut pixels = [0u8; 16];
        // SAFETY: The 2x2 shared texture and the destination buffer match.
        unsafe {
            send_get_bytes(
                target.as_raw(),
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc::MTLSize { width: 2, height: 2, depth: 1 },
                },
                &mut pixels,
                8,
            );
        }
        for (index, pixel) in pixels.chunks_exact(4).enumerate() {
            assert_eq!(
                pixel,
                [255, 0, 128, 255],
                "pixel {index} mismatch: got {pixel:?}"
            );
        }
        eprintln!(
            "root-signature graphics proof: 4 pixels == [255, 0, 128, 255] (CBV offset {})",
            layout.parameters()[0].argument_offset()
        );
    }

}
