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
//! signature order. Descriptor parameters occupy one 8-byte field containing a
//! 64-bit device address. Descriptor *tables* occupy one 8-byte field containing
//! the device address of an array of descriptors in GPU-visible memory:
//!
//! ```text
//! root CBV / SRV / UAV : field = address of the resource data
//! table of CBV/SRV/UAV : field = address of [u64 device address; count]
//! table of textures    : field = address of [_Desc { tex, tex, id }; count]  (stride 24)
//! table of samplers    : field = address of [_Desc { smp, id, id }; count]    (stride 24)
//! root constants       : inline bytes at the reflected offset (see below)
//! ```
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
    /// Table of constant buffer views: `[u64 device address; count]`.
    ConstantBufferView,
    /// Table of buffer shader resource views: `[u64 device address; count]`.
    ShaderResourceView,
    /// Table of unordered access views: `[u64 device address; count]`.
    UnorderedAccessView,
    /// Table of `Texture2D`-style descriptors: 24-byte `{texture, texture, id}`.
    Texture,
    /// Table of sampler states: 24-byte `{sampler, id, id}`.
    Sampler,
    /// `"Table"` in reflection without a caller-supplied range type.
    ///
    /// The converter reflection does not say which descriptor range a table
    /// declares. Callers attach the kind in root-signature order with
    /// [`RootSignatureLayout::with_descriptor_table_kinds`]; until then the
    /// table can still be bound, but with the address-table stride of 8 bytes.
    Unspecified,
}

impl DescriptorTableKind {
    /// Default stride of one descriptor of this kind in converted memory.
    pub fn descriptor_bytes(self) -> usize {
        match self {
            Self::Texture | Self::Sampler => TEXTURE_DESCRIPTOR_BYTES,
            Self::ConstantBufferView | Self::ShaderResourceView | Self::UnorderedAccessView => {
                DEVICE_ADDRESS_BYTES
            }
            Self::Unspecified => DEVICE_ADDRESS_BYTES,
        }
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

impl RootSignatureLayout {
    /// Parses Apple's `IRShaderReflectionCopyJSONString` output.
    ///
    /// Required field: `TopLevelArgumentBuffer`. `EntryPoint` and `ShaderType`
    /// are recorded when present. Every entry must have `EltOffset`, `Size`,
    /// `Slot`, `Space` and a `Type` of `Constant`, `CBV`, `SRV`, `UAV` or
    /// `Table`; descriptor fields must be 8-byte aligned and at least 8 bytes,
    /// root constants must be 4-byte aligned and 4-byte sized, and no two
    /// parameters may overlap.
    pub fn from_converter_reflection_json(json: &str) -> Result<Self, MetalError> {
        let root = parse_json(json)?;
        let entry_point = root
            .get("EntryPoint")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string();
        let stage = root
            .get("ShaderType")
            .and_then(Json::as_str)
            .map_or(ReflectedStage::Other, ReflectedStage::from_json);
        let entries = root
            .get("TopLevelArgumentBuffer")
            .and_then(Json::as_array)
            .ok_or(MetalError::InvalidArg(
                "converter reflection has no TopLevelArgumentBuffer array",
            ))?;
        if entries.len() > 64 {
            return Err(MetalError::InvalidArg(
                "converter reflection has more than 64 root parameters",
            ));
        }

        let mut parameters = Vec::with_capacity(entries.len());
        for entry in entries {
            let object = entry.as_object().ok_or(MetalError::InvalidArg(
                "TopLevelArgumentBuffer entry is not an object",
            ))?;
            let offset = object
                .iter()
                .find(|(key, _)| key == "EltOffset")
                .and_then(|(_, value)| value.as_u64())
                .ok_or(MetalError::InvalidArg(
                    "TopLevelArgumentBuffer entry has no EltOffset",
                ))?;
            let size = object
                .iter()
                .find(|(key, _)| key == "Size")
                .and_then(|(_, value)| value.as_u64())
                .ok_or(MetalError::InvalidArg(
                    "TopLevelArgumentBuffer entry has no Size",
                ))?;
            let slot = object
                .iter()
                .find(|(key, _)| key == "Slot")
                .and_then(|(_, value)| value.as_u64())
                .ok_or(MetalError::InvalidArg(
                    "TopLevelArgumentBuffer entry has no Slot",
                ))?;
            let space = object
                .iter()
                .find(|(key, _)| key == "Space")
                .and_then(|(_, value)| value.as_u64())
                .ok_or(MetalError::InvalidArg(
                    "TopLevelArgumentBuffer entry has no Space",
                ))?;
            let type_name = object
                .iter()
                .find(|(key, _)| key == "Type")
                .and_then(|(_, value)| value.as_str())
                .ok_or(MetalError::InvalidArg(
                    "TopLevelArgumentBuffer entry has no Type",
                ))?;

            let kind = match type_name {
                "Constant" => RootParameterKind::Constants,
                "CBV" => RootParameterKind::ConstantBufferView,
                "SRV" => RootParameterKind::ShaderResourceView,
                "UAV" => RootParameterKind::UnorderedAccessView,
                "Table" => RootParameterKind::DescriptorTable(DescriptorTableKind::Unspecified),
                _ => {
                    return Err(MetalError::InvalidArg(
                        "converter reflection has an unknown root parameter type",
                    ));
                }
            };
            parameters.push(RootParameter {
                kind,
                argument_offset: usize::try_from(offset).map_err(|_| {
                    MetalError::InvalidArg("root parameter offset does not fit usize")
                })?,
                size: usize::try_from(size)
                    .map_err(|_| MetalError::InvalidArg("root parameter size does not fit usize"))?,
                register: u32::try_from(slot).unwrap_or(u32::MAX),
                space: u32::try_from(space).unwrap_or(u32::MAX),
            });
        }

        Self::assemble(
            stage,
            entry_point,
            TOP_LEVEL_ARGUMENT_BUFFER_INDEX,
            parameters,
            None,
        )
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
        if argument_buffer_index >= MAX_METAL_BUFFER_ARGUMENT_INDEX {
            return Err(MetalError::InvalidArg(
                "argument buffer index is not a valid Metal buffer index",
            ));
        }

        parameters.sort_by_key(|parameter| parameter.argument_offset);
        let mut end = 0usize;
        let mut constants_count = 0usize;
        for parameter in &parameters {
            let is_constants = parameter.kind == RootParameterKind::Constants;
            if is_constants {
                constants_count += 1;
                if parameter.size == 0 || parameter.size > MAX_ROOT_CONSTANT_BYTES {
                    return Err(MetalError::InvalidArg(
                        "root constants size is outside 1..=4096 bytes",
                    ));
                }
                if !parameter.size.is_multiple_of(4)
                    || !parameter.argument_offset.is_multiple_of(4)
                {
                    return Err(MetalError::InvalidArg(
                        "root constants must be 4-byte aligned and sized",
                    ));
                }
            } else {
                if parameter.size < DEVICE_ADDRESS_BYTES {
                    return Err(MetalError::InvalidArg(
                        "descriptor argument-buffer slot is smaller than one address",
                    ));
                }
                if !parameter.argument_offset.is_multiple_of(DEVICE_ADDRESS_BYTES) {
                    return Err(MetalError::InvalidArg(
                        "descriptor argument-buffer offset must be 8-byte aligned",
                    ));
                }
            }
            if parameter.argument_offset < end {
                return Err(MetalError::InvalidArg(
                    "root parameters overlap in the argument buffer",
                ));
            }
            end = parameter
                .argument_offset
                .checked_add(parameter.size)
                .ok_or(MetalError::InvalidArg("root parameter extent overflows"))?;
            if end > MAX_ARGUMENT_BUFFER_BYTES {
                return Err(MetalError::InvalidArg(
                    "root parameters exceed the 64 KiB argument buffer",
                ));
            }
        }
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
}

impl<'a> ArgumentBufferBuilder<'a> {
    /// Fresh zero-filled builder for `layout`.
    pub fn new(layout: &'a RootSignatureLayout) -> Self {
        Self {
            layout,
            bytes: vec![0; layout.argument_buffer_len()],
            encoded: vec![false; layout.parameters().len()],
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
    /// Stride of one converted descriptor in bytes (8 or 24; see
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

    /// Writes one 8-byte device-address descriptor (CBV/SRV/UAV tables).
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
        self.count = self.count.saturating_add(1);
        Ok(())
    }

    /// Writes one 24-byte texture descriptor (`{texture, texture, 0}`).
    ///
    /// Converted MSL loads element 1, but the runtime writes both pointers, so
    /// both fields carry the texture handle.
    pub fn push_texture(&mut self, texture: u64) -> Result<(), MetalError> {
        if self.kind != DescriptorTableKind::Texture {
            return Err(MetalError::InvalidArg(
                "descriptor table kind is not a texture table",
            ));
        }
        if texture == 0 || !texture.is_multiple_of(DEVICE_ADDRESS_BYTES as u64) {
            return Err(MetalError::InvalidArg(
                "texture descriptor handle must be nonzero and 8-byte aligned",
            ));
        }
        self.bytes.extend_from_slice(&texture.to_le_bytes());
        self.bytes.extend_from_slice(&texture.to_le_bytes());
        self.bytes.extend_from_slice(&0u64.to_le_bytes());
        self.count = self.count.saturating_add(1);
        Ok(())
    }

    /// Writes one 24-byte sampler descriptor (`{sampler, 0, 0}`).
    pub fn push_sampler(&mut self, sampler: u64) -> Result<(), MetalError> {
        if self.kind != DescriptorTableKind::Sampler {
            return Err(MetalError::InvalidArg(
                "descriptor table kind is not a sampler table",
            ));
        }
        if sampler == 0 || !sampler.is_multiple_of(DEVICE_ADDRESS_BYTES as u64) {
            return Err(MetalError::InvalidArg(
                "sampler descriptor handle must be nonzero and 8-byte aligned",
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
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Self::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

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


#[cfg(test)]
mod tests {
    use super::{
        ArgumentBufferBuilder, BufferBinding, DescriptorTableBinding, DescriptorTableBuilder,
        DescriptorTableKind, EncoderKind, MAX_ARGUMENT_BUFFER_BYTES, MAX_ROOT_CONSTANT_BYTES,
        ReflectedStage, RootParameterKind, RootParameterSpec, RootSignatureLayout,
        SAMPLER_DESCRIPTOR_BYTES, TEXTURE_DESCRIPTOR_BYTES, TOP_LEVEL_ARGUMENT_BUFFER_INDEX,
    };
    use crate::transport::MetalError;
    use crate::wire;
    use crate::VmBufferId;

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
                r#"{"EntryPoint":"main","ShaderType":"Compute"}"#
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
                count: 6,
                stride: 8,
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
        assert_eq!(addresses.bytes().len(), 16);
        assert_eq!(&addresses.bytes()[0..8], &0x1000u64.to_le_bytes());
        assert_eq!(&addresses.bytes()[8..16], &0x2000u64.to_le_bytes());
        let table = addresses.binding(0x9000).expect("binding");
        assert_eq!(table.count, 2);
        assert_eq!(table.stride, 8);
        assert_eq!(table.extent().expect("extent"), 16);

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
            DescriptorTableBuilder::new(DescriptorTableKind::Sampler).push_sampler(0x1004),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            DescriptorTableBuilder::new(DescriptorTableKind::Sampler).binding(0x1000),
            Err(MetalError::InvalidArg(_))
        ));
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
}
