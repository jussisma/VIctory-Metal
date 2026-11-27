//! The translated root signature: one description of what the translated
//! shaders' bindings declare, and its D3D12 serialization in plain bytes.
//!
//! `victory-gpu`'s host pipelines build a root signature on a live D3D12
//! device (`pipeline::host`): the parameters come from [`HostShader`], the
//! blob from `D3D12SerializeVersionedRootSignature`. The native Metal path has
//! no D3D12 device. The converter (`victory-irconverter`) takes a root
//! signature as the witness of the translated bindings, and the title's root
//! arguments must fill the translated shaders' [`USER_DATA_SPACE`],
//! [`ROOT_CBV_SPACE`], [`MEMORY_SPACE`] and [`SAMPLER_SPACE`] slots. Both
//! paths need the same description, so it is derived here from
//! [`HostShader`] alone and serialized here too, without any device call.
//!
//! # The serialized form
//!
//! The bytes `D3D12SerializeVersionedRootSignature` returns (the payload of
//! the `RTS0` part of a DXIL container; `RTS0` is the container part's
//! FourCC, not a magic at the start of the payload) are little-endian
//! DWORDs:
//!
//! ```text
//! offset  0  version             2 = D3D_ROOT_SIGNATURE_VERSION_1_1
//! offset  4  parameter count     at most 64
//! offset  8  parameter offset    24: right after this header
//! offset 12  static sampler count
//! offset 16  sampler offset      after the parameter array and its records
//! offset 20  flags               D3D12_ROOT_SIGNATURE_FLAGS
//! ```
//!
//! Then the parameter array, 12 bytes per parameter: `{ type, visibility,
//! record offset }`, where `type` is `D3D12_ROOT_PARAMETER_TYPE` and the
//! record offset points into the blob. The records follow the array in
//! parameter order: 12 bytes `{ register, space, value count }` for root
//! constants, 12 bytes `{ register, space, flags }` for a root descriptor,
//! and `{ range count, range offset }` for a descriptor table immediately
//! followed by its ranges (24 bytes each: `{ type, count, register, space,
//! flags, offset in table }`). Last, the static samplers: 52 bytes each,
//! `D3D12_STATIC_SAMPLER_DESC`'s fields in declaration order.
//!
//! Every field is four bytes wide and every record a multiple of four, so no
//! padding is ever needed. This is measured byte for byte against the `RTS0`
//! parts `dxc` 1.9.2609.5 wrote (`toolchains/dxc` under Wine's `d3d12.dll`,
//! which serializes with `D3D12SerializeVersionedRootSignature`); the tests
//! carry those blobs and the root signature strings they came from. That is
//! also the blob `dxc` embeds in a DXIL container's `RTS0` part; the
//! translated containers declare no root signature of their own (their
//! bindings are the directly indexed heaps), so the witness comes from here.
//!
//! The version is 1.1: the translated root descriptors carry D3D12's
//! `DATA_VOLATILE` flag and the static samplers live in a nonzero register
//! space, neither of which version 1.0 can express. The version DWORD is the
//! `D3D_ROOT_SIGNATURE_VERSION` value (2), not the text "1.1".

use crate::ShaderError;

use super::{
    HostShader, MEMORY_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, StaticSampler, USER_DATA_SPACE,
    VIEW_MARKS, VIEW_WRITES,
};
use crate::native::HwStage;

/// D3D12's limit on root parameters.
pub const MAX_ROOT_PARAMETERS: usize = 64;
/// D3D12's limit on static samplers.
pub const MAX_STATIC_SAMPLERS: usize = 2032;
/// The largest blob this serializer writes. A translated signature is a few
/// hundred bytes; a hand-built description beyond this is refused rather than
/// sized.
pub const MAX_BLOB_BYTES: usize = 1 << 20;

/// The `D3D_ROOT_SIGNATURE_VERSION` value serialized for a 1.1 description.
const VERSION_1_1: u32 = 2;
/// `D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT`.
const FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT: u32 = 1;
/// `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`.
const FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED: u32 = 1024;
/// `D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED`.
const FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED: u32 = 2048;
/// `D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE`.
const DESCRIPTOR_FLAG_DATA_VOLATILE: u32 = 2;
/// `D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND`.
const RANGE_OFFSET_APPEND: u32 = u32::MAX;

/// Header: the six fields above.
const HEADER_BYTES: usize = 24;
/// One parameter-array entry.
const PARAMETER_BYTES: usize = 12;
/// One root-constant or root-descriptor record.
const RECORD_BYTES: usize = 12;
/// A descriptor table's record before its ranges.
const TABLE_BYTES: usize = 8;
/// One descriptor range.
const RANGE_BYTES: usize = 24;
/// One static sampler.
const SAMPLER_BYTES: usize = 52;

/// The shader stages a root signature's access can be limited to.
///
/// The numbers are D3D12's `D3D12_SHADER_VISIBILITY`; the description names
/// them so that a caller does not need the D3D12 headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    All,
    Vertex,
    Hull,
    Domain,
    Geometry,
    Pixel,
    Amplification,
    Mesh,
}

impl Visibility {
    /// The visibility of a stage's own parameters. The console's local and
    /// export programs run inside the merged graphics wave and have no D3D12
    /// stage of their own; like a compute program they see everything.
    pub fn for_stage(stage: HwStage) -> Visibility {
        match stage {
            HwStage::Vs => Visibility::Vertex,
            HwStage::Hs => Visibility::Hull,
            HwStage::Gs => Visibility::Geometry,
            HwStage::Ps => Visibility::Pixel,
            HwStage::Ms => Visibility::Mesh,
            HwStage::As => Visibility::Amplification,
            HwStage::Ls | HwStage::Es | HwStage::Cs => Visibility::All,
        }
    }

    /// `D3D12_SHADER_VISIBILITY`.
    fn bits(self) -> u32 {
        match self {
            Visibility::All => 0,
            Visibility::Vertex => 1,
            Visibility::Hull => 2,
            Visibility::Domain => 3,
            Visibility::Geometry => 4,
            Visibility::Pixel => 5,
            Visibility::Amplification => 6,
            Visibility::Mesh => 7,
        }
    }
}

/// The kinds of root descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    Cbv,
    Srv,
    Uav,
}

impl DescriptorKind {
    /// `D3D12_ROOT_PARAMETER_TYPE`.
    fn parameter_type(self) -> u32 {
        match self {
            DescriptorKind::Cbv => 2,
            DescriptorKind::Srv => 3,
            DescriptorKind::Uav => 4,
        }
    }
}

/// The kinds of descriptor range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeKind {
    Srv,
    Uav,
    Cbv,
    Sampler,
}

impl RangeKind {
    /// `D3D12_DESCRIPTOR_RANGE_TYPE`.
    fn bits(self) -> u32 {
        match self {
            RangeKind::Srv => 0,
            RangeKind::Uav => 1,
            RangeKind::Cbv => 2,
            RangeKind::Sampler => 3,
        }
    }
}

/// One range of a descriptor table.
///
/// A translated signature never declares a table (every binding is either a
/// root descriptor or an index into the directly indexed heaps), but the
/// description and its serialization cover one so the format is complete and
/// checked against a real serializer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorRange {
    pub kind: RangeKind,
    /// The first shader register of the range.
    pub register: u32,
    pub space: u32,
    /// Descriptors in the range; `u32::MAX` is unbounded.
    pub count: u32,
    /// The range's first slot in the table; `None` appends it after the
    /// previous range.
    pub offset: Option<u32>,
}

/// What one root parameter is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterKind {
    /// A block of 32-bit root constants.
    Constants { values: u32 },
    /// A root descriptor (the resource's GPU address).
    Descriptor {
        kind: DescriptorKind,
        /// `D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE`: the root argument may
        /// change between draws without a new root signature.
        volatile: bool,
    },
    /// A descriptor table.
    Table { ranges: Vec<DescriptorRange> },
}

impl ParameterKind {
    /// `D3D12_ROOT_PARAMETER_TYPE`.
    fn parameter_type(&self) -> u32 {
        match self {
            ParameterKind::Constants { .. } => 1,
            ParameterKind::Descriptor { kind, .. } => kind.parameter_type(),
            ParameterKind::Table { .. } => 0,
        }
    }
}

/// One root parameter, in signature order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootParameter {
    pub kind: ParameterKind,
    /// The parameter's shader register; unused by a table, whose ranges carry
    /// their own.
    pub register: u32,
    pub space: u32,
    pub visibility: Visibility,
    /// The stage whose translated shader declared it, or `None` for a
    /// parameter shared by every stage (the view-mark pair).
    pub stage: Option<HwStage>,
}

/// One static sampler, in signature order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootSampler {
    /// The console's sampler, decoded (its integers are the D3D12
    /// `D3D12_FILTER`, `D3D12_TEXTURE_ADDRESS_MODE`, `D3D12_COMPARISON_FUNC`
    /// and `D3D12_STATIC_BORDER_COLOR` values).
    pub sampler: StaticSampler,
    pub register: u32,
    pub space: u32,
    pub visibility: Visibility,
}

/// Root signature flags, by meaning rather than by D3D12 bit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SignatureFlags {
    /// The title's input layout feeds the vertex shader's inputs.
    pub allow_input_assembler_input_layout: bool,
    /// The translated shaders index the CBV/SRV/UAV heap directly.
    pub cbv_srv_uav_heap_directly_indexed: bool,
    /// They index the sampler heap directly.
    pub sampler_heap_directly_indexed: bool,
}

impl SignatureFlags {
    /// `D3D12_ROOT_SIGNATURE_FLAGS`.
    fn bits(self) -> u32 {
        let mut bits = 0;
        if self.allow_input_assembler_input_layout {
            bits |= FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        }
        if self.cbv_srv_uav_heap_directly_indexed {
            bits |= FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
        }
        if self.sampler_heap_directly_indexed {
            bits |= FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
        }
        bits
    }
}

/// One translated root signature: its parameters and static samplers in
/// order, independent of any D3D12 object or device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootSignature {
    flags: SignatureFlags,
    parameters: Vec<RootParameter>,
    samplers: Vec<RootSampler>,
}

impl RootSignature {
    /// An empty version 1.1 signature with these flags.
    pub fn new(flags: SignatureFlags) -> RootSignature {
        RootSignature {
            flags,
            parameters: Vec::new(),
            samplers: Vec::new(),
        }
    }

    pub fn flags(&self) -> SignatureFlags {
        self.flags
    }

    /// The parameters, in the order they occupy root parameter slots.
    pub fn parameters(&self) -> &[RootParameter] {
        &self.parameters
    }

    /// The static samplers, in the order the host declares them.
    pub fn static_samplers(&self) -> &[RootSampler] {
        &self.samplers
    }

    /// The root parameter indices of the view-mark table and the buffer
    /// views' written-pages bitmap, if a stage marks its writes
    /// ([`HostShader::view_marks`]).
    pub fn view_mark_parameters(&self) -> Option<(u32, u32)> {
        self.parameters.iter().enumerate().find_map(|(i, p)| {
            let marks = p.stage.is_none()
                && p.space == MEMORY_SPACE
                && p.register == VIEW_MARKS
                && matches!(
                    p.kind,
                    ParameterKind::Descriptor {
                        kind: DescriptorKind::Srv,
                        ..
                    }
                );
            marks.then(|| (i as u32, i as u32 + 1))
        })
    }

    /// Refuses a description this serializer cannot write.
    fn check(&self) -> Result<(), ShaderError> {
        if self.parameters.len() > MAX_ROOT_PARAMETERS {
            return Err(ShaderError::RootSignature("more than 64 root parameters"));
        }
        if self.samplers.len() > MAX_STATIC_SAMPLERS {
            return Err(ShaderError::RootSignature("more than 2032 static samplers"));
        }
        Ok(())
    }

    /// The bytes `D3D12SerializeVersionedRootSignature` returns for this
    /// description, as laid out in the module docs. No device is involved.
    pub fn to_versioned_blob(&self) -> Result<Vec<u8>, ShaderError> {
        self.check()?;
        let array = self.parameters.len() * PARAMETER_BYTES;
        let mut size = HEADER_BYTES + array;
        // Where each parameter's own record goes: right after the array, in
        // parameter order.
        let mut records = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            records.push(dword_offset(size)?);
            size = size
                .checked_add(record_bytes(parameter)?)
                .ok_or(ShaderError::RootSignature(
                    "the root signature's size does not fit memory",
                ))?;
            if size > MAX_BLOB_BYTES {
                return Err(ShaderError::RootSignature(
                    "the root signature is larger than 1 MiB",
                ));
            }
        }
        let sampler_offset = dword_offset(size)?;
        size = size
            .checked_add(self.samplers.len() * SAMPLER_BYTES)
            .ok_or(ShaderError::RootSignature(
                "the root signature's size does not fit memory",
            ))?;
        if size > MAX_BLOB_BYTES {
            return Err(ShaderError::RootSignature(
                "the root signature is larger than 1 MiB",
            ));
        }
        let mut blob = vec![0u8; size];

        put(&mut blob, 0, VERSION_1_1)?;
        put(&mut blob, 4, count(self.parameters.len())?)?;
        put(&mut blob, 8, HEADER_BYTES as u32)?;
        put(&mut blob, 12, count(self.samplers.len())?)?;
        put(&mut blob, 16, sampler_offset)?;
        put(&mut blob, 20, self.flags.bits())?;

        for (i, (parameter, &record)) in self.parameters.iter().zip(records.iter()).enumerate() {
            let at = HEADER_BYTES + i * PARAMETER_BYTES;
            put(&mut blob, at, parameter.kind.parameter_type())?;
            put(&mut blob, at + 4, parameter.visibility.bits())?;
            put(&mut blob, at + 8, record)?;

            let at = record as usize;
            match &parameter.kind {
                ParameterKind::Constants { values } => {
                    put(&mut blob, at, parameter.register)?;
                    put(&mut blob, at + 4, parameter.space)?;
                    put(&mut blob, at + 8, *values)?;
                }
                ParameterKind::Descriptor { volatile, .. } => {
                    put(&mut blob, at, parameter.register)?;
                    put(&mut blob, at + 4, parameter.space)?;
                    put(
                        &mut blob,
                        at + 8,
                        if *volatile {
                            DESCRIPTOR_FLAG_DATA_VOLATILE
                        } else {
                            0
                        },
                    )?;
                }
                ParameterKind::Table { ranges } => {
                    put(&mut blob, at, count(ranges.len())?)?;
                    let first = at + TABLE_BYTES;
                    put(&mut blob, at + 4, dword_offset(first)?)?;
                    for (j, range) in ranges.iter().enumerate() {
                        let at = first + j * RANGE_BYTES;
                        put(&mut blob, at, range.kind.bits())?;
                        put(&mut blob, at + 4, range.count)?;
                        put(&mut blob, at + 8, range.register)?;
                        put(&mut blob, at + 12, range.space)?;
                        put(&mut blob, at + 16, 0)?;
                        put(
                            &mut blob,
                            at + 20,
                            range.offset.unwrap_or(RANGE_OFFSET_APPEND),
                        )?;
                    }
                }
            }
        }

        for (i, sampler) in self.samplers.iter().enumerate() {
            let at = sampler_offset as usize + i * SAMPLER_BYTES;
            let s = sampler.sampler;
            put(&mut blob, at, s.filter)?;
            put(&mut blob, at + 4, s.address[0])?;
            put(&mut blob, at + 8, s.address[1])?;
            put(&mut blob, at + 12, s.address[2])?;
            put(&mut blob, at + 16, s.mip_lod_bias.to_bits())?;
            put(&mut blob, at + 20, s.max_anisotropy)?;
            put(&mut blob, at + 24, s.comparison)?;
            put(&mut blob, at + 28, s.border)?;
            put(&mut blob, at + 32, s.min_lod.to_bits())?;
            put(&mut blob, at + 36, s.max_lod.to_bits())?;
            put(&mut blob, at + 40, sampler.register)?;
            put(&mut blob, at + 44, sampler.space)?;
            put(&mut blob, at + 48, sampler.visibility.bits())?;
        }
        Ok(blob)
    }

    /// One stage's block: its user-data constants, root views, title-memory
    /// views and static samplers, in the order the host binds them.
    fn push_stage(&mut self, host: &HostShader) -> Result<(), ShaderError> {
        let visibility = Visibility::for_stage(host.stage);
        let stage = Some(host.stage);
        let dwords = host.user_data_dwords();
        if dwords > 0 {
            self.parameters.push(RootParameter {
                kind: ParameterKind::Constants { values: dwords },
                register: 0,
                space: USER_DATA_SPACE,
                visibility,
                stage,
            });
        }
        for (slots, kind) in [
            (&host.root_cbvs, DescriptorKind::Cbv),
            (&host.root_srvs, DescriptorKind::Srv),
            (&host.root_uavs, DescriptorKind::Uav),
        ] {
            for &slot in slots {
                self.parameters.push(RootParameter {
                    kind: ParameterKind::Descriptor {
                        kind,
                        volatile: true,
                    },
                    register: u32::from(slot),
                    space: ROOT_CBV_SPACE,
                    visibility,
                    stage,
                });
            }
        }
        for (used, kind, register) in [
            (host.memory, DescriptorKind::Srv, 0),
            (host.memory_writes, DescriptorKind::Uav, 0),
            (host.view_addresses, DescriptorKind::Srv, 1),
        ] {
            if used {
                self.parameters.push(RootParameter {
                    kind: ParameterKind::Descriptor {
                        kind,
                        volatile: true,
                    },
                    register,
                    space: MEMORY_SPACE,
                    visibility,
                    stage,
                });
            }
        }
        for (n, sampler) in host.static_samplers.iter().enumerate() {
            self.samplers.push(RootSampler {
                sampler: *sampler,
                register: u32::try_from(n).map_err(|_| {
                    ShaderError::RootSignature("a static sampler index does not fit 32 bits")
                })?,
                space: SAMPLER_SPACE,
                visibility,
            });
        }
        Ok(())
    }
}

impl Default for RootSignature {
    fn default() -> RootSignature {
        RootSignature::new(SignatureFlags::default())
    }
}

/// The root signature one translated stage's bindings declare: a block of
/// root constants for its user data at `b0` of [`USER_DATA_SPACE`], a root
/// descriptor per root CBV/SRV/UAV at [`ROOT_CBV_SPACE`], the title-memory
/// page table and written-pages bitmap and the view addresses at
/// [`MEMORY_SPACE`], and its immediate samplers at [`SAMPLER_SPACE`].
///
/// A pipeline's signature is the stages' blocks in stage order plus, if any
/// stage marks its writes, the shared view-mark pair
/// ([`translated_pipeline_root_signature`]).
pub fn translated_root_signature(host: &HostShader) -> Result<RootSignature, ShaderError> {
    let mut signature = RootSignature::default();
    signature.push_stage(host)?;
    signature.check()?;
    Ok(signature)
}

/// The root signature a translated pipeline's stages declare, in order, with
/// the view-mark pair once if any stage marks its writes.
pub fn translated_pipeline_root_signature(
    stages: &[&HostShader],
    flags: SignatureFlags,
) -> Result<RootSignature, ShaderError> {
    let mut signature = RootSignature::new(flags);
    for host in stages {
        signature.push_stage(host)?;
    }
    if stages.iter().any(|host| host.view_marks) {
        for (kind, register) in [
            (DescriptorKind::Srv, VIEW_MARKS),
            (DescriptorKind::Uav, VIEW_WRITES),
        ] {
            signature.parameters.push(RootParameter {
                kind: ParameterKind::Descriptor {
                    kind,
                    volatile: true,
                },
                register,
                space: MEMORY_SPACE,
                visibility: Visibility::All,
                stage: None,
            });
        }
    }
    signature.check()?;
    Ok(signature)
}

/// The bytes one parameter's own record occupies.
fn record_bytes(parameter: &RootParameter) -> Result<usize, ShaderError> {
    match &parameter.kind {
        ParameterKind::Table { ranges } => ranges
            .len()
            .checked_mul(RANGE_BYTES)
            .and_then(|bytes| bytes.checked_add(TABLE_BYTES))
            .ok_or(ShaderError::RootSignature(
                "a descriptor table does not fit a root signature",
            )),
        ParameterKind::Constants { .. } | ParameterKind::Descriptor { .. } => Ok(RECORD_BYTES),
    }
}

/// A byte offset as the 32-bit DWORD the format stores.
fn dword_offset(offset: usize) -> Result<u32, ShaderError> {
    u32::try_from(offset).map_err(|_| {
        ShaderError::RootSignature("an offset in the root signature does not fit 32 bits")
    })
}

/// A count as the 32-bit DWORD the format stores.
fn count(n: usize) -> Result<u32, ShaderError> {
    u32::try_from(n).map_err(|_| {
        ShaderError::RootSignature("a count in the root signature does not fit 32 bits")
    })
}

/// Writes one DWORD; a bad offset is an error, never a panic.
fn put(blob: &mut [u8], at: usize, value: u32) -> Result<(), ShaderError> {
    let Some(slot) = blob.get_mut(at..at.saturating_add(4)) else {
        return Err(ShaderError::RootSignature(
            "the serialized root signature overran its buffer",
        ));
    };
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::UserData;
    use crate::native::RootType;

    // The `RTS0` part of the DXIL container `dxc` 1.9.2609.5 wrote for these
    // root signatures (`dxc.exe -T cs_6_0` / `-T ps_6_0` under Wine, then the
    // part's bytes): a real `D3D12SerializeVersionedRootSignature`
    // serialization, which is what this module's serializer must reproduce.

    /// `RootFlags(0), SRV(t0), UAV(u5), CBV(b2)`.
    const MEASURED_DESCRIPTORS: [u8; 96] = [
        0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48,
        0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    /// `RootFlags(0), RootConstants(b3, num32BitConstants=4),
    /// DescriptorTable(SRV(t10, numDescriptors=2)), StaticSampler(s0,
    /// FILTER_MIN_MAG_MIP_LINEAR, WRAP, CLAMP, MIRROR)` — whose defaults
    /// `dxc` fills in (16x anisotropy, less-equal comparison, opaque white
    /// border, unbounded LOD).
    const MEASURED_CONSTANTS_TABLE_SAMPLER: [u8; 144] = [
        0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00, 0x5c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c,
        0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff,
        0xff, 0xff, 0x15, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x7f, 0x7f, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    /// `RootFlags(0),
    /// DescriptorTable(CBV(b0, numDescriptors=3), visibility=PIXEL),
    /// DescriptorTable(UAV(u2, numDescriptors=1), visibility=VERTEX)`.
    const MEASURED_TABLES: [u8; 112] = [
        0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00,
        0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x50,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
    ];

    /// `RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
    /// SAMPLER_HEAP_DIRECTLY_INDEXED), SRV(t0), StaticSampler(s0, space=3,
    /// FILTER_MIN_MAG_MIP_POINT, CLAMP, CLAMP, CLAMP)`.
    const MEASURED_HEAPS_AND_SAMPLER_SPACE: [u8; 100] = [
        0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x7f, 0x7f, 0x00, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    fn sampler(filter: u32, address: [u32; 3]) -> StaticSampler {
        StaticSampler {
            filter,
            address,
            mip_lod_bias: 0.0,
            max_anisotropy: 16,
            comparison: 4,
            border: 2,
            min_lod: 0.0,
            max_lod: f32::MAX,
        }
    }

    fn constants(values: u32, register: u32, space: u32) -> RootParameter {
        RootParameter {
            kind: ParameterKind::Constants { values },
            register,
            space,
            visibility: Visibility::All,
            stage: None,
        }
    }

    fn descriptor(
        kind: DescriptorKind,
        register: u32,
        space: u32,
        visibility: Visibility,
        volatile: bool,
    ) -> RootParameter {
        RootParameter {
            kind: ParameterKind::Descriptor { kind, volatile },
            register,
            space,
            visibility,
            stage: None,
        }
    }

    fn table(ranges: Vec<DescriptorRange>, visibility: Visibility) -> RootParameter {
        RootParameter {
            kind: ParameterKind::Table { ranges },
            register: 0,
            space: 0,
            visibility,
            stage: None,
        }
    }

    /// A translated stage that declares every kind of parameter.
    fn stage(stage: HwStage, view_marks: bool) -> HostShader {
        HostShader {
            stage,
            dxil: Vec::new(),
            user_data: vec![UserData {
                index: 2,
                slot: 1,
                root: RootType::Constants,
                dword: 0,
            }],
            samplers: Vec::new(),
            static_samplers: vec![sampler(0x15, [1, 2, 3]), sampler(0, [4, 5, 6])],
            root_cbvs: vec![0, 3],
            root_srvs: vec![1],
            root_uavs: vec![7],
            root_srv_bytes: Vec::new(),
            memory: true,
            memory_writes: true,
            view_addresses: true,
            heap: false,
            heap_writes: false,
            view_marks,
        }
    }

    /// What a reader of the blob sees; the description carries one thing the
    /// format does not, the stage that declared a parameter.
    #[derive(Debug, PartialEq, Eq)]
    struct Read {
        version: u32,
        flags: u32,
        sampler_offset: usize,
        parameters: Vec<ReadParameter>,
        samplers: Vec<[u32; 13]>,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ReadParameter {
        parameter_type: u32,
        visibility: u32,
        record: u32,
        /// Root constants or a root descriptor: register, space, count/flags.
        words: [u32; 3],
        ranges_at: u32,
        ranges: Vec<[u32; 6]>,
    }

    /// Parses the blob as the module docs lay it out, checking every offset.
    fn read(blob: &[u8]) -> Read {
        let u = |at: usize| -> u32 {
            let slot = blob
                .get(at..at + 4)
                .unwrap_or_else(|| panic!("read past the end at {at:#x}"));
            u32::from_le_bytes(slot.try_into().expect("4 bytes"))
        };
        assert_eq!(blob.len() % 4, 0, "a blob of whole DWORDs");
        let version = u(0);
        let parameters = u(4) as usize;
        let parameter_offset = u(8) as usize;
        let samplers = u(12) as usize;
        let sampler_offset = u(16) as usize;
        let flags = u(20);
        assert_eq!(
            parameter_offset, HEADER_BYTES,
            "the array follows the header"
        );
        assert_eq!(sampler_offset % 4, 0);
        let mut read_parameters = Vec::with_capacity(parameters);
        for i in 0..parameters {
            let at = parameter_offset + i * PARAMETER_BYTES;
            let parameter_type = u(at);
            let visibility = u(at + 4);
            let record = u(at + 8);
            assert_eq!(record as usize % 4, 0, "a record offset is a DWORD");
            let mut words = [0; 3];
            let mut ranges = Vec::new();
            let mut ranges_at = 0;
            if parameter_type == 0 {
                ranges_at = u(record as usize + 4);
                let n = u(record as usize) as usize;
                assert_eq!(
                    ranges_at as usize,
                    record as usize + TABLE_BYTES,
                    "a table's ranges follow its record"
                );
                for j in 0..n {
                    let at = ranges_at as usize + j * RANGE_BYTES;
                    ranges.push([
                        u(at),
                        u(at + 4),
                        u(at + 8),
                        u(at + 12),
                        u(at + 16),
                        u(at + 20),
                    ]);
                }
            } else {
                words = [
                    u(record as usize),
                    u(record as usize + 4),
                    u(record as usize + 8),
                ];
            }
            read_parameters.push(ReadParameter {
                parameter_type,
                visibility,
                record,
                words,
                ranges_at,
                ranges,
            });
        }
        let mut read_samplers = Vec::with_capacity(samplers);
        for i in 0..samplers {
            let at = sampler_offset + i * SAMPLER_BYTES;
            let mut words = [0; 13];
            for (k, word) in words.iter_mut().enumerate() {
                *word = u(at + 4 * k);
            }
            read_samplers.push(words);
        }
        assert_eq!(
            sampler_offset + samplers * SAMPLER_BYTES,
            blob.len(),
            "the blob ends with its static samplers"
        );
        Read {
            version,
            flags,
            sampler_offset,
            parameters: read_parameters,
            samplers: read_samplers,
        }
    }

    /// Reads `blob` and compares everything the format holds with `signature`,
    /// including that every offset is where the writer's layout puts it.
    fn compare(signature: &RootSignature, blob: &[u8]) {
        let read = read(blob);
        assert_eq!(read.version, VERSION_1_1);
        assert_eq!(read.flags, signature.flags().bits());
        assert_eq!(read.parameters.len(), signature.parameters().len());
        let mut at = HEADER_BYTES + signature.parameters().len() * PARAMETER_BYTES;
        for (parameter, read) in signature.parameters().iter().zip(read.parameters.iter()) {
            assert_eq!(read.parameter_type, parameter.kind.parameter_type());
            assert_eq!(read.visibility, parameter.visibility.bits());
            assert_eq!(
                read.record as usize, at,
                "records follow the array in order"
            );
            match &parameter.kind {
                ParameterKind::Constants { values } => {
                    assert_eq!(read.words, [parameter.register, parameter.space, *values]);
                    at += RECORD_BYTES;
                }
                ParameterKind::Descriptor { volatile, .. } => {
                    assert_eq!(
                        read.words,
                        [
                            parameter.register,
                            parameter.space,
                            if *volatile {
                                DESCRIPTOR_FLAG_DATA_VOLATILE
                            } else {
                                0
                            }
                        ]
                    );
                    at += RECORD_BYTES;
                }
                ParameterKind::Table { ranges } => {
                    assert_eq!(read.ranges_at as usize, at + TABLE_BYTES);
                    assert_eq!(read.ranges.len(), ranges.len());
                    for (range, words) in ranges.iter().zip(read.ranges.iter()) {
                        assert_eq!(
                            *words,
                            [
                                range.kind.bits(),
                                range.count,
                                range.register,
                                range.space,
                                0,
                                range.offset.unwrap_or(RANGE_OFFSET_APPEND),
                            ]
                        );
                    }
                    at += TABLE_BYTES + ranges.len() * RANGE_BYTES;
                }
            }
        }
        assert_eq!(read.sampler_offset, at, "samplers follow the parameters");
        assert_eq!(read.samplers.len(), signature.static_samplers().len());
        for (sampler, words) in signature.static_samplers().iter().zip(read.samplers.iter()) {
            let s = sampler.sampler;
            assert_eq!(
                *words,
                [
                    s.filter,
                    s.address[0],
                    s.address[1],
                    s.address[2],
                    s.mip_lod_bias.to_bits(),
                    s.max_anisotropy,
                    s.comparison,
                    s.border,
                    s.min_lod.to_bits(),
                    s.max_lod.to_bits(),
                    sampler.register,
                    sampler.space,
                    sampler.visibility.bits(),
                ]
            );
        }
    }

    #[test]
    fn the_blob_is_the_one_d3d12_writes() {
        let descriptors = RootSignature {
            flags: SignatureFlags::default(),
            parameters: vec![
                descriptor(DescriptorKind::Srv, 0, 0, Visibility::All, false),
                descriptor(DescriptorKind::Uav, 5, 0, Visibility::All, false),
                descriptor(DescriptorKind::Cbv, 2, 0, Visibility::All, false),
            ],
            samplers: Vec::new(),
        };
        assert_eq!(
            descriptors.to_versioned_blob().unwrap(),
            MEASURED_DESCRIPTORS
        );
        compare(&descriptors, &MEASURED_DESCRIPTORS);

        let constants_table_sampler = RootSignature {
            flags: SignatureFlags::default(),
            parameters: vec![
                constants(4, 3, 0),
                table(
                    vec![DescriptorRange {
                        kind: RangeKind::Srv,
                        register: 10,
                        space: 0,
                        count: 2,
                        offset: None,
                    }],
                    Visibility::All,
                ),
            ],
            samplers: vec![RootSampler {
                sampler: sampler(0x15, [1, 3, 2]),
                register: 0,
                space: 0,
                visibility: Visibility::All,
            }],
        };
        assert_eq!(
            constants_table_sampler.to_versioned_blob().unwrap(),
            MEASURED_CONSTANTS_TABLE_SAMPLER
        );
        compare(&constants_table_sampler, &MEASURED_CONSTANTS_TABLE_SAMPLER);

        let tables = RootSignature {
            flags: SignatureFlags::default(),
            parameters: vec![
                table(
                    vec![DescriptorRange {
                        kind: RangeKind::Cbv,
                        register: 0,
                        space: 0,
                        count: 3,
                        offset: None,
                    }],
                    Visibility::Pixel,
                ),
                table(
                    vec![DescriptorRange {
                        kind: RangeKind::Uav,
                        register: 2,
                        space: 0,
                        count: 1,
                        offset: None,
                    }],
                    Visibility::Vertex,
                ),
            ],
            samplers: Vec::new(),
        };
        assert_eq!(tables.to_versioned_blob().unwrap(), MEASURED_TABLES);
        compare(&tables, &MEASURED_TABLES);

        // Both directly indexed heap flags and a static sampler in a
        // nonzero register space: the translated signature's flags and
        // `SAMPLER_SPACE`.
        let heaps = RootSignature {
            flags: SignatureFlags {
                allow_input_assembler_input_layout: false,
                cbv_srv_uav_heap_directly_indexed: true,
                sampler_heap_directly_indexed: true,
            },
            parameters: vec![descriptor(
                DescriptorKind::Srv,
                0,
                0,
                Visibility::All,
                false,
            )],
            samplers: vec![RootSampler {
                sampler: sampler(0, [3, 3, 3]),
                register: 0,
                space: 3,
                visibility: Visibility::All,
            }],
        };
        assert_eq!(
            heaps.to_versioned_blob().unwrap(),
            MEASURED_HEAPS_AND_SAMPLER_SPACE
        );
        compare(&heaps, &MEASURED_HEAPS_AND_SAMPLER_SPACE);
    }

    #[test]
    fn a_stage_describes_its_bindings_in_binding_order() {
        let shader = stage(HwStage::Ps, false);
        let signature = translated_root_signature(&shader).expect("described");
        let expected: Vec<(ParameterKind, u32, u32)> = vec![
            (ParameterKind::Constants { values: 4 }, 0, USER_DATA_SPACE),
            (
                ParameterKind::Descriptor {
                    kind: DescriptorKind::Cbv,
                    volatile: true,
                },
                0,
                ROOT_CBV_SPACE,
            ),
            (
                ParameterKind::Descriptor {
                    kind: DescriptorKind::Cbv,
                    volatile: true,
                },
                3,
                ROOT_CBV_SPACE,
            ),
            (
                ParameterKind::Descriptor {
                    kind: DescriptorKind::Srv,
                    volatile: true,
                },
                1,
                ROOT_CBV_SPACE,
            ),
            (
                ParameterKind::Descriptor {
                    kind: DescriptorKind::Uav,
                    volatile: true,
                },
                7,
                ROOT_CBV_SPACE,
            ),
            (
                ParameterKind::Descriptor {
                    kind: DescriptorKind::Srv,
                    volatile: true,
                },
                0,
                MEMORY_SPACE,
            ),
            (
                ParameterKind::Descriptor {
                    kind: DescriptorKind::Uav,
                    volatile: true,
                },
                0,
                MEMORY_SPACE,
            ),
            (
                ParameterKind::Descriptor {
                    kind: DescriptorKind::Srv,
                    volatile: true,
                },
                1,
                MEMORY_SPACE,
            ),
        ];
        let described: Vec<(ParameterKind, u32, u32)> = signature
            .parameters()
            .iter()
            .map(|p| (p.kind.clone(), p.register, p.space))
            .collect();
        assert_eq!(described, expected);
        assert!(
            signature
                .parameters()
                .iter()
                .all(|p| p.stage == Some(HwStage::Ps) && p.visibility == Visibility::Pixel)
        );
        let samplers: Vec<(u32, u32)> = signature
            .static_samplers()
            .iter()
            .map(|s| (s.register, s.space))
            .collect();
        assert_eq!(samplers, vec![(0, SAMPLER_SPACE), (1, SAMPLER_SPACE)]);
        assert!(
            signature
                .static_samplers()
                .iter()
                .all(|s| s.visibility == Visibility::Pixel)
        );

        let blob = signature.to_versioned_blob().expect("serialized");
        compare(&signature, &blob);
        assert_eq!(
            signature.to_versioned_blob().unwrap(),
            blob,
            "deterministic"
        );
        assert_eq!(
            signature,
            translated_root_signature(&shader).unwrap(),
            "the same shader describes the same signature"
        );
    }

    #[test]
    fn view_marks_come_once_after_every_stage() {
        let vs = stage(HwStage::Vs, true);
        let ps = stage(HwStage::Ps, false);
        let flags = SignatureFlags {
            allow_input_assembler_input_layout: true,
            cbv_srv_uav_heap_directly_indexed: true,
            sampler_heap_directly_indexed: true,
        };
        let signature = translated_pipeline_root_signature(&[&vs, &ps], flags).expect("described");
        // Eight parameters and two samplers per stage, then the pair.
        assert_eq!(signature.parameters().len(), 8 + 8 + 2);
        assert_eq!(signature.parameters()[0].stage, Some(HwStage::Vs));
        assert_eq!(signature.parameters()[8].stage, Some(HwStage::Ps));
        let (marks, writes) = signature.view_mark_parameters().expect("the pair");
        assert_eq!((marks, writes), (16, 17));
        for (index, register) in [(16usize, VIEW_MARKS), (17, VIEW_WRITES)] {
            let parameter = &signature.parameters()[index];
            assert_eq!(parameter.register, register);
            assert_eq!(parameter.space, MEMORY_SPACE);
            assert_eq!(parameter.stage, None);
            assert_eq!(parameter.visibility, Visibility::All);
        }
        let blob = signature.to_versioned_blob().expect("serialized");
        compare(&signature, &blob);
        // D3D12's ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT and both direct-indexed
        // heap flags.
        assert_eq!(signature.flags().bits(), 1 | 1024 | 2048);
        assert_eq!(
            u32::from_le_bytes(blob[20..24].try_into().unwrap()),
            1 | 1024 | 2048
        );
    }

    #[test]
    fn too_many_parameters_are_refused() {
        let mut signature = RootSignature::new(SignatureFlags::default());
        for _ in 0..MAX_ROOT_PARAMETERS {
            signature.parameters.push(constants(1, 0, 0));
        }
        assert!(signature.to_versioned_blob().is_ok());
        signature.parameters.push(constants(1, 0, 0));
        assert_eq!(
            signature.to_versioned_blob(),
            Err(ShaderError::RootSignature("more than 64 root parameters"))
        );
        assert_eq!(MAX_ROOT_PARAMETERS, 64);
    }

    #[test]
    fn more_samplers_than_d3d12_allows_are_refused() {
        let mut signature = RootSignature::new(SignatureFlags::default());
        for i in 0..=MAX_STATIC_SAMPLERS {
            signature.samplers.push(RootSampler {
                sampler: sampler(0, [1, 2, 3]),
                register: i as u32,
                space: SAMPLER_SPACE,
                visibility: Visibility::All,
            });
        }
        assert_eq!(
            signature.to_versioned_blob(),
            Err(ShaderError::RootSignature("more than 2032 static samplers"))
        );
    }

    #[test]
    fn a_table_too_large_to_describe_is_refused() {
        let ranges = vec![
            DescriptorRange {
                kind: RangeKind::Srv,
                register: 0,
                space: 0,
                count: 1,
                offset: None,
            };
            MAX_BLOB_BYTES / RANGE_BYTES
        ];
        let mut signature = RootSignature::new(SignatureFlags::default());
        signature.parameters.push(table(ranges, Visibility::All));
        assert_eq!(
            signature.to_versioned_blob(),
            Err(ShaderError::RootSignature(
                "the root signature is larger than 1 MiB"
            ))
        );
    }
}
