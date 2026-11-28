//! Host-side Metal vertex layout derived from a translated vertex shader.
//!
//! Metal refuses to create a graphics pipeline whose vertex function declares
//! `[[attribute(n)]]` inputs unless the `MTLRenderPipelineDescriptor` carries a
//! matching `MTLVertexDescriptor`. A version 3 pipeline frame does not send the
//! guest's D3D12 input layout, but it does send each stage's complete DXIL
//! container, so the vertex stage's `ISG1` input signature is available at
//! registration time. This module turns that signature into the attribute
//! formats, offsets, buffer indices and strides the registry applies.
//!
//! # Attribute index assumption
//!
//! The `ISG1` rows are written in register order by the container builder (the
//! same order the console shader declares its inputs). Apple's Metal Shader
//! Converter does not number the first vertex input at 0: it starts the first
//! signature element at attribute index 11 and advances the index for every
//! element, including a system value it turns into a builtin. This is measured,
//! not assumed: the kept Boot #21 vertex signature with three non-system float
//! rows at elements 0/1/2 translates to `MTLFunction.vertexAttributes` indices
//! 11/12/13, and a shader with `SV_VertexID` first shifts the attributes to
//! 12/13. This module therefore assigns `index = 11 + element` to every
//! non-system element; a system value advances the index but produces no
//! attribute. An element whose index would exceed Metal's 30 is refused by
//! name rather than creating a descriptor Metal would reject.
//!
//! `SV_VertexID` and `SV_InstanceID` are Metal's built-in `[[vertex_id]]` and
//! `[[instance_id]]`; they consume no attribute slot and are skipped. Any other
//! system-value input has no Metal attribute mapping and is refused by name.
//!
//! # Buffer packing assumption
//!
//! The input signature does not carry the guest's per-slot byte offsets or
//! strides either. Every derived attribute is packed into vertex buffer 0 in
//! signature order, each offset 4-byte aligned, with one stride equal to the
//! packed element size. A draw that uses this descriptor must bind an
//! interleaved buffer with exactly that layout; a later IA-layout packet can
//! replace this packing without changing the pipeline descriptor shape.
//!
//! # Wire chunk
//!
//! The layout rides at the end of the existing `state` blob (the same bytes a
//! version 1/2 frame already carries), so the `BatchExecutor` trait does not
//! change:
//!
//! ```text
//! 0   "VMVD"
//! 4   version (1)
//! 5   attribute count (1..=31)
//! 6   per attribute, 11 bytes:
//!       +0 index (u8)
//!       +1 format (u8)
//!       +2 offset (u32 little-endian)
//!       +6 buffer index (u8)
//!       +7 stride (u32 little-endian)
//! ```
//!
//! A state blob without this chunk decodes exactly as before and sets no
//! vertex descriptor.

use victory_shader::{Container, FourCC};

/// `ISG1` element row size in bytes.
const SIGNATURE_ELEMENT_LEN: usize = 32;
/// Offset of the system-value type in one `ISG1` element.
const ELEMENT_SYSTEM_VALUE: usize = 12;
/// Offset of the D3D register component type in one `ISG1` element.
const ELEMENT_COMPONENT_TYPE: usize = 16;
/// Offset of the component mask in one `ISG1` element.
const ELEMENT_MASK: usize = 24;

/// Largest attribute count a Metal vertex descriptor accepts.
pub const MAX_VERTEX_ATTRIBUTES: usize = 31;
/// Largest vertex buffer index a Metal vertex descriptor accepts.
const MAX_VERTEX_BUFFERS: usize = 31;

/// Magic that starts the vertex-layout chunk appended to a state blob.
pub const VERTEX_LAYOUT_MAGIC: [u8; 4] = *b"VMVD";
/// Version of the vertex-layout chunk this module writes and reads.
pub const VERTEX_LAYOUT_VERSION: u8 = 1;
/// Header length of the chunk.
pub const VERTEX_LAYOUT_HEADER_LEN: usize = 6;
/// Encoded length of one attribute.
pub const VERTEX_ATTRIBUTE_LEN: usize = 11;

/// D3D register component type `UINT32`.
const D3D_COMPONENT_UINT: u32 = 1;
/// D3D register component type `SINT32`.
const D3D_COMPONENT_SINT: u32 = 2;
/// D3D register component type `FLOAT32`.
const D3D_COMPONENT_FLOAT: u32 = 3;

/// D3D system-value `SV_VertexID`.
const SV_VERTEX_ID: u32 = 6;
/// D3D system-value `SV_InstanceID`.
const SV_INSTANCE_ID: u32 = 8;

/// First Metal attribute index the converter assigns to a signature element.
///
/// Measured from the kept title shaders: three non-system inputs at elements
/// 0/1/2 become Metal attribute indices 11/12/13, and a leading
/// `SV_VertexID` shifts the following attributes to 12/13.
const CONVERTER_ATTRIBUTE_BASE: u8 = 11;
/// Largest Metal attribute index (the descriptor holds indices 0..=30).
const MAX_ATTRIBUTE_INDEX: u8 = 30;

/// First format code of each group; four consecutive codes are x1..x4.
const FLOAT32_BASE: u8 = 0x01;
const HALF_BASE: u8 = 0x05;
const UINT8_BASE: u8 = 0x09;
const SINT8_BASE: u8 = 0x0d;
const UNORM8_BASE: u8 = 0x11;
const SNORM8_BASE: u8 = 0x15;
const UINT16_BASE: u8 = 0x19;
const SINT16_BASE: u8 = 0x1d;
const UNORM16_BASE: u8 = 0x21;
const SNORM16_BASE: u8 = 0x25;
const UINT32_BASE: u8 = 0x29;
const SINT32_BASE: u8 = 0x2d;

/// One supported Metal vertex format, encoded as a stable one-byte code.
///
/// Codes `1..=48` cover twelve format groups of x1..x4 components:
/// 32-bit float, half, uint8, sint8, unorm8, snorm8, uint16, sint16, unorm16,
/// snorm16, uint32 and sint32. Anything else is rejected, never dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VertexFormat(u8);

impl VertexFormat {
    /// The 32-bit float group (`MTLVertexFormatFloat`..`Float4`).
    pub const FLOAT: u8 = FLOAT32_BASE;
    /// The half group (`MTLVertexFormatHalf`..`Half4`).
    pub const HALF: u8 = HALF_BASE;
    /// The uint8 group (`MTLVertexFormatUChar`..`UChar4`).
    pub const UINT8: u8 = UINT8_BASE;
    /// The sint8 group (`MTLVertexFormatChar`..`Char4`).
    pub const SINT8: u8 = SINT8_BASE;
    /// The unorm8 group (`MTLVertexFormatUCharNormalized`..`UChar4Normalized`).
    pub const UNORM8: u8 = UNORM8_BASE;
    /// The snorm8 group (`MTLVertexFormatCharNormalized`..`Char4Normalized`).
    pub const SNORM8: u8 = SNORM8_BASE;
    /// The uint16 group (`MTLVertexFormatUShort`..`UShort4`).
    pub const UINT16: u8 = UINT16_BASE;
    /// The sint16 group (`MTLVertexFormatShort`..`Short4`).
    pub const SINT16: u8 = SINT16_BASE;
    /// The unorm16 group (`MTLVertexFormatUShortNormalized`..`UShort4Normalized`).
    pub const UNORM16: u8 = UNORM16_BASE;
    /// The snorm16 group (`MTLVertexFormatShortNormalized`..`Short4Normalized`).
    pub const SNORM16: u8 = SNORM16_BASE;
    /// The uint32 group (`MTLVertexFormatUInt`..`UInt4`).
    pub const UINT32: u8 = UINT32_BASE;
    /// The sint32 group (`MTLVertexFormatInt`..`Int4`).
    pub const SINT32: u8 = SINT32_BASE;

    /// The format in `group` with `components` (1..=4) components.
    pub const fn with_components(group: u8, components: u8) -> Option<Self> {
        if components == 0 || components > 4 {
            return None;
        }
        let base = match group {
            FLOAT32_BASE | HALF_BASE | UINT8_BASE | SINT8_BASE | UNORM8_BASE | SNORM8_BASE
            | UINT16_BASE | SINT16_BASE | UNORM16_BASE | SNORM16_BASE | UINT32_BASE
            | SINT32_BASE => group,
            _ => return None,
        };
        Some(Self(base + components - 1))
    }

    /// Decodes one stable format code, rejecting every unsupported code.
    pub const fn from_code(code: u8) -> Option<Self> {
        if code >= FLOAT32_BASE && code <= SINT32_BASE + 3 {
            Some(Self(code))
        } else {
            None
        }
    }

    /// The stable one-byte code carried in the chunk.
    pub const fn code(self) -> u8 {
        self.0
    }

    /// Number of components in this format (1..=4).
    pub const fn components(self) -> u8 {
        (self.0 - 1) % 4 + 1
    }

    /// Bytes of one component.
    pub const fn component_size(self) -> u8 {
        const SIZES: [u8; 12] = [4, 2, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4];
        SIZES[((self.0 - 1) / 4) as usize]
    }

    /// Bytes one vertex attribute occupies.
    pub const fn byte_size(self) -> u32 {
        self.components() as u32 * self.component_size() as u32
    }

    /// The `MTLVertexFormat` value this format maps to.
    ///
    /// Metal has no x1..x3 variants for several families, so the single
    /// component forms use the dedicated scalar formats at `45..=56`; the
    /// x2..x4 forms use the packed-family values at `1..=36`.
    pub const fn mtl_format(self) -> usize {
        const FORMATS: [[usize; 4]; 12] = [
            [54, 28, 29, 30], // float
            [53, 25, 26, 27], // half
            [45, 1, 2, 3],    // uchar
            [46, 4, 5, 6],    // char
            [47, 7, 8, 9],    // uchar normalized
            [48, 10, 11, 12], // char normalized
            [49, 13, 14, 15], // ushort
            [50, 16, 17, 18], // short
            [51, 19, 20, 21], // ushort normalized
            [52, 22, 23, 24], // short normalized
            [56, 34, 35, 36], // uint
            [55, 31, 32, 33], // int
        ];
        FORMATS[((self.0 - 1) / 4) as usize][(self.components() - 1) as usize]
    }
}

/// One Metal vertex attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    /// Metal attribute index (`[[attribute(index)]]`).
    pub index: u8,
    /// Metal attribute format.
    pub format: VertexFormat,
    /// Byte offset inside the vertex buffer.
    pub offset: u32,
    /// Metal vertex buffer index.
    pub buffer_index: u8,
    /// Byte stride of the buffer this attribute reads.
    pub stride: u32,
}

/// A complete graphics vertex layout.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct VertexLayout {
    /// Attributes in Metal index order.
    attributes: Vec<VertexAttribute>,
}

/// Why a vertex layout could not be derived or decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VertexLayoutError {
    /// The input signature container is not a DXBC container.
    NotDxbc,
    /// The signature row is malformed or truncated.
    Malformed(&'static str),
    /// A non-builtin system-value input has no Metal attribute mapping.
    SystemValue { index: usize, system_value: u32 },
    /// The signature declares a component type this host cannot map.
    UnknownComponentType {
        /// Signature element index.
        index: usize,
        /// The raw D3D register component type.
        component_type: u32,
    },
    /// The signature declares more attributes than Metal accepts.
    TooManyAttributes(usize),
    /// A signature element maps to an attribute index Metal cannot hold.
    AttributeIndexOutOfRange {
        /// Signature element index.
        element: usize,
        /// The Metal attribute index it would need.
        index: u8,
    },
    /// A chunk carries a format code outside the supported set.
    UnknownFormat(u8),
    /// A chunk does not start with the vertex-layout magic.
    BadMagic,
    /// A chunk uses a version this host does not read.
    UnsupportedVersion(u8),
    /// A chunk's attribute count is outside `1..=31`.
    BadCount(usize),
    /// Two attributes claim the same Metal index.
    DuplicateIndex(u8),
    /// An attribute names a vertex buffer index Metal does not accept.
    InvalidBuffer(u8),
    /// A chunk has bytes after its declared attributes.
    TrailingBytes,
}

impl std::fmt::Display for VertexLayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotDxbc => f.write_str("the vertex stage container is not DXBC"),
            Self::Malformed(reason) => write!(f, "the vertex input signature is malformed: {reason}"),
            Self::SystemValue {
                index,
                system_value,
            } => write!(
                f,
                "input signature element {index} is system value {system_value}, which has no vertex attribute"
            ),
            Self::UnknownComponentType {
                index,
                component_type,
            } => write!(
                f,
                "input signature element {index} has unsupported component type {component_type}"
            ),
            Self::TooManyAttributes(count) => write!(
                f,
                "the vertex input signature declares {count} attributes; Metal accepts at most {MAX_VERTEX_ATTRIBUTES}"
            ),
            Self::AttributeIndexOutOfRange { element, index } => write!(
                f,
                "input signature element {element} needs Metal attribute index {index}, above {MAX_ATTRIBUTE_INDEX}"
            ),
            Self::UnknownFormat(code) => write!(f, "vertex attribute format code {code} is unsupported"),
            Self::BadMagic => f.write_str("the vertex layout chunk has no VMVD magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "vertex layout chunk version {version} is unsupported")
            }
            Self::BadCount(count) => write!(
                f,
                "the vertex layout chunk declares {count} attributes; expected 1..={MAX_VERTEX_ATTRIBUTES}"
            ),
            Self::DuplicateIndex(index) => {
                write!(f, "two vertex attributes claim index {index}")
            }
            Self::InvalidBuffer(index) => {
                write!(f, "vertex buffer index {index} exceeds Metal's limit")
            }
            Self::TrailingBytes => f.write_str("the vertex layout chunk has trailing bytes"),
        }
    }
}

impl std::error::Error for VertexLayoutError {}

impl VertexLayout {
    /// An empty layout: the vertex function needs no vertex descriptor.
    pub const fn empty() -> Self {
        Self {
            attributes: Vec::new(),
        }
    }

    /// The attributes in Metal index order.
    pub fn attributes(&self) -> &[VertexAttribute] {
        &self.attributes
    }

    /// Builds a layout from already-derived attributes, validating the bounds
    /// the chunk decoder and Metal both enforce.
    pub fn from_attributes(
        mut attributes: Vec<VertexAttribute>,
    ) -> Result<Self, VertexLayoutError> {
        validate_attributes(&mut attributes)?;
        Ok(Self { attributes })
    }

    /// Whether this layout has no attributes.
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    /// Appends this layout's wire chunk to an existing `state` blob.
    pub fn append_to(&self, state: &mut Vec<u8>) {
        state.extend_from_slice(&self.encode());
    }

    /// Encodes this layout as a complete wire chunk.
    ///
    /// A layout with no attributes encodes to an empty chunk; callers should
    /// use [`Self::append_to`] only for a non-empty layout.
    pub fn encode(&self) -> Vec<u8> {
        if self.attributes.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(
            VERTEX_LAYOUT_HEADER_LEN + VERTEX_ATTRIBUTE_LEN * self.attributes.len(),
        );
        out.extend_from_slice(&VERTEX_LAYOUT_MAGIC);
        out.push(VERTEX_LAYOUT_VERSION);
        out.push(u8::try_from(self.attributes.len()).expect("attribute count is bounded by 31"));
        for attribute in &self.attributes {
            out.push(attribute.index);
            out.push(attribute.format.code());
            out.extend_from_slice(&attribute.offset.to_le_bytes());
            out.push(attribute.buffer_index);
            out.extend_from_slice(&attribute.stride.to_le_bytes());
        }
        out
    }

    /// Decodes one complete vertex-layout chunk.
    pub fn decode(bytes: &[u8]) -> Result<Self, VertexLayoutError> {
        if bytes.len() < VERTEX_LAYOUT_HEADER_LEN {
            return Err(VertexLayoutError::Malformed(
                "the vertex layout chunk is truncated",
            ));
        }
        if bytes[..4] != VERTEX_LAYOUT_MAGIC {
            return Err(VertexLayoutError::BadMagic);
        }
        let version = bytes[4];
        if version != VERTEX_LAYOUT_VERSION {
            return Err(VertexLayoutError::UnsupportedVersion(version));
        }
        let count = usize::from(bytes[5]);
        if count == 0 || count > MAX_VERTEX_ATTRIBUTES {
            return Err(VertexLayoutError::BadCount(count));
        }
        let expected = VERTEX_LAYOUT_HEADER_LEN + VERTEX_ATTRIBUTE_LEN * count;
        match bytes.len().cmp(&expected) {
            std::cmp::Ordering::Less => {
                return Err(VertexLayoutError::Malformed(
                    "the vertex layout chunk is truncated",
                ));
            }
            std::cmp::Ordering::Greater => return Err(VertexLayoutError::TrailingBytes),
            std::cmp::Ordering::Equal => {}
        }

        let mut attributes = Vec::with_capacity(count);
        for i in 0..count {
            let at = VERTEX_LAYOUT_HEADER_LEN + VERTEX_ATTRIBUTE_LEN * i;
            let index = bytes[at];
            let format = VertexFormat::from_code(bytes[at + 1])
                .ok_or(VertexLayoutError::UnknownFormat(bytes[at + 1]))?;
            let offset = u32::from_le_bytes(
                bytes[at + 2..at + 6]
                    .try_into()
                    .expect("four offset bytes"),
            );
            let buffer_index = bytes[at + 6];
            let stride = u32::from_le_bytes(
                bytes[at + 7..at + 11]
                    .try_into()
                    .expect("four stride bytes"),
            );
            attributes.push(VertexAttribute {
                index,
                format,
                offset,
                buffer_index,
                stride,
            });
        }
        validate_attributes(&mut attributes)?;
        Ok(Self { attributes })
    }

    /// Derives a layout from a vertex stage's DXIL container.
    ///
    /// A payload that is not a DXBC container at all (for example a stub used
    /// by a converter test) yields an empty layout: the converter owns whether
    /// those bytes are usable, and this module must not claim an input
    /// signature that is not there. A DXBC container whose `ISG1` cannot be
    /// turned into a Metal layout is an explicit error.
    pub fn from_dxil_container(bytes: &[u8]) -> Result<Self, VertexLayoutError> {
        let container = match Container::parse(bytes) {
            Ok(container) => container,
            Err(_) if !bytes.starts_with(b"DXBC") => return Ok(Self::empty()),
            Err(_) => return Err(VertexLayoutError::NotDxbc),
        };
        let Some(part) = container.part(FourCC::ISG1) else {
            return Ok(Self::empty());
        };
        let count = read_u32(part.data, 0)? as usize;
        let base = read_u32(part.data, 4)? as usize;
        if count > MAX_VERTEX_ATTRIBUTES {
            return Err(VertexLayoutError::TooManyAttributes(count));
        }

        let mut attributes: Vec<VertexAttribute> = Vec::with_capacity(count);
        let mut next_offset = 0u32;
        for element in 0..count {
            let at = base
                .checked_add(SIGNATURE_ELEMENT_LEN * element)
                .ok_or(VertexLayoutError::Malformed(
                    "the input signature offset overflows",
                ))?;
            let system_value = read_u32(part.data, at + ELEMENT_SYSTEM_VALUE)?;
            if system_value != 0 {
                // Metal synthesises these two from the draw, so they occupy no
                // attribute slot in the descriptor.
                if system_value == SV_VERTEX_ID || system_value == SV_INSTANCE_ID {
                    continue;
                }
                return Err(VertexLayoutError::SystemValue {
                    index: element,
                    system_value,
                });
            }
            let component_type = read_u32(part.data, at + ELEMENT_COMPONENT_TYPE)?;
            let mask = *part.data.get(at + ELEMENT_MASK).ok_or(
                VertexLayoutError::Malformed("the input signature mask is truncated"),
            )?;
            let components = mask.count_ones();
            if components == 0 || components > 4 || mask != (1u8 << components) - 1 {
                return Err(VertexLayoutError::Malformed(
                    "an input signature mask is not 1..=4 contiguous components",
                ));
            }
            let components = components as u8;
            let group = match component_type {
                D3D_COMPONENT_UINT => UINT32_BASE,
                D3D_COMPONENT_SINT => SINT32_BASE,
                D3D_COMPONENT_FLOAT => FLOAT32_BASE,
                other => {
                    return Err(VertexLayoutError::UnknownComponentType {
                        index: element,
                        component_type: other,
                    });
                }
            };
            let format = VertexFormat::with_components(group, components)
                .expect("a validated component count builds a format");
            let index = CONVERTER_ATTRIBUTE_BASE + u8::try_from(element).expect("count is <= 31");
            if index > MAX_ATTRIBUTE_INDEX {
                return Err(VertexLayoutError::AttributeIndexOutOfRange { element, index });
            }
            let offset = align_four(next_offset);
            next_offset = offset
                .checked_add(format.byte_size())
                .ok_or(VertexLayoutError::Malformed(
                    "the packed vertex stride overflows",
                ))?;
            attributes.push(VertexAttribute {
                index,
                format,
                offset,
                buffer_index: 0,
                stride: 0,
            });
        }

        let stride = align_four(next_offset);
        for attribute in &mut attributes {
            attribute.stride = stride;
        }
        attributes.sort_by_key(|attribute| attribute.index);
        Ok(Self { attributes })
    }
}

/// Validates a layout's attributes and sorts them into Metal index order.
fn validate_attributes(attributes: &mut Vec<VertexAttribute>) -> Result<(), VertexLayoutError> {
    if attributes.is_empty() || attributes.len() > MAX_VERTEX_ATTRIBUTES {
        return Err(VertexLayoutError::BadCount(attributes.len()));
    }
    let mut seen = [false; MAX_VERTEX_ATTRIBUTES];
    let mut buffer_strides: [Option<u32>; MAX_VERTEX_BUFFERS] = [None; MAX_VERTEX_BUFFERS];
    for attribute in attributes.iter() {
        let slot = usize::from(attribute.index);
        if slot >= MAX_VERTEX_ATTRIBUTES {
            return Err(VertexLayoutError::BadCount(attributes.len()));
        }
        if seen[slot] {
            return Err(VertexLayoutError::DuplicateIndex(attribute.index));
        }
        seen[slot] = true;
        let buffer = usize::from(attribute.buffer_index);
        if buffer >= MAX_VERTEX_BUFFERS {
            return Err(VertexLayoutError::InvalidBuffer(attribute.buffer_index));
        }
        if attribute.stride == 0 {
            return Err(VertexLayoutError::Malformed("a vertex stride is zero"));
        }
        match buffer_strides[buffer] {
            Some(stride) if stride != attribute.stride => {
                return Err(VertexLayoutError::Malformed(
                    "attributes of one vertex buffer disagree on stride",
                ));
            }
            Some(_) => {}
            None => buffer_strides[buffer] = Some(attribute.stride),
        }
    }
    attributes.sort_by_key(|attribute| attribute.index);
    Ok(())
}

/// Rounds `value` up to the next 4-byte boundary.
const fn align_four(value: u32) -> u32 {
    (value + 3) & !3
}

fn read_u32(data: &[u8], at: usize) -> Result<u32, VertexLayoutError> {
    data.get(at..at + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .ok_or(VertexLayoutError::Malformed(
            "the input signature row is truncated",
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One synthetic `ISG1` element.
    struct Element {
        name: &'static str,
        semantic_index: u32,
        system_value: u32,
        component_type: u32,
        mask: u8,
    }

    /// Builds an `ISG1` part exactly as the container writer does: a count, the
    /// row base and 32-byte rows whose name offsets point into a name table.
    fn isg1(elements: &[Element]) -> Vec<u8> {
        let base = 8u32;
        let mut names = Vec::new();
        let mut name_offsets = Vec::new();
        let mut next = base + (SIGNATURE_ELEMENT_LEN * elements.len()) as u32;
        for element in elements {
            name_offsets.push(next);
            names.extend_from_slice(element.name.as_bytes());
            names.push(0);
            next += element.name.len() as u32 + 1;
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(elements.len() as u32).to_le_bytes());
        out.extend_from_slice(&base.to_le_bytes());
        for (index, element) in elements.iter().enumerate() {
            // stream, name offset, semantic index, system value, component
            // type, register, mask, read/write mask, reserved.
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&name_offsets[index].to_le_bytes());
            out.extend_from_slice(&element.semantic_index.to_le_bytes());
            out.extend_from_slice(&element.system_value.to_le_bytes());
            out.extend_from_slice(&element.component_type.to_le_bytes());
            out.extend_from_slice(&(index as u32).to_le_bytes());
            out.push(element.mask);
            out.push(element.mask);
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        out.extend_from_slice(&names);
        out
    }

    /// Wraps one part in the minimal DXBC container.
    fn container(fourcc: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let header = 0x20usize;
        let table = header + 4;
        let total = table + 8 + data.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"DXBC");
        out.extend_from_slice(&[0; 16]);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(table as u32).to_le_bytes());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    fn float(_register: u32, index: u32, mask: u8, name: &'static str) -> Element {
        Element {
            name,
            semantic_index: index,
            system_value: 0,
            component_type: D3D_COMPONENT_FLOAT,
            mask,
        }
    }

    #[test]
    fn chunk_round_trips_every_supported_format() {
        let groups = [
            VertexFormat::FLOAT,
            VertexFormat::HALF,
            VertexFormat::UINT8,
            VertexFormat::SINT8,
            VertexFormat::UNORM8,
            VertexFormat::SNORM8,
            VertexFormat::UINT16,
            VertexFormat::SINT16,
            VertexFormat::UNORM16,
            VertexFormat::SNORM16,
            VertexFormat::UINT32,
            VertexFormat::SINT32,
        ];
        let mut codes = Vec::new();
        for group in groups {
            for components in 1..=4 {
                let format = VertexFormat::with_components(group, components).unwrap();
                assert_eq!(format.components(), components);
                codes.push(format.code());
                let layout = VertexLayout {
                    attributes: vec![VertexAttribute {
                        index: 0,
                        format,
                        offset: 0,
                        buffer_index: 0,
                        stride: format.byte_size(),
                    }],
                };
                let encoded = layout.encode();
                assert_eq!(
                    encoded.len(),
                    VERTEX_LAYOUT_HEADER_LEN + VERTEX_ATTRIBUTE_LEN
                );
                assert_eq!(&encoded[..4], &VERTEX_LAYOUT_MAGIC);
                assert_eq!(encoded[4], VERTEX_LAYOUT_VERSION);
                assert_eq!(VertexLayout::decode(&encoded).unwrap(), layout);
                // Metal has a distinct format value for every supported code.
                assert!(format.mtl_format() <= 56, "code {:#x}", format.code());
            }
        }
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), 48, "twelve groups of four formats");
        assert_eq!(codes.first(), Some(&1));
        assert_eq!(codes.last(), Some(&0x30));

        // Appending to an arbitrary state blob leaves its prefix untouched.
        let layout = VertexLayout {
            attributes: vec![VertexAttribute {
                index: 0,
                format: VertexFormat::with_components(VertexFormat::FLOAT, 3).unwrap(),
                offset: 0,
                buffer_index: 0,
                stride: 12,
            }],
        };
        let mut blob = b"VPS1\x01".to_vec();
        layout.append_to(&mut blob);
        assert_eq!(&blob[..5], b"VPS1\x01");
        assert_eq!(VertexLayout::decode(&blob[5..]).unwrap(), layout);
    }

    #[test]
    fn two_element_signature_packs_offsets_and_order() {
        let bytes = container(
            b"ISG1",
            &isg1(&[
                float(0x0000, 0, 0x07, "POSITION"),
                float(0x0001, 0, 0x0f, "COLOR"),
            ]),
        );
        let layout = VertexLayout::from_dxil_container(&bytes).expect("two float inputs lay out");
        let attributes = layout.attributes();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0].index, 11);
        assert_eq!(attributes[0].format, VertexFormat::with_components(VertexFormat::FLOAT, 3).unwrap());
        assert_eq!(attributes[0].offset, 0);
        assert_eq!(attributes[0].buffer_index, 0);
        assert_eq!(attributes[0].stride, 28);
        assert_eq!(attributes[1].index, 12);
        assert_eq!(attributes[1].format, VertexFormat::with_components(VertexFormat::FLOAT, 4).unwrap());
        assert_eq!(attributes[1].offset, 12);
        assert_eq!(attributes[1].stride, 28);
    }

    #[test]
    fn integer_component_types_map_to_integer_formats() {
        let bytes = container(
            b"ISG1",
            &isg1(&[
                Element {
                    name: "BLENDINDICES",
                    semantic_index: 0,
                    system_value: 0,
                    component_type: D3D_COMPONENT_UINT,
                    mask: 0x03,
                },
                Element {
                    name: "BLENDWEIGHT",
                    semantic_index: 0,
                    system_value: 0,
                    component_type: D3D_COMPONENT_SINT,
                    mask: 0x01,
                },
            ]),
        );
        let layout = VertexLayout::from_dxil_container(&bytes).unwrap();
        assert_eq!(
            layout.attributes()[0].format,
            VertexFormat::with_components(VertexFormat::UINT32, 2).unwrap()
        );
        assert_eq!(
            layout.attributes()[1].format,
            VertexFormat::with_components(VertexFormat::SINT32, 1).unwrap()
        );
    }

    #[test]
    fn vertex_and_instance_ids_occupy_no_attribute_slot() {
        let bytes = container(
            b"ISG1",
            &isg1(&[
                float(0x0000, 0, 0x0f, "POSITION"),
                Element {
                    name: "SV_VertexID",
                    semantic_index: 0,
                    system_value: SV_VERTEX_ID,
                    component_type: D3D_COMPONENT_UINT,
                    mask: 0x01,
                },
                Element {
                    name: "SV_InstanceID",
                    semantic_index: 0,
                    system_value: SV_INSTANCE_ID,
                    component_type: D3D_COMPONENT_UINT,
                    mask: 0x01,
                },
            ]),
        );
        let layout = VertexLayout::from_dxil_container(&bytes).unwrap();
        assert_eq!(layout.attributes().len(), 1);
        assert_eq!(layout.attributes()[0].index, 11);
        assert_eq!(layout.attributes()[0].stride, 16);
    }

    #[test]
    fn a_leading_system_value_still_advances_the_attribute_index() {
        // Measured converter behaviour: `SV_VertexID` at element 0 consumes
        // attribute index 11, so the two real inputs land on 12 and 13.
        let bytes = container(
            b"ISG1",
            &isg1(&[
                Element {
                    name: "SV_VertexID",
                    semantic_index: 0,
                    system_value: SV_VERTEX_ID,
                    component_type: D3D_COMPONENT_UINT,
                    mask: 0x01,
                },
                float(0x0000, 0, 0x07, "POSITION"),
                float(0x0001, 0, 0x0f, "COLOR"),
            ]),
        );
        let layout = VertexLayout::from_dxil_container(&bytes).unwrap();
        let indices: Vec<u8> = layout.attributes().iter().map(|a| a.index).collect();
        assert_eq!(indices, [12, 13]);
    }

    #[test]
    fn a_non_builtin_system_value_is_refused_by_name() {
        let bytes = container(
            b"ISG1",
            &isg1(&[Element {
                name: "SV_Position",
                semantic_index: 0,
                system_value: 1,
                component_type: D3D_COMPONENT_FLOAT,
                mask: 0x0f,
            }]),
        );
        let error = VertexLayout::from_dxil_container(&bytes).unwrap_err();
        assert_eq!(
            error,
            VertexLayoutError::SystemValue {
                index: 0,
                system_value: 1
            }
        );
        assert!(error.to_string().contains("system value 1"));
        assert!(!error.to_string().contains("SV_"));
    }

    #[test]
    fn an_unknown_component_type_is_refused_by_name() {
        let bytes = container(
            b"ISG1",
            &isg1(&[Element {
                name: "POSITION",
                semantic_index: 0,
                system_value: 0,
                component_type: 9,
                mask: 0x07,
            }]),
        );
        assert_eq!(
            VertexLayout::from_dxil_container(&bytes).unwrap_err(),
            VertexLayoutError::UnknownComponentType {
                index: 0,
                component_type: 9
            }
        );
    }

    #[test]
    fn more_than_thirty_one_attributes_are_refused() {
        let elements: Vec<Element> = (0..32)
            .map(|index| float(0, index, 0x01, "TEXCOORD"))
            .collect();
        let bytes = container(b"ISG1", &isg1(&elements));
        assert_eq!(
            VertexLayout::from_dxil_container(&bytes).unwrap_err(),
            VertexLayoutError::TooManyAttributes(32)
        );
    }

    #[test]
    fn an_element_above_metal_attribute_index_thirty_is_refused() {
        // Element 20 would need index 31 (base 11), which Metal cannot hold.
        let elements: Vec<Element> = (0..21)
            .map(|index| float(0, index, 0x01, "TEXCOORD"))
            .collect();
        let bytes = container(b"ISG1", &isg1(&elements));
        assert_eq!(
            VertexLayout::from_dxil_container(&bytes).unwrap_err(),
            VertexLayoutError::AttributeIndexOutOfRange {
                element: 20,
                index: 31
            }
        );
    }

    #[test]
    fn an_unknown_chunk_format_is_refused_not_dropped() {
        let layout = VertexLayout {
            attributes: vec![VertexAttribute {
                index: 0,
                format: VertexFormat::with_components(VertexFormat::FLOAT, 3).unwrap(),
                offset: 0,
                buffer_index: 0,
                stride: 12,
            }],
        };
        let mut encoded = layout.encode();
        // Format byte of the first attribute follows the 6-byte header.
        encoded[VERTEX_LAYOUT_HEADER_LEN + 1] = 0x31;
        assert_eq!(
            VertexLayout::decode(&encoded).unwrap_err(),
            VertexLayoutError::UnknownFormat(0x31)
        );
    }

    #[test]
    fn a_chunk_with_little_endian_fields_decodes_exactly() {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&VERTEX_LAYOUT_MAGIC);
        encoded.push(VERTEX_LAYOUT_VERSION);
        encoded.push(1);
        encoded.push(7);
        encoded.push(VertexFormat::with_components(VertexFormat::UNORM8, 4).unwrap().code());
        encoded.extend_from_slice(&0x1234_5678u32.to_le_bytes());
        encoded.push(3);
        encoded.extend_from_slice(&0x0000_0100u32.to_le_bytes());
        let layout = VertexLayout::decode(&encoded).unwrap();
        assert_eq!(
            layout.attributes(),
            &[VertexAttribute {
                index: 7,
                format: VertexFormat::with_components(VertexFormat::UNORM8, 4).unwrap(),
                offset: 0x1234_5678,
                buffer_index: 3,
                stride: 0x100,
            }]
        );
    }

    #[test]
    fn a_truncated_or_extended_chunk_is_refused() {
        let layout = VertexLayout {
            attributes: vec![VertexAttribute {
                index: 0,
                format: VertexFormat::with_components(VertexFormat::HALF, 2).unwrap(),
                offset: 0,
                buffer_index: 0,
                stride: 4,
            }],
        };
        let encoded = layout.encode();
        assert!(VertexLayout::decode(&encoded[..encoded.len() - 1]).is_err());
        let mut extended = encoded.clone();
        extended.push(0);
        assert_eq!(
            VertexLayout::decode(&extended).unwrap_err(),
            VertexLayoutError::TrailingBytes
        );
        assert_eq!(
            VertexLayout::decode(&[]).unwrap_err(),
            VertexLayoutError::Malformed("the vertex layout chunk is truncated")
        );
        assert_eq!(
            VertexLayout::decode(b"NOPE").unwrap_err(),
            VertexLayoutError::Malformed("the vertex layout chunk is truncated")
        );
    }

    #[test]
    fn non_dxbc_payloads_keep_the_legacy_descriptorless_path() {
        assert!(VertexLayout::from_dxil_container(b"not a container").unwrap().is_empty());
    }
}
