//! Versioned root-argument plan (`VRAB`) carried beside a version 3 submission.
//!
//! A title sets a root argument by *root-parameter index*, which names a
//! register, a register space and an argument kind only in the title's own root
//! signature. The Wire v3 root commands (`SetRootConstants`, `SetRootBufferView`,
//! `SetDescriptorTable`) carry that index alone. The native backend runs the
//! converter's *translated* root signature instead, which reorders and augments
//! the parameters: per-stage user data in
//! [`USER_DATA_SPACE`](victory_shader::emit::USER_DATA_SPACE), root CBV/SRV/UAV
//! descriptors and descriptor tables in
//! [`ROOT_CBV_SPACE`](victory_shader::emit::ROOT_CBV_SPACE), the title-memory
//! page table in [`MEMORY_SPACE`](victory_shader::emit::MEMORY_SPACE) and static
//! samplers in [`SAMPLER_SPACE`](victory_shader::emit::SAMPLER_SPACE). Title
//! index `N` is therefore not translated slot `N`.
//!
//! The guest records the relation while the title binds and sends this plan
//! beside the encoded submission (see [`crate::ipc_host`] for the framing). The
//! host decodes it into [`RootParameterPlan`] and resolves each wire root
//! argument by its translated `(space, register, kind)` identity. A submission
//! with no plan section decodes exactly as before this frame existed: the host
//! then keeps its legacy slot lookup.
//!
//! The plan is versioned so the reader never mis-reads a newer record shape as
//! a legacy one: version 1 records carry no space and decode with the kind's
//! translated space, version 2 records carry it explicitly. Every bound is
//! checked before a record is read, and trailing bytes are refused, so a
//! hostile or truncated plan is a named refusal rather than a shifted read.

use victory_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};

/// Magic of the versioned root-argument plan (`VRAB`).
pub const ROOT_PLAN_MAGIC: [u8; 4] = *b"VRAB";

/// The plan version that carries the register space.
pub const ROOT_PLAN_VERSION: u8 = 2;

/// The plan version before the space field existed.
pub const ROOT_PLAN_VERSION_LEGACY: u8 = 1;

/// The most root arguments one encoded plan carries.
pub const MAX_ROOT_PLAN_ENTRIES: usize = 256;

/// The largest register space a plan entry may name.
///
/// D3D12 register spaces are sparse, but a plan is not a root signature: an
/// argument past this is not a space this backend can have translated.
pub const MAX_ROOT_PLAN_SPACE: u16 = 15;

/// Bytes of the plan header: magic, version, count.
const PLAN_HEADER_LEN: usize = 7;

/// Bytes of one version 2 record: `root_parameter(2) space(2) kind(1)
/// register(2) offset(4) size(4)`.
const PLAN_RECORD_V2_LEN: usize = 15;

/// Bytes of one version 1 record (no space): `root_parameter(2) kind(1)
/// register(2) offset(4) size(4)`.
const PLAN_RECORD_V1_LEN: usize = 13;

/// Largest encoded plan the host accepts: the header plus
/// [`MAX_ROOT_PLAN_ENTRIES`] version 2 records.
pub const MAX_ROOT_PLAN_LEN: usize = PLAN_HEADER_LEN + PLAN_RECORD_V2_LEN * MAX_ROOT_PLAN_ENTRIES;

/// Which translated root argument a captured title root parameter feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PlanArgumentKind {
    /// A 32-bit-constants block (`Set*Root32BitConstants`).
    Constants = 0,
    /// A root CBV, SRV or UAV (`Set*Root*View`).
    RootView = 1,
    /// A descriptor table of CBV, SRV or UAV ranges.
    DescriptorTable = 2,
    /// A descriptor table of sampler states.
    SamplerTable = 3,
}

impl PlanArgumentKind {
    /// The wire code of this kind.
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The kind a wire code names, or `None` for an unknown code.
    pub const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Constants,
            1 => Self::RootView,
            2 => Self::DescriptorTable,
            3 => Self::SamplerTable,
            _ => return None,
        })
    }

    /// The translated register space this kind is bound in.
    ///
    /// This is the mapping [`victory_shader::emit`] gives the translated root
    /// signature; it is the space a version 1 record (which has no space field)
    /// is decoded with.
    pub const fn translated_space(self) -> u16 {
        match self {
            Self::Constants => USER_DATA_SPACE as u16,
            Self::RootView | Self::DescriptorTable => ROOT_CBV_SPACE as u16,
            Self::SamplerTable => SAMPLER_SPACE as u16,
        }
    }

    /// A short name for errors and diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Constants => "root constants",
            Self::RootView => "root view",
            Self::DescriptorTable => "descriptor table",
            Self::SamplerTable => "sampler table",
        }
    }
}

/// One captured root argument as the translated root signature addresses it.
///
/// `root_parameter` keeps the title's root-parameter index, which is the slot
/// the wire's root commands carry. `space`, `kind` and `register` name the
/// translated parameter; `offset` and `size` describe the argument's extent
/// (constants bytes, or a view/table range).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootPlanEntry {
    /// Title root-parameter index the title set.
    pub root_parameter: u16,
    /// Register space the translated root signature binds it in.
    pub space: u16,
    /// What the argument is.
    pub kind: PlanArgumentKind,
    /// Register (views/tables) or the constant block's base register.
    pub register: u16,
    /// Constants block's starting dword; `0` for a view or table.
    pub offset: u32,
    /// Bytes of a constants block or view, or entries of a table.
    pub size: u32,
}

impl RootPlanEntry {
    /// Checks the entry against the spaces and shapes this backend names.
    pub fn validate(&self) -> Result<(), RootPlanError> {
        if self.space > MAX_ROOT_PLAN_SPACE {
            return Err(RootPlanError::UnknownSpace { space: self.space });
        }
        if self.kind == PlanArgumentKind::SamplerTable && self.space != SAMPLER_SPACE as u16 {
            return Err(RootPlanError::SamplerNonSamplerSpace { space: self.space });
        }
        if self.size == 0 {
            return Err(RootPlanError::Invalid("the root argument is empty"));
        }
        Ok(())
    }
}

/// The decoded root-argument plan of one submission.
///
/// Entries keep the guest's order and are *not* collapsed by
/// `root_parameter`: the same title slot recorded in two register spaces stays
/// two distinct entries. A wire root command carries no space, so
/// [`entry`](Self::entry) resolves the first entry for the slot the command
/// names, deterministically.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RootParameterPlan {
    entries: Vec<RootPlanEntry>,
}

impl RootParameterPlan {
    /// Builds a plan from validated entries.
    pub fn from_entries(entries: Vec<RootPlanEntry>) -> Result<Self, RootPlanError> {
        if entries.len() > MAX_ROOT_PLAN_ENTRIES {
            return Err(RootPlanError::TooMany(entries.len()));
        }
        for entry in &entries {
            entry.validate()?;
        }
        Ok(Self { entries })
    }

    /// Every entry, in the order the guest recorded them.
    pub fn entries(&self) -> &[RootPlanEntry] {
        &self.entries
    }

    /// The first entry recorded for title root parameter `root_parameter`.
    pub fn entry(&self, root_parameter: u16) -> Option<&RootPlanEntry> {
        self.entries
            .iter()
            .find(|entry| entry.root_parameter == root_parameter)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the plan carries no argument at all.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Why a root-argument plan could not be validated or decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootPlanError {
    /// The plan is malformed.
    Invalid(&'static str),
    /// The space is past [`MAX_ROOT_PLAN_SPACE`].
    UnknownSpace {
        /// The space the entry named.
        space: u16,
    },
    /// A sampler table is not in [`SAMPLER_SPACE`].
    SamplerNonSamplerSpace {
        /// The space the table named.
        space: u16,
    },
    /// The plan names a version this reader does not know.
    UnsupportedVersion(u8),
    /// The plan carries more than [`MAX_ROOT_PLAN_ENTRIES`].
    TooMany(usize),
    /// The plan ends in the middle of a record.
    Truncated,
}

impl std::fmt::Display for RootPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "invalid root-argument plan: {why}"),
            Self::UnknownSpace { space } => {
                write!(f, "root argument names unknown register space {space}")
            }
            Self::SamplerNonSamplerSpace { space } => write!(
                f,
                "a sampler table is in space {space}, not the sampler space {SAMPLER_SPACE}"
            ),
            Self::UnsupportedVersion(version) => {
                write!(f, "root-argument plan version {version} is not supported")
            }
            Self::TooMany(count) => write!(
                f,
                "the root-argument plan carries {count}, more than {MAX_ROOT_PLAN_ENTRIES}"
            ),
            Self::Truncated => write!(f, "the root-argument plan ends mid-record"),
        }
    }
}

impl std::error::Error for RootPlanError {}

/// Encodes a plan in the current (version 2) shape.
///
/// The host decodes this shape; the guest's own encoder writes the same bytes,
/// and the two are pinned by the shared record layout rather than a crate
/// dependency (the host cannot depend on `victory-gpu`).
pub fn encode_root_plan(entries: &[RootPlanEntry]) -> Result<Vec<u8>, RootPlanError> {
    if entries.len() > MAX_ROOT_PLAN_ENTRIES {
        return Err(RootPlanError::TooMany(entries.len()));
    }
    let mut out = Vec::with_capacity(PLAN_HEADER_LEN + PLAN_RECORD_V2_LEN * entries.len());
    out.extend_from_slice(&ROOT_PLAN_MAGIC);
    out.push(ROOT_PLAN_VERSION);
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for entry in entries {
        entry.validate()?;
        out.extend_from_slice(&entry.root_parameter.to_le_bytes());
        out.extend_from_slice(&entry.space.to_le_bytes());
        out.push(entry.kind.code());
        out.extend_from_slice(&entry.register.to_le_bytes());
        out.extend_from_slice(&entry.offset.to_le_bytes());
        out.extend_from_slice(&entry.size.to_le_bytes());
    }
    Ok(out)
}

/// Decodes a root-argument plan, accepting both the legacy (version 1, no
/// space) and current (version 2) record shapes.
///
/// An unknown version is refused before any record is read, a truncated plan is
/// refused before a record is partially used, and bytes after the last record
/// are refused rather than ignored.
pub fn decode_root_plan(bytes: &[u8]) -> Result<RootParameterPlan, RootPlanError> {
    let header = bytes.get(..PLAN_HEADER_LEN).ok_or(RootPlanError::Truncated)?;
    if header[..4] != ROOT_PLAN_MAGIC {
        return Err(RootPlanError::Invalid("the plan magic is not VRAB"));
    }
    let version = header[4];
    let count = usize::from(u16::from_le_bytes([header[5], header[6]]));
    if count > MAX_ROOT_PLAN_ENTRIES {
        return Err(RootPlanError::TooMany(count));
    }
    let record = match version {
        ROOT_PLAN_VERSION_LEGACY => PLAN_RECORD_V1_LEN,
        ROOT_PLAN_VERSION => PLAN_RECORD_V2_LEN,
        other => return Err(RootPlanError::UnsupportedVersion(other)),
    };
    let expected = PLAN_HEADER_LEN
        .checked_add(record.checked_mul(count).ok_or(RootPlanError::TooMany(count))?)
        .ok_or(RootPlanError::TooMany(count))?;
    if bytes.len() < expected {
        return Err(RootPlanError::Truncated);
    }
    if bytes.len() > expected {
        return Err(RootPlanError::Invalid(
            "the plan has trailing bytes after its last record",
        ));
    }
    let mut entries = Vec::with_capacity(count);
    let mut at = PLAN_HEADER_LEN;
    for _ in 0..count {
        let root_parameter = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let (space, kind_at) = match version {
            ROOT_PLAN_VERSION => (u16::from_le_bytes([bytes[at + 2], bytes[at + 3]]), at + 4),
            _ => (0, at + 2),
        };
        let code = bytes[kind_at];
        let kind = PlanArgumentKind::from_code(code)
            .ok_or(RootPlanError::Invalid("the plan names an unknown root-argument kind"))?;
        let space = if version == ROOT_PLAN_VERSION_LEGACY {
            kind.translated_space()
        } else {
            space
        };
        let register_at = kind_at + 1;
        let offset_at = register_at + 2;
        let size_at = offset_at + 4;
        let register = u16::from_le_bytes([bytes[register_at], bytes[register_at + 1]]);
        let offset =
            u32::from_le_bytes(bytes[offset_at..offset_at + 4].try_into().expect("four bytes"));
        let size = u32::from_le_bytes(bytes[size_at..size_at + 4].try_into().expect("four bytes"));
        let entry = RootPlanEntry {
            root_parameter,
            space,
            kind,
            register,
            offset,
            size,
        };
        entry.validate()?;
        entries.push(entry);
        at += record;
    }
    Ok(RootParameterPlan { entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(root_parameter: u16, space: u16, kind: PlanArgumentKind, register: u16) -> RootPlanEntry {
        RootPlanEntry {
            root_parameter,
            space,
            kind,
            register,
            offset: 0,
            size: 16,
        }
    }

    /// A plan of one root CBV at title slot 0 in the translated root-CBV space
    /// round-trips through the version 2 shape.
    #[test]
    fn a_version_two_plan_round_trips() {
        let entries = vec![
            entry(0, ROOT_CBV_SPACE as u16, PlanArgumentKind::RootView, 0),
            RootPlanEntry {
                root_parameter: 1,
                space: USER_DATA_SPACE as u16,
                kind: PlanArgumentKind::Constants,
                register: 0,
                offset: 2,
                size: 12,
            },
        ];
        let encoded = encode_root_plan(&entries).expect("the plan encodes");
        assert_eq!(&encoded[..4], b"VRAB");
        assert_eq!(encoded[4], ROOT_PLAN_VERSION);
        assert_eq!(
            usize::from(u16::from_le_bytes([encoded[5], encoded[6]])),
            2
        );
        assert_eq!(encoded.len(), PLAN_HEADER_LEN + 2 * PLAN_RECORD_V2_LEN);
        let decoded = decode_root_plan(&encoded).expect("the plan decodes");
        assert_eq!(decoded.entries(), entries.as_slice());
        assert_eq!(decoded.entry(1).map(|entry| entry.kind), Some(PlanArgumentKind::Constants));
        assert!(decoded.entry(9).is_none());
    }

    /// A version 1 record has no space field and decodes with the kind's
    /// translated space, which is where the pre-space encoder implicitly put it.
    #[test]
    fn a_legacy_plan_decodes_with_the_translated_space() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VRAB");
        bytes.push(ROOT_PLAN_VERSION_LEGACY);
        bytes.extend_from_slice(&2u16.to_le_bytes());
        // Root CBV at title slot 0, register 3, 16 bytes.
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.push(PlanArgumentKind::RootView.code());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&16u32.to_le_bytes());
        // Constants at title slot 1, offset 4, 8 bytes.
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.push(PlanArgumentKind::Constants.code());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());

        let decoded = decode_root_plan(&bytes).expect("a legacy plan decodes");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.entries()[0].space, ROOT_CBV_SPACE as u16);
        assert_eq!(decoded.entries()[1].space, USER_DATA_SPACE as u16);
    }

    /// The same title slot recorded in two spaces stays two distinct entries,
    /// which a map keyed by the slot alone would collapse.
    #[test]
    fn the_same_slot_in_two_spaces_stays_two_entries() {
        let entries = vec![
            entry(0, 0, PlanArgumentKind::Constants, 0),
            entry(0, 1, PlanArgumentKind::Constants, 0),
        ];
        let decoded =
            decode_root_plan(&encode_root_plan(&entries).expect("encode")).expect("decode");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.entries()[0].space, 0);
        assert_eq!(decoded.entries()[1].space, 1);
        // The wire command carries no space: the first record is the identity.
        assert_eq!(decoded.entry(0).map(|entry| entry.space), Some(0));
    }

    /// An empty plan is the header alone and is *not* the same thing as a
    /// submission with no plan section; both are legal and distinguishable.
    #[test]
    fn an_empty_plan_is_valid_and_empty() {
        let encoded = encode_root_plan(&[]).expect("an empty plan encodes");
        assert_eq!(encoded.len(), PLAN_HEADER_LEN);
        let decoded = decode_root_plan(&encoded).expect("an empty plan decodes");
        assert!(decoded.is_empty());
        assert!(decoded.entry(0).is_none());
    }

    /// A plan at the entry cap round-trips; one past it is refused before any
    /// record is written.
    #[test]
    fn a_plan_at_the_entry_cap_round_trips_and_one_past_it_is_refused() {
        let entries: Vec<RootPlanEntry> = (0..MAX_ROOT_PLAN_ENTRIES as u16)
            .map(|slot| entry(slot, ROOT_CBV_SPACE as u16, PlanArgumentKind::RootView, slot))
            .collect();
        let encoded = encode_root_plan(&entries).expect("the cap encodes");
        assert_eq!(encoded.len(), MAX_ROOT_PLAN_LEN);
        let decoded = decode_root_plan(&encoded).expect("the cap decodes");
        assert_eq!(decoded.len(), MAX_ROOT_PLAN_ENTRIES);

        let mut over = entries.clone();
        over.push(entry(999, ROOT_CBV_SPACE as u16, PlanArgumentKind::RootView, 0));
        assert_eq!(
            encode_root_plan(&over),
            Err(RootPlanError::TooMany(MAX_ROOT_PLAN_ENTRIES + 1))
        );
    }

    /// A truncated record, a bad magic, an unknown version and trailing bytes
    /// are each refused by name, never read as a shorter plan.
    #[test]
    fn malformed_plans_are_refused_by_name() {
        let encoded = encode_root_plan(&[entry(
            0,
            ROOT_CBV_SPACE as u16,
            PlanArgumentKind::RootView,
            0,
        )])
        .expect("encode");

        assert_eq!(
            decode_root_plan(&encoded[..PLAN_HEADER_LEN - 1]),
            Err(RootPlanError::Truncated)
        );
        assert_eq!(
            decode_root_plan(&encoded[..encoded.len() - 1]),
            Err(RootPlanError::Truncated)
        );

        let mut bad_magic = encoded.clone();
        bad_magic[0] = b'X';
        assert_eq!(
            decode_root_plan(&bad_magic),
            Err(RootPlanError::Invalid("the plan magic is not VRAB"))
        );

        let mut bad_version = encoded.clone();
        bad_version[4] = 9;
        assert_eq!(
            decode_root_plan(&bad_version),
            Err(RootPlanError::UnsupportedVersion(9))
        );

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert_eq!(
            decode_root_plan(&trailing),
            Err(RootPlanError::Invalid(
                "the plan has trailing bytes after its last record"
            ))
        );

        let mut bad_kind = encoded.clone();
        bad_kind[PLAN_HEADER_LEN + 4] = 7;
        assert_eq!(
            decode_root_plan(&bad_kind),
            Err(RootPlanError::Invalid(
                "the plan names an unknown root-argument kind"
            ))
        );

        let mut unknown_space = encoded.clone();
        unknown_space[PLAN_HEADER_LEN + 2..PLAN_HEADER_LEN + 4]
            .copy_from_slice(&16u16.to_le_bytes());
        assert_eq!(
            decode_root_plan(&unknown_space),
            Err(RootPlanError::UnknownSpace { space: 16 })
        );

        let mut empty_argument = encoded;
        let size_at = empty_argument.len() - 4;
        empty_argument[size_at..].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            decode_root_plan(&empty_argument),
            Err(RootPlanError::Invalid("the root argument is empty"))
        );
    }
}
