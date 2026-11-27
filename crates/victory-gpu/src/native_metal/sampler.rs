//! Guest sampler-state identity and registration bookkeeping (Cycle 07).
//!
//! A D3D12 sampler descriptor is converted to the reduced
//! [`SamplerStateDesc`] the host's `register_sampler` path can express, and
//! every distinct content gets one deterministic, nonzero handle. The handle is
//! a BLAKE3-keyed value with a fixed namespace byte, so the same descriptor
//! produces the same handle in every process and across a reconnect without a
//! lookup table surviving the session.
//!
//! The process-wide map is bounded: it deduplicates by content and refuses a
//! new descriptor once [`MAX_SAMPLER_STATES`] distinct states exist. It never
//! evicts and never turns an unknown handle into a default sampler; an
//! unregistered handle is an explicit error at the binding seam
//! ([`crate::native_metal::bindings::require_registered_handles`]).
//!
//! The host code is the reduced `MTLSamplerState` shape that
//! `DirectMetalTransport::register_sampler` accepts: min/mag filters and the
//! S/T/R address modes. Mip filtering, LOD bias/clamps, anisotropy and border
//! colour are not part of that shape; a comparison sampler is refused rather
//! than approximated.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

/// Largest number of distinct sampler contents the guest tracks.
///
/// The map is content-addressed, so a title reusing its sampler set costs one
/// entry each. A 257th distinct state is an explicit error, never a silent
/// eviction that would move a previously bound handle.
pub const MAX_SAMPLER_STATES: usize = 256;

/// Namespace byte of a sampler handle (`'S'`), so a sampler handle can never
/// be mistaken for a texture or allocation id by a reader of the wire.
const SAMPLER_HANDLE_TAG: u64 = 0x5300_0000_0000_0000;
const SAMPLER_HANDLE_MASK: u64 = 0x00ff_ffff_ffff_ffff;

const D3D12_FILTER_COMPARISON_BIT: u32 = 0x80;
const D3D12_FILTER_ANISOTROPIC: u32 = 0x55;
/// The eight non-comparison, non-anisotropic `D3D12_FILTER` values.
const D3D12_FILTERS: [u32; 8] = [0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15];

/// The reduced sampler content one host `MTLSamplerState` is created from.
///
/// Field values are Metal codes (`MTLSamplerMinMagFilter` 0/1 and
/// `MTLSamplerAddressMode` 0..=4), which are exactly the values the host passes
/// to `setMinFilter:`, `setMagFilter:` and `setS/T/RAddressMode:`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerStateDesc {
    /// `MTLSamplerMinMagFilter` for minification (`0` nearest, `1` linear).
    pub min_filter: u32,
    /// `MTLSamplerMinMagFilter` for magnification (`0` nearest, `1` linear).
    pub mag_filter: u32,
    /// `MTLSamplerAddressMode` for S.
    pub s_address_mode: u32,
    /// `MTLSamplerAddressMode` for T.
    pub t_address_mode: u32,
    /// `MTLSamplerAddressMode` for R.
    pub r_address_mode: u32,
}

impl SamplerStateDesc {
    /// Nearest filtering with clamp-to-edge addressing on every axis.
    pub const NEAREST_CLAMP: Self = Self {
        min_filter: 0,
        mag_filter: 0,
        s_address_mode: 0,
        t_address_mode: 0,
        r_address_mode: 0,
    };

    /// Validates Metal-coded fields from the wire or a caller.
    pub fn from_host_codes(
        min_filter: u32,
        mag_filter: u32,
        s_address_mode: u32,
        t_address_mode: u32,
        r_address_mode: u32,
    ) -> Result<Self, SamplerStateError> {
        if min_filter > 1 || mag_filter > 1 {
            return Err(SamplerStateError::InvalidFilter(min_filter.max(mag_filter)));
        }
        for (axis, mode) in [
            ("S", s_address_mode),
            ("T", t_address_mode),
            ("R", r_address_mode),
        ] {
            if mode > 4 {
                return Err(SamplerStateError::InvalidAddressMode { axis, mode });
            }
        }
        Ok(Self {
            min_filter,
            mag_filter,
            s_address_mode,
            t_address_mode,
            r_address_mode,
        })
    }

    /// Converts a `D3D12_SAMPLER_DESC` in its 13-word guest layout.
    ///
    /// Word 0 is `Filter`, words 1..=3 are `AddressU/V/W`; the remaining words
    /// are LOD, anisotropy and border colour, which the host sampler shape does
    /// not carry.
    pub fn from_d3d12(words: &[u32; 13]) -> Result<Self, SamplerStateError> {
        let filter = words[0];
        if filter & D3D12_FILTER_COMPARISON_BIT != 0 {
            return Err(SamplerStateError::ComparisonFilter);
        }
        // Anisotropic filters are linear/linear to the min/mag fields; the host
        // sampler shape has no maxAnisotropy, so the sample count is not
        // carried. Any other value has to be one of D3D12's eight filters.
        let (min_filter, mag_filter) = if filter == D3D12_FILTER_ANISOTROPIC {
            (1, 1)
        } else if D3D12_FILTERS.contains(&filter) {
            (u32::from(filter & 0x10 != 0), u32::from(filter & 0x4 != 0))
        } else {
            return Err(SamplerStateError::InvalidFilter(filter));
        };
        let address = |axis, mode| d3d12_address_mode(axis, mode);
        Self::from_host_codes(
            min_filter,
            mag_filter,
            address("U", words[1])?,
            address("V", words[2])?,
            address("W", words[3])?,
        )
    }

    /// The deterministic handle this content registers under.
    ///
    /// The value is the first eight bytes of a domain-separated BLAKE3 hash in
    /// the sampler namespace. It is stable across processes and reconnects, so
    /// a second frame that binds the same sampler names the handle the host
    /// already holds.
    pub fn handle(&self) -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"victory-sampler-state-v1");
        for value in [
            self.min_filter,
            self.mag_filter,
            self.s_address_mode,
            self.t_address_mode,
            self.r_address_mode,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        let digest = hasher.finalize();
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&digest.as_bytes()[..8]);
        let handle = SAMPLER_HANDLE_TAG | (u64::from_le_bytes(raw) & SAMPLER_HANDLE_MASK);
        // The namespace byte makes this nonzero; keep the invariant explicit.
        if handle == 0 {
            SAMPLER_HANDLE_TAG | 1
        } else {
            handle
        }
    }
}

/// Maps one `D3D12_TEXTURE_ADDRESS_MODE` to `MTLSamplerAddressMode`.
fn d3d12_address_mode(axis: &'static str, mode: u32) -> Result<u32, SamplerStateError> {
    Ok(match mode {
        1 => 2, // WRAP -> repeat
        2 => 3, // MIRROR -> mirrorRepeat
        3 => 0, // CLAMP -> clampToEdge
        4 => 4, // BORDER -> clampToZero (the border colour is not carried)
        5 => 1, // MIRROR_ONCE -> mirrorClampToEdge
        other => return Err(SamplerStateError::InvalidAddressMode { axis, mode: other }),
    })
}

/// Why a sampler descriptor could not be converted or registered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SamplerStateError {
    /// A comparison (`D3D12_FILTER_COMPARISON_*`) sampler has no host shape.
    ComparisonFilter,
    /// A filter value outside D3D12's documented filters.
    InvalidFilter(u32),
    /// An address mode outside `D3D12_TEXTURE_ADDRESS_MODE`.
    InvalidAddressMode {
        /// Heap axis the mode belongs to (`U`, `V` or `W`).
        axis: &'static str,
        /// The refused mode.
        mode: u32,
    },
    /// The content-addressed map holds [`MAX_SAMPLER_STATES`] distinct states.
    RegistryFull {
        /// The bound.
        capacity: usize,
    },
    /// Two different contents produced the same deterministic handle.
    HandleCollision(u64),
}

impl std::fmt::Display for SamplerStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComparisonFilter => write!(
                f,
                "a comparison sampler has no MTLSamplerState shape in the host registry"
            ),
            Self::InvalidFilter(filter) => {
                write!(
                    f,
                    "D3D12 filter 0x{filter:02x} is not a supported sampler filter"
                )
            }
            Self::InvalidAddressMode { axis, mode } => write!(
                f,
                "D3D12 texture address mode {mode} on axis {axis} has no Metal equivalent"
            ),
            Self::RegistryFull { capacity } => write!(
                f,
                "the guest sampler registry is full: {capacity} distinct states is the bound"
            ),
            Self::HandleCollision(handle) => {
                write!(f, "two sampler contents collided on handle 0x{handle:016x}")
            }
        }
    }
}

impl std::error::Error for SamplerStateError {}

struct Registry {
    by_desc: HashMap<SamplerStateDesc, u64>,
    by_handle: HashMap<u64, SamplerStateDesc>,
}

static REGISTRY: LazyLock<RwLock<Registry>> = LazyLock::new(|| {
    RwLock::new(Registry {
        by_desc: HashMap::new(),
        by_handle: HashMap::new(),
    })
});

fn registry() -> std::sync::RwLockWriteGuard<'static, Registry> {
    REGISTRY.write().unwrap_or_else(|error| error.into_inner())
}

/// Returns the stable handle for `desc`, registering its content once.
///
/// The same descriptor always returns the same handle. A new descriptor is
/// refused with [`SamplerStateError::RegistryFull`] once
/// [`MAX_SAMPLER_STATES`] distinct states exist; nothing is evicted.
pub fn sampler_handle(desc: &SamplerStateDesc) -> Result<u64, SamplerStateError> {
    let mut registry = registry();
    if let Some(handle) = registry.by_desc.get(desc) {
        return Ok(*handle);
    }
    if registry.by_desc.len() >= MAX_SAMPLER_STATES {
        return Err(SamplerStateError::RegistryFull {
            capacity: MAX_SAMPLER_STATES,
        });
    }
    let handle = desc.handle();
    if let Some(existing) = registry.by_handle.get(&handle) {
        if existing != desc {
            return Err(SamplerStateError::HandleCollision(handle));
        }
        return Ok(handle);
    }
    registry.by_desc.insert(*desc, handle);
    registry.by_handle.insert(handle, *desc);
    Ok(handle)
}

/// The content registered under `handle`, when this process derived it.
pub fn sampler_state(handle: u64) -> Option<SamplerStateDesc> {
    registry().by_handle.get(&handle).copied()
}

/// Number of distinct sampler contents this process tracks.
pub fn sampler_state_count() -> usize {
    registry().by_desc.len()
}

/// Forgets every tracked sampler content (session teardown and tests).
pub fn reset_sampler_states() {
    let mut registry = registry();
    registry.by_desc.clear();
    registry.by_handle.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d3d12_filters_and_address_modes_map_to_metal_codes() {
        // Linear min/mag with wrap addressing.
        let linear_wrap = SamplerStateDesc::from_d3d12(&[0x15, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .expect("linear wrap");
        assert_eq!(linear_wrap.min_filter, 1);
        assert_eq!(linear_wrap.mag_filter, 1);
        assert_eq!(linear_wrap.s_address_mode, 2);
        // Point/point with clamp.
        let point_clamp = SamplerStateDesc::from_d3d12(&[0x00, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .expect("point clamp");
        assert_eq!(
            point_clamp,
            SamplerStateDesc {
                min_filter: 0,
                mag_filter: 0,
                s_address_mode: 0,
                t_address_mode: 0,
                r_address_mode: 0,
            }
        );
        // Anisotropic is linear/linear; the filter is not carried further.
        let anisotropic = SamplerStateDesc::from_d3d12(&[0x55, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .expect("anisotropic");
        assert_eq!((anisotropic.min_filter, anisotropic.mag_filter), (1, 1));
        assert_eq!(anisotropic.s_address_mode, 3, "mirror -> mirrorRepeat");
    }

    #[test]
    fn comparison_and_unknown_sampler_fields_are_refused() {
        assert_eq!(
            SamplerStateDesc::from_d3d12(&[0x80, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            Err(SamplerStateError::ComparisonFilter)
        );
        assert!(matches!(
            SamplerStateDesc::from_d3d12(&[0x22, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            Err(SamplerStateError::InvalidFilter(0x22))
        ));
        assert!(matches!(
            SamplerStateDesc::from_d3d12(&[0x00, 9, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            Err(SamplerStateError::InvalidAddressMode { axis: "U", mode: 9 })
        ));
    }

    #[test]
    fn sampler_handles_are_content_addressed_and_bounded() {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|error| error.into_inner());
        reset_sampler_states();
        let desc = SamplerStateDesc::NEAREST_CLAMP;
        let first = sampler_handle(&desc).expect("the first state");
        let second = sampler_handle(&desc).expect("the same state");
        assert_eq!(first, second, "idempotent by content");
        assert_eq!(first & 0xff00_0000_0000_0000, SAMPLER_HANDLE_TAG);
        assert_ne!(first, 0);
        assert_eq!(sampler_state(first), Some(desc));
        assert_eq!(sampler_state_count(), 1);

        // A different content is a different handle.
        let linear = SamplerStateDesc {
            min_filter: 1,
            mag_filter: 1,
            ..SamplerStateDesc::NEAREST_CLAMP
        };
        let other = sampler_handle(&linear).expect("the second state");
        assert_ne!(other, first);

        reset_sampler_states();
        for index in 0..MAX_SAMPLER_STATES {
            sampler_handle(&desc_for(index)).expect("a state below the bound");
        }
        assert_eq!(sampler_state_count(), MAX_SAMPLER_STATES);
        assert_eq!(
            sampler_handle(&desc_for(MAX_SAMPLER_STATES)),
            Err(SamplerStateError::RegistryFull {
                capacity: MAX_SAMPLER_STATES
            }),
            "a 257th distinct state is refused, never evicted"
        );
        // The already-registered contents still resolve.
        assert_eq!(sampler_handle(&desc_for(0)), Ok(desc_for(0).handle()));
        reset_sampler_states();
    }

    /// A unique content per index for `0..=MAX_SAMPLER_STATES` (base-5 mixed
    /// radix over the five fields).
    fn desc_for(index: usize) -> SamplerStateDesc {
        SamplerStateDesc {
            min_filter: ((index / 250) % 2) as u32,
            mag_filter: ((index / 125) % 2) as u32,
            s_address_mode: ((index / 25) % 5) as u32,
            t_address_mode: ((index / 5) % 5) as u32,
            r_address_mode: (index % 5) as u32,
        }
    }
}
