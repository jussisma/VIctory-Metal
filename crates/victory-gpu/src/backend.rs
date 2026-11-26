//! Host GPU backend selection seam (`VICTORY_BACKEND=d3d12|metal`).
//!
//! Keeps desktop D3D12 / D3DMetal working as the default while Milestone M1–M4
//! bring up the Wine-decoupled native Metal engine (`victory-metal`) behind
//! `VICTORY_BACKEND=metal`.

/// Selected host GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuBackend {
    /// Existing D3D12.X -> host D3D12 / D3DMetal translation (default).
    #[default]
    D3d12,
    /// Experimental direct native Apple Metal backend (`victory-metal`).
    Metal,
}

impl GpuBackend {
    /// Parses `VICTORY_BACKEND` from an optional string value.
    ///
    /// Unset, empty, or unrecognized values fall back to [`GpuBackend::D3d12`].
    pub fn parse(value: Option<&str>) -> Self {
        let Some(raw) = value.map(str::trim).filter(|s| !s.is_empty()) else {
            return Self::D3d12;
        };
        if raw.eq_ignore_ascii_case("d3d12") {
            Self::D3d12
        } else if raw.eq_ignore_ascii_case("metal") {
            Self::Metal
        } else {
            tracing::warn!(
                value = raw,
                "unknown VICTORY_BACKEND value; falling back to d3d12"
            );
            Self::D3d12
        }
    }

    /// Reads `VICTORY_BACKEND` from the process environment (cached once).
    pub fn from_env() -> Self {
        static BACKEND: std::sync::OnceLock<GpuBackend> = std::sync::OnceLock::new();
        *BACKEND.get_or_init(|| {
            let val = std::env::var("VICTORY_BACKEND").ok();
            Self::parse(val.as_deref())
        })
    }

    /// Canonical lowercase backend name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::D3d12 => "d3d12",
            Self::Metal => "metal",
        }
    }
}

/// Whether the queue and present entry points own the native Metal path.
///
/// This is the single backend switch point outside [`GpuBackend::from_env`]:
/// entry points ask it instead of reading `VICTORY_BACKEND` themselves, so the
/// environment is read exactly once per process and every caller agrees.
pub fn metal_selected() -> bool {
    GpuBackend::from_env() == GpuBackend::Metal
}

#[cfg(test)]
mod tests {
    use super::GpuBackend;

    #[test]
    fn defaults_to_d3d12_when_unset_or_blank() {
        assert_eq!(GpuBackend::parse(None), GpuBackend::D3d12);
        assert_eq!(GpuBackend::parse(Some("")), GpuBackend::D3d12);
        assert_eq!(GpuBackend::parse(Some("   ")), GpuBackend::D3d12);
    }

    #[test]
    fn parses_known_backends_case_insensitively() {
        assert_eq!(GpuBackend::parse(Some("d3d12")), GpuBackend::D3d12);
        assert_eq!(GpuBackend::parse(Some("D3D12")), GpuBackend::D3d12);
        assert_eq!(GpuBackend::parse(Some("metal")), GpuBackend::Metal);
        assert_eq!(GpuBackend::parse(Some(" Metal ")), GpuBackend::Metal);
    }

    #[test]
    fn unknown_value_falls_back_to_d3d12() {
        assert_eq!(GpuBackend::parse(Some("vulkan")), GpuBackend::D3d12);
    }
}
