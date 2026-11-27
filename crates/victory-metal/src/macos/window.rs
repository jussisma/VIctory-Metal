//! Opt-in native `NSWindow` + `CAMetalLayer` presentation (ADR-020).
//!
//! Headless `IOSurface` presentation stays the default so CI and automated
//! gates never need a window server. Setting `VICTORY_METAL_WINDOW=1` (or a
//! size spec such as `1280x720`) asks the host daemon to present
//! [`crate::macos::present::NativeSwapchain`] drawables on screen instead.
//!
//! # Environment
//!
//! | value | meaning |
//! |---|---|
//! | unset / `0` / `off` / `false` | headless (default) |
//! | `1` / `on` / `true` | open a window at the default size |
//! | `<width>x<height>` | open a window at that drawable size |
//! | `title=<text>` (after `,`) | set the window title |
//!
//! # Status
//!
//! This module is the ADR-020 seam. [`NativeWindow::open`] is still a stub that
//! reports [`WindowError::Unsupported`]; the presentation backend fills it in
//! behind the same API. Nothing here may be required by the headless path, and
//! opening a window must never load D3DMetal.

/// Environment variable that opts into on-screen presentation.
pub const WINDOW_ENV: &str = "VICTORY_METAL_WINDOW";

/// Default drawable size when the environment only says "on".
pub const DEFAULT_WINDOW_WIDTH: u32 = 1280;
/// Default drawable size when the environment only says "on".
pub const DEFAULT_WINDOW_HEIGHT: u32 = 720;

/// Requested window geometry and title.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
    pub title: String,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: DEFAULT_WINDOW_WIDTH,
            height: DEFAULT_WINDOW_HEIGHT,
            title: "victory-metal".to_string(),
        }
    }
}

impl WindowConfig {
    /// Parses a `VICTORY_METAL_WINDOW` value.
    ///
    /// Returns `None` for the headless spellings, `Some(Err(..))` for a value
    /// that asks for a window but is malformed, and `Some(Ok(..))` otherwise.
    #[allow(clippy::result_unit_err)]
    pub fn parse(value: &str) -> Option<Result<Self, ()>> {
        let mut parts = value.split(',').map(str::trim);
        let head = parts.next().unwrap_or("").trim();
        let lower = head.to_ascii_lowercase();
        if head.is_empty() || matches!(lower.as_str(), "0" | "off" | "false" | "no") {
            return None;
        }
        let mut config = WindowConfig::default();
        if !matches!(lower.as_str(), "1" | "on" | "true" | "yes") {
            let Some((width, height)) = head.split_once(['x', 'X']) else {
                return Some(Err(()));
            };
            let Ok(width) = width.trim().parse::<u32>() else {
                return Some(Err(()));
            };
            let Ok(height) = height.trim().parse::<u32>() else {
                return Some(Err(()));
            };
            if width == 0 || height == 0 {
                return Some(Err(()));
            }
            config.width = width;
            config.height = height;
        }
        for part in parts {
            if let Some(title) = part.strip_prefix("title=") {
                if title.trim().is_empty() {
                    return Some(Err(()));
                }
                config.title = title.trim().to_string();
            } else if !part.is_empty() {
                return Some(Err(()));
            }
        }
        Some(Ok(config))
    }

    /// Reads the configuration from the process environment.
    pub fn from_env() -> Option<Result<Self, ()>> {
        std::env::var(WINDOW_ENV).ok().and_then(|v| Self::parse(&v))
    }

    /// True when on-screen presentation was requested.
    pub fn requested() -> bool {
        matches!(Self::from_env(), Some(_))
    }
}

/// Failure to open or present through the native window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowError {
    /// No window server session, or the backend is not implemented yet.
    Unsupported(&'static str),
    /// AppKit rejected a call (bad selector, nil object, wrong thread).
    AppKit(String),
    /// Metal or the swapchain refused the drawable.
    Metal(String),
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::Unsupported(why) => write!(f, "native window unsupported: {why}"),
            WindowError::AppKit(why) => write!(f, "AppKit: {why}"),
            WindowError::Metal(why) => write!(f, "Metal: {why}"),
        }
    }
}

impl std::error::Error for WindowError {}

/// An open on-screen presenter.
///
/// The API is intentionally pointer-free: a drawable is identified by its
/// `CAMetalLayer.nextDrawable` identity as a `u64` so the type stays usable
/// from the transport and from tests. The presentation backend replaces the
/// stub body and keeps this surface stable.
#[derive(Debug)]
pub struct NativeWindow {
    config: WindowConfig,
    drawable_serial: u64,
    presented: u64,
}

impl NativeWindow {
    /// Opens a window for `config`.
    ///
    /// Stub: reports [`WindowError::Unsupported`] until ADR-020 lands.
    pub fn open(config: WindowConfig) -> Result<Self, WindowError> {
        let _ = config;
        Err(WindowError::Unsupported(
            "the NSWindow + CAMetalLayer presenter is not implemented in this build",
        ))
    }

    /// The configuration this window was opened with.
    pub fn config(&self) -> &WindowConfig {
        &self.config
    }

    /// Drawables acquired so far.
    pub fn drawable_serial(&self) -> u64 {
        self.drawable_serial
    }

    /// Drawables presented so far.
    pub fn presented(&self) -> u64 {
        self.presented
    }

    /// Acquires the next drawable, if the layer can supply one.
    pub fn next_drawable(&mut self) -> Option<u64> {
        None
    }

    /// Presents the most recently acquired drawable.
    pub fn present(&mut self) -> Result<(), WindowError> {
        Err(WindowError::Unsupported("no drawable to present"))
    }

    /// Pumps one AppKit event cycle so the window stays responsive.
    pub fn pump_events(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headless_spellings_disable_the_window() {
        for value in ["", "0", "off", "false", "no"] {
            assert_eq!(WindowConfig::parse(value), None, "{value}");
        }
    }

    #[test]
    fn on_uses_default_geometry() {
        assert_eq!(WindowConfig::parse("1"), Some(Ok(WindowConfig::default())));
        assert_eq!(WindowConfig::parse("on"), Some(Ok(WindowConfig::default())));
        assert_eq!(WindowConfig::parse("true"), Some(Ok(WindowConfig::default())));
    }

    #[test]
    fn size_and_title_are_parsed() {
        let parsed = WindowConfig::parse("1920x1080,title=Victory").unwrap().unwrap();
        assert_eq!(parsed.width, 1920);
        assert_eq!(parsed.height, 1080);
        assert_eq!(parsed.title, "Victory");
    }

    #[test]
    fn malformed_specs_are_rejected() {
        assert_eq!(WindowConfig::parse("width"), Some(Err(())));
        assert_eq!(WindowConfig::parse("0x10"), Some(Err(())));
        assert_eq!(WindowConfig::parse("10x0"), Some(Err(())));
        assert_eq!(WindowConfig::parse("10x10,title="), Some(Err(())));
        assert_eq!(WindowConfig::parse("10x10,bogus"), Some(Err(())));
    }

    #[test]
    fn the_stub_reports_unsupported() {
        assert!(matches!(
            NativeWindow::open(WindowConfig::default()),
            Err(WindowError::Unsupported(_))
        ));
    }
}
