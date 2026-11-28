//! ADR-053: exact host encode counters and the opt-in draw / offscreen-RT
//! X-ray tracer.
//!
//! The counters are always on and are cheap (one relaxed atomic per encoded
//! command); they answer "did a real draw or dispatch reach Metal at all?".
//! The tracer is off unless `VICTORY_METAL_DRAW_TRACE` is a truthy value, and
//! it is bounded: at most [`MAX_TRACED_WORK`] draws/dispatches and
//! [`MAX_TRACED_PASSES`] render passes are described per host process, so a
//! long boot never grows its log without bound.
//!
//! The trace never records shader bytes, resource contents or title paths: it
//! describes the encoder call (kind, pipeline id, counts, viewport, scissor,
//! argument-buffer shape), the attachment geometry and format, whether that
//! attachment is the registered swapchain surface, and a bounded pixel summary
//! of the first traced offscreen render target.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Environment flag that arms the tracer. Off when unset.
pub const DRAW_TRACE_ENV: &str = "VICTORY_METAL_DRAW_TRACE";

/// Draws/dispatches the tracer describes per process.
pub const MAX_TRACED_WORK: usize = 10;

/// Render passes the tracer samples per process.
pub const MAX_TRACED_PASSES: u32 = 10;

/// Edge, in texels, of the bounded offscreen-RT pixel sample.
pub const RT_SAMPLE_EDGE: u32 = 64;

/// Environment flag that arms the tracer, read once.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| enabled_from(std::env::var(DRAW_TRACE_ENV).ok().as_deref()))
}

/// Pure form of [`enabled`]: `1`, `true`, `on` or `yes` arm the tracer.
pub fn enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
    )
}

/// The exact encoded-work counters of one host process (ADR-053).
///
/// They are atomics so a read-only planner path can count without taking the
/// device state mutex again; ordering is relaxed because the counters are
/// monotonic diagnostics, never synchronization.
#[derive(Default)]
pub struct EncodeCounters {
    /// Non-indexed draw calls encoded (including instanced ones).
    pub draws: AtomicU64,
    /// Indexed draw calls encoded (including instanced ones).
    pub draws_indexed: AtomicU64,
    /// Direct compute dispatches encoded.
    pub dispatches: AtomicU64,
    /// Mesh dispatches encoded (`dispatchThreadgroups` on the mesh stage).
    pub mesh_dispatches: AtomicU64,
    /// Indirect draw/dispatch calls encoded (one per GPU-driven command).
    pub indirect: AtomicU64,
    /// Render passes opened (a clear inside a pass opens a continuation).
    pub render_passes: AtomicU64,
    /// Explicit clear commands folded into pass load actions.
    pub clear_ops: AtomicU64,
}

impl EncodeCounters {
    /// Records one non-indexed draw.
    pub fn bump_draw(&self) {
        self.draws.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one indexed draw.
    pub fn bump_draw_indexed(&self) {
        self.draws_indexed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one direct compute dispatch.
    pub fn bump_dispatch(&self) {
        self.dispatches.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one mesh dispatch.
    pub fn bump_mesh_dispatch(&self) {
        self.mesh_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one indirect draw/dispatch call.
    pub fn bump_indirect(&self) {
        self.indirect.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one opened render pass.
    pub fn bump_render_pass(&self) {
        self.render_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one explicit clear folded into a load action.
    pub fn bump_clear(&self) {
        self.clear_ops.fetch_add(1, Ordering::Relaxed);
    }

    /// A consistent-enough copy of every counter.
    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            draws: self.draws.load(Ordering::Relaxed),
            draws_indexed: self.draws_indexed.load(Ordering::Relaxed),
            dispatches: self.dispatches.load(Ordering::Relaxed),
            mesh_dispatches: self.mesh_dispatches.load(Ordering::Relaxed),
            indirect: self.indirect.load(Ordering::Relaxed),
            render_passes: self.render_passes.load(Ordering::Relaxed),
            clear_ops: self.clear_ops.load(Ordering::Relaxed),
        }
    }
}

/// Plain copy of [`EncodeCounters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CounterSnapshot {
    /// Non-indexed draws encoded.
    pub draws: u64,
    /// Indexed draws encoded.
    pub draws_indexed: u64,
    /// Direct compute dispatches encoded.
    pub dispatches: u64,
    /// Mesh dispatches encoded.
    pub mesh_dispatches: u64,
    /// Indirect draw/dispatch calls encoded.
    pub indirect: u64,
    /// Render passes opened.
    pub render_passes: u64,
    /// Clears folded into load actions.
    pub clear_ops: u64,
}

impl CounterSnapshot {
    /// The seven counters as `telemetry.json` members (no trailing comma).
    pub fn json_fields(&self) -> String {
        format!(
            "\"draws_encoded\": {},\n  \"draw_indexed_encoded\": {},\n  \
             \"dispatches_encoded\": {},\n  \"mesh_dispatches_encoded\": {},\n  \
             \"indirect_encoded\": {},\n  \"render_passes_encoded\": {},\n  \
             \"clear_ops_encoded\": {}",
            self.draws,
            self.draws_indexed,
            self.dispatches,
            self.mesh_dispatches,
            self.indirect,
            self.render_passes,
            self.clear_ops
        )
    }

    /// Total draw-ish calls (draws, indexed draws, mesh dispatches).
    pub fn total_draws(&self) -> u64 {
        self.draws
            .saturating_add(self.draws_indexed)
            .saturating_add(self.mesh_dispatches)
    }
}

/// One attachment the tracer may sample.
#[derive(Debug, Clone, Copy)]
pub struct TraceTarget {
    /// Level-0 texture of the pass's first colour attachment.
    ///
    /// Stored as a plain address so the tracer state stays `Send`; the host
    /// casts it back while the registry still owns the texture.
    pub texture: usize,
    /// Attachment width in texels.
    pub width: u32,
    /// Attachment height in texels.
    pub height: u32,
    /// Attachment pixel format name.
    pub format: &'static str,
    /// Whether the texture is the registered swapchain surface.
    pub swapchain: bool,
}

/// One traced draw or dispatch.
#[derive(Debug, Clone, Copy)]
pub struct DrawRecord {
    /// One-based index among traced work (1..=10).
    pub ordinal: usize,
    /// `draw`, `draw_indexed`, `draw_indirect`, `dispatch`, `dispatch_indirect`
    /// or `mesh_dispatch`.
    pub kind: &'static str,
    /// Selected pipeline id.
    pub pipeline_id: u64,
    /// Vertices, indices or threadgroups the call names.
    pub count: u32,
    /// Instances (draws only; `1` otherwise).
    pub instances: u32,
    /// Viewport rectangle `(x, y, width, height)` in pixels, when set.
    pub viewport: Option<[u32; 4]>,
    /// Scissor rectangle `(x, y, width, height)` in pixels, when set.
    pub scissor: Option<[u32; 4]>,
    /// Reflected argument-buffer parameters of the active pipeline.
    pub abi_parameters: u32,
    /// Bytes of the stage argument buffers the call binds.
    pub abi_bytes: u32,
    /// Attachment the pass renders into, when known.
    pub target: Option<TraceTarget>,
}

impl DrawRecord {
    /// One log line, counts and shapes only.
    pub fn line(&self) -> String {
        let mut out = format!(
            "draw_trace ordinal={} kind={} pipeline={} count={} instances={}",
            self.ordinal, self.kind, self.pipeline_id, self.count, self.instances
        );
        if let Some([x, y, w, h]) = self.viewport {
            out.push_str(&format!(" viewport={x},{y},{w},{h}"));
        } else {
            out.push_str(" viewport=none");
        }
        if let Some([x, y, w, h]) = self.scissor {
            out.push_str(&format!(" scissor={x},{y},{w},{h}"));
        } else {
            out.push_str(" scissor=none");
        }
        out.push_str(&format!(
            " abi_parameters={} abi_bytes={}",
            self.abi_parameters, self.abi_bytes
        ));
        match self.target {
            Some(target) => out.push_str(&format!(
                " rt={} {}x{} {}",
                if target.swapchain {
                    "swapchain"
                } else {
                    "offscreen"
                },
                target.width,
                target.height,
                target.format
            )),
            None => out.push_str(" rt=none"),
        }
        out
    }
}

/// Bounded per-process tracer state.
#[derive(Default)]
pub struct TraceState {
    /// Traced calls, in encode order.
    pub records: Vec<DrawRecord>,
    /// Render passes opened so far.
    pub passes: u32,
    /// Passes whose pixels were sampled.
    pub sampled_passes: u32,
}

impl TraceState {
    /// Whether [`MAX_TRACED_WORK`] calls were already described.
    pub fn saturated(&self) -> bool {
        self.records.len() >= MAX_TRACED_WORK
    }

    /// Appends and formats one record when the process still traces.
    pub fn record(&mut self, record: DrawRecord) -> Option<String> {
        if self.saturated() {
            return None;
        }
        let line = record.line();
        self.records.push(record);
        Some(line)
    }

    /// Counts one opened pass and reports whether it should be sampled.
    pub fn next_pass(&mut self) -> bool {
        self.passes = self.passes.saturating_add(1);
        if self.sampled_passes >= MAX_TRACED_PASSES {
            return false;
        }
        self.sampled_passes = self.sampled_passes.saturating_add(1);
        true
    }
}

/// Bounded summary of a sampled `RGBA8`/`BGRA8` attachment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PixelStats {
    /// Bytes that are not zero.
    pub nonzero_bytes: u64,
    /// Bytes sampled.
    pub sampled_bytes: u64,
    /// Distinct byte values, saturated at 256.
    pub distinct_values: u32,
}

impl PixelStats {
    /// One log line.
    pub fn line(&self) -> String {
        format!(
            "draw_trace_rt nonzero_bytes={} sampled_bytes={} distinct_values={}",
            self.nonzero_bytes, self.sampled_bytes, self.distinct_values
        )
    }
}

/// Summarizes a `RT_SAMPLE_EDGE`-square grid of evenly spaced texels from a
/// whole `RGBA8`/`BGRA8` attachment.
///
/// The input is the level-0 image (`width * height * 4` bytes, tightly packed).
/// At most [`RT_SAMPLE_EDGE`]² texels are read, so a 4K attachment costs the
/// same bounded summary as a 64×64 one.
pub fn sample_grid(bytes: &[u8], width: u32, height: u32) -> PixelStats {
    if width == 0 || height == 0 {
        return PixelStats::default();
    }
    let row_bytes = width as usize * 4;
    if bytes.len() < row_bytes * height as usize {
        return PixelStats::default();
    }
    let edge = RT_SAMPLE_EDGE.min(width).min(height);
    let step_x = (width / edge).max(1) as usize;
    let step_y = (height / edge).max(1) as usize;
    let mut sample = Vec::with_capacity((edge * edge * 4) as usize);
    for row in 0..edge as usize {
        let y = (row * step_y).min(height as usize - 1);
        for column in 0..edge as usize {
            let x = (column * step_x).min(width as usize - 1);
            let at = y * row_bytes + x * 4;
            sample.extend_from_slice(&bytes[at..at + 4]);
        }
    }
    pixel_stats(&sample)
}

/// Summarizes a sampled attachment without copying it.
pub fn pixel_stats(bytes: &[u8]) -> PixelStats {
    let mut nonzero = 0u64;
    let mut seen = [false; 256];
    let mut distinct = 0u32;
    for &byte in bytes {
        if byte != 0 {
            nonzero += 1;
        }
        let slot = usize::from(byte);
        if !seen[slot] {
            seen[slot] = true;
            distinct += 1;
        }
    }
    PixelStats {
        nonzero_bytes: nonzero,
        sampled_bytes: bytes.len() as u64,
        distinct_values: distinct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tracer_is_off_without_a_truthy_flag() {
        assert!(!enabled_from(None));
        assert!(!enabled_from(Some("")));
        assert!(!enabled_from(Some("0")));
        assert!(!enabled_from(Some("false")));
        for value in ["1", "true", " on ", "yes", "ON"] {
            assert!(enabled_from(Some(value)), "{value:?} arms the tracer");
        }
    }

    #[test]
    fn counters_bump_independently_and_snapshot() {
        let counters = EncodeCounters::default();
        counters.bump_draw();
        counters.bump_draw_indexed();
        counters.bump_dispatch();
        counters.bump_mesh_dispatch();
        counters.bump_indirect();
        counters.bump_render_pass();
        counters.bump_clear();
        let snapshot = counters.snapshot();
        assert_eq!(snapshot.draws, 1);
        assert_eq!(snapshot.draws_indexed, 1);
        assert_eq!(snapshot.total_draws(), 3);
        assert_eq!(snapshot.dispatches, 1);
        assert_eq!(snapshot.mesh_dispatches, 1);
        assert_eq!(snapshot.indirect, 1);
        assert_eq!(snapshot.render_passes, 1);
        assert_eq!(snapshot.clear_ops, 1);
        assert!(snapshot.json_fields().contains("\"draws_encoded\": 1"));
    }

    #[test]
    fn trace_state_bounds_records_and_passes() {
        let mut state = TraceState::default();
        for ordinal in 1..=MAX_TRACED_WORK {
            let record = DrawRecord {
                ordinal,
                kind: "draw",
                pipeline_id: u64::from(ordinal as u32),
                count: 3,
                instances: 1,
                viewport: None,
                scissor: None,
                abi_parameters: 0,
                abi_bytes: 0,
                target: None,
            };
            assert!(state.record(record).is_some());
        }
        assert!(state.saturated());
        assert!(
            state
                .record(DrawRecord {
                    ordinal: 11,
                    kind: "draw",
                    pipeline_id: 0,
                    count: 1,
                    instances: 1,
                    viewport: None,
                    scissor: None,
                    abi_parameters: 0,
                    abi_bytes: 0,
                    target: None,
                })
                .is_none()
        );
        for _ in 0..MAX_TRACED_PASSES {
            assert!(state.next_pass());
        }
        assert!(!state.next_pass());
    }

    #[test]
    fn a_record_names_counts_shapes_and_attachment() {
        let record = DrawRecord {
            ordinal: 2,
            kind: "draw_indexed",
            pipeline_id: 9,
            count: 36,
            instances: 4,
            viewport: Some([0, 0, 1280, 720]),
            scissor: Some([0, 0, 640, 360]),
            abi_parameters: 3,
            abi_bytes: 40,
            target: Some(TraceTarget {
                texture: 0,
                width: 1920,
                height: 1080,
                format: "rgba8unorm",
                swapchain: false,
            }),
        };
        let line = record.line();
        assert!(line.contains("kind=draw_indexed"));
        assert!(line.contains("count=36"));
        assert!(line.contains("viewport=0,0,1280,720"));
        assert!(line.contains("rt=offscreen 1920x1080 rgba8unorm"));
    }

    #[test]
    fn pixel_stats_are_exact_and_bounded() {
        let clear = vec![0u8; 4096];
        assert_eq!(pixel_stats(&clear).nonzero_bytes, 0);
        assert_eq!(pixel_stats(&clear).distinct_values, 1);
        let mut drawn = vec![0u8; 4096];
        drawn[100] = 0xff;
        drawn[101] = 0x7f;
        let stats = pixel_stats(&drawn);
        assert_eq!(stats.nonzero_bytes, 2);
        assert_eq!(stats.distinct_values, 3);
        assert_eq!(stats.sampled_bytes, 4096);
    }
}
