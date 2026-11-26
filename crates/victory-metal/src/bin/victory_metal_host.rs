//! Standalone ARM64 `victory-metal-host` daemon (ADR-016).
//!
//! Reserved by the orchestrator. This binary hosts the native Metal engine over
//! the authenticated loopback session so a guest started with
//! `VICTORY_BACKEND=metal` can auto-connect, submit Wire v3 work and present
//! without any Wine or D3DMetal component.

fn main() {
    eprintln!("victory-metal-host: not implemented yet (reserved by the orchestrator)");
    std::process::exit(2);
}
