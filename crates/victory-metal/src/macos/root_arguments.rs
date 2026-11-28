//! Kind-compatible resolution of wire root arguments to reflected parameters.
//!
//! ADR-062a: a descriptor-table parameter carries the converter's range kind,
//! and a wire table must never resolve to a reflected parameter of an
//! incompatible kind (an `Srv`/`Uav`/`Cbv` table onto a `Sampler` parameter,
//! or a sampler table onto a view parameter). Cycle 17 left 40 live refusals of
//! exactly that shape, named
//! `descriptor table wire kind Srv cannot fill a reflected Sampler range`.
//!
//! This module owns the single decision used by `device::flush_abi` (and its
//! descriptor-table build path), so the compatibility rule and the fallback
//! search can be tested without a device. Seam extraction only: the behaviour
//! is unchanged from `device::abi_parameter_index`, which this replaces.

use crate::macos::bindings::{RootParameterKind, RootSignatureLayout};

/// Whether a wire argument kind can address a reflected parameter kind.
///
/// A table is a table; converter reflection cannot distinguish its range kinds
/// (`Table` is stored as `Unspecified`), so those match any table.
pub(crate) fn kinds_compatible(declared: RootParameterKind, named: RootParameterKind) -> bool {
    match (declared, named) {
        (RootParameterKind::DescriptorTable(_), RootParameterKind::DescriptorTable(_)) => true,
        (left, right) => left == right,
    }
}

/// The layout index a wire argument names in one stage layout.
///
/// The resolver is `(space, register, kind)`. The wire also carries the flat
/// index the title named, which is exactly the converter's parameter index on
/// the legacy registration path (a gate pipeline registered from supplied
/// reflection has no versioned signature map). When the triple is not declared
/// — or a table's reflected triple is `u32::MAX` — the wire index is the
/// deterministic identity, never a silent drop.
pub(crate) fn parameter_index(
    layout: &RootSignatureLayout,
    wire_slot: usize,
    space: u32,
    register: u32,
    kind: RootParameterKind,
) -> Option<usize> {
    // Converter reflection reports a descriptor table's Slot/Space as
    // `u32::MAX`, so every table's reflected triple is identical: a triple
    // lookup would bind the first table twice. The wire index is that path's
    // only identity, and it is checked first whenever the layout agrees with it
    // (an unknown `u32::MAX` field, or the exact declared pair).
    if let Some(parameter) = layout.parameters().get(wire_slot)
        && kinds_compatible(parameter.kind(), kind)
        && (parameter.space() == space || parameter.space() == u32::MAX)
        && (parameter.register() == register || parameter.register() == u32::MAX)
    {
        return Some(wire_slot);
    }
    if let Ok(Some(found)) = layout.parameter_by(space, register, kind) {
        return Some(found);
    }
    None
}
