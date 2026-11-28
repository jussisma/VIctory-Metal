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
//! search can be tested without a device.

use crate::macos::bindings::{DescriptorTableKind, RootParameterKind, RootSignatureLayout};
use crate::wire_v3::DescriptorKind;

/// Whether a wire descriptor-table kind can fill a declared range kind.
///
/// This is the single shape matrix: the converter uses one 24-byte
/// `{device_address, 0, 0}` descriptor for every buffer range, so a
/// CBV/SRV/UAV range accepts a `Cbv`/`Srv`/`Uav` wire table; a `Texture` range
/// accepts the texture-id (`Srv`/`Uav`) shapes only; a `Sampler` range accepts
/// sampler ids only; and `Unspecified` (reflection's plain `"Table"`) accepts
/// every wire kind because [`crate::macos::device`]'s shape check selects from
/// the entry variants. A genuine mismatch is never coerced.
pub(crate) fn wire_kind_fills_table(declared: DescriptorTableKind, wire: DescriptorKind) -> bool {
    match declared {
        DescriptorTableKind::ConstantBufferView
        | DescriptorTableKind::ShaderResourceView
        | DescriptorTableKind::UnorderedAccessView => {
            matches!(
                wire,
                DescriptorKind::Cbv | DescriptorKind::Srv | DescriptorKind::Uav
            )
        }
        DescriptorTableKind::Texture => matches!(wire, DescriptorKind::Srv | DescriptorKind::Uav),
        DescriptorTableKind::Sampler => wire == DescriptorKind::Sampler,
        DescriptorTableKind::Unspecified => true,
    }
}

/// Whether a wire argument kind can address a reflected parameter kind.
///
/// A converter-reflected table is reported only as `"Table"` and is stored as
/// [`DescriptorTableKind::Unspecified`], so an unspecified table range matches
/// any concrete table kind. Two *concrete* table kinds must agree: a `Sampler`
/// range never matches a buffer or texture view range (and the reverse), and a
/// `Texture` range never matches an address-buffer range. Every non-table kind
/// stays exact.
pub(crate) fn kinds_compatible(declared: RootParameterKind, named: RootParameterKind) -> bool {
    match (declared, named) {
        (
            RootParameterKind::DescriptorTable(declared),
            RootParameterKind::DescriptorTable(named),
        ) => {
            declared == named
                || declared == DescriptorTableKind::Unspecified
                || named == DescriptorTableKind::Unspecified
        }
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
///
/// ADR-062a: the ordinal fast path is taken only when the parameter there is
/// kind-compatible with `kind`. Otherwise a descriptor table is resolved by
/// [`unique_compatible_table`] instead of `parameter_by`: the latter binds the
/// first wildcard table it meets and would let a `u32::MAX` sampler win over the
/// view table the wire named.
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
    // (an unknown `u32::MAX` field, or the exact declared pair) *and* the
    // parameter's kind is compatible.
    if let Some(parameter) = layout.parameters().get(wire_slot)
        && kinds_compatible(parameter.kind(), kind)
        && (parameter.space() == space || parameter.space() == u32::MAX)
        && (parameter.register() == register || parameter.register() == u32::MAX)
    {
        return Some(wire_slot);
    }
    if matches!(kind, RootParameterKind::DescriptorTable(_)) {
        return unique_compatible_table(layout, space, register, kind);
    }
    if let Ok(Some(found)) = layout.parameter_by(space, register, kind) {
        return Some(found);
    }
    None
}

/// The unique descriptor-table parameter that can carry a table of `kind`.
///
/// ADR-062a: once the ordinal fast path has missed on kind, the only safe
/// fallback is a *unique* candidate. A parameter qualifies when its kind is
/// compatible with `kind` and its reflected `(space, register)` is the
/// requested pair or a `u32::MAX` wildcard (converter reflection reports every
/// table that way). Two or more candidates are ambiguous: returning either would
/// be a guess, so this returns `None` and the caller records a counted refusal.
pub(crate) fn unique_compatible_table(
    layout: &RootSignatureLayout,
    space: u32,
    register: u32,
    kind: RootParameterKind,
) -> Option<usize> {
    let mut found = None;
    for (index, parameter) in layout.parameters().iter().enumerate() {
        if !matches!(parameter.kind(), RootParameterKind::DescriptorTable(_)) {
            continue;
        }
        if !kinds_compatible(parameter.kind(), kind) {
            continue;
        }
        if parameter.space() != space && parameter.space() != u32::MAX {
            continue;
        }
        if parameter.register() != register && parameter.register() != u32::MAX {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(index);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::{kinds_compatible, parameter_index, wire_kind_fills_table};
    use crate::macos::bindings::{
        DescriptorTableKind, ReflectedStage, RootParameterKind, RootParameterSpec,
        RootSignatureLayout,
    };
    use crate::wire_v3::DescriptorKind;

    const SRV: RootParameterKind =
        RootParameterKind::DescriptorTable(DescriptorTableKind::ShaderResourceView);
    const SAMPLER: RootParameterKind =
        RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler);
    const TEXTURE: RootParameterKind =
        RootParameterKind::DescriptorTable(DescriptorTableKind::Texture);

    /// A fragment layout with one descriptor table per `(kind, space, register)`
    /// declaration, flattened in declaration order.
    fn table_layout(parameters: &[(RootParameterKind, u32, u32)]) -> RootSignatureLayout {
        let specs: Vec<RootParameterSpec> = parameters
            .iter()
            .map(|(kind, space, register)| RootParameterSpec {
                kind: *kind,
                size: 8,
                register: *register,
                space: *space,
            })
            .collect();
        RootSignatureLayout::from_parameters(ReflectedStage::Fragment, &specs)
            .expect("a test root signature must build")
    }

    #[test]
    fn an_srv_table_skips_a_sampler_at_the_wire_ordinal() {
        // Reflection flattens the sampler first; the wire names the SRV view
        // table at the same ordinal and the same `(space, register)`.
        let layout = table_layout(&[(SAMPLER, 0, 0), (SRV, 0, 0)]);
        assert_eq!(parameter_index(&layout, 0, 0, 0, SRV), Some(1));
    }

    #[test]
    fn a_sampler_table_resolves_to_the_sampler() {
        let layout = table_layout(&[(SAMPLER, 0, 0), (SRV, 0, 0)]);
        assert_eq!(parameter_index(&layout, 0, 0, 0, SAMPLER), Some(0));
        // Even when the ordinal names the view parameter, the only compatible
        // table is the sampler.
        assert_eq!(parameter_index(&layout, 1, 0, 0, SAMPLER), Some(0));
    }

    #[test]
    fn a_non_table_ordinal_still_resolves_the_table_the_wire_names() {
        // The ordinal is a root view, not a table; the table the wire names is
        // the second parameter.
        let layout = table_layout(&[(RootParameterKind::ConstantBufferView, 0, 0), (SRV, 0, 0)]);
        assert_eq!(parameter_index(&layout, 0, 0, 0, SRV), Some(1));
    }

    #[test]
    fn two_kind_compatible_candidates_are_ambiguous() {
        let layout = table_layout(&[(SAMPLER, 0, 0), (SRV, 0, 0), (SRV, 0, 0)]);
        assert_eq!(
            parameter_index(&layout, 0, 0, 0, SRV),
            None,
            "two SRV candidates must refuse rather than guess"
        );
    }

    #[test]
    fn a_texture_range_and_a_buffer_range_do_not_cross_match() {
        let texture = table_layout(&[(TEXTURE, 0, 0)]);
        assert_eq!(
            parameter_index(&texture, 0, 0, 0, SRV),
            None,
            "an SRV table cannot take a Texture parameter"
        );
        let view = table_layout(&[(SRV, 0, 0)]);
        assert_eq!(
            parameter_index(&view, 0, 0, 0, TEXTURE),
            None,
            "a Texture table cannot take an SRV parameter"
        );
        assert!(!kinds_compatible(TEXTURE, SRV));
        assert!(!kinds_compatible(SRV, TEXTURE));
        // Sampler never crosses a buffer or texture view range in either
        // direction.
        assert!(!kinds_compatible(SAMPLER, SRV));
        assert!(!kinds_compatible(SRV, SAMPLER));
        assert!(!kinds_compatible(SAMPLER, TEXTURE));
        assert!(!kinds_compatible(TEXTURE, SAMPLER));
    }

    #[test]
    fn unspecified_parameters_keep_matching_any_concrete_kind() {
        let unspecified = RootParameterKind::DescriptorTable(DescriptorTableKind::Unspecified);
        let unspecified_layout = table_layout(&[(unspecified, 0, 0)]);
        for kind in [SRV, SAMPLER, TEXTURE] {
            assert_eq!(
                parameter_index(&unspecified_layout, 0, 0, 0, kind),
                Some(0),
                "an Unspecified reflected table keeps matching {kind:?}"
            );
        }
        // The reverse: a wire `Unspecified` table still matches a concrete
        // reflected parameter.
        let concrete = table_layout(&[(SRV, 0, 0)]);
        assert_eq!(parameter_index(&concrete, 0, 0, 0, unspecified), Some(0));
    }

    /// ADR-062: the shape matrix that both the resolver search and
    /// `device::check_table_kind` consult. Every address range accepts every
    /// address wire kind, a texture range accepts texture ids only, a sampler
    /// range is exact, and `Unspecified` accepts every wire kind.
    #[test]
    fn the_wire_shape_matrix_is_exact() {
        for declared in [
            DescriptorTableKind::ConstantBufferView,
            DescriptorTableKind::ShaderResourceView,
            DescriptorTableKind::UnorderedAccessView,
        ] {
            for wire in [
                DescriptorKind::Cbv,
                DescriptorKind::Srv,
                DescriptorKind::Uav,
            ] {
                assert!(
                    wire_kind_fills_table(declared, wire),
                    "{declared:?} must accept a {wire:?} buffer table"
                );
            }
            assert!(
                !wire_kind_fills_table(declared, DescriptorKind::Sampler),
                "{declared:?} must refuse a sampler table"
            );
        }
        for wire in [DescriptorKind::Srv, DescriptorKind::Uav] {
            assert!(wire_kind_fills_table(DescriptorTableKind::Texture, wire));
        }
        for wire in [DescriptorKind::Cbv, DescriptorKind::Sampler] {
            assert!(!wire_kind_fills_table(DescriptorTableKind::Texture, wire));
        }
        assert!(wire_kind_fills_table(
            DescriptorTableKind::Sampler,
            DescriptorKind::Sampler
        ));
        for wire in [
            DescriptorKind::Cbv,
            DescriptorKind::Srv,
            DescriptorKind::Uav,
        ] {
            assert!(!wire_kind_fills_table(DescriptorTableKind::Sampler, wire));
        }
        for wire in [
            DescriptorKind::Cbv,
            DescriptorKind::Srv,
            DescriptorKind::Uav,
            DescriptorKind::Sampler,
        ] {
            assert!(wire_kind_fills_table(
                DescriptorTableKind::Unspecified,
                wire
            ));
        }
    }

    #[test]
    fn non_table_kinds_stay_exact() {
        use crate::macos::bindings::RootParameterKind as Kind;
        assert!(kinds_compatible(Kind::Constants, Kind::Constants));
        assert!(!kinds_compatible(Kind::Constants, Kind::ConstantBufferView));
        assert!(!kinds_compatible(
            Kind::ShaderResourceView,
            Kind::UnorderedAccessView
        ));
    }
}
