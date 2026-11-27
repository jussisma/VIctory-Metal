//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (Grand Theft Auto VI asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: Grand Theft Auto VI's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<Upload>>,
    /// The host lists the calls record into, in order: one, and one more
    /// for every wait the title records after commands ([`wait`]). Shared
    /// with the queues that hold them until their waits hold
    /// (`crate::hold`), so they outlive a list the title releases first.
    #[cfg(windows)]
    hosts: std::sync::Mutex<Vec<std::sync::Arc<crate::host::HostList>>>,
    /// Which of them the calls record into.
    #[cfg(windows)]
    segment: std::sync::atomic::AtomicUsize,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
    /// The native Metal producer when `VICTORY_BACKEND=metal` (ADR-008):
    /// this list records Wire v3 submissions instead of host D3D12 work.
    #[cfg(windows)]
    metal: std::sync::Mutex<Option<crate::native_metal::producer::MetalList>>,
    /// Guest backing pins taken for this list's native bindings, retired by
    /// the queue after the GPU completion that runs them
    /// ([`take_retained`]).
    #[cfg(windows)]
    metal_retained: std::sync::Mutex<Vec<u64>>,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
pub(crate) fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
}

/// The host list a title list's calls record into (its current segment),
/// if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    let current: *const crate::host::HostList = std::sync::Arc::as_ptr(hosts.get(k)?);
    // SAFETY: the list holds this host list until it is reset or released
    // (an `Arc` does not move it), and the title records into a list from
    // one thread at a time.
    Some(unsafe { &*current })
}

/// A title list's host lists, in order: what a submission of it runs.
#[cfg(windows)]
pub(crate) fn segments(this: This) -> Vec<std::sync::Arc<crate::host::HostList>> {
    if this.is_null() {
        return Vec::new();
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    hosts.iter().take(k + 1).cloned().collect()
}

/// List identifiers for the native path; never zero.
#[cfg(windows)]
fn metal_list_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Relaxed)
}

/// Runs `f` on the native producer of a title list, when `VICTORY_BACKEND`
/// selects Metal.
///
/// The backend is read from the cached `GpuBackend` once per call; when the
/// backend is `D3d12` the closure never runs and the host recording below it
/// stays untouched. A refusal is remembered by the [`MetalList`] and returns
/// its HRESULT from `Close`; nothing falls back to D3D12/D3DMetal (ADR-008).
#[cfg(windows)]
pub(crate) fn with_metal(
    this: This,
    f: impl FnOnce(
        &mut crate::native_metal::producer::MetalProducer,
    ) -> Result<(), crate::native_metal::producer::ProducerError>,
) {
    if crate::backend::GpuBackend::from_env() != crate::backend::GpuBackend::Metal || this.is_null()
    {
        return;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let mut metal = list.metal.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(metal) = metal.as_mut() {
        let result = f(metal.producer_mut());
        if let Err(error) = result {
            // Record the refusal where it happens: `Close` returns the
            // HRESULT and the list submits nothing, so waiting until
            // `ExecuteCommandLists` would lose the opcode.
            crate::native_metal::telemetry::note_unsupported_opcode(error.opcode());
            if let Some(operation) = error.operation() {
                crate::native_metal::telemetry::note_named_refusal(operation);
            }
            metal.note_error(error);
        }
    }
}

/// The native submissions of a list finished by `Close`, for the queue/bridge
/// task to carry over the wire.
#[cfg(windows)]
#[allow(dead_code)]
pub(crate) fn take_submissions(this: This) -> Vec<crate::native_metal::producer::SubmissionV3> {
    if this.is_null() {
        return Vec::new();
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let mut metal = list.metal.lock().unwrap_or_else(|e| e.into_inner());
    metal.as_mut().map(|metal| metal.take()).unwrap_or_default()
}

/// Calls the native producer refused on this list so far.
///
/// A refused list submits nothing: `Close` returns the first refusal's
/// HRESULT and the batch the queue would carry is empty. The count lets the
/// queue attribute such a frame instead of reporting a silent gap; the refused
/// opcode itself is not retained anywhere readable (it lives in `MetalList`'s
/// private `failed` field), so the telemetry histogram cannot name it yet.
#[cfg(windows)]
#[allow(dead_code)]
pub(crate) fn metal_failures(this: This) -> u64 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let metal = list.metal.lock().unwrap_or_else(|e| e.into_inner());
    metal.as_ref().map(|metal| metal.failures()).unwrap_or(0)
}

/// The guest backing pins this list's native bindings hold, for the queue to
/// retire ([`crate::native_metal::bindings::GuestRangeResolver::retire`])
/// after the GPU completion that ran them.
#[cfg(windows)]
#[allow(dead_code)]
pub(crate) fn take_retained(this: This) -> Vec<u64> {
    if this.is_null() {
        return Vec::new();
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    std::mem::take(&mut *list.metal_retained.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Remembers pins a recorded native binding holds until the queue retires
/// them.
#[cfg(windows)]
fn retain_on_list(this: This, tokens: &[u64]) {
    if tokens.is_empty() || this.is_null() {
        return;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    list.metal_retained
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .extend_from_slice(tokens);
}

/// Refuses a native binding that could not be resolved, with the precise
/// reason in the log and a fixed operation name on the wire path.
#[cfg(windows)]
fn refuse_binding(this: This, why: &'static str, error: impl std::fmt::Display) {
    tracing::warn!(error = %error, "{why}");
    with_metal(this, |producer| producer.refuse(why, None));
}

/// Refuses a binding whose guest allocation the host has not registered.
///
/// The operation is expressible, so the title sees `E_FAIL`, not `E_NOTIMPL`:
/// nothing about the command list is unimplemented, the backing is simply not
/// visible to the host (no share file, no resolver range, or a refused kind 2).
#[cfg(windows)]
fn refuse_unregistered(this: This, why: &'static str, error: impl std::fmt::Display) {
    tracing::warn!(error = %error, "{why}");
    crate::native_metal::telemetry::note_registration_failure();
    with_metal(this, |_| {
        Err(crate::native_metal::producer::ProducerError::unregistered(why))
    });
}

/// Refuses a native binding whose guest range could not be resolved.
///
/// The operation is expressible; the guest address names no range the resolver
/// can pin, so the title sees `E_FAIL` (not `E_NOTIMPL`) with the exact
/// resolver error in the log.
#[cfg(windows)]
fn refuse_unresolved(this: This, why: &'static str, error: impl std::fmt::Display) {
    tracing::warn!(error = %error, "{why}");
    with_metal(this, |_| {
        Err(crate::native_metal::producer::ProducerError::unresolved(why))
    });
}

/// Maps a lazy first-bind registration refusal to the producer error a copy or
/// indirect call records.
///
/// A missing resolver or an unresolvable range is `ProducerError::unresolved`;
/// every other refusal (no share backing, a non-increasing id, a host refusal)
/// is `ProducerError::unregistered`, because the operation is expressible and
/// only the host-visible backing is missing. Both are `E_FAIL`; the reason is
/// logged, counted and returned to the title, never skipped or fallen back.
#[cfg(windows)]
fn registration_refusal(
    operation: &'static str,
    error: crate::native_metal::queue::BufferRegistrationError,
) -> crate::native_metal::producer::ProducerError {
    use crate::native_metal::producer::ProducerError;
    use crate::native_metal::queue::BufferRegistrationError;

    tracing::warn!(%error, "{operation}");
    match error {
        BufferRegistrationError::NoResolver | BufferRegistrationError::Unresolved { .. } => {
            ProducerError::unresolved(operation)
        }
        _ => {
            crate::native_metal::telemetry::note_registration_failure();
            ProducerError::unregistered(operation)
        }
    }
}

/// Why a compute root buffer view could not be resolved for recording.
#[cfg(any(windows, test))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum ComputeBindingRefusal {
    /// The resolver could not map the guest address to a share range.
    Unresolved(crate::native_metal::bindings::ResolveError),
    /// The host's kind 2 registry does not know the resolved allocation.
    Unregistered(String),
}

#[cfg(any(windows, test))]
impl ComputeBindingRefusal {
    /// The producer error the recording is refused with.
    ///
    /// Both cases are `E_FAIL`: the operation is expressible, the guest
    /// backing is what is missing.
    fn producer_error(&self, operation: &'static str) -> crate::native_metal::producer::ProducerError {
        match self {
            Self::Unresolved(_) => {
                crate::native_metal::producer::ProducerError::unresolved(operation)
            }
            Self::Unregistered(_) => {
                crate::native_metal::producer::ProducerError::unregistered(operation)
            }
        }
    }
}

/// Resolves and pins one compute root CBV/SRV/UAV and checks the host
/// registry.
///
/// The guest call carries only the GPU virtual address (the compute root
/// signature declares the extent), so the view covers the rest of the page
/// window the address resolves into, like the graphics path. `registered`
/// answers whether the allocation is visible to the host; a refusal retires the
/// pin the resolution took, so a refused call leaves no backing pinned.
#[cfg(any(windows, test))]
fn resolve_compute_root_view(
    resolver: &dyn crate::native_metal::bindings::GuestRangeResolver,
    slot: u16,
    address: u64,
    registered: impl FnOnce(u64) -> Result<(), String>,
) -> Result<
    crate::native_metal::bindings::Resolved<victory_metal::wire_v3::RootBufferView>,
    ComputeBindingRefusal,
> {
    use crate::native_metal::bindings;
    let resolved = bindings::root_buffer_view_page(resolver, slot, address)
        .map_err(ComputeBindingRefusal::Unresolved)?;
    if let Err(why) = registered(resolved.entry.allocation_id) {
        bindings::retire_all(resolver, &resolved.tokens);
        return Err(ComputeBindingRefusal::Unregistered(why));
    }
    Ok(resolved)
}

/// Resolves and records a compute root CBV/SRV/UAV as a Wire v3 root buffer
/// view, pinning its guest backing until the queue retires the list.
///
/// This is the compute mirror of [`bind_graphics_root_view`]: an unresolved
/// range or an allocation the host has not registered is `E_FAIL` with a named
/// reason, and an empty range is refused by the producer with its opcode.
#[cfg(windows)]
fn bind_compute_root_view(this: This, slot: u32, address: u64) {
    use crate::native_metal::{bindings, queue};
    let Ok(slot) = u16::try_from(slot) else {
        with_metal(this, |p| {
            p.refuse(
                "SetComputeRoot*View: the root parameter slot does not fit the wire",
                Some(victory_metal::wire_v3::OP_SET_ROOT_BUFFER_VIEW),
            )
        });
        return;
    };
    let Some(resolver) = bindings::resolver() else {
        refuse_unresolved(
            this,
            "a compute root view needs the guest range resolver",
            "VICTORY_BACKEND=metal without the xmem share resolver",
        );
        return;
    };
    let resolved = match resolve_compute_root_view(&*resolver, slot, address, |allocation_id| {
        queue::register_allocation_entry(allocation_id)
            .map_err(|error| error.to_string())
    }) {
        Ok(resolved) => resolved,
        Err(refusal) => {
            match &refusal {
                ComputeBindingRefusal::Unresolved(error) => {
                    tracing::warn!(error = %error, "a compute root view could not be resolved");
                }
                ComputeBindingRefusal::Unregistered(error) => {
                    tracing::warn!(
                        error = %error,
                        "a compute root view's allocation is not registered with the host"
                    );
                    crate::native_metal::telemetry::note_registration_failure();
                }
            }
            let operation = match &refusal {
                ComputeBindingRefusal::Unresolved(_) => {
                    "a compute root view could not be resolved"
                }
                ComputeBindingRefusal::Unregistered(_) => {
                    "a compute root view's allocation is not registered with the host"
                }
            };
            with_metal(this, |_| Err(refusal.producer_error(operation)));
            return;
        }
    };
    let (view, tokens) = resolved.into_parts();
    let mut recorded = None;
    with_metal(this, |producer| {
        let result = producer.set_compute_root_buffer_view(view);
        recorded = Some(result.is_ok());
        result
    });
    match recorded {
        Some(true) => retain_on_list(this, &tokens),
        _ => bindings::retire_all(&*resolver, &tokens),
    }
}

/// Resolves and records a graphics root CBV/SRV/UAV as a Wire v3 root buffer
/// view, pinning its guest backing until the queue retires the list.
///
/// The guest call carries only the GPU virtual address (the root signature
/// declares the extent), so the view covers the rest of the page window the
/// address resolves into. The Wire root view's offset is the exact
/// in-allocation byte offset of the address; the host binds that offset.
#[cfg(windows)]
fn bind_graphics_root_view(this: This, slot: u32, address: u64) {
    use crate::native_metal::bindings;
    let Ok(slot) = u16::try_from(slot) else {
        refuse_binding(
            this,
            "a root view slot does not fit the wire",
            "slot exceeds 65535",
        );
        return;
    };
    let Some(resolver) = bindings::resolver() else {
        refuse_binding(
            this,
            "a root view needs the guest range resolver",
            "VICTORY_BACKEND=metal without the xmem share resolver",
        );
        return;
    };
    let resolved = match bindings::root_buffer_view_page(&*resolver, slot, address) {
        Ok(resolved) => resolved,
        Err(error) => {
            refuse_binding(this, "a root view could not be resolved", error);
            return;
        }
    };
    let (view, tokens) = resolved.into_parts();
    if let Err(error) = crate::native_metal::queue::register_allocation_entry(view.allocation_id) {
        bindings::retire_all(&*resolver, &tokens);
        refuse_unregistered(
            this,
            "a root view's allocation is not registered with the host",
            error,
        );
        return;
    }
    let mut recorded = None;
    with_metal(this, |producer| {
        let result = producer.set_graphics_root_buffer_view(view);
        recorded = Some(result.is_ok());
        result
    });
    match recorded {
        Some(true) => retain_on_list(this, &tokens),
        _ => bindings::retire_all(&*resolver, &tokens),
    }
}

/// Resolves and records a title list's vertex buffers, pinning their backing.
///
/// Each view's GPU virtual address goes through the `GuestRangeResolver` seam,
/// so the wire carries the share allocation id, generation and exact offset
/// instead of the address. The host validates every binding against its kind 2
/// registry at execution, so an allocation that is not registered is refused
/// here (and its pins released) rather than producing a submission the host
/// must reject: no silent skip and no D3DMetal fallback.
#[cfg(windows)]
fn bind_vertex_buffers(this: This, start: u32, views: &[MetalVertexView]) {
    use crate::native_metal::{bindings, queue};
    let Some(resolver) = bindings::resolver() else {
        refuse_binding(
            this,
            "a vertex buffer needs the guest range resolver",
            "VICTORY_BACKEND=metal without the xmem share resolver",
        );
        return;
    };
    let mut entries = Vec::with_capacity(views.len());
    let mut tokens = Vec::new();
    for view in views {
        let resolved = match bindings::vertex_buffer(
            &*resolver,
            view.address,
            u64::from(view.size),
            view.stride,
        ) {
            Ok(resolved) => resolved,
            Err(error) => {
                bindings::retire_all(&*resolver, &tokens);
                refuse_binding(this, "a vertex buffer could not be resolved", error);
                return;
            }
        };
        let (entry, mut held) = resolved.into_parts();
        if let Err(error) = queue::register_allocation_entry(entry.allocation_id) {
            bindings::retire_all(&*resolver, &held);
            bindings::retire_all(&*resolver, &tokens);
            refuse_unregistered(
                this,
                "a vertex buffer's allocation is not registered with the host",
                error,
            );
            return;
        }
        entries.push(entry);
        tokens.append(&mut held);
    }
    let mut recorded = None;
    with_metal(this, |producer| {
        let result = producer.set_vertex_buffers_bound(start, &entries);
        recorded = Some(result.is_ok());
        result
    });
    match recorded {
        Some(true) => retain_on_list(this, &tokens),
        _ => bindings::retire_all(&*resolver, &tokens),
    }
}

/// Resolves and records a title list's index buffer (`IASetIndexBuffer`).
#[cfg(windows)]
fn bind_index_buffer(this: This, address: u64, len: u64, format: u32) {
    use crate::native_metal::{bindings, producer, queue};
    let index_format = match producer::index_format(format) {
        Ok(index_format) => index_format,
        Err(error) => {
            with_metal(this, |_| Err(error));
            return;
        }
    };
    let Some(resolver) = bindings::resolver() else {
        refuse_binding(
            this,
            "an index buffer needs the guest range resolver",
            "VICTORY_BACKEND=metal without the xmem share resolver",
        );
        return;
    };
    let resolved = match bindings::index_buffer(&*resolver, address, len, index_format) {
        Ok(resolved) => resolved,
        Err(error) => {
            refuse_binding(this, "an index buffer could not be resolved", error);
            return;
        }
    };
    let (entry, tokens) = resolved.into_parts();
    if let Err(error) = queue::register_allocation_entry(entry.allocation_id) {
        bindings::retire_all(&*resolver, &tokens);
        refuse_unregistered(
            this,
            "an index buffer's allocation is not registered with the host",
            error,
        );
        return;
    }
    let mut recorded = None;
    with_metal(this, |producer| {
        let result = producer.set_index_buffer_bound(entry);
        recorded = Some(result.is_ok());
        result
    });
    match recorded {
        Some(true) => retain_on_list(this, &tokens),
        _ => bindings::retire_all(&*resolver, &tokens),
    }
}

/// One descriptor slot as the guest's descriptor memory describes it.
///
/// The slot record (`[tag, index, generation]`) is read at `handle`; a slot
/// that holds no live view is [`DescriptorSlot::Empty`], while a live view this
/// backend cannot put in a table entry is [`DescriptorSlot::Unresolvable`] so
/// the enumeration can stop there instead of mis-reading the next slot.
///
/// The entry shape follows the resource the descriptor names, never the root
/// parameter alone: a buffer SRV/UAV becomes a buffer range, a texture SRV/UAV
/// names the texture's registered handle (registering it as a sampled texture
/// when the session has not seen it), and a sampler names the sampler handle
/// the guest content registry derived.
#[cfg(windows)]
fn descriptor_slot(handle: u64) -> crate::native_metal::bindings::DescriptorSlot {
    use crate::descriptor::View;
    use crate::native_metal::bindings::{DescriptorSlot, GuestBinding};
    use victory_metal::wire_v3::DescriptorKind;

    let Some(view) = crate::descriptor::view(handle) else {
        return DescriptorSlot::Empty;
    };
    match &*view {
        View::Constant(address, size) => {
            if *address == 0 || *size == 0 {
                return DescriptorSlot::Unresolvable("a constant buffer descriptor is null");
            }
            DescriptorSlot::Live(
                DescriptorKind::Cbv,
                GuestBinding {
                    address: *address,
                    len: u64::from(*size),
                },
            )
        }
        View::Shader(resource, _) => match buffer_entry(*resource, DescriptorKind::Srv) {
            Ok((kind, binding)) => DescriptorSlot::Live(kind, binding),
            Err(_) => match texture_entry(*resource, 0) {
                Ok(handle) => DescriptorSlot::Texture(DescriptorKind::Srv, handle),
                Err(why) => DescriptorSlot::Unresolvable(why),
            },
        },
        View::Unordered(resource, _, _) => match buffer_entry(*resource, DescriptorKind::Uav) {
            Ok((kind, binding)) => DescriptorSlot::Live(kind, binding),
            Err(_) => match texture_entry(*resource, 3) {
                Ok(handle) => DescriptorSlot::Texture(DescriptorKind::Uav, handle),
                Err(why) => DescriptorSlot::Unresolvable(why),
            },
        },
        View::Sampler(words) => match sampler_entry(words) {
            Ok(handle) => DescriptorSlot::Sampler(handle),
            Err(why) => DescriptorSlot::Unresolvable(why),
        },
        View::ConsoleSampler(_) => DescriptorSlot::Unresolvable(
            "a console sampler descriptor has no D3D12-to-Metal mapping",
        ),
        View::Placed(unordered, resource, desc) => {
            placed_slot(*unordered, resource, desc, register_texture_handle)
        }
    }
}

/// ADR-040's placed-view census: two fixed resolution counters, never one per
/// view. A refusal goes through the existing named-refusal census
/// ([`crate::native_metal::telemetry::note_named_refusal`]) under its own
/// static string, so a boot can compare the two totals.
#[cfg(any(windows, test))]
mod placed_census {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// A placed buffer view resolved to a guest buffer range.
    pub(super) const BUFFER_RESOLVED: usize = 0;
    /// A placed texture view resolved to a registered host handle.
    pub(super) const TEXTURE_RESOLVED: usize = 1;

    static COUNTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

    /// Counts one placed resolution and logs the whole census at the first
    /// one and at each power of two after it: bounded log volume, and enough
    /// for a boot to show the refused tables became resolutions.
    pub(super) fn note(outcome: usize) {
        let count = COUNTS[outcome].fetch_add(1, Relaxed) + 1;
        if count.is_power_of_two() {
            let [buffers, textures] = std::array::from_fn(|i| COUNTS[i].load(Relaxed));
            tracing::info!(
                placed_buffer_resolved = buffers,
                placed_texture_resolved = textures,
                "placed view census"
            );
        }
    }

    #[cfg(test)]
    pub(super) fn counts() -> [u64; 2] {
        std::array::from_fn(|i| COUNTS[i].load(Relaxed))
    }

    #[cfg(test)]
    pub(super) fn reset() {
        for count in &COUNTS {
            count.store(0, Relaxed);
        }
    }
}

/// The resource-description dimension of a buffer
/// (`D3D12_RESOURCE_DIMENSION_BUFFER`).
#[cfg(any(windows, test))]
const PLACED_BUFFER: u32 = 1;

/// A placed view's named refusal: counted under its own string in the
/// existing refusal census, never returned as a generic one.
#[cfg(any(windows, test))]
fn refuse_placed(why: &'static str) -> crate::native_metal::bindings::DescriptorSlot {
    crate::native_metal::telemetry::note_named_refusal(why);
    crate::native_metal::bindings::DescriptorSlot::Unresolvable(why)
}

/// Resolves one placed view (ADR-040) to a guest buffer range or a registered
/// host texture handle.
///
/// A placed view names no resource object, so its bytes come from the words
/// the title passed to device 0x230/0x238:
///
/// * the resource description (words 0-13 of `resource`) is the same 56-byte
///   layout a title resource uses: word 0 the dimension (1 buffer, 2-4
///   textures), words 4-5 the buffer's byte length (its whole footprint);
/// * the view description's memory address (a guest GPU virtual address) is
///   words 10-11 of `desc`;
/// * a buffer view's union is words 4-9 (the console's 40-byte view
///   description puts the shader-4-component mapping at word 2 and a
///   console-only dword at 3): first element words 4-5, element count word 6,
///   structure stride word 7, flags word 8.
///
/// The resource description's dimension decides buffer against texture, never
/// the root parameter. A buffer resolves to `(address + first * element,
/// count * element)`, bounded by the resource footprint; a view with no
/// element count (or a format the table does not know) reads the whole
/// buffer. A texture resolves to its guest memory address, registered with the
/// host on first use exactly as a resource-backed texture is. Every failure is
/// a named refusal: a zero address, a zero or oversized length, a range past
/// the footprint, or a mismatched view dimension is never guessed.
#[cfg(any(windows, test))]
fn placed_slot(
    unordered: bool,
    resource: &[u32; 14],
    desc: &[u32; 16],
    register_texture: impl FnOnce(
        &crate::resource::TextureTableRegistration,
    ) -> Result<u64, &'static str>,
) -> crate::native_metal::bindings::DescriptorSlot {
    use crate::native_metal::bindings::{DescriptorSlot, GuestBinding};
    use victory_metal::wire_v3::DescriptorKind;

    let dimension = resource[0];
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let footprint = u64::from(resource[4]) | u64::from(resource[5]) << 32;
    let view_dimension = desc[1];
    let first = u64::from(desc[4]) | u64::from(desc[5]) << 32;
    let count = u64::from(desc[6]);
    let stride = desc[7];

    if dimension == PLACED_BUFFER {
        if address == 0 {
            return refuse_placed("a placed buffer view names no device memory");
        }
        let kind = if unordered {
            match view_dimension {
                // An unordered view has no constant-buffer form.
                0 => return refuse_placed("a placed view's view dimension is unknown"),
                1 => DescriptorKind::Uav,
                2..=11 => {
                    return refuse_placed("a placed buffer view is described as a texture");
                }
                _ => return refuse_placed("a placed view's view dimension is unknown"),
            }
        } else {
            match view_dimension {
                // A constant-buffer description carries no view dimension at
                // all (word 1 is zero): it names the buffer's whole footprint.
                0 => {
                    if footprint == 0 {
                        return refuse_placed("a placed buffer view has no bytes");
                    }
                    if footprint > victory_metal::wire::MAX_RESOURCE_LEN {
                        return refuse_placed("a placed buffer view is larger than a wire binding");
                    }
                    placed_census::note(placed_census::BUFFER_RESOLVED);
                    return DescriptorSlot::Live(
                        DescriptorKind::Cbv,
                        GuestBinding {
                            address,
                            len: footprint,
                        },
                    );
                }
                1 => DescriptorKind::Srv,
                2..=11 => {
                    return refuse_placed("a placed buffer view is described as a texture");
                }
                _ => return refuse_placed("a placed view's view dimension is unknown"),
            }
        };
        return placed_buffer_slot(kind, address, footprint, first, count, stride, desc[0]);
    }

    if (2..=4).contains(&dimension) {
        if address == 0 {
            return refuse_placed("a placed texture view names no device memory");
        }
        if view_dimension == 1 {
            return refuse_placed("a placed texture view is described as a buffer");
        }
        if !(2..=11).contains(&view_dimension) {
            return refuse_placed("a placed view's view dimension is unknown");
        }
        let kind = if unordered {
            DescriptorKind::Uav
        } else {
            DescriptorKind::Srv
        };
        let usage = if unordered { 3 } else { 0 };
        return match crate::resource::placed_texture_registration(address, resource, usage) {
            Ok(registration) => match register_texture(&registration) {
                Ok(handle) => {
                    placed_census::note(placed_census::TEXTURE_RESOLVED);
                    DescriptorSlot::Texture(kind, handle)
                }
                Err(why) => refuse_placed(why),
            },
            Err(why) => refuse_placed(why),
        };
    }

    refuse_placed("a placed view's resource dimension is unknown")
}

/// A placed buffer view's range inside its resource: `(address + first *
/// element, count * element)`, bounded by the resource footprint.
///
/// `first`, `count` and `stride` are the console view description's buffer
/// union (words 4-7), `format` its word 0. A view with no element count, or a
/// format this build does not know, reads the whole buffer (footprint) exactly
/// as the host path does. Every out-of-range case is a named refusal.
#[cfg(any(windows, test))]
fn placed_buffer_slot(
    kind: victory_metal::wire_v3::DescriptorKind,
    address: u64,
    footprint: u64,
    first: u64,
    count: u64,
    stride: u32,
    format: u32,
) -> crate::native_metal::bindings::DescriptorSlot {
    use crate::native_metal::bindings::{DescriptorSlot, GuestBinding};

    let element = if stride > 0 {
        Some(u64::from(stride))
    } else {
        crate::resource::element(format).map(|(bytes, _)| bytes)
    };
    let range = element
        .filter(|_| count > 0)
        .and_then(|element| {
            Some((
                first.checked_mul(element)?,
                count.checked_mul(element)?,
            ))
        });
    // A view with no element count (or a format the table does not know)
    // reads the whole buffer.
    let (offset, len) = range.unwrap_or((0, footprint));
    if len == 0 {
        return refuse_placed("a placed buffer view has no bytes");
    }
    if len > victory_metal::wire::MAX_RESOURCE_LEN {
        return refuse_placed("a placed buffer view is larger than a wire binding");
    }
    let Some(end) = offset.checked_add(len) else {
        return refuse_placed("a placed buffer view's range overflows its address");
    };
    if end > footprint {
        return refuse_placed(
            "a placed buffer view's range is not within its resource footprint",
        );
    }
    let Some(address) = address.checked_add(offset) else {
        return refuse_placed("a placed buffer view's range overflows its address");
    };
    placed_census::note(placed_census::BUFFER_RESOLVED);
    DescriptorSlot::Live(kind, GuestBinding { address, len })
}

/// A texture resource's host registration handle for a table entry.
///
/// The handle is the resource's guest memory base, the same identity kind 8
/// uses. A texture the session has not registered is registered as a sampled
/// texture here (idempotent), so an SRV table never binds a blank or guessed
/// handle. A session without a live host connection is an explicit error.
#[cfg(windows)]
fn texture_entry(resource: usize, usage: u8) -> Result<u64, &'static str> {
    let this = resource as This;
    let Some(registration) = crate::resource::texture_table_registration(this, usage) else {
        return Err("a descriptor names neither a buffer nor a texture");
    };
    let registration = match registration {
        Ok(registration) => registration,
        Err(why) => {
            tracing::warn!(why, "texture SRV table");
            return Err("a sampled texture has no host registration code");
        }
    };
    register_texture_handle(&registration)
}

/// Registers a texture with the host on first use and returns its handle.
///
/// This is the one registration path for a descriptor table entry: a
/// resource-backed texture's first bind ([`texture_entry`]) and a placed
/// texture view ([`placed_slot`]) both come through it, so neither can bind a
/// handle the session has not registered. Registration is idempotent and
/// bounded by the host's content map; a refusal is an explicit error, never a
/// default texture.
#[cfg(windows)]
fn register_texture_handle(
    registration: &crate::resource::TextureTableRegistration,
) -> Result<u64, &'static str> {
    use crate::native_metal::queue;

    let handle = registration.handle;
    if queue::texture_registered(handle) {
        return Ok(handle);
    }
    match queue::register_texture(
        handle,
        1,
        registration.dimension,
        registration.format,
        registration.usage,
        registration.sample_count,
        registration.width,
        registration.height,
        registration.depth,
        registration.mip_levels,
        registration.array_layers,
    ) {
        Ok(true) => Ok(handle),
        Ok(false) => {
            tracing::warn!(
                handle = format_args!("{handle:#x}"),
                "the host refused a sampled texture registration"
            );
            Err("the host refused a sampled texture registration")
        }
        Err(error) => {
            tracing::warn!(
                handle = format_args!("{handle:#x}"),
                %error,
                "a sampled texture could not be registered"
            );
            Err("a sampled texture could not be registered with the host")
        }
    }
}

/// A D3D12 sampler descriptor's registered handle for a table entry.
#[cfg(windows)]
fn sampler_entry(words: &[u32; 13]) -> Result<u64, &'static str> {
    use crate::native_metal::{queue, sampler::SamplerStateDesc};

    let desc = match SamplerStateDesc::from_d3d12(words) {
        Ok(desc) => desc,
        Err(error) => {
            tracing::warn!(%error, "a D3D12 sampler has no host equivalent");
            return Err("a sampler descriptor has no host equivalent");
        }
    };
    match queue::register_sampler_state(&desc) {
        Ok(handle) => Ok(handle),
        Err(error) => {
            tracing::warn!(%error, "a sampler could not be registered with the host");
            Err("a sampler could not be registered with the host")
        }
    }
}

/// A buffer resource's guest range as a descriptor table entry.
#[cfg(windows)]
fn buffer_entry(
    resource: usize,
    kind: victory_metal::wire_v3::DescriptorKind,
) -> Result<
    (
        victory_metal::wire_v3::DescriptorKind,
        crate::native_metal::bindings::GuestBinding,
    ),
    &'static str,
> {
    let this = resource as This;
    let Some((base, size)) = crate::resource::buffer_bytes(this) else {
        return Err("a texture descriptor cannot be carried by a buffer table");
    };
    if base.is_null() || size == 0 {
        return Err("a buffer descriptor is empty");
    }
    Ok((
        kind,
        crate::native_metal::bindings::GuestBinding {
            address: base as u64,
            len: size,
        },
    ))
}

/// Resolves and records a graphics descriptor table, pinning every referenced
/// buffer's guest backing until the queue retires the list.
///
/// `base_register` is the table's root parameter slot: this backend's local
/// convention (root views and root constants already use the root parameter
/// slot as the Metal buffer index).
///
/// The table's entries are enumerated from the base handle by walking
/// consecutive live slots of the same descriptor kind at the kind's real guest
/// increment (`descriptor.rs`'s 32 bytes for views, 16 for samplers). The run
/// stops at the first empty, unresolvable or different-kind slot; a different
/// kind is the boundary of the next root parameter's table in the same heap, so
/// a genuinely mixed range must be split into separate root parameters (wire
/// v3 carries one kind per table). Each slot's entry shape follows the resource
/// it names: a buffer range, a registered texture handle or a registered
/// sampler handle.
#[cfg(windows)]
fn bind_graphics_descriptor_table(this: This, slot: u32, handle: u64) {
    use crate::native_metal::bindings;
    let Ok(slot) = u16::try_from(slot) else {
        refuse_binding(
            this,
            "a descriptor table slot does not fit the wire",
            "slot exceeds 65535",
        );
        return;
    };
    if handle == 0 {
        refuse_binding(
            this,
            "a descriptor table handle is zero",
            "SetGraphicsRootDescriptorTable(null)",
        );
        return;
    }
    let Some(resolver) = bindings::resolver() else {
        refuse_binding(
            this,
            "a descriptor table needs the guest range resolver",
            "VICTORY_BACKEND=metal without the xmem share resolver",
        );
        return;
    };
    let run = match bindings::enumerate_descriptor_table_typed(
        handle,
        descriptor_slot,
    ) {
        Ok(run) => run,
        Err(error) => {
            refuse_binding(this, "a descriptor table could not be enumerated", error);
            return;
        }
    };
    let resolved = match bindings::descriptor_table_typed(
        &*resolver, handle, slot, slot, run.kind, &run.entries,
    ) {
        Ok(resolved) => resolved,
        Err(error) => {
            refuse_binding(this, "a descriptor table could not be resolved", error);
            return;
        }
    };
    let (table, tokens) = resolved.into_parts();
    if let Err(error) = bindings::require_registered_handles(
        &table,
        crate::native_metal::queue::texture_registered,
        crate::native_metal::queue::sampler_registered,
    ) {
        bindings::retire_all(&*resolver, &tokens);
        refuse_binding(
            this,
            "a descriptor table names a handle this session did not register",
            error,
        );
        return;
    }
    for entry in &table.bindings {
        let Some(binding) = entry.as_buffer() else {
            continue;
        };
        if let Err(error) =
            crate::native_metal::queue::register_allocation_entry(binding.allocation_id)
        {
            bindings::retire_all(&*resolver, &tokens);
            refuse_unregistered(
                this,
                "a descriptor table's allocation is not registered with the host",
                error,
            );
            return;
        }
    }
    let mut recorded = None;
    with_metal(this, |producer| {
        let result = producer.set_graphics_descriptor_table(table);
        recorded = Some(result.is_ok());
        result
    });
    match recorded {
        Some(true) => retain_on_list(this, &tokens),
        _ => bindings::retire_all(&*resolver, &tokens),
    }
}

/// Resolves and records a compute descriptor table, pinning every referenced
/// buffer's guest backing until the queue retires the list.
///
/// The rules are the graphics table's: the root parameter slot is the Metal
/// buffer index, entries are enumerated from the base handle by consecutive
/// same-kind live slots at the kind's guest increment, and the run stops at the
/// first empty, unresolvable or different-kind slot. A different kind is the
/// next root parameter's table boundary (a genuinely mixed range must be split
/// into separate root parameters). Each slot's entry shape follows the resource
/// it names: a buffer range, a registered texture handle or a registered
/// sampler handle. A table whose first entry is not expressible increments the
/// opcode histogram with [`victory_metal::wire_v3::OP_DESCRIPTOR_TABLE`].
#[cfg(windows)]
fn bind_compute_descriptor_table(this: This, slot: u32, handle: u64) {
    use crate::native_metal::{bindings, queue};
    let Ok(slot) = u16::try_from(slot) else {
        with_metal(this, |p| {
            p.refuse(
                "SetComputeRootDescriptorTable: the table slot does not fit the wire",
                Some(victory_metal::wire_v3::OP_DESCRIPTOR_TABLE),
            )
        });
        return;
    };
    if handle == 0 {
        with_metal(this, |p| {
            p.refuse(
                "SetComputeRootDescriptorTable: the table handle is zero",
                Some(victory_metal::wire_v3::OP_DESCRIPTOR_TABLE),
            )
        });
        return;
    }
    let Some(resolver) = bindings::resolver() else {
        refuse_unresolved(
            this,
            "a compute descriptor table needs the guest range resolver",
            "VICTORY_BACKEND=metal without the xmem share resolver",
        );
        return;
    };
    let run = match bindings::enumerate_descriptor_table_typed(handle, descriptor_slot) {
        Ok(run) => run,
        Err(error) => {
            tracing::warn!(%error, "a compute descriptor table could not be enumerated");
            with_metal(this, |p| {
                p.refuse(
                    "SetComputeRootDescriptorTable: the table could not be enumerated",
                    Some(victory_metal::wire_v3::OP_DESCRIPTOR_TABLE),
                )
            });
            return;
        }
    };
    let resolved = match bindings::descriptor_table_typed(
        &*resolver,
        handle,
        slot,
        slot,
        run.kind,
        &run.entries,
    ) {
        Ok(resolved) => resolved,
        Err(error) => {
            refuse_unresolved(this, "a compute descriptor table could not be resolved", error);
            return;
        }
    };
    let (table, tokens) = resolved.into_parts();
    if let Err(error) = bindings::require_registered_handles(
        &table,
        queue::texture_registered,
        queue::sampler_registered,
    ) {
        bindings::retire_all(&*resolver, &tokens);
        refuse_unregistered(
            this,
            "a compute descriptor table names a handle this session did not register",
            error,
        );
        return;
    }
    for entry in &table.bindings {
        let Some(binding) = entry.as_buffer() else {
            continue;
        };
        if let Err(error) = queue::register_allocation_entry(binding.allocation_id) {
            bindings::retire_all(&*resolver, &tokens);
            refuse_unregistered(
                this,
                "a compute descriptor table's allocation is not registered with the host",
                error,
            );
            return;
        }
    }
    let mut recorded = None;
    with_metal(this, |producer| {
        let result = producer.set_compute_descriptor_table(table);
        recorded = Some(result.is_ok());
        result
    });
    match recorded {
        Some(true) => retain_on_list(this, &tokens),
        _ => bindings::retire_all(&*resolver, &tokens),
    }
}

/// The provisional native identity of a title resource: its guest allocation
/// base address, generation 1.
///
/// The IPC bridge owns the allocation registry (`WIRE-V3-HOST-BRIDGE`); until
/// it registers a range, the guest address is the key both sides can agree
/// on, and nonzero.
#[cfg(windows)]
fn metal_resource(this: This) -> Option<crate::native_metal::producer::ResourceId> {
    use crate::native_metal::producer::ResourceId;
    let address = match crate::resource::texture_memory(this) {
        Some((base, _)) => base as u64,
        None => crate::resource::buffer_bytes(this)?.0 as u64,
    };
    ResourceId::from_guest_address(address).ok()
}

/// One texture subresource as the wire names it, from a D3D12 subresource
/// index.
#[cfg(windows)]
fn metal_subresource(
    this: This,
    index: u32,
) -> Result<victory_metal::wire_v3::TextureSubresource, crate::native_metal::producer::ProducerError>
{
    use crate::native_metal::producer::{ProducerError, ResourceId};
    let (base, mip_level, slice, _extent) =
        crate::resource::texture_subresource(this, index).ok_or(
            ProducerError::unsupported_opcode(
                "a texture subresource index names no subresource",
                victory_metal::wire_v3::OP_COPY_TEXTURE_REGION,
            ),
        )?;
    let resource = ResourceId::from_guest_address(base)?;
    Ok(victory_metal::wire_v3::TextureSubresource {
        allocation_id: resource.allocation_id,
        generation: resource.generation,
        mip_level,
        slice,
    })
}

/// Whole-resource byte length for a barrier range, when the guest knows it.
#[cfg(windows)]
fn metal_resource_len(this: This) -> Option<u64> {
    if let Some((_, size)) = crate::resource::buffer_bytes(this) {
        return Some(size.max(1));
    }
    let (_, desc) = crate::resource::texture_memory(this)?;
    Some(crate::resource::texture_layout(&desc)?.size.max(1))
}

/// `crate::draw::VertexView` as the title passes it: the same `repr(C)`
/// layout, mirrored here because its fields are private to `draw`.
#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct MetalVertexView {
    address: u64,
    size: u32,
    stride: u32,
}
#[cfg(windows)]
const _: () = assert!(size_of::<MetalVertexView>() == size_of::<crate::draw::VertexView>());

/// `crate::draw::IndexView`, mirrored the same way.
#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct MetalIndexView {
    address: u64,
    size: u32,
    format: u32,
}
#[cfg(windows)]
const _: () = assert!(size_of::<MetalIndexView>() == size_of::<crate::draw::IndexView>());

/// Waits recorded after commands: the title list goes on in a new host
/// list, which runs once the wait holds.
static CUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Lists reset or released while a queue still held some of their host
/// lists: D3D12 lets a list be reset as soon as it is submitted (Grand
/// Theft Auto VI does, 0-21 times a world frame).
static EARLY_RESETS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Cuts and early resets since the last call (the frame line).
pub(crate) fn take_cuts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (CUTS.swap(0, Relaxed), EARLY_RESETS.swap(0, Relaxed))
}

/// `Wait32BitValueX(address, comparison, reference, mask, flags)` and
/// `Wait64BitValueX(address, comparison, reference, flags)`: the GPU waits
/// until the value compares true. Recorded at the start of the current host
/// list, or, after commands, at the start of a new one the rest of the
/// title list records into (the queue runs each once its waits hold,
/// `crate::hold`); bottom-of-pipe writes recorded before it land when the
/// part before it has run.
extern "system" fn wait32(this: This, address: u64, comparison: u32, reference: u32, mask: u32, _flags: u32) {
    wait(this, crate::hold::ValueWait { address, bytes: 4, comparison, reference: u64::from(reference), mask: u64::from(mask) });
}
pub(crate) extern "system" fn wait64(this: This, address: u64, comparison: u32, reference: u64, _flags: u32) {
    wait(this, crate::hold::ValueWait { address, bytes: 8, comparison, reference, mask: u64::MAX });
}

fn wait(this: This, wait: crate::hold::ValueWait) {
    #[cfg(windows)]
    {
        let Some(current) = host_list(this) else { return };
        if !current.is_empty() {
            let Some(next) = next_segment(this) else { return };
            CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            current.close();
            // SAFETY: the title passes one of our lists.
            let list = unsafe { &*this.cast::<List>() };
            list.state.lock().unwrap_or_else(|e| e.into_inner()).cut(next);
            next.defer_wait(wait);
            return;
        }
        current.defer_wait(wait);
    }
    #[cfg(not(windows))]
    let _ = (this, wait);
}

/// Moves a title list's recording on to its next host list (made, or one
/// it held since an earlier recording, reset then): that host list.
#[cfg(windows)]
fn next_segment<'a>(this: This) -> Option<&'a crate::host::HostList> {
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed) + 1;
    let mut hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    if hosts.len() <= k {
        let made = crate::host::host()?.new_list().ok()?;
        hosts.push(std::sync::Arc::new(made));
    }
    list.segment.store(k, std::sync::atomic::Ordering::Relaxed);
    let next: *const crate::host::HostList = std::sync::Arc::as_ptr(&hosts[k]);
    drop(hosts);
    // SAFETY: as in `host_list`.
    Some(unsafe { &*next })
}

/// After the title list's last submission: its host lists start again,
/// and a host list a queue still holds (`crate::hold`) is left to the
/// queue and replaced. The last one left so, which should keep what the
/// title list owned.
#[cfg(windows)]
fn restart(this: This) -> Option<std::sync::Arc<crate::host::HostList>> {
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let used = list.segment.swap(0, std::sync::atomic::Ordering::Relaxed);
    let mut hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    let mut left = None;
    for (k, h) in hosts.iter_mut().enumerate() {
        if std::sync::Arc::strong_count(h) > 1 {
            EARLY_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(segment = k, "a list reset while a queue still holds part of it: that part keeps its own host list");
            let spare = SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop();
            match spare.map(Ok).or_else(|| crate::host::host().map(|host| host.new_list())) {
                Some(Ok(fresh)) => left = Some(std::mem::replace(h, std::sync::Arc::new(fresh))),
                _ => continue,
            }
        } else if k <= used {
            h.reset();
        }
    }
    left
}

/// Host lists [`restart`] left to a queue, back once they ran: the next
/// ones it hands out (a host list and its allocators are not free to make;
/// Grand Theft Auto VI resets up to 21 lists a frame while they are held).
#[cfg(windows)]
static SPARE: std::sync::Mutex<Vec<crate::host::HostList>> = std::sync::Mutex::new(Vec::new());

/// A host list a queue has run: one no title list holds any more is reset
/// and kept for [`restart`].
#[cfg(windows)]
pub(crate) fn recycle(list: std::sync::Arc<crate::host::HostList>) {
    const KEPT: usize = 64;
    if let Ok(list) = std::sync::Arc::try_unwrap(list) {
        list.reset();
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if spare.len() < KEPT {
            spare.push(list);
        }
    }
}

/// What a title list owned that the host list a queue still holds, if any,
/// may read: its upload memory (list 0x4a0) and its draw state's dynamic
/// descriptor slots and title-memory page table. Freed once that host list
/// (the list's last held one, so the last to run) is gone.
#[cfg(windows)]
fn leave(to: Option<std::sync::Arc<crate::host::HostList>>, uploads: Vec<Upload>, state: crate::draw::State) {
    struct Kept(#[allow(dead_code)] Vec<Upload>, #[allow(dead_code)] crate::draw::State);
    // SAFETY: only dropped, with the host list, after it has run.
    unsafe impl Send for Kept {}
    if let Some(held) = to {
        let kept = Kept(uploads, state);
        held.defer_after(move || {
            let _ = &kept;
        });
    }
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, pipeline, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, pipeline, iid, out)
}

fn new_list(
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    #[cfg(windows)]
    let metal_selected =
        crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal;
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        hosts: std::sync::Mutex::new(if metal_selected {
            // No host D3D12 command list exists on the native path.
            Vec::new()
        } else {
            crate::host::host()
                .and_then(|h| match h.new_list() {
                    Ok(list) => Some(std::sync::Arc::new(list)),
                    Err(e) => {
                        tracing::warn!(error = %e, "no host command list");
                        None
                    }
                })
                .into_iter()
                .collect()
        }),
        #[cfg(windows)]
        segment: Default::default(),
        #[cfg(windows)]
        state: {
            let mut state = crate::draw::State::default();
            state.set_pipeline(pipeline);
            std::sync::Mutex::new(state)
        },
        #[cfg(windows)]
        metal: std::sync::Mutex::new(if metal_selected {
            match crate::native_metal::producer::MetalList::new(metal_list_id()) {
                Ok(metal) => Some(metal),
                Err(error) => {
                    tracing::warn!(%error, "no native Metal producer for this list");
                    None
                }
            }
        } else {
            None
        }),
        #[cfg(windows)]
        metal_retained: Default::default(),
    });
    let hr = object::hand_out(LIST, list, iid, &LIST_IIDS, out);
    if hr == S_OK {
        crate::pipeline::bound(pipeline);
    }
    hr
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    // A queue may still hold some of its host lists; they are its own, and
    // the last of them takes what the list owned.
    #[cfg(windows)]
    if !this.is_null() && object::refs(this) == 1 {
        // SAFETY: the title passes one of our lists.
        let list = unsafe { &*this.cast::<List>() };
        let held = list.hosts.lock().unwrap_or_else(|e| e.into_inner()).iter().rfind(|h| std::sync::Arc::strong_count(h) > 1).cloned();
        if held.is_some() {
            EARLY_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!("a list released while a queue still holds part of it: that part runs on its own");
            let uploads = std::mem::take(&mut *list.uploads.lock().unwrap_or_else(|e| e.into_inner()));
            let state = std::mem::take(&mut *list.state.lock().unwrap_or_else(|e| e.into_inner()));
            leave(held, uploads, state);
        }
    }
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    #[cfg(windows)]
    {
        if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
            // The native path submits nothing on refusal; Close returns the
            // failure HRESULT instead of falling back (ADR-008).
            if !this.is_null() {
                // SAFETY: the title passes one of our lists.
                let list = unsafe { &*this.cast::<List>() };
                let mut metal = list.metal.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(metal) = metal.as_mut()
                    && let Err(error) = metal.close()
                {
                    return error.hresult();
                }
            }
            return S_OK;
        }
    }
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // A queue may still hold what the list recorded (`restart`, `leave`).
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    let uploads = std::mem::take(&mut *uploads.lock().unwrap_or_else(|e| e.into_inner()));
    #[cfg(windows)]
    {
        let held = restart(this);
        // SAFETY: the title passes one of our lists.
        let list = unsafe { &*this.cast::<List>() };
        if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
            let mut metal = list.metal.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(metal) = metal.as_mut()
                && let Err(error) = metal.reset()
            {
                metal.note_error(error);
            }
        }
        let mut state = list.state.lock().unwrap_or_else(|e| e.into_inner());
        let old = std::mem::take(&mut *state);
        state.reset(pipeline);
        drop(state);
        leave(held, uploads, old);
    }
    #[cfg(not(windows))]
    drop(uploads);
    crate::pipeline::bound(pipeline);
    #[cfg(not(windows))]
    let _ = this;
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    let host = |r: This| crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(&texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// A transition barrier between two read-only states (diagnostics: such a
/// barrier on a buffer orders nothing, but still becomes a global UAV
/// barrier). A transition has the subresource and the state before packed
/// in `second` (low, high 32 bits) and the state after in `last`. States
/// are the PC bits (the console's compression hints are higher); common
/// (none) is not counted, as it may stand for a write the title promoted
/// from.
#[cfg(any(all(windows, feature = "diagnostics"), test))]
fn read_only_transition(b: &Barrier) -> bool {
    // Vertex and constant buffer, index buffer, depth read, non-pixel and
    // pixel shader resource, indirect argument, copy source, resolve
    // source, ray tracing acceleration structure read, shading rate source.
    const READ_ONLY: u32 =
        0x1 | 0x2 | 0x20 | 0x40 | 0x80 | 0x200 | 0x800 | 0x2000 | 0x40_0000 | 0x100_0000;
    const PC: u32 = 0x01ff_ffff;
    let read_only = |state: u32| {
        let state = state & PC;
        state != 0 && state & !READ_ONLY == 0
    };
    b.kind == 0 && read_only((b.second >> 32) as u32) && read_only(b.last as u32)
}

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
        if !barriers.is_null() && count > 0 {
            // SAFETY: the title passes `count` barriers.
            let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
            with_metal(this, |producer| {
                let mut entries = Vec::with_capacity(barriers.len());
                for barrier in barriers {
                    // D3D12_RESOURCE_STATE_UNORDERED_ACCESS.
                    const UNORDERED_ACCESS: u32 = 0x8;
                    let (handle, before, after) = match barrier.kind {
                        0 => {
                            // A split barrier's first half records no transition.
                            const BEGIN_ONLY: u32 = 1;
                            if barrier.flags & BEGIN_ONLY != 0 {
                                return producer
                                    .refuse(
                                        "ResourceBarrier: split barrier begin",
                                        Some(victory_metal::wire_v3::OP_RESOURCE_BARRIER),
                                    );
                            }
                            (
                                barrier.first,
                                (barrier.second >> 32) as u32,
                                barrier.last as u32,
                            )
                        }
                        1 => return producer.refuse(
                            "ResourceBarrier: aliasing barrier",
                            Some(victory_metal::wire_v3::OP_RESOURCE_BARRIER),
                        ),
                        2 if barrier.first.is_null() => {
                            return producer.refuse(
                                "ResourceBarrier: global UAV barrier",
                                Some(victory_metal::wire_v3::OP_RESOURCE_BARRIER),
                            );
                        }
                        2 => (barrier.first, UNORDERED_ACCESS, UNORDERED_ACCESS),
                        _ => return producer.refuse(
                            "ResourceBarrier: unknown barrier type",
                            Some(victory_metal::wire_v3::OP_RESOURCE_BARRIER),
                        ),
                    };
                    let resource = metal_resource(handle).ok_or(
                        crate::native_metal::producer::ProducerError::unsupported(
                            "ResourceBarrier: resource without a guest allocation",
                        ),
                    )?;
                    entries.push(crate::native_metal::producer::BarrierEntry {
                        allocation_id: resource.allocation_id,
                        generation: resource.generation,
                        offset: 0,
                        length: metal_resource_len(handle).unwrap_or(1),
                        before_state: before,
                        after_state: after,
                    });
                }
                producer.resource_barrier(&entries)
            });
        }
        return 0;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("barrier kinds {:?}", barriers.iter().map(|b| b.kind).collect::<Vec<_>>()));
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        let on_buffer = |b: &Barrier| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        };
        let buffer_barrier = barriers.iter().any(on_buffer);
        let global_uav = barriers.iter().any(|b| b.kind == 2 && b.first.is_null());
        if buffer_barrier {
            host.push(UAV_BARRIER);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                let (buffers, read_only) = barriers
                    .iter()
                    .filter(|b| on_buffer(b))
                    .fold((0, 0), |(n, r), b| {
                        (n + 1, r + u64::from(read_only_transition(b)))
                    });
                add(C::BufferBarriers, buffers);
                add(C::BufferReadOnly, read_only);
                add(C::BufferUav, 1);
                add(C::BufferUavSkippable, u64::from(buffers == read_only));
            }
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
        if buffer_barrier || global_uav {
            list.sync_widened();
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(&texture, state);
        let barrier = crate::host::transition(&texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Buffer to buffer: both are host memory, copied at submission.
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
        with_metal(this, |producer| {
            use crate::native_metal::queue;

            let (src_address, src_size) = crate::resource::buffer_bytes(src).ok_or(
                crate::native_metal::producer::ProducerError::unsupported(
                    "CopyBufferRegion: source is not a buffer",
                ),
            )?;
            let (dst_address, dst_size) = crate::resource::buffer_bytes(dst).ok_or(
                crate::native_metal::producer::ProducerError::unsupported(
                    "CopyBufferRegion: destination is not a buffer",
                ),
            )?;
            if src_offset.checked_add(len).is_none_or(|end| end > src_size)
                || dst_offset.checked_add(len).is_none_or(|end| end > dst_size)
            {
                return producer.refuse(
                    "CopyBufferRegion: range outside its buffer",
                    Some(victory_metal::wire_v3::OP_COPY_BUFFER),
                );
            }
            // Both ranges are resolved and registered before the copy is
            // recorded: a copy names share allocations, never guest addresses,
            // and a source or destination the host cannot map is refused by
            // name (no silent skip, no D3DMetal fallback).
            let (source, destination) = queue::register_copy_buffers(
                (src_address as u64, src_size),
                (dst_address as u64, dst_size),
            )
            .map_err(|error| {
                registration_refusal(
                    "CopyBufferRegion: a buffer is not registered with the host",
                    error,
                )
            })?;
            let src = crate::native_metal::producer::ResourceId::from_range(&source)?;
            let dst = crate::native_metal::producer::ResourceId::from_range(&destination)?;
            producer.copy_buffer(src, src_offset, dst, dst_offset, len)
        });
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        list.defer_copy(to, from, len as usize);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
        use crate::native_metal::queue;

        with_metal(this, |producer| {
            if let (Some((to, to_size)), Some((from, from_size))) = (
                crate::resource::buffer_bytes(dst),
                crate::resource::buffer_bytes(src),
            ) {
                let length = to_size.min(from_size);
                if length == 0 {
                    return producer.refuse(
                        "CopyResource: empty buffer",
                        Some(victory_metal::wire_v3::OP_COPY_BUFFER),
                    );
                }
                // Register the source and destination before recording; the
                // returned ranges name the share allocations the wire carries.
                let (source, destination) = queue::register_copy_buffers(
                    (from as u64, from_size),
                    (to as u64, to_size),
                )
                .map_err(|error| {
                    registration_refusal(
                        "CopyResource: a buffer is not registered with the host",
                        error,
                    )
                })?;
                let to = crate::native_metal::producer::ResourceId::from_range(&destination)?;
                let from = crate::native_metal::producer::ResourceId::from_range(&source)?;
                return producer.copy_buffer(from, 0, to, 0, length);
            }
            producer.refuse(
                "CopyResource of a texture (subresource lowering not wired)",
                None,
            )
        });
        return;
    }
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if crate::texture::title_written(&from) {
                crate::texture::upload(list, src, &from);
            }
            ensure_state(list, dst, COPY_DEST);
            ensure_state(list, src, COPY_SOURCE);
            crate::texture::host_writes(list, dst);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: an open list and live textures.
            unsafe { list.list.CopyResource(&to, &from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            list.defer_copy(to, from, a.min(b) as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// One `CopyTextureRegion` buffer-footprint upload lowered to the wire's
/// pitched copy: the first source row, its pitch and the destination box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UploadCopy {
    source: crate::native_metal::producer::ResourceId,
    destination: victory_metal::wire_v3::TextureSubresource,
    offset: u64,
    bytes_per_row: u32,
    extent: [u32; 3],
    origin: [u32; 3],
}

/// Lowers a guest buffer footprint (`D3D12_PLACED_SUBRESOURCE_FOOTPRINT`
/// source) into Wire v3's single-slice pitched copy.
///
/// `source_box` is the guest `[left, top, front, right, bottom, back]` box in
/// footprint texels; `None` copies the whole footprint. A box outside the
/// footprint, deeper than the one slice a wire command carries, or whose
/// first row is not 4-byte aligned is refused with a name instead of being
/// approximated.
fn upload_copy(
    source: crate::native_metal::producer::ResourceId,
    destination: victory_metal::wire_v3::TextureSubresource,
    destination_extent: [u32; 3],
    destination_origin: [u32; 3],
    footprint: &crate::resource::PlacedFootprint,
    source_box: Option<[u32; 6]>,
) -> Result<UploadCopy, &'static str> {
    if footprint.row_pitch == 0
        || footprint.width == 0
        || footprint.height == 0
        || footprint.depth == 0
    {
        return Err("CopyTextureRegion: an empty buffer footprint");
    }
    let (at, size) = match source_box {
        Some([left, top, front, right, bottom, back]) => {
            let (Some(width), Some(height), Some(depth)) = (
                right.checked_sub(left),
                bottom.checked_sub(top),
                back.checked_sub(front),
            ) else {
                return Err("CopyTextureRegion: an inverted buffer-upload box");
            };
            ([left, top, front], [width, height, depth])
        }
        None => (
            [0, 0, 0],
            [footprint.width, footprint.height, footprint.depth],
        ),
    };
    if size.contains(&0) {
        return Err("CopyTextureRegion: an empty buffer-upload box");
    }
    if size[2] != 1 {
        return Err("CopyTextureRegion: a buffer upload of more than one depth slice");
    }
    let rows = footprint
        .rows(at, size)
        .ok_or("CopyTextureRegion: the upload box is outside its footprint")?;
    let Some(&(first, _)) = rows.first() else {
        return Err("CopyTextureRegion: the upload box is outside its footprint");
    };
    let offset = footprint
        .offset
        .checked_add(first)
        .ok_or("CopyTextureRegion: the upload offset overflows")?;
    if !offset.is_multiple_of(4) {
        return Err("CopyTextureRegion: the upload offset is not 4-byte aligned");
    }
    for ((start, extent), limit) in destination_origin.into_iter().zip(size).zip(destination_extent)
    {
        if start.checked_add(extent).is_none_or(|end| end > limit) {
            return Err("CopyTextureRegion: the upload box is outside its destination subresource");
        }
    }
    Ok(UploadCopy {
        source,
        destination,
        offset,
        bytes_per_row: footprint.row_pitch,
        extent: size,
        origin: destination_origin,
    })
}

/// Texture to texture on the GPU; buffer to texture through an upload
/// buffer filled at submission; texture to buffer through readback memory
/// ([`read_back`]).
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
        use crate::native_metal::producer::ProducerError;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(&dst), Some(&src)) = (unsafe { dst.as_ref() }, unsafe { src.as_ref() }) else {
            return;
        };
        let copy_refused = |why: &'static str| {
            ProducerError::unsupported_opcode(why, victory_metal::wire_v3::OP_COPY_TEXTURE_REGION)
        };
        with_metal(this, |producer| {
            // A buffer footprint is an upload (buffer -> texture) or a
            // readback (texture -> buffer). The upload has a Wire v3 command;
            // a readback still does not.
            if src.kind == 1 && dst.kind == 0 {
                let (address, size) = crate::resource::buffer_bytes(src.resource).ok_or(
                    copy_refused("CopyTextureRegion: an upload source is not a buffer"),
                )?;
                let to = metal_subresource(dst.resource, dst.footprint.offset as u32)?;
                let to_extent =
                    crate::resource::texture_subresource(dst.resource, dst.footprint.offset as u32)
                        .map(|(_, _, _, extent)| extent)
                        .ok_or(copy_refused("CopyTextureRegion: destination subresource"))?;
                let footprint = src.footprint;
                let needed = footprint
                    .bytes()
                    .ok_or(copy_refused("CopyTextureRegion: an upload footprint's format"))?;
                if footprint.offset.checked_add(needed).is_none_or(|end| end > size) {
                    return producer.refuse(
                        "CopyTextureRegion: the upload footprint is outside its buffer",
                        Some(victory_metal::wire_v3::OP_COPY_BUFFER_TO_TEXTURE),
                    );
                }
                // The staging buffer is a guest buffer like any other: resolve
                // and register its share allocation before the upload names it.
                let source = crate::native_metal::queue::register_guest_buffer(
                    address as u64,
                    size,
                )
                .map_err(|error| {
                    registration_refusal(
                        "CopyTextureRegion: the upload buffer is not registered with the host",
                        error,
                    )
                })?;
                let source = crate::native_metal::producer::ResourceId::from_range(&source)?;
                // SAFETY: the title passes a source box or null.
                let copy = upload_copy(
                    source,
                    to,
                    to_extent,
                    [x, y, z],
                    &footprint,
                    unsafe { source_box.as_ref() }.copied(),
                )
                .map_err(|why| {
                    ProducerError::unsupported_opcode(
                        why,
                        victory_metal::wire_v3::OP_COPY_BUFFER_TO_TEXTURE,
                    )
                })?;
                return producer.copy_buffer_to_texture(
                    copy.source,
                    copy.destination,
                    copy.offset,
                    copy.bytes_per_row,
                    copy.extent,
                    copy.origin,
                );
            }
            // A buffer readback (texture -> buffer) has no wire command yet.
            if dst.kind != 0 || src.kind != 0 {
                return Err(copy_refused(
                    "CopyTextureRegion: a buffer footprint (readback) is not expressible",
                ));
            }
            let to = metal_subresource(dst.resource, dst.footprint.offset as u32)?;
            let from = metal_subresource(src.resource, src.footprint.offset as u32)?;
            let to_extent = crate::resource::texture_subresource(
                dst.resource,
                dst.footprint.offset as u32,
            )
            .map(|(_, _, _, extent)| extent)
            .ok_or(copy_refused("CopyTextureRegion: destination subresource"))?;
            let from_extent = crate::resource::texture_subresource(
                src.resource,
                src.footprint.offset as u32,
            )
            .map(|(_, _, _, extent)| extent)
            .ok_or(copy_refused("CopyTextureRegion: source subresource"))?;
            // A null box is the whole source subresource; a title box is
            // [left, top, front, right, bottom, back].
            let (origin, size) = match unsafe { source_box.as_ref() } {
                Some(&[left, top, front, right, bottom, back]) => (
                    [x, y, z],
                    [
                        right.saturating_sub(left).max(1),
                        bottom.saturating_sub(top).max(1),
                        back.saturating_sub(front).max(1),
                    ],
                ),
                None => {
                    let size = [
                        from_extent[0].saturating_sub(x).max(1),
                        from_extent[1].saturating_sub(y).max(1),
                        from_extent[2].saturating_sub(z).max(1),
                    ];
                    ([x, y, z], size)
                }
            };
            if x.checked_add(size[0]).is_none_or(|end| end > to_extent[0])
                || y.checked_add(size[1]).is_none_or(|end| end > to_extent[1])
                || z.checked_add(size[2]).is_none_or(|end| end > to_extent[2])
            {
                return Err(copy_refused(
                    "CopyTextureRegion: destination box outside its subresource",
                ));
            }
            producer.copy_texture_region(from, to, origin, [x, y, z], size)
        });
        return;
    }
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            if dst.kind == FOOTPRINT && src.kind != FOOTPRINT {
                read_back(list, &dst, [x, y, z], &src, source_box);
            }
            return;
        };
        let mut from = src;
        let _upload; // keeps the source alive while this call records it
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if src.kind == FOOTPRINT {
            let Some(bytes) = src.footprint.bytes() else {
                return;
            };
            let Some(data) = buffer_range(src.resource, src.footprint.offset, bytes) else {
                return;
            };
            let Some((buffer, at, staging)) = list.upload(bytes) else {
                return;
            };
            // The list owns this mapped D3D12 upload buffer. Wine can report
            // its address as MEM_FREE, so use the host-upload path.
            list.defer_fill(staging, data, bytes as usize);
            from.resource = buffer.as_raw().cast_const().cast();
            from.footprint.offset = at;
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live buffer.
                read = Some((unsafe { buffer.GetGPUVirtualAddress() } + at, bytes));
            }
            _upload = buffer;
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            if crate::texture::title_written(&texture) {
                crate::texture::upload(list, src.resource, &texture);
            }
            ensure_state(list, src.resource, COPY_SOURCE);
            from.resource = texture.as_raw().cast_const().cast();
        }
        ensure_state(list, dst.resource, COPY_DEST);
        crate::texture::host_writes(list, dst.resource);
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(src.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

#[cfg(test)]
impl CopyLocation {
    /// A placed footprint in `resource` (tests).
    pub(crate) fn placed(resource: This, footprint: crate::resource::PlacedFootprint) -> Self {
        Self { resource, kind: 1, footprint }
    }
    /// Subresource 0 of `resource` (tests).
    pub(crate) fn whole(resource: This) -> Self {
        Self { resource, kind: 0, footprint: crate::resource::PlacedFootprint::new(0, 0, 0, 0, 0) }
    }
}

/// Texture read backs since the last frame-rate line: how many, bytes
/// (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) static READ_BACKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Buffer footprint to texture copies since the last frame-rate line: how
/// many, bytes, and how many of them were larger than a scratch chunk (a
/// buffer of their own; the rest take the list's scratch) (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) static FOOTPRINT_UPLOADS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// (texture, buffer) pairs read back so far, each logged once (diagnostics).
#[cfg(feature = "diagnostics")]
static READ_BACK_PAIRS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// Texture to buffer (a read back): the region is copied on the host into
/// readback memory laid out as the title's footprint, and once the list has
/// run its rows go into the title's buffer (the title reads them on the CPU
/// after its fence).
#[cfg(windows)]
pub(crate) fn read_back(list: &crate::host::HostList, dst: &CopyLocation, at: [u32; 3], src: &CopyLocation, source_box: *const [u32; 6]) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_BOX, D3D12_RESOURCE_DIMENSION_TEXTURE3D, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
    };
    use windows::core::Interface;
    macro_rules! refused {
        ($why:literal) => {{
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(concat!("texture to buffer copy refused: ", $why)));
            return;
        }};
    }
    let (Some(host), Some(texture)) = (crate::host::host(), crate::resource::host_resource(src.resource)) else {
        refused!("no host texture");
    };
    // SAFETY: the title's box (same layout as the host's) or null; a live texture.
    let (b, desc) = unsafe { (source_box.cast::<D3D12_BOX>().as_ref(), texture.GetDesc()) };
    let size = match b {
        Some(b) => [b.right.saturating_sub(b.left), b.bottom.saturating_sub(b.top), b.back.saturating_sub(b.front)],
        None => {
            // A subresource index shares the footprint's first dword.
            let mip = src.footprint.offset as u32 % u32::from(desc.MipLevels.max(1));
            let depth = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { u32::from(desc.DepthOrArraySize) } else { 1 };
            [((desc.Width >> mip) as u32).max(1), (desc.Height >> mip).max(1), (depth >> mip).max(1)]
        }
    };
    // The host takes footprints of 256-byte pitches only (as the console).
    if !dst.footprint.row_pitch.is_multiple_of(256) {
        refused!("a row pitch not a multiple of 256");
    }
    // shortcut: depth and stencil planes copy out in typeless formats of
    // their own; refused until a title is seen reading one back.
    if desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
        refused!("a depth or stencil texture");
    }
    // The host copies into a footprint only within a buffer that holds all
    // of it, so readback memory spans the footprint, not just the region.
    let (Some(rows), Some(len)) = (dst.footprint.rows(at, size), dst.footprint.bytes()) else {
        refused!("a region outside its footprint");
    };
    let (Some(title), Some((buffer, data))) = (buffer_range(dst.resource, dst.footprint.offset, len), list.readback(host, len)) else {
        return;
    };
    if crate::texture::title_written(&texture) {
        crate::texture::upload(list, src.resource, &texture);
    }
    ensure_state(list, src.resource, COPY_SOURCE);
    let (mut to, mut from) = (*dst, *src);
    to.resource = buffer.as_raw().cast_const().cast();
    to.footprint.offset = 0;
    from.resource = texture.as_raw().cast_const().cast();
    #[cfg(feature = "diagnostics")]
    {
        let text = || {
            let mem = crate::resource::texture_memory(src.resource).map_or(0, |(at, _)| at as usize);
            format!(
                "read back {:?} (mem {mem:#x}, {}x{} f{} sub {}) {size:?} at {at:?} into {:?}+{:#x}, {len} bytes",
                src.resource,
                desc.Width,
                desc.Height,
                desc.Format.0,
                src.footprint.offset as u32,
                dst.resource,
                dst.footprint.offset
            )
        };
        READ_BACKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        READ_BACKS[1].fetch_add(len, std::sync::atomic::Ordering::Relaxed);
        let pair = (src.resource as usize, dst.resource as usize);
        let mut seen = READ_BACK_PAIRS.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(&pair) {
            seen.push(pair);
            tracing::info!("a texture read back (first of this pair): {}", text());
        }
        list.note_op(text);
    }
    // SAFETY: an open list; both locations in the host's layout, naming live
    // resources; the region lies in the footprint (checked).
    unsafe {
        list.list.CopyTextureRegion(
            (&raw const to).cast(),
            at[0],
            at[1],
            at[2],
            (&raw const from).cast(),
            b.map(std::ptr::from_ref),
        )
    };
    let (data, title) = (data as usize, title as usize);
    list.defer_after(move || {
        for &(at, n) in &rows {
            let (at, n) = (at as usize, n as usize);
            let _writing = crate::watch::writing(title + at, n);
            crate::host::write_mapped((title + at) as *mut u8, (data + at) as *const u8, n);
        }
    });
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
        let _ = format;
        with_metal(this, |producer| {
            let to = metal_subresource(dst, dst_sub)?;
            let from = metal_subresource(src, src_sub)?;
            producer.resolve_subresource(from, to)
        });
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        ensure_state(list, dst, RESOLVE_DEST);
        ensure_state(list, src, RESOLVE_SOURCE);
        crate::texture::host_writes(list, dst);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = DXGI_FORMAT(format as i32);
            if (a.Width, a.Height) == (b.Width, b.Height) {
                list.list
                    .ResolveSubresource(&to, dst_sub, &from, src_sub, format);
            } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: a.Width.min(b.Width) as i32,
                    bottom: a.Height.min(b.Height) as i32,
                };
                list1.ResolveSubresourceRegion(
                    &to,
                    dst_sub,
                    0,
                    0,
                    &from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
        // SAFETY: the title passes four floats (null is tolerated).
        let color = unsafe { color.as_ref() }.copied();
        with_metal(this, |producer| {
            let Some(color) = color else {
                return producer.refuse(
                "ClearRenderTargetView: no clear colour",
                Some(victory_metal::wire_v3::OP_BEGIN_RENDER_PASS),
            );
            };
            if count > 0 || !rects.is_null() {
                return producer.refuse(
                "ClearRenderTargetView: partial rectangles",
                Some(victory_metal::wire_v3::OP_BEGIN_RENDER_PASS),
            );
            }
            let record = crate::descriptor::rtv_record(handle).ok_or(
                crate::native_metal::producer::ProducerError::unsupported(
                    "ClearRenderTargetView: no recorded view",
                ),
            )?;
            let resource = metal_resource(record.resource).ok_or(
                crate::native_metal::producer::ProducerError::unsupported(
                    "ClearRenderTargetView: no guest allocation",
                ),
            )?;
            producer.clear_render_target(resource, color)
        });
        return;
    }
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(&texture, record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.list.ResourceBarrier(&[crate::host::transition(
                    &texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
        with_metal(this, |producer| {
            if count > 0 || !rects.is_null() {
                return producer.refuse(
                "ClearDepthStencilView: partial rectangles",
                Some(victory_metal::wire_v3::OP_BEGIN_RENDER_PASS),
            );
            }
            let record = crate::descriptor::dsv_record(handle).ok_or(
                crate::native_metal::producer::ProducerError::unsupported(
                    "ClearDepthStencilView: no recorded view",
                ),
            )?;
            let resource = metal_resource(record.resource).ok_or(
                crate::native_metal::producer::ProducerError::unsupported(
                    "ClearDepthStencilView: no guest allocation",
                ),
            )?;
            producer.clear_depth_stencil(resource, flags & 1 != 0, depth, flags & 2 != 0, stencil)
        });
        return;
    }
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        ensure_state(list, record.resource, DEPTH_WRITE);
        let dsv = host.dsv(&texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list.ClearDepthStencilView(
                dsv,
                D3D12_CLEAR_FLAGS(flags as i32 & 3),
                depth,
                stencil,
                rects,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (`copy_mapped`:
/// an unmapped source page reads as zeros). Overlapping ranges move with
/// `memmove` when both ranges are committed.
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
    {
        if src.abs_diff(dst) >= len {
            list.defer_copy(dst as *mut u8, src as *const u8, len as usize);
        } else {
            list.defer_move(dst as *mut u8, src as *const u8, len as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags, which the
/// host has no use for (the ray-tracing builder dispatches this way).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, _flags: u32) {
    #[cfg(windows)]
    state::dispatch(this, x, y, z);
    #[cfg(not(windows))]
    {
        let _ = (this, x, y, z);
        count_dispatch();
    }
}

/// `BeginQuery`/`EndQuery(heap, type, index)`: queries are answered when
/// resolved.
extern "system" fn query_mark(_: This, _heap: This, _kind: u32, _index: u32) {}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// each query's result into the buffer once the list has run. Timestamps
/// (type 2) are the GPU clock then; occlusion counts (0) say a million
/// samples passed and binary occlusion (1) visible, so nothing is culled
/// by a query.
/// shortcut: no query runs on the host GPU; results are plausible values,
/// not measured ones (time spans come out as the submission's length).
extern "system" fn resolve_queries(
    this: This,
    _heap: This,
    kind: u32,
    _first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    let Some(end) = offset.checked_add(u64::from(count) * bytes) else {
        return;
    };
    if end > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
        set_native_pipeline(this, pipeline);
        return;
    }
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// `SetPipelineState` on the native path: the list names the host's own
/// pipeline id, never the title's pointer.
///
/// Every PSO family is registered on first use (`pipeline::native_pipeline`):
/// a compute PSO uses the original kind 4 v1 frame, and a graphics or mesh PSO
/// uses the version 2 frame with one library per stage, the encoded
/// `PipelineState`, the ABI and the root-signature bytes. The id the host
/// answers with (kind 129) is what the producer records for
/// `SetComputePipeline`, `SetPipeline` or `SetMeshPipeline`.
///
/// A refusal is an explicit HRESULT naming the stage. The underlying reason
/// (`vertex`, `pixel`, `mesh` or an unsupported stage) is in the log; nothing
/// is silently skipped and nothing falls back to D3DMetal.
#[cfg(windows)]
fn set_native_pipeline(list: This, pipeline: This) {
    use victory_metal::wire_v3::{
        OP_SET_COMPUTE_PIPELINE, OP_SET_GRAPHICS_PIPELINE, OP_SET_MESH_PIPELINE,
    };

    if pipeline.is_null() {
        with_metal(list, |producer| {
            producer.refuse("SetPipelineState(null)", Some(OP_SET_GRAPHICS_PIPELINE))
        });
        return;
    }
    let stages = crate::pipeline::stage_names(pipeline);
    if stages.iter().all(|stage| *stage == "compute") {
        match crate::pipeline::native_pipeline(pipeline) {
            Ok(id) => {
                with_metal(list, |producer| producer.set_compute_pipeline(id));
            }
            Err(error) => {
                with_metal(list, |producer| {
                    producer.refuse(
                        "SetPipelineState: the compute PSO could not be registered with the host",
                        Some(OP_SET_COMPUTE_PIPELINE),
                    )
                });
                tracing::warn!(error, "native compute pipeline registration refused");
            }
        }
        return;
    }
    let (stage, opcode) = if stages.iter().any(|stage| *stage == "mesh") {
        ("mesh", OP_SET_MESH_PIPELINE)
    } else {
        ("graphics", OP_SET_GRAPHICS_PIPELINE)
    };
    match crate::pipeline::native_pipeline(pipeline) {
        Ok(id) => {
            tracing::info!(?pipeline, stage, id, "native pipeline registered");
            with_metal(list, |producer| {
                if stage == "mesh" {
                    producer.set_mesh_pipeline(id)
                } else {
                    producer.set_pipeline(id)
                }
            });
        }
        Err(error) => {
            tracing::warn!(
                ?pipeline,
                stage,
                %error,
                "VICTORY_BACKEND=metal: the {stage} PSO could not be registered with the host"
            );
            crate::native_metal::telemetry::note_registration_failure();
            // The refusal names the failing stage, not just the family: the
            // HRESULT the title sees must say which shader blocked it.
            let operation = if error.contains("Vertex") {
                "SetPipelineState: the vertex stage could not be registered with the host"
            } else if error.contains("Pixel") {
                "SetPipelineState: the pixel stage could not be registered with the host"
            } else if error.contains("Mesh") {
                "SetPipelineState: the mesh stage could not be registered with the host"
            } else if stage == "mesh" {
                "SetPipelineState: the mesh PSO could not be registered with the host"
            } else {
                "SetPipelineState: the graphics PSO could not be registered with the host"
            };
            with_metal(list, |producer| producer.refuse(operation, Some(opcode)));
        }
    }
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. Host lists order
/// their own work and every submission is waited for, so there is nothing
/// to do.
extern "system" fn sync(_: This, _flags: u32, _control: u32, _address: u64, _size: u64) {}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. Grand Theft Auto VI's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size)
        .ok()
        .and_then(|s| s.checked_add(align))
    else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let Some(block) = Upload::new(len) else {
        return std::ptr::null_mut();
    };
    let start = block.start();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    at
}

/// Memory [`allocate_upload`] hands out, zeroed, freed when the list is
/// reset. On Windows it lies in [4 GiB, 2^40), where translated shaders
/// find title memory by address (`crate::memory`; the console's GPU
/// addresses have 40 bits), and is a buffer of its own to the host copies
/// (`crate::shadow`), so its copy is exactly it.
struct Upload {
    #[cfg(windows)]
    at: usize,
    #[cfg(windows)]
    len: usize,
    #[cfg(not(windows))]
    block: Vec<u8>,
}

impl Upload {
    #[cfg(windows)]
    fn new(len: usize) -> Option<Upload> {
        use windows_sys::Win32::System::Memory::{
            MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE,
            MemExtendedParameterAddressRequirements, PAGE_READWRITE, VirtualAlloc2,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 32) as *mut _,
            HighestEndingAddress: ((1usize << 40) - 1) as *mut _,
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // Watched like the title's pools: copies take only written pages.
        // SAFETY: a fresh allocation; the parameter points at `requirements`,
        // which outlives the call.
        let at = unsafe {
            VirtualAlloc2(
                std::ptr::null_mut(),
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                &mut parameter,
                1,
            )
        } as usize;
        if at == 0 {
            return None;
        }
        crate::shadow::register(at, at as u64, len as u64);
        UPLOADS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        UPLOADS[1].fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
        Some(Upload { at, len })
    }

    #[cfg(not(windows))]
    fn new(len: usize) -> Option<Upload> {
        Some(Upload {
            block: vec![0; len],
        })
    }

    fn start(&self) -> *mut u8 {
        #[cfg(windows)]
        return self.at as *mut u8;
        #[cfg(not(windows))]
        return self.block.as_ptr().cast_mut();
    }
}

#[cfg(windows)]
impl Drop for Upload {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        crate::shadow::unregister(self.at, self.at as u64);
        UPLOADS[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        UPLOADS[1].fetch_sub(self.len as u64, std::sync::atomic::Ordering::Relaxed);
        // Not guarded any more: memory allocated here next is new.
        crate::watch::title_memory_remapped(self.at, self.len);
        // SAFETY: our allocation; lists that still use its copy hold that.
        unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
    }
}

// SAFETY: plain memory, handed out once.
#[cfg(windows)]
unsafe impl Send for Upload {}

/// Live list upload blocks and their bytes (the statistics).
#[cfg(windows)]
pub(crate) static UPLOADS: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Draw state the host has no use for (depth bounds, depth order and shading
/// rate hints): any arguments, nothing returned.
///
/// Every table slot that used to point here is now wired to a named handler
/// or refused by name; the stub remains only as the sentinel the table audit
/// ([`tests::no_list_slot_is_a_silent_state_drop`]) asserts against, so a
/// silent drop added later fails the suite.
#[allow(dead_code)]
extern "system" fn ignored_state(_: This) {}

extern "system" fn depth_bounds(this: This, min: f32, max: f32) {
    #[cfg(windows)]
    with_state(this, |list, s| s.set_depth_bounds(list, min, max));
    #[cfg(not(windows))]
    let _ = (this, min, max);
}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, metal_resource, with_metal, with_state};
    use crate::backend::GpuBackend;
    use crate::draw::{IndexView, VertexView};
    use crate::native_metal::producer::{
        DepthTarget, ProducerError, RenderTarget, RootSignatureStage,
    };
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    fn metal() -> bool {
        GpuBackend::from_env() == GpuBackend::Metal
    }

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        if metal() {
            with_metal(this, |p| p.set_topology_value(topology));
            return;
        }
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        if metal() {
            let views = items(viewports, count);
            with_metal(this, |p| match views {
                [view] => p.set_viewport(
                    view.TopLeftX,
                    view.TopLeftY,
                    view.Width,
                    view.Height,
                    view.MinDepth,
                    view.MaxDepth,
                ),
                _ => p.refuse(
                    "RSSetViewports: more than one viewport",
                    Some(victory_metal::wire_v3::OP_SET_VIEWPORT),
                ),
            });
            return;
        }
        with_state(this, |list, s| s.viewports(list, items(viewports, count)));
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        if metal() {
            let rects = items(rects, count);
            with_metal(this, |p| match rects {
                [rect] => p.set_scissor_edges(rect.left, rect.top, rect.right, rect.bottom),
                _ => p.refuse(
                    "RSSetScissorRects: more than one rectangle",
                    Some(victory_metal::wire_v3::OP_SET_SCISSOR),
                ),
            });
            return;
        }
        with_state(this, |list, s| s.scissors(list, items(rects, count)));
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        // SAFETY: the title passes four floats or null.
        with_state(this, |list, s| {
            s.blend_factor(list, unsafe { factor.as_ref() })
        });
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        with_state(this, |list, s| s.stencil_ref(list, reference));
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        let slots = items(heaps, count);
        // The native backend has no host D3D12 list, so `with_state` returns
        // before recording: audit the title's binding directly, else the bind
        // is dropped silently and the census stays at zero.
        #[cfg(windows)]
        if metal() {
            crate::draw::State::record_native_heaps(slots);
            return;
        }
        with_state(this, |_, s| s.set_heaps(slots));
    }

    /// `SetComputeRootSignature(root signature)`: recorded on the producer.
    ///
    /// Wire v3 has no root-signature command: the host binds the PSO's own
    /// root signature when it registers that PSO, so the commit is carried as
    /// producer state (a later PSO comparison can see it). A null pointer
    /// unbinds the signature, as D3D12 allows.
    pub(super) extern "system" fn compute_root_signature(this: This, signature: This) {
        if metal() {
            with_metal(this, |p| {
                p.commit_root_signature(RootSignatureStage::Compute, signature as usize as u64)
            });
        }
    }

    /// `SetGraphicsRootSignature(root signature)`: the graphics twin of
    /// [`compute_root_signature`].
    pub(super) extern "system" fn graphics_root_signature(this: This, signature: This) {
        if metal() {
            with_metal(this, |p| {
                p.commit_root_signature(RootSignatureStage::Graphics, signature as usize as u64)
            });
        }
    }

    /// `D3D12_DISCARD_REGION`: rectangles, then the subresource range.
    #[repr(C)]
    pub(super) struct DiscardRegion {
        rects: u32,
        rectangles: *const RECT,
        first_subresource: u32,
        subresources: u32,
    }

    /// `DiscardResource(resource, region)`: discarded contents are undefined,
    /// so clearing the bound target to zero is a valid outcome and reuses the
    /// existing clear lowering. A partial discard (rectangles or a subresource
    /// range) has no Wire v3 equivalent and is refused by name.
    pub(super) extern "system" fn discard_resource(
        this: This,
        resource: This,
        region: *const DiscardRegion,
    ) {
        if !metal() {
            let _ = (this, resource, region);
            return;
        }
        with_metal(this, |producer| {
            use victory_metal::wire_v3::OP_BEGIN_RENDER_PASS;
            let discard_refused = |why: &'static str| {
                ProducerError::unsupported_opcode(why, OP_BEGIN_RENDER_PASS)
            };
            if resource.is_null() {
                return producer.refuse("DiscardResource: no resource", Some(OP_BEGIN_RENDER_PASS));
            }
            // SAFETY: the title passes a region description or null.
            if let Some(region) = unsafe { region.as_ref() } {
                // D3D12's "all subresources" is `-1`; a range that names the
                // only subresource is the same discard.
                let whole_range = region.first_subresource == 0
                    && (region.subresources == u32::MAX
                        || (region.subresources == 1
                            && crate::resource::texture_subresource(resource, 1).is_none()));
                if region.rects != 0 || !region.rectangles.is_null() || !whole_range {
                    return Err(discard_refused(
                        "DiscardResource: a partial discard (rectangles or subresources)",
                    ));
                }
            }
            let target = metal_resource(resource)
                .ok_or(discard_refused("DiscardResource: no guest allocation"))?;
            match producer.clear_render_target(target, [0.0; 4]) {
                Ok(()) => Ok(()),
                Err(_) => producer
                    .clear_depth_stencil(target, true, 1.0, true, 0)
                    .map_err(|_| discard_refused(
                        "DiscardResource: the resource is not a bound render target",
                    )),
            }
        });
    }

    /// 0x248: a pixel-shader depth-order hint. Wire v3 carries no command for
    /// it, so the producer records the value and counts it; the hint is never
    /// a refusal and never drops a draw.
    pub(super) extern "system" fn pixel_depth_order(this: This, order: u32) {
        if metal() {
            with_metal(this, |producer| {
                producer.set_pixel_shader_depth_order(order)
            });
        }
    }

    /// 0x3e0: a shading-rate hint. Wire v3 carries no command for it, so the
    /// producer records the value and counts it; the hint is never a refusal
    /// and never drops a draw.
    pub(super) extern "system" fn shading_rate(this: This, flags: u32) {
        if metal() {
            with_metal(this, |producer| producer.set_shading_rate(flags));
        }
    }

    /// 0x470 `(u64)` on newer lists: an unknown state flush. Wire v3 has no
    /// command for it, so it is refused by name instead of silently dropped.
    pub(super) extern "system" fn unknown_state_flush(this: This, _value: u64) {
        if metal() {
            with_metal(this, |producer| {
                producer.refuse(
                    "SetStateX (0x470): unknown state with no Wire v3 command",
                    None,
                )
            });
        }
    }
    /// `ClearUnorderedAccessViewUint(GPU handle, CPU handle, resource,
    /// values, rectangle count, rectangles)`.
    pub(super) extern "system" fn clear_unordered_uint(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, false, items(rects, count));
    }
    /// `ClearUnorderedAccessViewFloat`, the same with four floats.
    pub(super) extern "system" fn clear_unordered_float(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, true, items(rects, count));
    }
    fn clear_unordered(this: This, cpu: u64, values: *const [u32; 4], float: bool, rects: &[RECT]) {
        // SAFETY: the title passes four values (null is tolerated).
        let Some(&values) = (unsafe { values.as_ref() }) else {
            return;
        };
        with_state(this, |list, s| {
            s.clear_unordered(list, cpu, values, float, rects);
        });
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        if metal() {
            super::bind_compute_descriptor_table(this, slot, handle);
            return;
        }
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    /// Descriptor tables and root views carry guest GPU addresses; the
    /// native path resolves them through the installed
    /// `crate::native_metal::bindings` resolver, pins the backing ranges and
    /// records the Wire v3 entry. A table whose layout cannot be determined
    /// is refused, never guessed.
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        if metal() {
            super::bind_graphics_descriptor_table(this, slot, handle);
            return;
        }
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        if metal() {
            with_metal(this, |p| {
                p.set_compute_root_constants(slot, &[value], offset)
            });
            return;
        }
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        if metal() {
            with_metal(this, |p| {
                p.set_graphics_root_constants(slot, &[value], offset)
            });
            return;
        }
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        if metal() {
            let values = items(values, count);
            with_metal(this, |p| p.set_compute_root_constants(slot, values, offset));
            return;
        }
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        if metal() {
            let values = items(values, count);
            with_metal(this, |p| {
                p.set_graphics_root_constants(slot, values, offset)
            });
            return;
        }
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        if metal() {
            super::bind_compute_root_view(this, slot, address);
            return;
        }
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        if metal() {
            super::bind_graphics_root_view(this, slot, address);
            return;
        }
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        if metal() {
            // SAFETY: the title passes a view or null; `MetalIndexView` has
            // the same `repr(C)` layout.
            let view = unsafe { view.cast::<super::MetalIndexView>().as_ref() }.copied();
            match view {
                Some(view) => {
                    super::bind_index_buffer(this, view.address, u64::from(view.size), view.format)
                }
                // A null view is D3D12's legal unbind: clear the recorded
                // binding (no wire command) and count it, never a refusal.
                None => with_metal(this, |p| p.unbind_index_buffer()),
            }
            return;
        }
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        if metal() {
            if views.is_null() {
                with_metal(this, |p| p.clear_vertex_buffers(start, count));
            } else {
                super::bind_vertex_buffers(
                    this,
                    start,
                    items(views.cast::<super::MetalVertexView>(), count),
                );
            }
            return;
        }
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        if metal() {
            with_metal(this, |p| {
                let mut colors = Vec::with_capacity(handles.len());
                for &handle in &handles {
                    let record =
                        crate::descriptor::rtv_record(handle).ok_or(ProducerError::unsupported(
                            "OMSetRenderTargets: render target with no recorded view",
                        ))?;
                    let resource =
                        metal_resource(record.resource).ok_or(ProducerError::unsupported(
                            "OMSetRenderTargets: render target without a guest allocation",
                        ))?;
                    colors.push(RenderTarget::new(resource));
                }
                let depth = if depth == 0 {
                    None
                } else {
                    let record =
                        crate::descriptor::dsv_record(depth).ok_or(ProducerError::unsupported(
                            "OMSetRenderTargets: depth target with no recorded view",
                        ))?;
                    let resource =
                        metal_resource(record.resource).ok_or(ProducerError::unsupported(
                            "OMSetRenderTargets: depth target without a guest allocation",
                        ))?;
                    Some(DepthTarget::new(resource))
                };
                p.set_render_targets(&colors, depth)
            });
            return;
        }
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        if metal() {
            with_metal(this, |p| p.draw(vertices, instances, first, first_instance));
            return;
        }
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        if metal() {
            with_metal(this, |p| p.dispatch(x, y, z));
            return;
        }
        with_state(this, |list, s| s.dispatch(list, x, y, z));
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
        }
        if metal() {
            with_metal(this, |p| p.dispatch_mesh(x, y, z));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// shortcut: dispatches whose group count is in GPU memory (indirect)
    /// are dropped; make them ExecuteIndirect when the title uses them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if metal() {
            with_metal(this, |p| {
                if indirect != 0 {
                    p.refuse(
                        "DispatchMeshX: indirect mesh dispatch",
                        Some(victory_metal::wire_v3::OP_EXECUTE_INDIRECT_DISPATCH_MESH),
                    )
                } else {
                    p.dispatch_mesh(x, 1, 1)
                }
            });
            return;
        }
        if indirect != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("indirect mesh dispatches dropped"));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        if metal() {
            with_metal(this, |p| {
                p.draw_indexed(indices, instances, first, base, first_instance)
            });
            return;
        }
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is a distinct no-op entry point.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;

    /// One named entry point per state method, so no slot aliases the
    /// `ignored_state` sentinel: the table audit can tell a wired slot from a
    /// silent one on every platform. These are never called: the guest calls
    /// only exist on Windows.
    macro_rules! no_op_state {
        ($($name:ident),* $(,)?) => {
            $(
                pub(super) extern "system" fn $name(_: super::This) {}
            )*
        };
    }
    no_op_state!(
        blend_factor,
        clear_unordered_float,
        clear_unordered_uint,
        compute_constant,
        compute_constants,
        compute_root_signature,
        compute_table,
        compute_view,
        discard_resource,
        graphics_constant,
        graphics_constants,
        graphics_root_signature,
        graphics_table,
        graphics_view,
        heaps,
        index_buffer,
        pixel_depth_order,
        scissors,
        shading_rate,
        stencil_ref,
        targets,
        topology,
        unknown_state_flush,
        vertex_buffers,
        viewports,
    );
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. Grand Theft Auto VI's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    #[cfg(windows)]
    if crate::backend::GpuBackend::from_env() == crate::backend::GpuBackend::Metal {
        use crate::native_metal::producer::{ProducerError, ResourceId};
        use victory_metal::wire_v3::{IndirectArguments, IndirectCountBuffer};
        with_metal(this, |producer| {
            if max == 0 {
                // D3D12 executes nothing; no wire command is needed.
                return Ok(());
            }
            let Some((stride, args)) = crate::signature::layout(signature) else {
                return producer.refuse("ExecuteIndirect: signature is not one of ours", None);
            };
            // A signature that also changes root arguments or geometry is
            // not expressible: the wire command carries only the argument
            // range.
            let [[kind, ..]] = args else {
                return producer.refuse(
                    "ExecuteIndirect: a multi-argument signature",
                    Some(victory_metal::wire_v3::OP_EXECUTE_INDIRECT_DRAW),
                );
            };
            let Some((address, size)) = crate::resource::buffer_bytes(arguments) else {
                return producer.refuse(
                    "ExecuteIndirect: arguments are not a buffer",
                    Some(victory_metal::wire_v3::OP_EXECUTE_INDIRECT_DRAW),
                );
            };
            if offset.checked_add(u64::from(max).saturating_mul(u64::from(stride))).is_none_or(|end| end > size)
            {
                return producer.refuse(
                    "ExecuteIndirect: argument range outside its buffer",
                    Some(victory_metal::wire_v3::OP_EXECUTE_INDIRECT_DRAW),
                );
            }
            let count_binding = if count.is_null() {
                None
            } else {
                let Some((counter, count_size)) = crate::resource::buffer_bytes(count) else {
                    return producer.refuse(
                        "ExecuteIndirect: count is not a buffer",
                        Some(victory_metal::wire_v3::OP_EXECUTE_INDIRECT_DRAW),
                    );
                };
                if count_offset.checked_add(4).is_none_or(|end| end > count_size) {
                    return producer.refuse(
                        "ExecuteIndirect: count outside its buffer",
                        Some(victory_metal::wire_v3::OP_EXECUTE_INDIRECT_DRAW),
                    );
                }
                Some((counter as u64, count_size))
            };
            // The argument buffer (and the optional count buffer) are guest
            // buffers: resolve and register both, ascending by allocation id,
            // before the indirect command names them. The wire carries share
            // allocations, never guest addresses.
            let mut buffers = vec![(address as u64, size)];
            if let Some(counter) = count_binding {
                buffers.push(counter);
            }
            let mut resolved = crate::native_metal::queue::register_guest_buffers(&buffers)
                .map_err(|error| {
                    registration_refusal(
                        "ExecuteIndirect: a buffer is not registered with the host",
                        error,
                    )
                })?;
            let count = match count_binding {
                None => None,
                Some(_) => {
                    let counter = resolved.pop().expect("the count range was resolved");
                    let counter = ResourceId::from_range(&counter)?;
                    Some(IndirectCountBuffer {
                        allocation_id: counter.allocation_id,
                        generation: counter.generation,
                        offset: count_offset,
                    })
                }
            };
            let allocation = ResourceId::from_range(&resolved[0])?;
            let arguments = IndirectArguments {
                allocation_id: allocation.allocation_id,
                generation: allocation.generation,
                offset,
                max_count: max,
                count,
            };
            // D3D12 indirect argument types: draw 0, indexed draw 1,
            // dispatch 2, mesh dispatch 9; 105 is the console's single-pass
            // dispatch, whose record is the three group counts.
            match *kind {
                0 => producer.execute_indirect_draw(arguments),
                1 => producer.execute_indirect_draw_indexed(arguments),
                2 | 105 => producer.execute_indirect_dispatch(arguments),
                9 => {
                    let Some(pipeline) = producer.mesh_pipeline() else {
                        return Err(ProducerError::MissingPipeline { stage: "mesh" });
                    };
                    producer.execute_indirect_dispatch_mesh(pipeline, arguments)
                }
                _ => Err(ProducerError::unsupported_opcode(
                    "ExecuteIndirect: argument type has no native command",
                    victory_metal::wire_v3::OP_EXECUTE_INDIRECT_DRAW,
                )),
            }
        });
        return;
    }
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE {
            DISPATCH
        } else {
            *kind
        };
        with_state(this, |list, s| {
            ran = s.execute_indirect(list, (kind, stride, max), address, counter);
        });
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}


/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. Every host use of a
/// title texture (clears, copies) moves its host copy to the state it needs
/// first, so only their ordering matters: one barrier on every shader
/// write (the title's buffer barriers order compute passes).
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    #[cfg(windows)]
    if count > 0
        && let Some(list) = host_list(this)
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        // Which kinds of barriers (0 global, 1 texture, 2 buffer) and how many.
        #[cfg(feature = "diagnostics")]
        if !_groups.is_null() {
            // SAFETY: the title passes `count` groups of 16 bytes: kind, count, pointer.
            let groups = unsafe { std::slice::from_raw_parts(_groups.cast::<[u32; 4]>(), count as usize) };
            list.note_op(|| format!("barrier groups {:?}", groups.iter().map(|g| (g[0], g[1])).collect::<Vec<_>>()));
            crate::copystat::add(
                crate::copystat::C::GroupBarriers,
                groups.iter().map(|g| u64::from(g[1])).sum(),
            );
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GroupCalls, 1);
        // SAFETY: the same layout; no resource named.
        let barrier =
            unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        list.sync_widened();
    }
    #[cfg(not(windows))]
    let _ = (this, count);
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}
pub(crate) extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 52] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, state::compute_root_signature as *const ()), // 0x108 compute root signature
    (34, state::graphics_root_signature as *const ()), // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, depth_bounds as *const ()),              // 0x220
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait32 as *const ()),                    // 0x238 (newer 0x248)
    (72, wait64 as *const ()),                    // 0x240 (newer 0x250)
    (73, state::pixel_depth_order as *const ()),  // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, state::shading_rate as *const ()),      // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 12] = [
    (53, state::clear_unordered_uint as *const ()), // 0x1a8
    (54, state::clear_unordered_float as *const ()), // 0x1b0
    (55, state::discard_resource as *const ()),     // 0x1b8 DiscardResource
    (28, query_mark as *const ()),                  // 0xe0 BeginQuery
    (29, query_mark as *const ()),                  // 0xe8 EndQuery
    (30, resolve_queries as *const ()),             // 0xf0
    (78, dispatch_x as *const ()),                  // 0x270
    (98, state::dispatch_mesh_x as *const ()),      // 0x310
    (99, driver_hint as *const ()),                 // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// Grand Theft Auto VI's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from Grand Theft Auto VI's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown and Wire v3 has no command for it, so
/// it is refused by name (never silently ignored).
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, state::unknown_state_flush as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(all(test, windows))]
pub(crate) fn has_newer_table(list: This) -> bool {
    // SAFETY: one of our lists; its first field is the table. The device
    // tests can switch layouts, so use this list's table.
    let table = unsafe { *list.cast::<*const *const ()>() };
    table == NEWER_LIST_FUNCTIONS.0.as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tracing::{Event, Metadata, Subscriber, span};

    struct PipelineBinds(Arc<AtomicUsize>);

    impl Subscriber for PipelineBinds {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.is_event()
                && metadata.target() == "victory_gpu::pipeline"
                && metadata.fields().field("pipeline").is_some()
                && metadata.fields().field("frame").is_some()
        }

        fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
            unreachable!()
        }

        fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
        fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
        fn enter(&self, _: &span::Id) {}
        fn exit(&self, _: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            if self.enabled(event.metadata()) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[test]
    fn lists_bind_the_initial_pipeline_on_creation() {
        use crate::pipeline::{ComputeHead, create_compute};

        // A synthetic compute description matching the public ABI.
        #[repr(C)]
        struct ComputeDesc {
            root: This,
            code: *const u8,
            size: usize,
        }

        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let desc = ListDesc {
            kind: 0,
            flags: 0,
            nodes: 0,
            barrier_batch: 0,
        };
        let bytes = [0xa5, 0xb6, 0xc7];
        let compute = ComputeDesc {
            root: null,
            code: bytes.as_ptr(),
            size: bytes.len(),
        };
        let compute = std::ptr::from_ref(&compute).cast::<ComputeHead>();
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        let binds = Arc::new(AtomicUsize::new(0));
        let subscriber = tracing::Dispatch::new(PipelineBinds(binds.clone()));

        tracing::dispatcher::with_default(&subscriber, || {
            for console in [false, true] {
                let create = |pipeline, out| {
                    if console {
                        create_list_x(null, &desc, allocator, pipeline, &LIST_IIDS[1], out)
                    } else {
                        create_list(null, 0, 0, allocator, pipeline, &LIST_IIDS[1], out)
                    }
                };
                let mut pipeline = null;
                let mut list = null;
                assert_eq!(
                    create_compute(null, compute, &PAGEABLE_IIDS[0], &mut pipeline),
                    S_OK
                );
                let before = binds.load(Ordering::Relaxed);

                assert_eq!(create(null, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before);
                assert_eq!(release(list), 0);

                assert_eq!(create(pipeline, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before + 1);
                assert_eq!(release(list), 0);
                assert_eq!(release(pipeline), 0);
            }
        });
        assert_eq!(release(allocator), 0);
    }

    /// ClearUnorderedAccessViewUint/Float and DiscardResource come before
    /// the methods newer lists insert: the AMD upscaler calls them there.
    #[test]
    fn unordered_access_clears_have_the_same_slots_in_both_layouts() {
        for table in [&LIST_FUNCTIONS, &NEWER_LIST_FUNCTIONS] {
            assert_eq!(table.0[0x1a8 / 8], state::clear_unordered_uint as *const ());
            assert_eq!(
                table.0[0x1b0 / 8],
                state::clear_unordered_float as *const ()
            );
            assert_eq!(table.0[0x1b8 / 8], state::discard_resource as *const ());
        }
    }

    /// ADR-030: no list slot may be the silent `ignored_state` stub. A future
    /// method wired to it (a state drop with no translation and no refusal)
    /// fails here instead of disappearing at runtime.
    #[test]
    fn no_list_slot_is_a_silent_state_drop() {
        for (layout, table) in [
            ("legacy", &LIST_FUNCTIONS),
            ("newer", &NEWER_LIST_FUNCTIONS),
        ] {
            for (slot, method) in table.0.iter().enumerate() {
                assert_ne!(
                    *method,
                    ignored_state as *const (),
                    "{layout} list slot {slot:#x} is the silent ignored_state stub"
                );
            }
        }
    }

    /// Every state once dropped silently now names its handler: root
    /// signatures are recorded, DiscardResource reuses the clear path, and the
    /// remaining console hints are refused by name.
    #[test]
    fn former_ignored_slots_bind_their_named_handlers() {
        for (layout, table, newer) in [
            ("legacy", &LIST_FUNCTIONS, false),
            ("newer", &NEWER_LIST_FUNCTIONS, true),
        ] {
            let slot = |legacy: usize| if newer { newer_slot(legacy) } else { legacy };
            assert_eq!(
                table.0[0x108 / 8],
                state::compute_root_signature as *const (),
                "{layout}: compute root signature"
            );
            assert_eq!(
                table.0[0x110 / 8],
                state::graphics_root_signature as *const (),
                "{layout}: graphics root signature"
            );
            assert_eq!(
                table.0[0x1b8 / 8],
                state::discard_resource as *const (),
                "{layout}: DiscardResource"
            );
            assert_eq!(
                table.0[slot(73)],
                state::pixel_depth_order as *const (),
                "{layout}: pixel-shader depth order at {:#x}",
                slot(73) * 8
            );
            assert_eq!(
                table.0[slot(124)],
                state::shading_rate as *const (),
                "{layout}: shading rate at {:#x}",
                slot(124) * 8
            );
        }
        assert_eq!(
            NEWER_LIST_FUNCTIONS.0[142],
            state::unknown_state_flush as *const (),
            "0x470 state flush"
        );
    }

    /// A guest buffer footprint lowers to one pitched wire copy: the first
    /// row's byte offset (footprint offset plus rows and texels), the row
    /// pitch, the texel box and the destination mip/slice/origin. Boxes the
    /// wire cannot carry are refused by name.
    #[test]
    fn a_buffer_footprint_lowers_to_one_pitched_copy() {
        use crate::native_metal::producer::ResourceId;
        use victory_metal::wire_v3::TextureSubresource;

        let source = ResourceId::new(0x1000, 1).expect("source identity");
        let destination = TextureSubresource {
            allocation_id: 0x2000,
            generation: 1,
            mip_level: 1,
            slice: 2,
        };
        // 64x32 R8G8B8A8 rows at a 256-byte pitch, starting 0x100 into the
        // staging buffer.
        let footprint = crate::resource::PlacedFootprint::new(0x100, 28, 64, 32, 256);
        let copy = upload_copy(
            source,
            destination,
            [64, 32, 1],
            [0, 0, 0],
            &footprint,
            None,
        )
        .expect("the whole footprint");
        assert_eq!(copy.offset, 0x100);
        assert_eq!(copy.bytes_per_row, 256);
        assert_eq!(copy.extent, [64, 32, 1]);
        assert_eq!(copy.origin, [0, 0, 0]);
        assert_eq!(copy.destination, destination);
        assert_eq!(copy.source, source);

        // A box at texel (8, 4): four pitched rows plus eight RGBA texels.
        let boxed = upload_copy(
            source,
            destination,
            [32, 16, 1],
            [0, 0, 0],
            &footprint,
            Some([8, 4, 0, 24, 20, 1]),
        )
        .expect("a boxed upload");
        assert_eq!(boxed.offset, 0x100 + 4 * 256 + 8 * 4);
        assert_eq!(boxed.extent, [16, 16, 1]);

        // A volume footprint needs one command per slice.
        let volume = crate::resource::PlacedFootprint {
            offset: 0,
            format: 28,
            width: 8,
            height: 8,
            depth: 2,
            row_pitch: 32,
        };
        assert_eq!(
            upload_copy(source, destination, [8, 8, 1], [0, 0, 0], &volume, None),
            Err("CopyTextureRegion: a buffer upload of more than one depth slice")
        );
        // A box outside the footprint is refused, not clipped.
        assert_eq!(
            upload_copy(
                source,
                destination,
                [64, 32, 1],
                [0, 0, 0],
                &footprint,
                Some([60, 0, 0, 68, 4, 1]),
            ),
            Err("CopyTextureRegion: the upload box is outside its footprint")
        );
        // The destination subresource bounds are checked too.
        assert_eq!(
            upload_copy(source, destination, [32, 16, 1], [8, 0, 0], &footprint, None),
            Err("CopyTextureRegion: the upload box is outside its destination subresource")
        );
    }

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as Grand Theft Auto VI asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, below 2^40 (where
        // shaders find title memory), gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        #[cfg(windows)]
        assert!((1 << 32..1 << 40).contains(&(p as usize)), "{p:p}");
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(
            allocate_upload(list, 8, 0x30).is_null(),
            "not a power of two"
        );
        assert_eq!(reset(list, allocator, null), S_OK);
        assert!(
            // SAFETY: the list is ours.
            unsafe { &(*list.cast::<List>()).uploads }
                .lock()
                .unwrap()
                .is_empty()
        );
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// An offset near `u64::MAX` must not overflow and write outside the
    /// query destination buffer.
    #[test]
    fn resolve_queries_rejects_overflowing_offset() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        let buffer = crate::resource::test_buffer(64);
        // Used to overflow `offset + count * 8` (panic in debug, wrap in release).
        resolve_queries(list, null, 0, 0, 2, buffer, u64::MAX - 8);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for object in [buffer, list, allocator] {
            assert_eq!(release(object), 0);
        }
    }

    /// Transitions between two read-only states are told from those that
    /// write, from common, and from other barrier kinds.
    #[test]
    fn read_only_transitions_need_both_states_read_only() {
        const COMMON: u32 = 0;
        const INDEX: u32 = 0x2;
        const UNORDERED: u32 = 0x8;
        const NON_PIXEL: u32 = 0x40;
        const PIXEL: u32 = 0x80;
        const COPY_DEST: u32 = 0x400;
        const GENERIC_READ: u32 = 0xac3;
        // The console's compression hints live above the PC bits.
        const HINT: u32 = 1 << 26;
        let transition = |kind, before: u32, after: u32| Barrier {
            kind,
            flags: 0,
            first: 8 as This,
            second: 0xffff_ffff | (u64::from(before) << 32),
            last: u64::from(after),
        };
        assert!(read_only_transition(&transition(0, NON_PIXEL, PIXEL)));
        let both = |before, after| read_only_transition(&transition(0, before, after));
        assert!(both(GENERIC_READ, INDEX | PIXEL));
        assert!(both(NON_PIXEL | HINT, PIXEL));
        assert!(!read_only_transition(&transition(0, UNORDERED, PIXEL)));
        assert!(!read_only_transition(&transition(0, PIXEL, UNORDERED)));
        assert!(!read_only_transition(&transition(0, NON_PIXEL, COPY_DEST)));
        assert!(!read_only_transition(&transition(0, COMMON, PIXEL)));
        assert!(!read_only_transition(&transition(0, NON_PIXEL, COMMON)));
        assert!(!read_only_transition(&transition(2, NON_PIXEL, PIXEL)));
        assert!(!read_only_transition(&UAV_BARRIER));
    }

    // ------------------------------------------------ compute root bindings

    /// A resolver over one synthetic share page for the compute-root helper.
    struct ComputePageResolver {
        address: u64,
        pins: std::sync::Mutex<Vec<u64>>,
        next: std::sync::atomic::AtomicU64,
    }

    impl ComputePageResolver {
        fn new(address: u64) -> Self {
            Self {
                address,
                pins: std::sync::Mutex::new(Vec::new()),
                next: std::sync::atomic::AtomicU64::new(1),
            }
        }

        fn live(&self) -> usize {
            self.pins.lock().unwrap_or_else(|e| e.into_inner()).len()
        }
    }

    impl crate::native_metal::bindings::GuestRangeResolver for ComputePageResolver {
        fn resolve(
            &self,
            guest_address: u64,
            len: u64,
        ) -> Result<crate::native_metal::bindings::GuestRange, crate::native_metal::bindings::ResolveError>
        {
            use crate::native_metal::bindings::{GuestRange, ResolveError, PAGE};
            let within = guest_address.checked_sub(self.address);
            let fits = within.is_some_and(|within| {
                within < PAGE && len != 0 && within.saturating_add(len) <= PAGE
            });
            if !fits {
                return Err(ResolveError::Unmapped {
                    address: guest_address,
                    len,
                });
            }
            Ok(GuestRange {
                allocation_id: 7,
                generation: 1,
                byte_offset: 0,
                size: PAGE,
            })
        }

        fn retain(
            &self,
            _range: &crate::native_metal::bindings::GuestRange,
        ) -> Result<u64, crate::native_metal::bindings::ResolveError> {
            let token = self.next.fetch_add(1, Ordering::Relaxed);
            self.pins
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(token);
            Ok(token)
        }

        fn retire(&self, token: u64) -> Result<(), crate::native_metal::bindings::ResolveError> {
            let mut pins = self.pins.lock().unwrap_or_else(|e| e.into_inner());
            match pins.iter().position(|held| *held == token) {
                Some(index) => {
                    pins.remove(index);
                    Ok(())
                }
                None => Err(crate::native_metal::bindings::ResolveError::Retire(
                    "the token is unknown or already retired",
                )),
            }
        }
    }

    /// A compute root view resolves to the page window at the exact offset,
    /// pins its backing once, and both refusal kinds are `E_FAIL` with a named
    /// reason: an unregistered allocation and an unmapped address.
    #[test]
    fn compute_root_views_pin_and_refuse_unregistered_or_unmapped() {
        use crate::native_metal::bindings::PAGE;
        use crate::native_metal::producer::ProducerError;

        let resolver = ComputePageResolver::new(0x1_0000_0000);
        let address = resolver.address + 0x40;
        let resolved = resolve_compute_root_view(&resolver, 2, address, |allocation_id| {
            (allocation_id == 7)
                .then_some(())
                .ok_or_else(|| "no share file".to_string())
        })
        .expect("a registered allocation");
        assert_eq!(resolved.entry.slot, 2);
        assert_eq!(resolved.entry.allocation_id, 7);
        assert_eq!(resolved.entry.generation, 1);
        assert_eq!(resolved.entry.offset, 0x40, "the in-page byte is exact");
        assert_eq!(resolved.entry.length, PAGE - 0x40);
        assert_eq!(resolved.tokens.len(), 1);
        assert_eq!(resolver.live(), 1);

        let refusal = resolve_compute_root_view(&resolver, 0, resolver.address, |_| {
            Err("no share file".to_string())
        })
        .expect_err("an unregistered allocation is refused");
        assert!(matches!(refusal, ComputeBindingRefusal::Unregistered(_)));
        let error = refusal.producer_error(
            "a compute root view's allocation is not registered with the host",
        );
        assert_eq!(
            error,
            ProducerError::Unregistered {
                operation: "a compute root view's allocation is not registered with the host"
            }
        );
        assert_eq!(error.hresult(), 0x8000_4005_u32 as i32);
        assert_eq!(
            resolver.live(),
            1,
            "the refused binding's pin was released"
        );

        let refusal = resolve_compute_root_view(&resolver, 0, resolver.address + 2 * PAGE, |_| {
            Ok(())
        })
        .expect_err("an unmapped address is refused");
        assert!(matches!(refusal, ComputeBindingRefusal::Unresolved(_)));
        let error = refusal.producer_error("a compute root view could not be resolved");
        assert_eq!(
            error,
            ProducerError::Unresolved {
                operation: "a compute root view could not be resolved"
            }
        );
        assert_eq!(error.hresult(), 0x8000_4005_u32 as i32);
        assert_eq!(resolver.live(), 1, "an unresolved range pins nothing");

        // Clean up the pin the successful resolution holds.
        crate::native_metal::bindings::retire_all(&resolver, &resolved.tokens);
        assert_eq!(resolver.live(), 0);
    }

    /// A placed buffer's resource description: `len` bytes.
    fn placed_buffer_resource(len: u64) -> [u32; 14] {
        let mut resource = [0u32; 14];
        resource[0] = 1; // D3D12_RESOURCE_DIMENSION_BUFFER
        resource[4] = len as u32;
        resource[5] = (len >> 32) as u32;
        resource
    }

    /// A placed texture's resource description: `width` x `height`, one mip,
    /// one array slice, one sample.
    fn placed_texture_resource(width: u64, height: u32, format: u32) -> [u32; 14] {
        let mut resource = [0u32; 14];
        resource[0] = 3; // D3D12_RESOURCE_DIMENSION_TEXTURE2D
        resource[4] = width as u32;
        resource[5] = (width >> 32) as u32;
        resource[6] = height;
        resource[7] = 1 | (1 << 16); // one array slice, one mip
        resource[8] = format;
        resource[9] = 1; // one sample
        resource
    }

    /// A placed view description: the console union begins at word 4.
    fn placed_view_desc(
        address: u64,
        view_dimension: u32,
        format: u32,
        first: u64,
        count: u32,
        stride: u32,
    ) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = format;
        desc[1] = view_dimension;
        desc[2] = 0x1688; // D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
        desc[4] = first as u32;
        desc[5] = (first >> 32) as u32;
        desc[6] = count;
        desc[7] = stride;
        desc[10] = address as u32;
        desc[11] = (address >> 32) as u32;
        desc
    }

    #[test]
    fn a_placed_buffer_view_resolves_to_its_exact_range() {
        use crate::native_metal::bindings::{DescriptorSlot, GuestBinding};
        use victory_metal::wire_v3::DescriptorKind;

        const ADDRESS: u64 = 0x1_2345_6000;
        let resource = placed_buffer_resource(0x400);
        // An SRV of 0x40 elements of 16 bytes, at the buffer's start.
        let view = placed_view_desc(ADDRESS, 1, 0, 0, 0x40, 16);
        assert_eq!(
            placed_slot(false, &resource, &view, |_| panic!("a buffer binds no texture")),
            DescriptorSlot::Live(
                DescriptorKind::Srv,
                GuestBinding {
                    address: ADDRESS,
                    len: 0x400,
                },
            )
        );
        // The same view as a UAV.
        assert_eq!(
            placed_slot(true, &resource, &view, |_| panic!("a buffer binds no texture")),
            DescriptorSlot::Live(
                DescriptorKind::Uav,
                GuestBinding {
                    address: ADDRESS,
                    len: 0x400,
                },
            )
        );
        // A constant-buffer description (no view dimension): the whole
        // footprint, at the view's exact address.
        let cbv = placed_view_desc(ADDRESS, 0, 0, 0, 0, 0);
        assert_eq!(
            placed_slot(false, &resource, &cbv, |_| panic!("a buffer binds no texture")),
            DescriptorSlot::Live(
                DescriptorKind::Cbv,
                GuestBinding {
                    address: ADDRESS,
                    len: 0x400,
                },
            )
        );
        // A view off the buffer's start keeps its exact in-resource offset.
        let offset = placed_view_desc(ADDRESS, 1, 0, 2, 8, 4);
        assert_eq!(
            placed_slot(false, &resource, &offset, |_| panic!("a buffer binds no texture")),
            DescriptorSlot::Live(
                DescriptorKind::Srv,
                GuestBinding {
                    address: ADDRESS + 8,
                    len: 32,
                },
            )
        );
    }

    #[test]
    fn a_placed_texture_view_yields_its_registration_handle() {
        use crate::native_metal::bindings::DescriptorSlot;
        use victory_metal::wire_v3::DescriptorKind;

        const ADDRESS: u64 = 0x2_3456_7000;
        let resource = placed_texture_resource(4, 4, 28);
        let view = placed_view_desc(ADDRESS, 4, 28, 0, 0, 0);
        let mut registered = 0;
        for (unordered, usage, kind) in [
            (false, 0u8, DescriptorKind::Srv),
            (true, 3u8, DescriptorKind::Uav),
        ] {
            let slot = placed_slot(unordered, &resource, &view, |registration| {
                assert_eq!(registration.handle, ADDRESS, "the guest memory address");
                assert_eq!(registration.usage, usage);
                assert_eq!(registration.width, 4);
                assert_eq!(registration.height, 4);
                assert_eq!(registration.mip_levels, 1);
                registered += 1;
                Ok(registration.handle)
            });
            assert_eq!(slot, DescriptorSlot::Texture(kind, ADDRESS));
        }
        assert_eq!(registered, 2);

        // A host refusal is the same named refusal the resource-backed path
        // returns, never a guessed handle.
        let refused = placed_slot(false, &resource, &view, |_| {
            Err("the host refused a sampled texture registration")
        });
        assert_eq!(
            refused,
            DescriptorSlot::Unresolvable("the host refused a sampled texture registration")
        );
    }

    #[test]
    fn placed_view_refusals_are_named() {
        use crate::native_metal::bindings::DescriptorSlot;

        let buffer = placed_buffer_resource(64);
        let assert_refused = |slot, why: &'static str| {
            assert_eq!(slot, DescriptorSlot::Unresolvable(why));
        };

        // No device memory.
        assert_refused(
            placed_slot(
                false,
                &buffer,
                &placed_view_desc(0, 1, 0, 0, 16, 4),
                |_| unreachable!(),
            ),
            "a placed buffer view names no device memory",
        );
        // A range past the resource footprint.
        assert_refused(
            placed_slot(
                false,
                &buffer,
                &placed_view_desc(0x1000, 1, 0, 0, 32, 4),
                |_| unreachable!(),
            ),
            "a placed buffer view's range is not within its resource footprint",
        );
        // An unknown resource dimension.
        let mut unknown = buffer;
        unknown[0] = 99;
        assert_refused(
            placed_slot(
                false,
                &unknown,
                &placed_view_desc(0x1000, 1, 0, 0, 1, 4),
                |_| unreachable!(),
            ),
            "a placed view's resource dimension is unknown",
        );
        // A buffer resource named through a texture view description.
        assert_refused(
            placed_slot(
                false,
                &buffer,
                &placed_view_desc(0x1000, 4, 0, 0, 0, 0),
                |_| unreachable!(),
            ),
            "a placed buffer view is described as a texture",
        );
        // A zero-length buffer.
        assert_refused(
            placed_slot(
                false,
                &placed_buffer_resource(0),
                &placed_view_desc(0x1000, 1, 0, 0, 0, 0),
                |_| unreachable!(),
            ),
            "a placed buffer view has no bytes",
        );
        // A texture resource named through a buffer view description.
        let texture = placed_texture_resource(4, 4, 28);
        assert_refused(
            placed_slot(
                false,
                &texture,
                &placed_view_desc(0x1000, 1, 28, 0, 0, 0),
                |_| unreachable!(),
            ),
            "a placed texture view is described as a buffer",
        );
        // A texture with no device memory.
        assert_refused(
            placed_slot(
                false,
                &texture,
                &placed_view_desc(0, 4, 28, 0, 0, 0),
                |_| unreachable!(),
            ),
            "a placed texture view names no device memory",
        );
    }

    #[test]
    fn a_placed_buffer_slot_enumerates_in_a_buffer_only_table() {
        use crate::native_metal::bindings::{
            DescriptorSlot, GuestBinding, TableRunError, enumerate_descriptor_table,
        };
        use victory_metal::wire_v3::DescriptorKind;

        const BASE: u64 = 0x3000_0000;
        const ADDRESS: u64 = 0x1_2345_6000;
        let buffer = placed_buffer_resource(0x400);
        let view = placed_view_desc(ADDRESS, 1, 0, 0, 0x40, 16);
        let run = enumerate_descriptor_table(BASE, |handle| {
            if handle == BASE {
                placed_slot(false, &buffer, &view, |_| unreachable!())
            } else {
                DescriptorSlot::Empty
            }
        })
        .expect("a placed buffer slot is a live buffer entry");
        assert_eq!(run.kind, DescriptorKind::Srv);
        assert_eq!(
            run.entries,
            vec![GuestBinding {
                address: ADDRESS,
                len: 0x400,
            }]
        );

        // The same buffer-only table still refuses a placed texture slot.
        let texture = placed_texture_resource(4, 4, 28);
        let texture_view = placed_view_desc(ADDRESS, 4, 28, 0, 0, 0);
        let error = enumerate_descriptor_table(BASE, |handle| {
            if handle == BASE {
                placed_slot(false, &texture, &texture_view, |r| Ok(r.handle))
            } else {
                DescriptorSlot::Empty
            }
        })
        .expect_err("the buffer-only enumerator cannot carry a registered resource");
        assert!(matches!(error, TableRunError::Unsupported(_)));
    }

    #[test]
    fn a_mixed_placed_run_is_refused_not_mis_bound() {
        use crate::native_metal::bindings::{
            DescriptorEntry, DescriptorSlot, TableRunBoundary, TableRunError,
            VIEW_DESCRIPTOR_INCREMENT, enumerate_descriptor_table_typed,
        };
        use victory_metal::wire_v3::DescriptorKind;

        const BASE: u64 = 0x4000_0000;
        const ADDRESS: u64 = 0x1_2345_6000;
        let buffer = placed_buffer_resource(0x400);
        let buffer_view = placed_view_desc(ADDRESS, 1, 0, 0, 0x40, 16);
        let texture = placed_texture_resource(4, 4, 28);
        let texture_view = placed_view_desc(ADDRESS, 4, 28, 0, 0, 0);
        let slot = |handle: u64| ((handle - BASE) / VIEW_DESCRIPTOR_INCREMENT) as usize;

        // A placed UAV buffer and a placed SRV texture end the run as a kind
        // change, never as one table with the texture mis-bound into it.
        let run = enumerate_descriptor_table_typed(BASE, |handle| match slot(handle) {
            0 => placed_slot(true, &buffer, &buffer_view, |_| unreachable!()),
            1 => placed_slot(false, &texture, &texture_view, |r| Ok(r.handle)),
            _ => DescriptorSlot::Empty,
        })
        .expect("the homogeneous prefix enumerates");
        assert_eq!(run.kind, DescriptorKind::Uav);
        assert_eq!(run.len(), 1);
        assert_eq!(
            run.boundary,
            TableRunBoundary::KindChange {
                kind: DescriptorKind::Srv
            }
        );
        assert_eq!(
            run.require_homogeneous(),
            Err(TableRunError::MixedKind {
                run_kind: DescriptorKind::Uav,
                next_kind: DescriptorKind::Srv,
            })
        );

        // Two same-kind placed SRV entries (one buffer, one texture) enumerate
        // both shapes, so the encoder refuses the mixed table instead of
        // binding a prefix.
        let run = enumerate_descriptor_table_typed(BASE, |handle| match slot(handle) {
            0 => placed_slot(false, &buffer, &buffer_view, |_| unreachable!()),
            1 => placed_slot(false, &texture, &texture_view, |r| Ok(r.handle)),
            _ => DescriptorSlot::Empty,
        })
        .expect("the mixed run enumerates");
        assert_eq!(run.len(), 2);
        assert!(matches!(run.entries[0], DescriptorEntry::Buffer(_)));
        assert!(matches!(run.entries[1], DescriptorEntry::Texture(_)));
    }

    #[test]
    fn the_placed_census_counts_resolutions_and_names_refusals() {
        placed_census::reset();
        const BUFFER_REFUSAL: &str = "a placed buffer view names no device memory";

        let buffer = placed_buffer_resource(64);
        let buffer_view = placed_view_desc(0x1000, 1, 0, 0, 16, 4);
        let _ = placed_slot(false, &buffer, &buffer_view, |_| unreachable!());
        let texture = placed_texture_resource(4, 4, 28);
        let texture_view = placed_view_desc(0x2000, 4, 28, 0, 0, 0);
        let _ = placed_slot(false, &texture, &texture_view, |r| Ok(r.handle));
        assert_eq!(placed_census::counts(), [1, 1]);

        let before = crate::native_metal::telemetry::named_refusal_histogram()
            .get(BUFFER_REFUSAL)
            .copied()
            .unwrap_or(0);
        let _ = placed_slot(
            false,
            &buffer,
            &placed_view_desc(0, 1, 0, 0, 1, 4),
            |_| unreachable!(),
        );
        let after = crate::native_metal::telemetry::named_refusal_histogram()
            .get(BUFFER_REFUSAL)
            .copied()
            .unwrap_or(0);
        assert_eq!(after, before + 1, "the refusal is in the named census");
        assert_eq!(placed_census::counts(), [1, 1], "a refusal is not a resolution");
    }
}
