//! Persistent `MTLResidencySet` bindless residency (ADR-013).
//!
//! This module owns attachment of live shared Metal buffers and registered
//! textures to one persistent residency set bound to the command queue, with an
//! explicit `useResources` fallback when the runtime does not expose
//! `MTLResidencySet`.
//!
//! # Native path
//!
//! [`ResidencySet::new`] asks the device `respondsToSelector:` for
//! `newResidencySetWithDescriptor:error:` before calling it. When the runtime
//! has `MTLResidencySet`, allocations are added with `addAllocation:`,
//! [`ResidencySet::attach_to_queue`] binds the set to the transport's
//! `MTLCommandQueue` with `addResidencySet:`, and [`ResidencySet::commit`]
//! publishes the set. [`ResidencySet::is_native`] then reports `true` and
//! [`ResidencySet::use_resources`] performs no per-encoder calls.
//!
//! # Fallback path
//!
//! When the selector, class or factory is missing, the constructor records the
//! exact failing selector with its `file:line` in
//! [`ResidencySet::fallback_reason`] and keeps every allocation in an ordered
//! `Vec` instead. [`ResidencySet::use_resources`] then issues one
//! `useResource:usage:` call per recorded allocation on the encoder. This
//! fallback is explicitly `O(n)` per encoder: [`ResidencySet::is_native`]
//! returns `false`, so no caller can mistake it for `O(1)` bindless residency.
//! Missing selectors never panic.

use crate::abi::VmBufferId;
use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, ObjcObject};
use crate::macos::registry::TextureRegistry;
use crate::macos::texture::VmTextureId;
use crate::transport::MetalError;

/// `MTLResourceUsageRead`.
const MTL_RESOURCE_USAGE_READ: usize = 1;
/// `MTLResourceUsageWrite`.
const MTL_RESOURCE_USAGE_WRITE: usize = 2;
/// Initial capacity requested from `MTLResidencySetDescriptor`.
const INITIAL_RESIDENCY_CAPACITY: usize = 64;

/// Errors produced by [`ResidencySet`].
#[derive(Debug, thiserror::Error)]
pub enum ResidencyError {
    /// The buffer handle is not registered on the transport.
    #[error("buffer {0} is not registered")]
    UnknownBuffer(u64),
    /// The texture handle is not registered in the texture registry.
    #[error("texture {0} is not registered")]
    UnknownTexture(u64),
    /// The requested length is zero or larger than the registered buffer.
    #[error("length {requested} is invalid for buffer {id} of {registered} bytes")]
    InvalidLength {
        /// Buffer handle.
        id: u64,
        /// Requested length.
        requested: usize,
        /// Registered length.
        registered: usize,
    },
    /// The host does not implement the requested residency operation.
    #[error("residency backend does not support: {0}")]
    Unsupported(String),
    /// A Metal object could not be created or resolved.
    #[error(transparent)]
    Metal(#[from] MetalError),
}

/// One allocation kept alive by the residency set.
struct ResidencyAllocation {
    object: ObjcObject,
    len: usize,
}

/// Persistent residency set with an explicit `useResource` fallback.
pub struct ResidencySet {
    // Keeps the originating device alive for the lifetime of the set. The
    // native `MTLResidencySet` also retains it; the explicit retain documents
    // the ownership and keeps the field available for future device wiring.
    _device: ObjcObject,
    native: Option<ObjcObject>,
    allocations: Vec<ResidencyAllocation>,
    fallback_reason: Option<String>,
}

impl ResidencySet {
    /// Creates a residency set bound to `device`, or records why it cannot.
    ///
    /// A missing `MTLResidencySet` surface is not an error: the constructor
    /// returns the documented fallback and [`Self::is_native`] reports `false`.
    pub fn new(device: &DirectMetalTransport) -> Result<Self, ResidencyError> {
        let raw_device = device.device_raw();
        let device_object = retain(raw_device)?;

        let factory_selector = objc::sel(c"newResidencySetWithDescriptor:error:");
        // SAFETY: `raw_device` is a live MTLDevice; the call is a pure query.
        if !unsafe { objc::msg_send_responds_to(raw_device, factory_selector) } {
            return Ok(Self::fallback(
                device_object,
                format!(
                    "{}:{}: MTLDevice does not respond to `newResidencySetWithDescriptor:error:`",
                    file!(),
                    line!()
                ),
            ));
        }
        let class = objc::class(c"MTLResidencySetDescriptor");
        if class.is_null() {
            return Ok(Self::fallback(
                device_object,
                format!(
                    "{}:{}: MTLResidencySetDescriptor class is unavailable",
                    file!(),
                    line!()
                ),
            ));
        }
        let raw_descriptor = unsafe { objc::msg_send_id_0(class, objc::sel(c"new")) };
        let Some(descriptor) = ObjcObject::from_retained(raw_descriptor) else {
            return Ok(Self::fallback(
                device_object,
                format!(
                    "{}:{}: MTLResidencySetDescriptor creation returned null",
                    file!(),
                    line!()
                ),
            ));
        };
        // SAFETY: `descriptor` is a live descriptor and the selectors take a
        // documented `NSUInteger`/`NSString *`.
        unsafe {
            if objc::msg_send_responds_to(descriptor.as_raw(), objc::sel(c"setInitialCapacity:")) {
                objc::msg_send_void_usize(
                    descriptor.as_raw(),
                    objc::sel(c"setInitialCapacity:"),
                    INITIAL_RESIDENCY_CAPACITY,
                );
            }
            if let Some(label) = objc::nsstring("victory-bindless-residency") {
                objc::msg_send_void_1(descriptor.as_raw(), objc::sel(c"setLabel:"), label.as_raw());
            }
        }

        let mut err = objc::empty_id();
        // SAFETY: `raw_device` is a live MTLDevice, `descriptor` is a live
        // MTLResidencySetDescriptor and `err` receives an `NSError *`.
        let raw_set = unsafe {
            objc::msg_send_id_err(raw_device, factory_selector, descriptor.as_raw(), &mut err)
        };
        match ObjcObject::from_retained(raw_set) {
            Some(set) => Ok(Self {
                _device: device_object,
                native: Some(set),
                allocations: Vec::new(),
                fallback_reason: None,
            }),
            None => {
                // SAFETY: Metal populated `err` when the factory failed.
                let message = unsafe { objc::nserror_message(err) };
                Ok(Self::fallback(
                    device_object,
                    format!(
                        "{}:{}: `newResidencySetWithDescriptor:error:` failed: {message}",
                        file!(),
                        line!()
                    ),
                ))
            }
        }
    }

    fn fallback(device: ObjcObject, reason: String) -> Self {
        Self {
            _device: device,
            native: None,
            allocations: Vec::new(),
            fallback_reason: Some(reason),
        }
    }

    /// Whether a real `MTLResidencySet` backs this object.
    pub fn is_native(&self) -> bool {
        self.native.is_some()
    }

    /// Exact selector and `file:line` that forced the fallback path.
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    /// Number of allocations recorded so far (both backends).
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Whether no allocation has been recorded.
    pub fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }

    /// Records a live Metal allocation, retaining it for the set's lifetime.
    ///
    /// A duplicate object is ignored so the ordered fallback list stays unique.
    fn add_allocation(&mut self, object: ObjcObject, len: usize) {
        if self
            .allocations
            .iter()
            .any(|allocation| allocation.object.as_raw() == object.as_raw())
        {
            return;
        }
        if let Some(set) = self.native.as_ref() {
            // SAFETY: `set` is a live MTLResidencySet and `object` is a live
            // MTLBuffer/MTLTexture conforming to MTLAllocation.
            unsafe {
                objc::msg_send_void_1(set.as_raw(), objc::sel(c"addAllocation:"), object.as_raw());
            }
        }
        self.allocations.push(ResidencyAllocation { object, len });
    }

    /// Adds a registered buffer, validating `len` against the registration.
    ///
    /// This takes the transport lock briefly to resolve and retain the
    /// `MTLBuffer`; a future `submit_v3` that already holds the lock must call
    /// the crate-internal `add_buffer_raw` instead.
    pub fn add_buffer(
        &mut self,
        transport: &DirectMetalTransport,
        id: VmBufferId,
        len: usize,
    ) -> Result<(), ResidencyError> {
        let (object, registered_len) = retain_registered_buffer(transport, id)?;
        if len == 0 || len > registered_len {
            return Err(ResidencyError::InvalidLength {
                id: id.0,
                requested: len,
                registered: registered_len,
            });
        }
        self.add_allocation(object, len);
        Ok(())
    }

    /// Adds a registered buffer from an already-resolved raw `MTLBuffer`.
    ///
    /// Callers must hold the transport state lock and keep the buffer alive.
    #[allow(dead_code)]
    pub(crate) fn add_buffer_raw(
        &mut self,
        buffer: objc::Id,
        len: usize,
    ) -> Result<(), ResidencyError> {
        let object = retain(buffer)?;
        self.add_allocation(object, len);
        Ok(())
    }

    /// Adds a texture created through [`TextureRegistry`].
    pub fn add_texture(
        &mut self,
        registry: &TextureRegistry,
        id: u64,
    ) -> Result<(), ResidencyError> {
        let raw = registry.raw(id).ok_or(ResidencyError::UnknownTexture(id))?;
        let object = retain(raw)?;
        self.add_allocation(object, 0);
        Ok(())
    }

    /// Adds a texture created through [`DirectMetalTransport::create_texture_2d`].
    pub fn add_vm_texture(
        &mut self,
        transport: &DirectMetalTransport,
        id: VmTextureId,
    ) -> Result<(), ResidencyError> {
        let object = transport.retain_texture_2d(id)?;
        self.add_allocation(object, 0);
        Ok(())
    }

    /// Adds a texture from an already-resolved raw `MTLTexture`.
    #[allow(dead_code)]
    pub(crate) fn add_texture_raw(&mut self, texture: objc::Id) -> Result<(), ResidencyError> {
        let object = retain(texture)?;
        self.add_allocation(object, 0);
        Ok(())
    }

    /// Adds every allocation recorded by `other`.
    ///
    /// `MTLResidencySet` does not nest sets, so this is an explicit merge: the
    /// allocations are retained by `self` and, on the native path, added to the
    /// backing set with `addAllocation:`.
    pub fn add_residency_set(&mut self, other: &ResidencySet) {
        for allocation in &other.allocations {
            let object = retain(allocation.object.as_raw());
            if let Ok(object) = object {
                self.add_allocation(object, allocation.len);
            }
        }
    }

    /// Publishes pending native additions (`commit`).
    ///
    /// A no-op on the fallback path.
    pub fn commit(&self) {
        if let Some(set) = self.native.as_ref() {
            // SAFETY: `set` is a live MTLResidencySet.
            unsafe { objc::msg_send_void_0(set.as_raw(), objc::sel(c"commit")) };
        }
    }

    /// Requests that the native set's allocations become resident.
    ///
    /// A no-op on the fallback path.
    pub fn request_residency(&self) {
        if let Some(set) = self.native.as_ref() {
            // SAFETY: `set` is a live MTLResidencySet.
            unsafe { objc::msg_send_void_0(set.as_raw(), objc::sel(c"requestResidency")) };
        }
    }

    /// Ends the residency request made by [`Self::request_residency`].
    ///
    /// A no-op on the fallback path.
    pub fn end_residency(&self) {
        if let Some(set) = self.native.as_ref() {
            // SAFETY: `set` is a live MTLResidencySet.
            unsafe { objc::msg_send_void_0(set.as_raw(), objc::sel(c"endResidency")) };
        }
    }

    /// Binds the native set to the transport's command queue.
    ///
    /// Returns `Ok(())` on the fallback path (there is nothing to bind) and an
    /// explicit [`ResidencyError::Unsupported`] when the native set exists but
    /// the queue does not implement `addResidencySet:`.
    pub fn attach_to_queue(&self, transport: &DirectMetalTransport) -> Result<(), ResidencyError> {
        let Some(set) = self.native.as_ref() else {
            return Ok(());
        };
        let queue = transport.queue_raw();
        if !unsafe { objc::msg_send_responds_to(queue, objc::sel(c"addResidencySet:")) } {
            return Err(ResidencyError::Unsupported(format!(
                "{}:{}: MTLCommandQueue does not respond to `addResidencySet:`",
                file!(),
                line!()
            )));
        }
        // SAFETY: `queue` is a live MTLCommandQueue and `set` is a live
        // MTLResidencySet created from the same device.
        unsafe {
            objc::msg_send_void_1(queue, objc::sel(c"addResidencySet:"), set.as_raw());
        }
        Ok(())
    }

    /// Declares every recorded allocation to a non-native encoder.
    ///
    /// Native residency sets do not need per-encoder calls (the bound set makes
    /// their allocations resident); the fallback issues one `useResource:usage:`
    /// per allocation, which is exactly the `O(n)` cost the fallback documents.
    #[allow(dead_code)]
    pub(crate) fn use_resources(&self, encoder: objc::Id) {
        if self.is_native() {
            return;
        }
        for allocation in &self.allocations {
            // SAFETY: `encoder` is a live compute/render/blit encoder and every
            // recorded object is a live MTLResource.
            unsafe {
                objc::msg_send_use_resource(
                    encoder,
                    allocation.object.as_raw(),
                    MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE,
                );
            }
        }
    }

    /// Whether the buffer registered at `id` is part of this residency set.
    pub fn contains_buffer(
        &self,
        transport: &DirectMetalTransport,
        id: VmBufferId,
    ) -> Result<bool, ResidencyError> {
        let (object, _len) = retain_registered_buffer(transport, id)?;
        Ok(self.contains_raw(object.as_raw()))
    }

    /// Whether the texture registered at `id` is part of this residency set.
    pub fn contains_texture(
        &self,
        registry: &TextureRegistry,
        id: u64,
    ) -> Result<bool, ResidencyError> {
        let raw = registry.raw(id).ok_or(ResidencyError::UnknownTexture(id))?;
        Ok(self.contains_raw(raw))
    }

    fn contains_raw(&self, resource: objc::Id) -> bool {
        if let Some(set) = self.native.as_ref() {
            // SAFETY: `set` is a live MTLResidencySet, `resource` is a live
            // MTLAllocation handle, and `containsAllocation:` returns BOOL.
            unsafe {
                type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id) -> u8;
                let f: Fn = std::mem::transmute(objc::msg_send_entry());
                if f(set.as_raw(), objc::sel(c"containsAllocation:"), resource) != 0 {
                    return true;
                }
            }
        }
        self.allocations
            .iter()
            .any(|allocation| allocation.object.as_raw() == resource)
    }

    /// Total bytes declared by explicit-length buffer allocations.
    pub fn reserved_bytes(&self) -> usize {
        self.allocations.iter().fold(0usize, |total, allocation| {
            total.saturating_add(allocation.len)
        })
    }
}

fn retain(raw: objc::Id) -> Result<ObjcObject, ResidencyError> {
    if raw.is_null() {
        return Err(ResidencyError::Metal(MetalError::InternalError(
            "cannot retain a null Metal object".to_string(),
        )));
    }
    let retained = unsafe { objc::msg_send_id_0(raw, objc::sel(c"retain")) };
    ObjcObject::from_retained(retained).ok_or_else(|| {
        ResidencyError::Metal(MetalError::InternalError(
            "Metal retain returned null".to_string(),
        ))
    })
}

fn retain_registered_buffer(
    transport: &DirectMetalTransport,
    id: VmBufferId,
) -> Result<(ObjcObject, usize), ResidencyError> {
    let state = transport.lock_state()?;
    let registered = state
        .buffers
        .get(&id.0)
        .ok_or(ResidencyError::UnknownBuffer(id.0))?;
    let len = registered.len;
    let raw = registered.buffer.as_raw();
    let object = retain(raw)?;
    Ok((object, len))
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use super::*;
    use crate::transport::MetalTransport;

    unsafe extern "C" {
        fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
        fn free(ptr: *mut c_void);
    }

    const ALIGN: usize = 16 * 1024;
    const BYTES: usize = 16 * 1024;

    struct PageGuard(*mut c_void);

    impl Drop for PageGuard {
        fn drop(&mut self) {
            // SAFETY: `self.0` came from `posix_memalign` and is no longer registered.
            unsafe { free(self.0) };
        }
    }

    fn register_page(transport: &DirectMetalTransport) -> (PageGuard, VmBufferId) {
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation.
        let rc = unsafe { posix_memalign(&mut raw, ALIGN, BYTES) };
        assert_eq!(rc, 0, "posix_memalign failed");
        let guard = PageGuard(raw);
        // SAFETY: The allocation is live for `BYTES` bytes and outlives the test.
        let id = unsafe { transport.register_zerocopy(raw.cast::<u8>(), BYTES) }
            .expect("register_zerocopy should succeed");
        (guard, id)
    }

    #[test]
    fn native_backend_is_reported_truthfully() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let set = ResidencySet::new(&transport).expect("residency set creation");
        match set.fallback_reason() {
            Some(reason) => {
                assert!(!set.is_native());
                assert!(
                    reason.contains("newResidencySetWithDescriptor")
                        || reason.contains("MTLResidencySetDescriptor"),
                    "unexpected fallback reason: {reason}"
                );
                eprintln!("residency fallback: {reason}");
            }
            None => {
                assert!(set.is_native());
                assert_eq!(set.allocation_count(), 0);
            }
        }
    }

    #[test]
    fn fallback_records_allocations_in_order_and_deduplicates() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let device = retain(transport.device_raw()).expect("retain device");
        let mut set =
            ResidencySet::fallback(device, "test: fallback forced by the unit test".to_string());
        assert!(!set.is_native());
        assert!(set.is_empty());

        let (_guard, id) = register_page(&transport);
        set.add_buffer(&transport, id, BYTES).expect("add buffer");
        assert_eq!(set.allocation_count(), 1);
        assert!(set.contains_buffer(&transport, id).unwrap());

        // A duplicate add must not grow the ordered list.
        set.add_buffer(&transport, id, BYTES)
            .expect("duplicate add");
        assert_eq!(set.allocation_count(), 1);

        // There is no native selector on this forced fallback, so the mutators
        // must be harmless no-ops.
        set.commit();
        set.request_residency();
        set.end_residency();

        assert!(matches!(
            set.add_buffer(&transport, id, 0),
            Err(ResidencyError::InvalidLength { .. })
        ));
        assert!(matches!(
            set.add_buffer(&transport, id, BYTES + 1),
            Err(ResidencyError::InvalidLength { .. })
        ));
        assert!(matches!(
            set.add_buffer(&transport, VmBufferId(u64::MAX), 4),
            Err(ResidencyError::UnknownBuffer(_))
        ));

        // Merging another set retains its allocations without duplication.
        let device = retain(transport.device_raw()).expect("retain device");
        let mut other = ResidencySet::fallback(device, "test: merge source".to_string());
        other.add_buffer(&transport, id, BYTES).expect("add buffer");
        set.add_residency_set(&other);
        assert_eq!(set.allocation_count(), 1);

        transport
            .unregister_buffer(id)
            .expect("unregister_buffer should succeed");
    }

    #[test]
    fn fallback_attaches_and_uses_resources_without_panicking() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let device = retain(transport.device_raw()).expect("retain device");
        let set = ResidencySet::fallback(device, "test: encoder fallback".to_string());
        // The fallback has no native set, so queue attachment is a no-op.
        set.attach_to_queue(&transport)
            .expect("fallback attach is a no-op");

        // A real encoder exercises the O(n) useResource loop with no allocations
        // (the loop body must not run) and never panics.
        let queue = transport.queue_raw();
        let cb = unsafe { objc::msg_send_id_0(queue, objc::sel(c"commandBuffer")) };
        assert!(!cb.is_null());
        let encoder = unsafe { objc::msg_send_id_0(cb, objc::sel(c"computeCommandEncoder")) };
        assert!(!encoder.is_null());
        set.use_resources(encoder);
        unsafe {
            objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }
    }
}
