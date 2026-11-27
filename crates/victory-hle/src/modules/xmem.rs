//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, ERROR_INVALID_ADDRESS, HANDLE, INVALID_HANDLE_VALUE,
    SetLastError,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_DECOMMIT,
    MEM_EXTENDED_PARAMETER, MEM_PRESERVE_PLACEHOLDER, MEM_PRIVATE, MEM_RELEASE,
    MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEM_RESET,
    MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS,
    PAGE_READWRITE, PAGE_WRITECOPY,
    SEC_COMMIT, SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2,
    VirtualFree as HostVirtualFree, VirtualProtect, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

#[cfg(test)]
mod e2e;
mod share;

/// Installs the native backend's guest-range resolver when this session runs
/// the Metal backend.
///
/// Called from console device creation, so a session that never creates a
/// shared physical-page section can still resolve guest bindings. The install
/// is idempotent, and the D3D12 reference backend never installs it.
pub(crate) fn install_share_resolver() {
    share::install_if_metal();
}

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// `MEM_RESET_UNDEFINED`, beside the host's `MEM_RESET`: windows-sys does
/// not name it.
const MEM_RESET_UNDEFINED: u32 = 0x0100_0000;
/// The protections whose pages a decommit or reset may zero through the view.
const WRITABLE: u32 = PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (Grand Theft
/// Auto VI's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let requested_kind = kind;
    let requested_protect = protect;
    // The console has no separate virtual-free export: a title releases,
    // decommits or resets a region by passing the matching allocation type
    // here, as it would to `VirtualFree`. A region this module handed out is
    // ours to act on; every other address keeps the host's answer.
    if kind & MEM_RELEASE != 0 {
        if virtual_release(base as usize) {
            trace_virtual(size, requested_kind, requested_protect, base, base, false, "released");
            return base;
        }
        // SAFETY: the title's own release call, unchanged.
        if !base.is_null() && unsafe { HostVirtualFree(base, 0, MEM_RELEASE) } != 0 {
            trace_virtual(size, requested_kind, requested_protect, base, base, false, "released by the host");
            return base;
        }
        return virtual_failure("release names no tracked reservation", ERROR_INVALID_ADDRESS);
    }
    // `MEM_DECOMMIT` shares its value with `MEM_REPLACE_PLACEHOLDER`, which a
    // title also passes to split a 64 KiB placeholder: only a call that
    // names no reserve or commit is a decommit.
    if kind & MEM_DECOMMIT != 0 && kind & (MEM_RESERVE | MEM_COMMIT) == 0 {
        if virtual_decommit(base as usize, size) {
            trace_virtual(size, requested_kind, requested_protect, base, base, false, "decommitted");
            return base;
        }
        // SAFETY: the title's own decommit call, unchanged.
        if !base.is_null() && unsafe { HostVirtualFree(base, size, MEM_DECOMMIT) } != 0 {
            trace_virtual(size, requested_kind, requested_protect, base, base, false, "decommitted by the host");
            return base;
        }
        return virtual_failure("decommit leaves its reservation", ERROR_INVALID_ADDRESS);
    }
    if kind & (MEM_RESET | MEM_RESET_UNDEFINED) != 0 {
        if virtual_reset(base as usize, size, kind & MEM_RESET_UNDEFINED != 0) {
            trace_virtual(size, requested_kind, requested_protect, base, base, false, "reset");
            return base;
        }
        // SAFETY: the title's own reset call; the parameter points nowhere.
        if !base.is_null()
            && unsafe {
                VirtualAlloc2(
                    GetCurrentProcess(),
                    base,
                    size,
                    MEM_RESET,
                    PAGE_NOACCESS,
                    std::ptr::null_mut(),
                    0,
                )
            } != std::ptr::null_mut()
        {
            trace_virtual(size, requested_kind, requested_protect, base, base, false, "reset by the host");
            return base;
        }
        return virtual_failure("reset names no committed range", ERROR_INVALID_ADDRESS);
    }
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let requested = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    let mut protect = requested;
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (Grand Theft Auto VI's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            trace_virtual(size, requested_kind, requested_protect, base, base, false, "placeholder split");
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
    }
    // ADR-045: exact reserve/commit over host-visible backing. A reservation
    // owns address space only; every commit maps a view of a share file over
    // exactly the pages it names, so the same bytes are readable at the guest
    // VA and by the native host. A reserve alone registers nothing and stays
    // `MEM_RESERVE` to the title (that is what its own assertion checks).
    // Any request this path cannot serve falls back to the host allocation
    // below, exactly as before, so a title's working call never turns into a
    // null one because of the share path.
    if share::configured() && !placeholder {
        if let Some(rounded) = size.checked_next_multiple_of(PAGE) {
            let served = if kind & MEM_COMMIT != 0 {
                virtual_commit(
                    base,
                    rounded,
                    flags,
                    requested,
                    &mut requirements,
                    constrained,
                    highest,
                )
            } else if let Some(at) = virtual_reserve(
                base,
                rounded,
                flags,
                requested,
                &mut requirements,
                constrained,
                highest,
                false,
            ) {
                Served::At(at, false)
            } else {
                Served::Unserved
            };
            match served {
                Served::At(at, registered) => {
                    trace_virtual(
                        size,
                        requested_kind,
                        requested_protect,
                        base,
                        at,
                        registered,
                        "virtual region",
                    );
                    return at;
                }
                Served::Refused(reason) => {
                    return virtual_failure(reason, ERROR_INVALID_ADDRESS);
                }
                Served::Unserved => {}
            }
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    if constrained && victory_gpu::fault_tracking() {
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); a placeholder commit has a
        // share range instead and needs no watch. Nor is anything under
        // Rosetta, where Wine's watch faults on first writes Rosetta cannot
        // resume (`victory_gpu::fault_tracking`).
        kind |= MEM_WRITE_WATCH;
    }
    let allocate = |parameter: &mut MEM_EXTENDED_PARAMETER| unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    let mut at = allocate(&mut parameter);
    // A 32-bit request that found no room below 4 GiB: the guards there
    // make way for it.
    if at.is_null() && constrained && highest < FOUR_GIB {
        at = low_guards::retry(size, || allocate(&mut parameter));
    }
    if at.is_null() {
        let error = std::io::Error::last_os_error();
        if kind & MEM_COMMIT != 0 {
            commit_refused(size, "virtual memory", error);
        } else {
            tracing::warn!(
                size,
                kind = format_args!("{kind:#x}"),
                flags = format_args!("{flags:#x}"),
                %error,
                "XMemVirtualAlloc refused"
            );
        }
    }
    let mut registered = false;
    if !at.is_null()
        && victory_gpu::alias_buffers()
        && kind & MEM_COMMIT != 0
        && kind & MEM_RESERVE_PLACEHOLDER == 0
        && protect == PAGE_READWRITE
        && (at as usize).is_multiple_of(PAGE)
        && size >= PAGE
        && size.is_multiple_of(PAGE)
    {
        registered = victory_gpu::alias_region(at as usize, size, false);
    }
    trace_virtual(
        size,
        requested_kind,
        requested_protect,
        base,
        at,
        registered,
        "host memory",
    );
    at
}

/// One bounded debug line per `XMemVirtualAlloc`, whatever the call did: the
/// requested kind and protection, the returned base and whether a
/// host-visible range was registered for it. Debug level, so the default
/// boot log stays quiet and a `RUST_LOG=debug` boot can diff the title's
/// whole memory sequence.
fn trace_virtual(
    size: usize,
    kind: u32,
    protect: u32,
    base: *mut c_void,
    at: *mut c_void,
    registered: bool,
    note: &'static str,
) {
    tracing::debug!(
        size,
        kind = format_args!("{kind:#x}"),
        protect = format_args!("{protect:#x}"),
        base = format_args!("{base:p}"),
        at = format_args!("{at:p}"),
        registered,
        note,
        "XMemVirtualAlloc"
    );
}


/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (Grand Theft Auto VI's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        // ADR-045: a virtual region's state follows its commits, not the
        // placeholder that holds its address space. A committed page reports
        // `MEM_COMMIT` with the protection it asked for; an uncommitted gap
        // reports `MEM_RESERVE`/`PAGE_NOACCESS`. A page is never reported
        // committed unless its bytes are really mapped.
        if let Some(state) = reservation_state(address as usize) {
            info.host.BaseAddress = state.base as *mut c_void;
            info.host.AllocationBase = state.allocation_base as *mut c_void;
            info.host.AllocationProtect = state.allocation_protect;
            info.host.RegionSize = state.size;
            info.host.State = state.state;
            info.host.Protect = state.protect;
            info.host.Type = MEM_PRIVATE;
        }
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (Grand Theft Auto VI does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// ADR-036: the range an allocation landed in, never its value (addresses are
/// ASLR-dependent, so a trace may compare classes but not pointers).
fn address_class(pointer: *mut c_void) -> u64 {
    let at = pointer as usize as u64;
    match at {
        0 => 0,
        _ if at < 1 << 32 => 1,
        _ if at < 1 << 40 => 2,
        _ => 3,
    }
}

/// One bounded trace line per allocation when `VICTORY_GUEST_ABI_TRACE` is on.
/// Values: `[size, attributes, address_class, used_title_hook]`. A title that
/// asserts on its own allocator's return is then visible as class 0.
fn trace_alloc(size: usize, attributes: u64, pointer: *mut c_void, hook: usize) {    victory_gpu::native_metal::abi_trace::record(
        "XMemAlloc",
        &[
            size as u64,
            attributes,
            address_class(pointer),
            u64::from(hook != 0),
        ],
    );
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    // ADR-036: the request is recorded before the title's hook runs, so an
    // allocator that faults on its own table still leaves the size, the
    // attributes and which allocator answered visible.
    victory_gpu::native_metal::abi_trace::record(
        "XMemAllocReq",
        &[
            size as u64,
            attributes,
            u64::from(hooks()[0] != 0),
        ],
    );
    match hooks()[0] {
        0 => {
            let pointer = XMemAllocDefault(size, attributes);
            trace_alloc(size, attributes, pointer, 0);
            pointer
        }
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            let pointer = alloc(size, attributes);
            trace_alloc(size, attributes, pointer, hook);
            pointer
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
    victory_gpu::native_metal::abi_trace::record(
        "XMemFree",
        &[address_class(address), attributes],
    );
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (Grand Theft Auto VI uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        tracing::warn!(size, "XMemAllocDefault: the host heap is out of memory");
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            // A share-backed virtual region is ours to release; every other
            // reservation stays the host's.
            if !virtual_release(address as usize) {
                victory_gpu::unalias_region(address as usize);
                HostVirtualFree(address, 0, MEM_RELEASE);
            }
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

pub(super) struct Section {
    handle: HANDLE,
    /// The allocation's real size: what the guest may map and resolve. A
    /// share file is rounded up to whole pages, so this can be smaller than
    /// [`Section::map_size`]; the difference is padding the guest never sees.
    size: usize,
    /// The length the section maps, always a non-zero multiple of 64 KiB
    /// pages for a share-backed section. Equal to `size` otherwise.
    map_size: usize,
    /// Optional shared CPU/GPU backing. The file still supplies Wine's
    /// placeholder-view metadata.
    #[cfg(feature = "diagnostics")]
    shared: Option<victory_gpu::PhysicalHeap>,
    /// The opt-in host-visible file behind this section (`share`), if the
    /// provider was configured and succeeded. Pagefile and diagnostic
    /// sections have none, and never claim one. Read by
    /// `share::resolve_guest_range`.
    #[allow(dead_code)]
    share: Option<Arc<share::Backing>>,
    /// Our own view of the whole section, mapped the first time a
    /// DirectStorage request writes its pages and kept while the section
    /// lives (0: it could not be mapped). A view per request cost a map
    /// and an unmap, and under Wine each holds the process's one
    /// virtual-memory lock across a wineserver request: every page fault
    /// and protection change on other threads waits for it (#50).
    whole: OnceLock<usize>,
}

// SAFETY: a section handle may be used from any thread, and the view is
// plain memory.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    /// The address of our view of the whole section, mapped the first time;
    /// `None` if the host would not map it.
    fn whole(&self) -> Option<usize> {
        #[cfg(feature = "diagnostics")]
        if let Some(shared) = &self.shared {
            return Some(shared.cpu_address());
        }
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let at = *self.whole.get_or_init(|| {
            // SAFETY: our section; the view is ours until the section drops.
            unsafe { MapViewOfFile(self.handle, FILE_MAP_WRITE, 0, 0, self.map_size) }.Value
                as usize
        });
        (at != 0).then_some(at)
    }
}

/// A section of `size` bytes for an allocation of physical pages: a file the
/// opt-in share provider names when `VICTORY_METAL_SHARE_DIR` is set (in
/// every build, diagnostics included), the pagefile when the provider is
/// unconfigured or genuinely fails, or in tests a file a synthetic caller
/// provides. A failure is logged with its category and still falls back
/// explicitly: it is never reported as a shared section.
fn new_section(size: usize) -> Option<Arc<Section>> {
    #[cfg(test)]
    if let Some(section) = tests::file_section(size) {
        return Some(section);
    }
    // `VICTORY_PHYSICAL_TILES` is an independent GPU-heap opt-in: a section
    // takes the share path whether or not that switch is on, and a share
    // section has no GPU host heap of its own.
    match share::section(size) {
        Ok(section) => {
            tracing::debug!(
                size,
                map_size = section.map_size,
                live = share::live(),
                "host-visible XMem section"
            );
            return Some(section);
        }
        Err(share::Failure::Unconfigured) => {}
        Err(error) => tracing::warn!(
            size,
            %error,
            "host-visible XMem backing failed; keeping the pagefile section"
        ),
    }
    pagefile_section(size)
}

/// A section over `file`: its pages are the file's pages. `share` records a
/// provider backing the file, if any. The file must outlive every view of
/// the section (callers keep it, or a `share::Backing`, alive).
#[cfg(test)]
fn file_section(
    file: &std::fs::File,
    size: usize,
    share: Option<Arc<share::Backing>>,
) -> Option<Arc<Section>> {
    file_section_sized(file, size, size, share)
}

/// A section over `file` reporting `size` to the guest but mapping
/// `map_size`: the provider rounds an allocation up to whole pages, and the
/// extra tail must stay padding the guest cannot resolve.
fn file_section_sized(
    file: &std::fs::File,
    size: usize,
    map_size: usize,
    share: Option<Arc<share::Backing>>,
) -> Option<Arc<Section>> {
    debug_assert!(map_size >= size && map_size.is_multiple_of(PAGE));
    use std::os::windows::io::AsRawHandle;
    // SAFETY: plain file-backed section creation; the caller keeps the file
    // open while the section and its views live.
    let handle = unsafe {
        CreateFileMappingW(
            file.as_raw_handle() as HANDLE,
            std::ptr::null(),
            PAGE_READWRITE,
            (map_size as u64 >> 32) as u32,
            map_size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        return None;
    }
    Some(Arc::new(Section {
        handle,
        size,
        map_size,
        // The file is the backing, not the GPU's host heap.
        #[cfg(feature = "diagnostics")]
        shared: None,
        share,
        whole: OnceLock::new(),
    }))
}

/// A pagefile-backed section: the documents' physical pages, committed when
/// first mapped or written.
fn pagefile_section(size: usize) -> Option<Arc<Section>> {
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        return None;
    }
    Some(Arc::new(Section {
        handle,
        size,
        map_size: size,
        #[cfg(feature = "diagnostics")]
        shared: victory_gpu::physical_heap(size),
        share: None,
        whole: OnceLock::new(),
    }))
}

// ------------------------------------------------ virtual regions (ADR-045)

/// One committed run of a [`Reservation`]: `len` bytes starting `offset`
/// bytes into the reservation, mapped by a view of `section` that starts at
/// `file_offset` inside it.
///
/// Each commit gets its own section, sized to that commit: a multi-GiB
/// reservation that commits one page creates a one-page file, not a
/// multi-GiB one. A protection change splits a run while keeping its section
/// and file offset, so the bytes stay where they were.
#[derive(Clone)]
pub(super) struct Committed {
    pub(super) len: usize,
    /// The guest protection the commit asked for.
    pub(super) protect: u32,
    /// The section whose bytes this run is.
    pub(super) section: Arc<Section>,
    /// Where this run starts inside `section`; always a whole page and
    /// `file_offset + len <= section.map_size`.
    pub(super) file_offset: usize,
}

/// A virtual region `XMemVirtualAlloc` handed out (ADR-045): the whole
/// reservation is one host placeholder, so a commit can map a view over
/// exactly the pages it names. A plain `MEM_RESERVE` owns no bytes at all.
pub(super) struct Reservation {
    pub(super) base: usize,
    /// The guest-visible length: the requested size rounded up to a whole
    /// 64 KiB page, as the console rounds.
    pub(super) size: usize,
    /// The committed runs, by offset from `base`.
    pub(super) commits: BTreeMap<usize, Committed>,
    /// The guest flags the reservation was requested with (diagnostics).
    #[allow(dead_code)]
    pub(super) flags: u64,
    /// What the reservation reports as `AllocationProtect`.
    pub(super) allocation_protect: u32,
    /// The address space is a host placeholder, so a commit can replace
    /// exactly its pages with a section view. A host that refuses placeholders
    /// keeps a plain reservation, whose commits stay on the host path.
    pub(super) placeholder: bool,
    /// A 64 KiB-page reservation for `XMemMapPhysicalPages`: its pages are
    /// shown by physical-page views, so a plain gap is "not mapped" rather
    /// than "reserved but uncommitted".
    pub(super) physical_pages: bool,
}

impl Reservation {
    /// The committed run holding `offset`, if any.
    pub(super) fn committed_at(&self, offset: usize) -> Option<(usize, &Committed)> {
        let (start, run) = self.commits.range(..=offset).next_back()?;
        (offset < start.checked_add(run.len)?).then_some((*start, run))
    }

    /// The runs of `[offset, offset + len)` no commit covers, in order.
    fn gaps(&self, offset: usize, len: usize) -> Vec<(usize, usize)> {
        let end = offset.saturating_add(len);
        let mut gaps = Vec::new();
        let mut cursor = offset;
        while cursor < end {
            match self.committed_at(cursor) {
                Some((start, run)) => cursor = start.saturating_add(run.len).min(end),
                None => {
                    let next = self
                        .commits
                        .range(cursor..)
                        .next()
                        .map_or(end, |(&start, _)| start.min(end));
                    let stop = next.max(cursor);
                    gaps.push((cursor, stop - cursor));
                    cursor = stop;
                }
            }
        }
        gaps
    }

    /// The committed pieces of `[offset, offset + len)` as
    /// `(start, length, protect)`, in order.
    fn pieces(&self, offset: usize, len: usize) -> Vec<(usize, usize, u32)> {
        let end = offset.saturating_add(len);
        self.commits
            .range(..end)
            .filter(|(start, run)| start.saturating_add(run.len) > offset)
            .map(|(&start, run)| {
                let from = offset.max(start);
                let to = (start + run.len).min(end);
                (from, to - from, run.protect)
            })
            .collect()
    }

    /// Splits the recorded runs around `[offset, offset + len)` and returns
    /// each intersection, so the caller acts on exactly the bytes it names
    /// without mistaking a partly covered run for the whole of it. Each piece
    /// keeps its section and the right file offset.
    fn partition(&mut self, offset: usize, len: usize) -> Vec<(usize, usize, Committed)> {
        let end = offset.saturating_add(len);
        let affected: Vec<(usize, Committed)> = self
            .commits
            .range(..end)
            .filter(|(start, run)| start.saturating_add(run.len) > offset)
            .map(|(&start, run)| (start, run.clone()))
            .collect();
        let mut pieces = Vec::with_capacity(affected.len());
        for (start, run) in affected {
            let stop = start + run.len;
            let (from, to) = (offset.max(start), end.min(stop));
            self.commits.remove(&start);
            if start < from {
                self.commits.insert(
                    start,
                    Committed {
                        len: from - start,
                        ..run.clone()
                    },
                );
            }
            if stop > to {
                self.commits.insert(
                    to,
                    Committed {
                        len: stop - to,
                        file_offset: run.file_offset + (to - start),
                        ..run.clone()
                    },
                );
            }
            if to > from {
                let piece = Committed {
                    len: to - from,
                    file_offset: run.file_offset + (from - start),
                    ..run
                };
                self.commits.insert(from, piece.clone());
                pieces.push((from, to - from, piece));
            }
        }
        pieces
    }

    /// Sets the protection recorded for `[offset, offset + len)`.
    fn reprotect(&mut self, offset: usize, len: usize, protect: u32) {
        for (from, length, run) in self.partition(offset, len) {
            self.commits.insert(
                from,
                Committed {
                    len: length,
                    protect,
                    ..run
                },
            );
        }
    }
}

/// What `XMemVirtualQuery` reports for one address of a virtual region.
struct ReservationState {
    base: usize,
    allocation_base: usize,
    allocation_protect: u32,
    size: usize,
    state: u32,
    protect: u32,
}

/// What the share-backed path did with an `XMemVirtualAlloc` request.
enum Served {
    /// Served here: the returned base and whether a host-visible range was
    /// registered for it.
    At(*mut c_void, bool),
    /// Not this path's to serve: the caller keeps the host allocation,
    /// exactly as it did before ADR-045.
    Unserved,
    /// Ours to refuse: a commit inside a region that already holds committed
    /// bytes could not be mapped, so returning a pointer would hand the title
    /// pages that are not there.
    Refused(&'static str),
}

impl Physical {
    /// The virtual region holding `at`, if any.
    pub(super) fn reservation_at(&self, at: usize) -> Option<(usize, &Reservation)> {
        self.reservations
            .range(..=at)
            .next_back()
            .filter(|(base, region)| at < base.saturating_add(region.size))
            .map(|(&base, region)| (base, region))
    }

    /// The virtual region holding `at`, mutably.
    fn reservation_mut(&mut self, at: usize) -> Option<(usize, &mut Reservation)> {
        let base = self.reservation_at(at).map(|(base, _)| base)?;
        self.reservations
            .get_mut(&base)
            .map(|region| (base, region))
    }
}

/// Bounded, named counters for the virtual region path (ADR-045): counts and
/// bytes only, never contents.
static VIRTUAL_RESERVES: AtomicU64 = AtomicU64::new(0);
static VIRTUAL_COMMITS: AtomicU64 = AtomicU64::new(0);
static VIRTUAL_DECOMMITS: AtomicU64 = AtomicU64::new(0);
static VIRTUAL_RELEASES: AtomicU64 = AtomicU64::new(0);
static VIRTUAL_RESETS: AtomicU64 = AtomicU64::new(0);
static VIRTUAL_COMMITTED_BYTES: AtomicU64 = AtomicU64::new(0);
static VIRTUAL_REFUSALS: AtomicU64 = AtomicU64::new(0);
static LAST_REFUSAL: Mutex<&'static str> = Mutex::new("");

/// The virtual region counters, for a boot trace and for tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct VirtualCounters {
    pub(crate) reserves: u64,
    pub(crate) commits: u64,
    pub(crate) decommits: u64,
    pub(crate) releases: u64,
    pub(crate) resets: u64,
    pub(crate) committed_bytes: u64,
    pub(crate) refusals: u64,
}

#[allow(dead_code)]
pub(crate) fn virtual_counters() -> VirtualCounters {
    VirtualCounters {
        reserves: VIRTUAL_RESERVES.load(Ordering::Relaxed),
        commits: VIRTUAL_COMMITS.load(Ordering::Relaxed),
        decommits: VIRTUAL_DECOMMITS.load(Ordering::Relaxed),
        releases: VIRTUAL_RELEASES.load(Ordering::Relaxed),
        resets: VIRTUAL_RESETS.load(Ordering::Relaxed),
        committed_bytes: VIRTUAL_COMMITTED_BYTES.load(Ordering::Relaxed),
        refusals: VIRTUAL_REFUSALS.load(Ordering::Relaxed),
    }
}

/// The last refusal's reason: a fixed, bounded phrase (at most 120 chars).
#[allow(dead_code)]
pub(crate) fn last_refusal() -> &'static str {
    *LAST_REFUSAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Records why a virtual operation was refused and answers null, with the
/// matching Windows error.
fn virtual_failure(reason: &'static str, error: u32) -> *mut c_void {
    debug_assert!(reason.len() <= 120, "a refusal reason is bounded");
    *LAST_REFUSAL.lock().unwrap_or_else(|e| e.into_inner()) = reason;
    VIRTUAL_REFUSALS.fetch_add(1, Ordering::Relaxed);
    tracing::debug!(reason, error, "XMemVirtualAlloc refused");
    // SAFETY: writes this thread's last-error value, as the host API would.
    unsafe { SetLastError(error) };
    std::ptr::null_mut()
}

/// Reserves `size` bytes of address space (one host placeholder) and records
/// it. `None` when the host refuses, so the caller can keep its host path.
fn virtual_reserve(
    base: *mut c_void,
    size: usize,
    flags: u64,
    allocation_protect: u32,
    requirements: &mut MEM_ADDRESS_REQUIREMENTS,
    constrained: bool,
    highest: usize,
    physical_pages: bool,
) -> Option<*mut c_void> {
    // A base inside a region this module already reserved is a split of it:
    // the address space is reserved already, so there is nothing to do.
    if !base.is_null() {
        let physical = physical();
        if let Some((region_base, region)) = physical.reservation_at(base as usize)
            && (base as usize)
                .checked_add(size)
                .is_some_and(|end| end <= region_base + region.size)
        {
            return Some(base);
        }
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (requirements as *mut MEM_ADDRESS_REQUIREMENTS).cast();
    let mut allocate = |placeholder: bool| {
        // SAFETY: the title's own base and size; the parameter points at
        // `requirements`, which outlives every call here.
        unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                base,
                size,
                MEM_RESERVE
                    | if placeholder {
                        MEM_RESERVE_PLACEHOLDER
                    } else {
                        0
                    },
                PAGE_NOACCESS,
                if constrained {
                    &raw mut parameter
                } else {
                    std::ptr::null_mut()
                },
                constrained.into(),
            )
        }
    };
    let mut at = allocate(true);
    let mut placeholder = true;
    if at.is_null() {
        // A host without placeholder support keeps the plain reservation the
        // previous code made; its commits then stay on the host path.
        placeholder = false;
        at = allocate(false);
    }
    if at.is_null() && constrained && highest < FOUR_GIB {
        at = low_guards::retry(size, || allocate(placeholder));
    }
    if at.is_null() {
        return None;
    }
    let base = at as usize;
    physical().reservations.insert(
        base,
        Reservation {
            base,
            size,
            commits: BTreeMap::new(),
            flags,
            allocation_protect,
            placeholder,
            physical_pages,
        },
    );
    VIRTUAL_RESERVES.fetch_add(1, Ordering::Relaxed);
    tracing::debug!(
        size,
        at = format_args!("{base:#x}"),
        placeholder,
        "XMemVirtualAlloc reserved"
    );
    Some(at)
}

/// A commit: maps the requested pages over their placeholders and records
/// them, or a fresh reserve-and-commit when `base` is null.
fn virtual_commit(
    base: *mut c_void,
    size: usize,
    flags: u64,
    protect: u32,
    requirements: &mut MEM_ADDRESS_REQUIREMENTS,
    constrained: bool,
    highest: usize,
) -> Served {
    if size == 0 {
        return Served::Refused("a commit size is zero");
    }
    if base.is_null() {
        let Some(at) = virtual_reserve(
            std::ptr::null_mut(),
            size,
            flags,
            protect,
            requirements,
            constrained,
            highest,
            false,
        ) else {
            return Served::Unserved;
        };
        if commit_pages(at as usize, size, protect) {
            return Served::At(at, true);
        }
        // Nothing was committed: give the address space back and let the host
        // allocation serve the request exactly as it did before.
        virtual_release(at as usize);
        return Served::Unserved;
    }
    let at = base as usize;
    // Win32 rounds a commit's address down and its end up to whole pages; the
    // returned base is the rounded one.
    let start = at & !(PAGE - 1);
    let Some(len) = at
        .checked_add(size)
        .and_then(|end| end.checked_next_multiple_of(PAGE))
        .and_then(|end| end.checked_sub(start))
    else {
        return Served::Refused("commit size overflows");
    };
    // A commit only belongs inside a reservation this module made (the host
    // cannot turn a placeholder into committed memory). Anything else keeps
    // the host path, exactly as before.
    let tracked = {
        let physical = physical();
        physical.reservation_at(at).map(|(base, region)| {
            let end = base.saturating_add(region.size);
            let shown = physical
                .views
                .range(..end)
                .any(|(&view, (size, _))| view.saturating_add(*size) > base);
            (
                base,
                !region.commits.is_empty(),
                shown,
                region.placeholder,
                region.physical_pages,
            )
        })
    };
    let Some((region_base, has_commits, shown, placeholder, physical_pages)) = tracked else {
        return Served::Unserved;
    };
    if commit_pages(start, len, protect) {
        return Served::At(start as *mut c_void, true);
    }
    // The share path could not map these pages. If the region still holds no
    // committed bytes, turn it back into a plain host reservation and let the
    // host commit: the title gets exactly the memory it got before, and the
    // range is a host range with no share backing rather than a silently
    // reserved one.
    if !has_commits && !shown && placeholder && !physical_pages && downgrade(region_base) {
        return Served::Unserved;
    }
    Served::Refused("a commit could not be mapped over its reservation")
}

/// Turns a placeholder reservation that has committed nothing back into a
/// plain host reservation, so the host can commit it as it did before.
fn downgrade(base: usize) -> bool {
    let size = {
        let mut physical = physical();
        let Some(region) = physical.reservations.remove(&base) else {
            return false;
        };
        if !region.commits.is_empty() {
            physical.reservations.insert(base, region);
            return false;
        }
        region.size
    };
    // Nothing is committed, so no bytes are lost: free the placeholder and
    // take a plain reservation for the same address space.
    // SAFETY: the placeholder this module made.
    if unsafe { HostVirtualFree(base as *mut c_void, 0, MEM_RELEASE) } == 0 {
        tracing::warn!(
            at = format_args!("{base:#x}"),
            error = %std::io::Error::last_os_error(),
            "dropping a placeholder reservation failed"
        );
        return false;
    }
    // SAFETY: the range is free again; the base was ours a moment ago.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base as *const c_void,
            size,
            MEM_RESERVE,
            PAGE_NOACCESS,
            std::ptr::null_mut(),
            0,
        )
    };
    if at.is_null() {
        tracing::warn!(
            size,
            at = format_args!("{base:#x}"),
            error = %std::io::Error::last_os_error(),
            "re-reserving a virtual region without a placeholder failed"
        );
        return false;
    }
    tracing::debug!(
        size,
        at = format_args!("{base:#x}"),
        "XMemVirtualAlloc reservation without placeholders"
    );
    true
}

/// The section backing one committed run: the share provider when it serves
/// the size, else the pagefile. The pagefile fallback keeps the run readable
/// and writable but cannot be named to the host
/// ([`share::ResolveError::Pagefile`]).
fn commit_backing(size: usize) -> Option<Arc<Section>> {
    #[cfg(test)]
    if let Some(section) = tests::file_section(size) {
        return Some(section);
    }
    match share::section(size) {
        Ok(section) => return Some(section),
        Err(share::Failure::Unconfigured) => {}
        Err(error) => tracing::warn!(
            size,
            %error,
            "host-visible commit backing failed; keeping the pagefile"
        ),
    }
    pagefile_section(size)
}

/// The placeholder holding `[at, at + len)`, as `(base, end)`. Windows splits a
/// placeholder only at its edges, so a middle run takes two splits; the caller
/// needs the bounds to know which edges to cut.
fn placeholder_bounds(at: usize, len: usize) -> Option<(usize, usize)> {
    // SAFETY: a zeroed `MEMORY_BASIC_INFORMATION` is valid.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: reads this process's address-space bookkeeping, not `at` itself.
    let queried = unsafe {
        VirtualQueryEx(
            GetCurrentProcess(),
            at as *const c_void,
            &mut info,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    if queried == 0 || info.State != MEM_RESERVE {
        return None;
    }
    let base = info.BaseAddress as usize;
    let end = base.checked_add(info.RegionSize)?;
    let stop = at.checked_add(len)?;
    (at >= base && stop <= end).then_some((base, end))
}

/// Maps `[at, at + len)` of one reservation: a view of `section` at
/// `file_offset`, over the placeholder there.
fn map_commit(section: &Section, at: usize, file_offset: usize, len: usize) -> bool {
    // Carve exactly `[at, at + len)` out of its placeholder first: a head split
    // (when the run starts inside the placeholder) and then the run itself,
    // each preserving both halves. A single call only splits at one edge, so a
    // middle run would not become a replaceable placeholder.
    let Some((base, end)) = placeholder_bounds(at, len) else {
        tracing::warn!(
            at = format_args!("{at:#x}"),
            len,
            "a committed virtual run is not inside one placeholder"
        );
        return false;
    };
    unsafe {
        if at > base {
            // SAFETY: the placeholder's head; both halves stay placeholders.
            HostVirtualFree(
                base as *mut c_void,
                at - base,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
        if at + len < end {
            // SAFETY: the run itself, preserved as the placeholder to replace;
            // its tail becomes a placeholder of its own.
            HostVirtualFree(
                at as *mut c_void,
                len,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
    }
    // A data view cannot be mapped with every protection a commit may ask for,
    // so map it read-write and apply the protection afterwards.
    // SAFETY: `[at, at + len)` is now one placeholder of this reservation.
    let view = unsafe {
        MapViewOfFile3(
            section.handle,
            GetCurrentProcess(),
            at as *const c_void,
            file_offset as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != at {
        tracing::warn!(
            at = format_args!("{at:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping a committed virtual run failed"
        );
        return false;
    }
    true
}

/// Returns one committed run to its placeholder, keeping the reservation.
fn unmap_commit(at: usize, len: usize) -> bool {
    // SAFETY: the view mapped at `at`; it becomes a placeholder again.
    let unmapped = unsafe {
        UnmapViewOfFile2(
            GetCurrentProcess(),
            MEMORY_MAPPED_VIEW_ADDRESS {
                Value: at as *mut c_void,
            },
            MEM_PRESERVE_PLACEHOLDER,
        )
    };
    if unmapped == 0 {
        tracing::warn!(
            at = format_args!("{at:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "unmapping a committed virtual run failed"
        );
        return false;
    }
    victory_gpu::title_memory_remapped(at, len);
    true
}

/// The parts of `[from, from + len)` no physical-page view already shows:
/// those pages are committed already and are not ours to map.
fn without_views(physical: &Physical, from: usize, len: usize) -> Vec<(usize, usize)> {
    let end = from.saturating_add(len);
    let mut runs = vec![(from, len)];
    for (&view_base, (view_len, _)) in physical.views.range(..end) {
        let view_end = view_base.saturating_add(*view_len);
        if view_end <= from {
            continue;
        }
        let mut next = Vec::with_capacity(runs.len() + 1);
        for (run_at, run_len) in runs {
            let run_end = run_at.saturating_add(run_len);
            if view_end <= run_at || view_base >= run_end {
                next.push((run_at, run_len));
                continue;
            }
            if run_at < view_base {
                next.push((run_at, view_base - run_at));
            }
            if view_end < run_end {
                next.push((view_end, run_end - view_end));
            }
        }
        runs = next;
    }
    runs
}

/// Maps and records `[at, at + size)` inside its reservation.
fn commit_pages(at: usize, size: usize, protect: u32) -> bool {
    if !at.is_multiple_of(PAGE) || size == 0 {
        virtual_failure("a commit is not page aligned", ERROR_INVALID_ADDRESS);
        return false;
    }
    // Plan under the lock: the runs no committed view and no physical-page
    // view already covers, and the committed pieces whose protection changes.
    let plan = {
        let physical = physical();
        let Some((base, region)) = physical.reservation_at(at) else {
            virtual_failure("commit names no reservation", ERROR_INVALID_ADDRESS);
            return false;
        };
        if !region.placeholder {
            virtual_failure(
                "the reservation holds no placeholder for a committed view",
                ERROR_INVALID_ADDRESS,
            );
            return false;
        }
        let offset = at - base;
        let Some(end) = offset.checked_add(size) else {
            virtual_failure("commit size overflows", ERROR_INVALID_ADDRESS);
            return false;
        };
        if end > region.size {
            virtual_failure("commit past the reservation", ERROR_INVALID_ADDRESS);
            return false;
        }
        let mut runs = Vec::new();
        for (gap_offset, gap_len) in region.gaps(offset, size) {
            runs.extend(without_views(&physical, base + gap_offset, gap_len));
        }
        (base, runs, region.pieces(offset, size))
    };
    let (base, runs, pieces) = plan;
    // Map one section per run: sized to the run, so a small commit into a huge
    // reservation costs a small file.
    let mut mapped: Vec<(usize, usize, Committed)> = Vec::with_capacity(runs.len());
    for (run_at, run_len) in runs {
        let Some(section) = commit_backing(run_len) else {
            for (at, len, _) in mapped {
                unmap_commit(at, len);
            }
            tracing::warn!(
                size = run_len,
                "a committed virtual run has no backing section"
            );
            return false;
        };
        if !map_commit(&section, run_at, 0, run_len) {
            for (at, len, _) in mapped {
                unmap_commit(at, len);
            }
            return false;
        }
        mapped.push((
            run_at,
            run_len,
            Committed {
                len: run_len,
                protect,
                section,
                file_offset: 0,
            },
        ));
    }
    let Some(end) = at.checked_add(size) else {
        return false;
    };
    {
        let mut physical = physical();
        let Some((_, region)) = physical.reservation_mut(at) else {
            for (at, len, _) in mapped {
                unmap_commit(at, len);
            }
            return false;
        };
        for (piece_offset, piece_len, old) in pieces {
            if old != protect {
                region.reprotect(piece_offset, piece_len, protect);
            }
        }
        for (run_at, _, run) in mapped {
            region.commits.insert(run_at - base, run);
        }
        drop(physical);
        if protect != PAGE_READWRITE {
            // Best-effort: a data view grants the commit's protection through
            // `VirtualProtect`, which an executable page may refuse.
            let mut previous = PAGE_NOACCESS;
            // SAFETY: every page of the range is now a mapped view.
            if unsafe {
                VirtualProtect(at as *const c_void, size, protect, &mut previous)
            } == 0
            {
                tracing::debug!(
                    at = format_args!("{at:#x}"),
                    size,
                    protect = format_args!("{protect:#x}"),
                    error = %std::io::Error::last_os_error(),
                    "a committed run keeps read-write instead of its requested protection"
                );
            }
        }
    }
    victory_gpu::title_memory_remapped(at, end - at);
    VIRTUAL_COMMITS.fetch_add(1, Ordering::Relaxed);
    VIRTUAL_COMMITTED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    tracing::debug!(
        at = format_args!("{at:#x}"),
        size,
        protect = format_args!("{protect:#x}"),
        "XMemVirtualAlloc committed"
    );
    true
}

/// Decommits `[at, at + size)` inside a reservation: its pages become
/// inaccessible but keep their address space. The bytes are zeroed through
/// the view first, so a re-commit reads zeros deterministically.
fn virtual_decommit(at: usize, size: usize) -> bool {
    let Some(len) = size.checked_next_multiple_of(PAGE) else {
        virtual_failure("decommit size overflows", ERROR_INVALID_ADDRESS);
        return false;
    };
    if !at.is_multiple_of(PAGE) || len == 0 {
        virtual_failure("decommit is not page aligned", ERROR_INVALID_ADDRESS);
        return false;
    }
    let mut physical = physical();
    let Some((base, region)) = physical.reservation_mut(at) else {
        virtual_failure("decommit names no reservation", ERROR_INVALID_ADDRESS);
        return false;
    };
    let offset = at - base;
    let Some(end) = offset.checked_add(len) else {
        virtual_failure("decommit overflows", ERROR_INVALID_ADDRESS);
        return false;
    };
    if end > region.size {
        virtual_failure("decommit past the reservation", ERROR_INVALID_ADDRESS);
        return false;
    }
    // Physical-page views in the range go back to placeholders too.
    unmap(&mut physical, at, len);
    let Some((base, region)) = physical.reservation_mut(at) else {
        virtual_failure("decommit names no reservation", ERROR_INVALID_ADDRESS);
        return false;
    };
    let affected: Vec<(usize, Committed)> = region
        .commits
        .range(..end)
        .filter(|(start, run)| start.saturating_add(run.len) > offset)
        .map(|(&start, run)| (start, run.clone()))
        .collect();
    for (start, run) in affected {
        let stop = start + run.len;
        let (zero_from, zero_to) = (offset.max(start), end.min(stop));
        if zero_to > zero_from && run.protect & WRITABLE != 0 {
            // SAFETY: the run is a live read-write view covering the bytes.
            unsafe {
                std::ptr::write_bytes((base + zero_from) as *mut u8, 0, zero_to - zero_from)
            };
        }
        unmap_commit(base + start, run.len);
        region.commits.remove(&start);
        // A run reaching past the decommitted range keeps its remainder.
        if start < zero_from {
            let left = zero_from - start;
            if map_commit(&run.section, base + start, run.file_offset, left) {
                region.commits.insert(
                    start,
                    Committed {
                        len: left,
                        ..run.clone()
                    },
                );
            }
        }
        if stop > zero_to {
            let right = stop - zero_to;
            if map_commit(
                &run.section,
                base + zero_to,
                run.file_offset + (zero_to - start),
                right,
            ) {
                region.commits.insert(
                    zero_to,
                    Committed {
                        len: right,
                        file_offset: run.file_offset + (zero_to - start),
                        ..run
                    },
                );
            }
        }
    }
    drop(physical);
    VIRTUAL_DECOMMITS.fetch_add(1, Ordering::Relaxed);
    tracing::debug!(
        at = format_args!("{at:#x}"),
        len,
        "XMemVirtualAlloc decommitted"
    );
    true
}

/// Applies `MEM_RESET`/`MEM_RESET_UNDEFINED` to a committed range: the pages
/// stay committed but their old contents are discarded.
fn virtual_reset(at: usize, size: usize, undefined: bool) -> bool {
    let Some(len) = size.checked_next_multiple_of(PAGE) else {
        virtual_failure("reset size overflows", ERROR_INVALID_ADDRESS);
        return false;
    };
    if !at.is_multiple_of(PAGE) || len == 0 {
        virtual_failure("reset is not page aligned", ERROR_INVALID_ADDRESS);
        return false;
    }
    let mut physical = physical();
    let Some((base, region)) = physical.reservation_mut(at) else {
        virtual_failure("reset names no reservation", ERROR_INVALID_ADDRESS);
        return false;
    };
    let offset = at - region.base;
    let Some(end) = offset.checked_add(len) else {
        virtual_failure("reset overflows", ERROR_INVALID_ADDRESS);
        return false;
    };
    if end > region.size {
        virtual_failure("reset past the reservation", ERROR_INVALID_ADDRESS);
        return false;
    }
    // Every byte must already be committed: only then may it be reset.
    let mut cursor = offset;
    while cursor < end {
        match region.committed_at(cursor) {
            Some((start, run)) => cursor = (start + run.len).min(end),
            None => break,
        }
    }
    if cursor < end {
        virtual_failure("reset does not name committed pages", ERROR_INVALID_ADDRESS);
        return false;
    }
    for (from, length, run) in region.partition(offset, len) {
        if !undefined && run.protect & WRITABLE != 0 {
            // SAFETY: the piece is a live read-write view covering the bytes.
            unsafe { std::ptr::write_bytes((base + from) as *mut u8, 0, length) };
        }
        region.commits.insert(from, run);
    }
    drop(physical);
    VIRTUAL_RESETS.fetch_add(1, Ordering::Relaxed);
    tracing::debug!(at = format_args!("{at:#x}"), len, undefined, "XMemVirtualAlloc reset");
    true
}

/// Frees a virtual region: every committed and physical-page view returns to
/// a placeholder, then the address space is released.
fn virtual_release(at: usize) -> bool {
    let mut physical = physical();
    let Some(region) = physical.reservations.remove(&at) else {
        return false;
    };
    unmap(&mut physical, at, region.size);
    let commits: Vec<(usize, usize)> = region
        .commits
        .iter()
        .map(|(&offset, run)| (at + offset, run.len))
        .collect();
    drop(physical);
    for (run_at, len) in commits {
        unmap_commit(run_at, len);
    }
    victory_gpu::unalias_region(at);
    // SAFETY: the placeholder reservation this module made, with every view
    // gone; a whole reservation's split pieces are released together.
    let freed = unsafe { HostVirtualFree(at as *mut c_void, 0, MEM_RELEASE) };
    if freed == 0 {
        tracing::warn!(
            at = format_args!("{at:#x}"),
            error = %std::io::Error::last_os_error(),
            "releasing a virtual region reservation failed"
        );
    }
    VIRTUAL_RELEASES.fetch_add(1, Ordering::Relaxed);
    tracing::debug!(
        at = format_args!("{at:#x}"),
        size = region.size,
        committed = region.commits.len(),
        "XMemVirtualAlloc released"
    );
    true
}

/// The state `XMemVirtualQuery` reports for `at` when it lies in a virtual
/// region; `None` for every other address (the host's own answer stands,
/// physical-page views included).
fn reservation_state(at: usize) -> Option<ReservationState> {
    let physical = physical();
    // A physical-page view inside a region is the host's to report, as
    // before: the virtual region only answers for the pages it backs.
    if physical
        .views
        .range(..=at)
        .next_back()
        .is_some_and(|(base, (size, _))| at < base.saturating_add(*size))
    {
        return None;
    }
    let (base, region) = physical.reservation_at(at)?;
    // A 64 KiB physical-page reservation with no virtual commit keeps the
    // host's answer (its views were checked above).
    if region.physical_pages && region.commits.is_empty() {
        return None;
    }
    let offset = at - base;
    if let Some((start, run)) = region.committed_at(offset) {
        let stop = (start + run.len).min(region.size);
        return Some(ReservationState {
            base: base + start,
            allocation_base: base,
            allocation_protect: region.allocation_protect,
            size: stop - start,
            state: MEM_COMMIT,
            protect: run.protect,
        });
    }
    // A reserved gap: up to the next commit or the end of the region.
    let next = region
        .commits
        .range(offset..)
        .next()
        .map_or(region.size, |(&start, _)| start);
    let start = region
        .commits
        .range(..offset)
        .next_back()
        .map_or(0, |(&start, run)| start + run.len);
    Some(ReservationState {
        base: base + start,
        allocation_base: base,
        allocation_protect: region.allocation_protect,
        size: next.saturating_sub(start),
        state: MEM_RESERVE,
        protect: PAGE_NOACCESS,
    })
}

impl Drop for Section {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        if let Some(&at) = self.whole.get()
            && at != 0
        {
            // SAFETY: our view of the section; nothing borrows it any more
            // (views of it hold the section).
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                })
            };
        }
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Where each slice (section, page index) is mapped: the addresses of
    /// its page in every view showing it (the GPU side's change tracking
    /// follows writes through one view to the others,
    /// `victory_gpu::title_memory_aliased`).
    shown: HashMap<(usize, usize), Vec<usize>>,
    /// Address space `XMemVirtualAlloc` handed out, by base: reservations
    /// own no bytes; each committed run is a view of its own share file
    /// (ADR-045).
    reservations: BTreeMap<usize, Reservation>,
}

impl Slice {
    /// The key of page `k` of the slices from this one on in `shown`.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

/// Every address of a title view showing a page that title memory at
/// `[at, at + len)` shows, except the page's own address: pushed to `out`
/// as (page address, other address). Tests read `shown` back with it; the
/// GPU side's change tracking (`victory_gpu::watch`) marks writes there too.
#[cfg(test)]
fn aliases(at: usize, len: usize, out: &mut Vec<(usize, usize)>) {
    let physical = physical();
    let end = at.saturating_add(len);
    let first = physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| *base + size > at)
        .map(|(base, _)| *base);
    let starts = first.into_iter().chain(physical.views.range(at.saturating_add(1)..end).map(|(b, _)| *b));
    for base in starts.collect::<Vec<_>>() {
        let (size, slice) = &physical.views[&base];
        let (from, to) = (at.max(base), end.min(base + size));
        for page in (from / PAGE..to.div_ceil(PAGE)).map(|p| p * PAGE) {
            let k = (page - base) / PAGE;
            if let Some(addresses) = physical.shown.get(&slice.key(k)) {
                out.extend(addresses.iter().filter(|&&a| a != page).map(|&a| (page, a)));
            }
        }
    }
}

/// The addresses of title views showing `pages` (page numbers): writes into
/// them through our own views must be reported there.
fn shown_at(pages: &[usize]) -> Vec<usize> {
    let physical = physical();
    pages
        .iter()
        .filter_map(|p| physical.pages.get(p))
        .filter_map(|slice| physical.shown.get(&slice.key(0)))
        .flatten()
        .copied()
        .collect()
}

/// Reports writes into physical pages through our own views at every title
/// view showing them (DirectStorage's page destinations).
fn report_written(shown: &[usize]) {
    for &page in shown {
        #[cfg(test)]
        REPORTED.with(|r| r.borrow_mut().push(page));
        drop(victory_gpu::kernel_will_write(page, PAGE));
    }
}

#[cfg(test)]
thread_local! {
    /// The title addresses this thread's writes into physical pages reported.
    static REPORTED: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A section of `size` bytes, its pages committed when first mapped or
    // written (`commit`): Grand Theft Auto VI asks for 6.1 GiB at once,
    // which a busy host often cannot commit (error 1450).
    let Some(section) = new_section(size) else {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    };
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: Grand Theft Auto VI keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// How many of `pages` are no allocated page (diagnostics: a DirectStorage
/// destination that could not be written).
pub(crate) fn unknown_pages(pages: &[usize]) -> usize {
    let physical = physical();
    pages.iter().filter(|p| !physical.pages.contains_key(p)).count()
}

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        physical.pages.remove(&unsafe { *pages.add(i) });
    }
    1
}

/// Consecutive slices of one section, committed and writable: through our
/// view of the whole section, or through a view of their own if the host
/// would not map all of it, unmapped when dropped.
struct View {
    at: usize,
    len: usize,
    /// The view this maps for itself, if any.
    own: Option<MEMORY_MAPPED_VIEW_ADDRESS>,
    /// Keeps the section, and with it the whole view, while this lives.
    _section: Arc<Section>,
}

impl View {
    /// `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        let section = slice.section.clone();
        if let Some(whole) = section.whole() {
            let at = whole + offset as usize;
            return commit(at, len).then_some(View {
                at,
                len,
                own: None,
                _section: section,
            });
        }
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let own = unsafe {
            MapViewOfFile(
                section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if own.Value.is_null() {
            return None;
        }
        let view = View {
            at: own.Value as usize,
            len,
            own: Some(own),
            _section: section,
        };
        commit(view.at, len).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at as *mut u8, self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        if let Some(own) = self.own {
            // SAFETY: the view this mapped.
            unsafe { UnmapViewOfFile(own) };
        }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// Through each section's whole view ([`View`]), one run of consecutive
/// pages at a time. False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    // Where the pages show once written (see `PageWindow`).
    report_written(&shown_at(pages.get(offset / PAGE..(offset + data.len()).div_ceil(PAGE)).unwrap_or(&[])));
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
    /// The pages (page numbers) the bytes reach: the title views showing
    /// them when the window is done are told. Looked up then, not when the
    /// window opens: a decode takes milliseconds, and the title may map a
    /// page at another address meanwhile (its GPU heap); a copy that took
    /// the page there before the bytes landed would never hear of them. An
    /// address that stops showing a page meanwhile was told by the unmap.
    pages: Vec<usize>,
}

impl Drop for PageWindow {
    fn drop(&mut self) {
        report_written(&shown_at(&self.pages));
    }
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        within: offset % PAGE,
        len,
        pages: pages.get(offset / PAGE..(offset + len).div_ceil(PAGE)).unwrap_or(&[]).to_vec(),
    })
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let base = address as usize;
    let mut physical = physical();
    if !pages.is_null() {
        // SAFETY: the title passes `n` page numbers.
        let pages = unsafe { std::slice::from_raw_parts(pages, n) };
        if pages.iter().zip((base..).step_by(PAGE)).all(|(page, at)| {
            physical
                .pages
                .get(page)
                .is_some_and(|slice| shows(&physical, at, slice))
        }) {
            return address;
        }
    }
    unmap(&mut physical, base, n * PAGE);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(&mut physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (Grand Theft Auto VI passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`): Grand Theft Auto
/// VI maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let mut physical = physical();
    if from != 0
        && (0..len).step_by(PAGE).all(|k| {
            slice_at(&physical, from + k).is_some_and(|slice| shows(&physical, to + k, &slice))
        })
    {
        return true;
    }
    unmap(&mut physical, to, len);
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let Some(slice) = slice_at(&physical, from + k) else {
            return false;
        };
        if !map(&mut physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// The page a title view shows at `at`, if any.
fn slice_at(physical: &Physical, at: usize) -> Option<Slice> {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < *base + size)
        .map(|(base, (_, s))| Slice {
            index: s.index + (at - base) / PAGE,
            section: s.section.clone(),
        })
}

/// Whether a title view shows `slice` at `at` already. Mapping it there
/// again would change nothing, and would cost an unmap and a map, which
/// under Wine each hold the process's one virtual-memory lock across a
/// wineserver request: every page fault and protection change on other
/// threads waits for them (#50).
fn shows(physical: &Physical, at: usize, slice: &Slice) -> bool {
    slice_at(physical, at).is_some_and(|s| s.key(0) == slice.key(0))
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    #[cfg(test)]
    tests::MAPS.with(|m| m.set(m.get() + 1));
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = unsafe {
        HostVirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    let mapped = commit(base, len);
    #[cfg(feature = "diagnostics")]
    let mapped = mapped
        && slice.section.shared.as_ref().is_none_or(|shared| {
            // SAFETY: this live view is ours. The slice will hold the
            // backing heap until the view is unmapped.
            let aliased = unsafe { shared.alias_view(base, slice.index * PAGE, len) };
            if !aliased {
                tracing::warn!(
                    address = format_args!("{base:#x}"),
                    len,
                    "sharing the mapped XMem view with the host heap failed"
                );
            }
            aliased
        });
    if !mapped {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    for k in 0..len / PAGE {
        let shown = physical.shown.entry(slice.key(k)).or_default();
        // Pages shown elsewhere already: guarded here too if they are there.
        victory_gpu::title_memory_aliased(base + k * PAGE, shown);
        shown.push(base + k * PAGE);
    }
    physical.views.insert(base, (len, slice));
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    commit_refused(len, "physical pages", std::io::Error::last_os_error());
    false
}

/// Reports memory the host would not give: at the Windows commit limit to
/// `crashes.log`, since the title usually faults soon after on the null it
/// gets; otherwise as a warning.
fn commit_refused(len: usize, what: &str, error: std::io::Error) {
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        victory_loader::log_crash(&commit_limit_message(
            len,
            what,
            &error,
            system_commit().as_ref(),
        ));
    } else {
        tracing::warn!(len, %error, "committing {what} failed");
    }
}

/// Reports to `crashes.log` that the host allocator refused `len_bytes` of
/// APU heap memory. There is no Windows error code to give: `std::alloc`
/// allocates with `HeapAlloc`, which does not set one when it fails.
pub(crate) fn report_allocator_refusal(len_bytes: usize) {
    victory_loader::log_crash(&allocator_refusal_message(
        len_bytes,
        system_commit().as_ref(),
    ));
}

/// The system commit, if Windows reports it.
fn system_commit() -> Option<PERFORMANCE_INFORMATION> {
    let mut info = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
    (unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0).then_some(info)
}

fn commit_limit_message(
    len: usize,
    what: &str,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    refusal_message(
        format_args!(
            "Windows commit limit reached while committing {len} bytes of {what}: {error}"
        ),
        info,
    )
}

fn allocator_refusal_message(len_bytes: usize, info: Option<&PERFORMANCE_INFORMATION>) -> String {
    refusal_message(
        format_args!("the host allocator refused {len_bytes} bytes of APU heap memory"),
        info,
    )
}

/// What the host refused, then the system commit and what the user can do.
fn refusal_message(
    refusal: std::fmt::Arguments<'_>,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!("VIctory: {refusal}.\n");
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        for k in 0..size / PAGE {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
        }
        victory_gpu::title_memory_remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// The file the next allocation in this thread makes its section
        /// from; production allocates from the pagefile or the share
        /// provider. `share`'s tests use it to prove such a section is not
        /// reported as host-visible.
        pub(super) static SECTION_FILE: std::cell::RefCell<Option<std::fs::File>> =
            const { std::cell::RefCell::new(None) };
    }

    /// A section over the test's own file, if it installed one for this
    /// thread. The file must outlive every view of the section.
    pub(super) fn file_section(size: usize) -> Option<Arc<Section>> {
        SECTION_FILE.with(|file| {
            file.borrow()
                .as_ref()
                .and_then(|file| super::file_section(file, size, None))
        })
    }

    #[test]
    fn file_backed_pages_keep_bytes_at_a_nonzero_section_offset() {
        // Three pages of a real file: the page at section offset 64 KiB has
        // a distinct pattern, so reading it cannot read page 0.
        const MARK: usize = 0x100;
        const WRITTEN: usize = 0x180;
        let file = tempfile::tempfile().expect("a temporary file");
        file.set_len((3 * PAGE) as u64).expect("three pages");
        let page1: Vec<u8> = (0..64).map(|i| 0x40 + i as u8).collect();
        let mut page0 = [0u8; 64];
        page0.fill(0x11);
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut out = &file;
            out.seek(SeekFrom::Start((PAGE + MARK) as u64)).unwrap();
            out.write_all(&page1).unwrap();
            out.seek(SeekFrom::Start(MARK as u64)).unwrap();
            out.write_all(&page0).unwrap();
        }
        SECTION_FILE.with(|slot| *slot.borrow_mut() = Some(file.try_clone().expect("a copy")));

        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()),
            1,
            "an allocation from the injected section"
        );
        SECTION_FILE.with(|slot| *slot.borrow_mut() = None);
        assert_eq!(n, 3);

        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let bytes = |at: *mut c_void, offset: usize, len: usize| {
            // SAFETY: `at` is a live mapped view holding at least `offset + len` bytes.
            unsafe { std::slice::from_raw_parts(at.cast::<u8>().add(offset), len).to_vec() }
        };

        // Map page 1 (section offset 65536) through the existing HLE path
        // and read the file's original bytes there.
        assert_eq!(XMemMapPhysicalPages(a, 1, pages[1..].as_ptr()), a);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(bytes(a, MARK, 64), page1, "page 1 at section offset 64 KiB");
        assert_ne!(
            bytes(a, MARK, 64),
            page0,
            "page 1 is not page 0 of the file"
        );

        // A write through this view is visible through an alias, and
        // survives the alias's unmap and remap.
        let written: Vec<u8> = (0..16).map(|i| 0xe0 + i as u8).collect();
        // SAFETY: `a` is mapped read-write for a whole page and `WRITTEN + 16`
        // is inside it; `written` is a distinct live local buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                written.as_ptr(),
                a.cast::<u8>().add(WRITTEN),
                written.len(),
            )
        };
        assert!(map_alias(b as u64, a as u64, PAGE as u64));
        assert_eq!(bytes(b, MARK, 64), page1);
        assert_eq!(bytes(b, WRITTEN, 16), written);
        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert_eq!(bytes(a, MARK, 64), page1);
        assert_eq!(bytes(a, WRITTEN, 16), written);
        assert!(map_alias(b as u64, a as u64, PAGE as u64));
        assert_eq!(bytes(b, MARK, 64), page1);
        assert_eq!(bytes(b, WRITTEN, 16), written);

        // Unmapped and mapped again from the same section, both the file's
        // original page and our written bytes are still there.
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(XMemMapPhysicalPages(a, 1, pages[1..].as_ptr()), a);
        assert_eq!(bytes(a, MARK, 64), page1);
        assert_eq!(bytes(a, WRITTEN, 16), written);

        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        drop(file);
    }

    /// Opt-in cross-process proof: the actual XMem path writes a payload
    /// through a `map_alias` view of a file-backed physical page whose
    /// section offset is 64 KiB, flushes it, then publishes an atomic
    /// marker. `tools/macos/metal-poc/xmem-arm64.sh` runs a native arm64
    /// process that waits for the marker and reads the same file through a
    /// no-copy Metal buffer. Without both variables the test does nothing,
    /// so the default suite is unchanged.
    #[test]
    fn external_fixture_alias_write_for_native_gpu() {
        use windows_sys::Win32::System::Memory::FlushViewOfFile;

        const OFFSET: usize = PAGE; // 64 KiB from the start of the fixture.
        const GUARD: usize = 0x200; // Original page-1 bytes, left untouched.
        const PAYLOAD: usize = 0x400;
        const WORDS: usize = 1024;
        const SENTINEL_AT: usize = 0x2000;
        const SENTINEL: u32 = 0xC0DE_D00D;

        let Ok(fixture) = std::env::var("VICTORY_XMEM_GPU_FIXTURE") else {
            eprintln!("skipped: VICTORY_XMEM_GPU_FIXTURE is not set");
            return;
        };
        let marker =
            std::env::var("VICTORY_XMEM_GPU_MARKER").expect("a marker for the native GPU process");

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&fixture)
            .expect("the fixture file");
        let length = file.metadata().expect("fixture metadata").len() as usize;
        assert!(length >= 2 * PAGE, "the fixture needs two 64 KiB pages");
        let original = std::fs::read(&fixture).expect("the fixture bytes");
        assert_eq!(&original[..16], [0xA5; 16], "page 0 is the sentinel page");
        let page1 = &original[OFFSET..OFFSET + PAGE];
        assert_ne!(&page1[..16], &original[..16], "page 1 is not page 0");
        // The driver fills page 1 with a known byte pattern; the guard word
        // must still be that pattern after the HLE write.
        let guard: Vec<u8> = (0..4).map(|i| 0x80 | ((GUARD + i) as u8 & 0x3f)).collect();
        assert_eq!(&page1[GUARD..GUARD + 4], guard);

        // Make the next physical-page allocation in this thread use the
        // fixture, then map its second page (section offset 64 KiB).
        SECTION_FILE.with(|slot| *slot.borrow_mut() = Some(file.try_clone().expect("a copy")));
        let mut pages = [0usize; 2];
        let mut n = 2;
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()),
            1,
            "an allocation from the fixture"
        );
        SECTION_FILE.with(|slot| *slot.borrow_mut() = None);
        assert_eq!(n, 2);

        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 1, pages[1..].as_ptr()), a);
        let view = |at: *mut c_void| {
            // SAFETY: `at` is a live mapped view of one whole page.
            unsafe { std::slice::from_raw_parts(at.cast::<u8>(), PAGE).to_vec() }
        };
        // The view shows section offset 64 KiB, not the file's first page.
        assert_eq!(&view(a)[GUARD..GUARD + 4], guard);

        // The GPU's page-mapping copy, then a write through the alias.
        assert!(map_alias(b as u64, a as u64, PAGE as u64));
        let payload: Vec<u8> = (0..WORDS)
            .flat_map(|i| {
                let word = 0x51E3_0000u32 ^ (i as u32).wrapping_mul(0x9E37_79B9);
                word.to_le_bytes()
            })
            .collect();
        // SAFETY: `b` is the live one-page alias mapped just above.
        let alias = unsafe { std::slice::from_raw_parts_mut(b.cast::<u8>(), PAGE) };

        alias[PAYLOAD..PAYLOAD + payload.len()].copy_from_slice(&payload);
        alias[SENTINEL_AT..SENTINEL_AT + 4].copy_from_slice(&SENTINEL.to_le_bytes());
        // The source view and the untouched guard see the alias write.
        assert_eq!(&view(a)[PAYLOAD..PAYLOAD + payload.len()], payload);
        assert_eq!(
            &view(a)[SENTINEL_AT..SENTINEL_AT + 4],
            SENTINEL.to_le_bytes()
        );
        assert_eq!(&view(a)[GUARD..GUARD + 4], guard);
        // SAFETY: the whole page is a live mapped view of the section.
        assert_ne!(unsafe { FlushViewOfFile(b.cast(), PAGE) }, 0);
        file.sync_all().expect("flushing the fixture");

        // The bytes are in the file itself, at the identified offset.
        let on_disk = std::fs::read(&fixture).expect("the fixture after the write");
        assert_eq!(&on_disk[..OFFSET], &original[..OFFSET]);
        assert_eq!(
            &on_disk[OFFSET + PAYLOAD..OFFSET + PAYLOAD + payload.len()],
            payload
        );
        assert_eq!(
            &on_disk[OFFSET + SENTINEL_AT..OFFSET + SENTINEL_AT + 4],
            SENTINEL.to_le_bytes()
        );
        assert_eq!(&on_disk[OFFSET + GUARD..OFFSET + GUARD + 4], guard);

        // Atomic handoff: the native process only maps the file after the
        // marker appears, so it cannot observe a partial write.
        let temporary = format!("{marker}.tmp");
        std::fs::write(&temporary, format!("offset={OFFSET}\n")).expect("the marker");
        std::fs::rename(&temporary, &marker).expect("publishing the marker");

        let mut sum = 0u32;
        let mut xor = 0u32;
        for word in payload.as_chunks::<4>().0 {
            let word = u32::from_le_bytes(*word);
            sum = sum.wrapping_add(word);
            xor ^= word;
        }
        println!(
            "HLE_WROTE offset={OFFSET} page={} sum={sum:#010x} xor={xor:#010x} sentinel={SENTINEL:#010x}",
            pages[1]
        );

        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        // Not under Rosetta (`victory_gpu::fault_tracking`).
        assert_eq!(watched == 0, victory_gpu::fault_tracking());
        // SAFETY: our own reservation.
        assert_ne!(unsafe { HostVirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { HostVirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, "physical pages", &error, Some(&info));
        assert!(report.contains("4194304 bytes of physical pages"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, "virtual memory", &error, None);
        assert!(fallback.contains("4194304 bytes of virtual memory"));
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    thread_local! {
        /// Views this thread mapped for the title (`map`).
        pub(super) static MAPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[test]
    fn mapping_what_shows_already_maps_nothing() {
        let pages = allocate(2);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                2 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let maps = || MAPS.with(std::cell::Cell::get);
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        let before = maps();
        // The same pages at the same address: nothing to do.
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        let scattered = [a, a.wrapping_byte_add(PAGE)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(scattered.as_ptr(), 2, pages.as_ptr()),
            1
        );
        assert_eq!(maps(), before);
        // The GPU's page-mapping copy, made twice: the second maps nothing.
        assert!(map_alias(b as u64, a as u64, 2 * PAGE as u64));
        assert_eq!(maps(), before + 2);
        assert!(map_alias(b as u64, a as u64, 2 * PAGE as u64));
        assert!(map_alias(
            b.wrapping_byte_add(PAGE) as u64,
            a.wrapping_byte_add(PAGE) as u64,
            PAGE as u64
        ));
        assert_eq!(maps(), before + 2);
        // Another page, or one page of the range changed, maps again.
        assert!(map_alias(
            b as u64,
            a.wrapping_byte_add(PAGE) as u64,
            PAGE as u64
        ));
        assert_eq!(maps(), before + 3);
        // SAFETY: both mapped read-write, showing the second page.
        unsafe {
            a.wrapping_byte_add(PAGE).cast::<u8>().write(6);
            assert_eq!(b.cast::<u8>().read(), 6);
        }
        // Pages swapped at `a`: both differ, so both map again.
        let swapped = [pages[1], pages[0]];
        assert_eq!(XMemMapPhysicalPages(a, 2, swapped.as_ptr()), a);
        assert_eq!(maps(), before + 5);
        // SAFETY: mapped read-write; `a` shows the second page now.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 6);
        for at in [a, b] {
            assert_eq!(XMemMapPhysicalPages(at, 2, std::ptr::null()), at);
        }
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    #[test]
    fn allocator_refusal_report_names_apu_heap_and_system_commit() {
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = allocator_refusal_message(4 * MIB, Some(&info));
        assert!(report.contains("host allocator refused 4194304 bytes of APU heap memory"));
        assert!(report.contains("System commit: 2 MiB of 3 MiB in use"));
        assert!(report.contains("Windows paging file size"));
        let fallback = allocator_refusal_message(4 * MIB, None);
        assert!(fallback.contains("4194304 bytes of APU heap memory"));
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        #[cfg(feature = "diagnostics")]
        if std::env::var("VICTORY_PHYSICAL_TILES").as_deref() == Ok("1") {
            let physical = physical();
            assert!(
                physical.pages[&pages[0]].section.shared.is_some(),
                "the test must use shared physical pages"
            );
        }
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Each mapping knows the other (the GPU side's change tracking).
        let others = |p: *mut c_void| {
            let mut out = Vec::new();
            aliases(p as usize, PAGE, &mut out);
            out.into_iter().map(|(_, a)| a).collect::<Vec<_>>()
        };
        assert_eq!(others(at(a, 2)), [at(b, 1) as usize]);
        assert_eq!(others(at(b, 1)), [at(a, 2) as usize]);
        assert!(others(a).is_empty());
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        let mut both = others(at(a, 2));
        both.sort_unstable();
        assert_eq!(both, [b as usize, at(b, 1) as usize]);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(others(at(a, 2)), [at(b, 1) as usize], "unmapped views are forgotten");
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through our view of its section.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// A page mapped at a second address while a window writes it is
    /// reported there too when the window is done (the title maps streamed
    /// pages into its GPU heap while their reads decode); an address that
    /// stopped showing it meanwhile is not.
    #[test]
    fn page_windows_report_where_the_pages_show_when_done() {
        let pages = allocate(1);
        let reserve = || XMemVirtualAlloc(std::ptr::null_mut(), PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        REPORTED.with(|r| r.borrow_mut().clear());
        let mut window = page_window(&pages, 0, 16).expect("one run");
        // Mapped at `b`, unmapped from `a`, while the bytes are written.
        assert_eq!(XMemMapPhysicalPages(b, 1, pages.as_ptr()), b);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        window.bytes().fill(3);
        drop(window);
        assert_eq!(REPORTED.with(|r| r.borrow().clone()), [b as usize]);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 3);
        // `write_physical` the same way.
        REPORTED.with(|r| r.borrow_mut().clear());
        assert!(write_physical(&pages, 0, &[4]));
        assert_eq!(REPORTED.with(|r| r.borrow().clone()), [b as usize]);
        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn page_writes_share_one_view_of_their_section() {
        let a = allocate(3);
        let slice = |p| physical().pages.get(&p).cloned().expect("known page");
        let section = Arc::downgrade(&slice(a[0]).section);
        // Every request on the section goes through the same view, each
        // page at its place in it, with no view of its own to map and unmap.
        let (first, third) = (
            View::map(&slice(a[0]), 1).expect("mapped"),
            View::map(&slice(a[2]), 1).expect("mapped"),
        );
        assert_eq!(third.at - first.at, 2 * PAGE);
        assert!(first.own.is_none() && third.own.is_none());
        let at = first.at;
        drop((first, third));
        assert_eq!(View::map(&slice(a[1]), 2).expect("mapped").at, at + PAGE);
        assert!(write_physical(&a, 2 * PAGE + 3, &[4, 5]));
        assert_eq!(page_bytes(a[2])[3..5], [4, 5]);
        // The view lives as long as the section: until its pages are freed.
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert!(section.upgrade().is_none());
    }
}

/// The space below 4 GiB, which the title's 32-bit memory (`XMemVirtualAlloc`
/// with the 32-bit address flag) has to fit in.
const FOUR_GIB: usize = 1 << 32;

/// Guards over the space below 4 GiB. Windows places allocations bottom up,
/// so the host's own memory (the driver's, heaps, the shader and pipeline
/// caches read at startup) filled it first: with a large pipeline cache only
/// 22 MB was left, in pieces, when Grand Theft Auto VI asked for a 185 MB
/// 32-bit placeholder at startup, the request was refused
/// (`ERROR_NOT_ENOUGH_MEMORY`) and the title stopped at an assertion. The
/// debug layer's memory did the same. So the free space below 4 GiB is
/// reserved at startup, past [`low_guards::SLACK`] left for the host, and a
/// 32-bit request that finds no room releases guards and tries again.
pub mod low_guards {
    use std::sync::Mutex;

    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualAlloc,
        VirtualFree, VirtualQuery,
    };

    use super::FOUR_GIB;

    /// Free space left to the host below 4 GiB (the lowest free space).
    pub(crate) const SLACK: usize = 256 << 20;
    /// The allocation granularity of Windows.
    const GRANULE: usize = 64 << 10;

    /// The guards, `(base, bytes)`.
    static GUARDS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

    fn guards() -> std::sync::MutexGuard<'static, Vec<(usize, usize)>> {
        GUARDS.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The free runs in `[from, to)`, granule-aligned.
    fn free_runs(from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        let mut at = from.max(GRANULE);
        while at < to {
            // SAFETY: plain data filled in by the query.
            let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
                break;
            }
            let base = m.BaseAddress as usize;
            let end = (base + m.RegionSize).min(to);
            if m.State == MEM_FREE {
                let (start, end) = (
                    base.max(from).next_multiple_of(GRANULE),
                    end & !(GRANULE - 1),
                );
                if end > start {
                    runs.push((start, end - start));
                }
            }
            at = base + m.RegionSize;
        }
        runs
    }

    /// Reserves `[base, base + bytes)` as a guard; false if something took it.
    fn guard(base: usize, bytes: usize) -> bool {
        // SAFETY: a reservation of free address space; nothing refers to it.
        let at = unsafe { VirtualAlloc(base as *const _, bytes, MEM_RESERVE, PAGE_NOACCESS) };
        at as usize == base
    }

    /// Reserves the free space below 4 GiB but the lowest [`SLACK`]. Call
    /// once, before the host allocates much (the GPU device, the caches).
    pub fn install() {
        let mut guards = guards();
        let mut slack = SLACK;
        for (mut base, mut bytes) in free_runs(0, FOUR_GIB) {
            let left = slack.min(bytes);
            (base, bytes, slack) = (base + left, bytes - left, slack - left);
            if bytes > 0 && guard(base, bytes) {
                guards.push((base, bytes));
            }
        }
        let total: usize = guards.iter().map(|g| g.1).sum();
        tracing::info!(
            guards = guards.len(),
            mb = total >> 20,
            "space below 4 GiB kept for the title"
        );
    }

    /// Try the smallest fitting guards first, then try smaller guards in
    /// descending size order. A replacement guard cannot be retried here.
    pub(crate) fn retry(
        bytes: usize,
        mut allocate: impl FnMut() -> *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};

        // Preserve the allocation error across address queries and reservations.
        // SAFETY: reads this thread's last-error value.
        let mut error = unsafe { GetLastError() };
        let mut guards = guards();
        let mut candidates = guards.clone();
        candidates.sort_unstable_by_key(|&(_, size)| {
            let fits = size >= bytes;
            (!fits, if fits { size } else { usize::MAX - size })
        });
        for (base, size) in candidates {
            // SAFETY: our own reservation, with no allocations inside it.
            if unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) } == 0 {
                continue;
            }
            guards.retain(|&(b, _)| b != base);
            let at = allocate();
            // SAFETY: reads this thread's last-error value.
            error = unsafe { GetLastError() };
            for (start, len) in free_runs(base, base + size) {
                if guard(start, len) {
                    guards.push((start, len));
                }
            }
            if !at.is_null() {
                return at;
            }
        }
        // SAFETY: sets this thread's last-error value.
        unsafe { SetLastError(error) };
        std::ptr::null_mut()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        static TEST: Mutex<()> = Mutex::new(());

        #[test]
        fn a_refused_retry_leaves_the_guards_in_place() {
            let _one = TEST.lock().unwrap();
            // SAFETY: a fresh reservation, released at once.
            let hole =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * GRANULE, MEM_RESERVE, PAGE_NOACCESS) };
            assert!(!hole.is_null());
            let first = hole as usize;
            // SAFETY: ours.
            assert_ne!(unsafe { VirtualFree(hole, 0, MEM_RELEASE) }, 0);
            assert!(guard(first, 4 * GRANULE));
            guards().push((first, 4 * GRANULE));
            let mut attempts = 0;
            let result = retry(4 * GRANULE, || {
                attempts += 1;
                std::ptr::null_mut()
            });
            assert!(result.is_null());
            assert_eq!(attempts, 1);
            assert!(guards().contains(&(first, 4 * GRANULE)));
            guards().retain(|&(base, _)| base != first);
            // SAFETY: the guard the retry put back, ours.
            assert_ne!(unsafe { VirtualFree(hole, 0, MEM_RELEASE) }, 0);
        }

        /// A released guard is guarded again around what took part of it.
        #[test]
        fn a_released_guard_is_guarded_again_around_what_took_it() {
            let _one = TEST.lock().unwrap();
            // A known hole of four granules: reserved, then freed.
            // SAFETY: a fresh reservation, released at once.
            let hole =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * GRANULE, MEM_RESERVE, PAGE_NOACCESS) }
                    as usize;
            assert_ne!(hole, 0);
            // SAFETY: ours.
            unsafe { VirtualFree(hole as *mut _, 0, MEM_RELEASE) };
            assert!(guard(hole, 4 * GRANULE));
            guards().push((hole, 4 * GRANULE));
            let taken = retry(GRANULE, || {
                // SAFETY: a reservation of the first granule at a known address.
                unsafe { VirtualAlloc(hole as *const _, GRANULE, MEM_RESERVE, PAGE_NOACCESS) }
            }) as usize;
            assert_eq!(taken, hole);
            assert!(guards().contains(&(hole + GRANULE, 3 * GRANULE)));
            guards().retain(|&(b, _)| b != hole + GRANULE);
            // SAFETY: ours.
            unsafe {
                VirtualFree(taken as *mut _, 0, MEM_RELEASE);
                VirtualFree((hole + GRANULE) as *mut _, 0, MEM_RELEASE);
            }
        }
    }
}
