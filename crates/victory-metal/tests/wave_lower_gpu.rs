//! Lane-tagged hardware proof for the wave64 lowering (`emit::lower::halves`
//! and `emit::lower::barriers`).
//!
//! Each test builds a synthetic console compute shader with the same
//! hand-encoded GFX10 words the `victory-shader` emit tests use (a root table
//! load, a UAV store, and the console's own ballot, lane-read, reduction,
//! prefix and LDS sequences), translates it through `victory_shader::emit` to
//! DXIL, converts the DXIL with Apple's IRConverter, dispatches the console
//! threadgroup on the Metal GPU, and asserts every lane's tagged dwords equal
//! the console semantics computed in Rust.
//!
//! # Binding the converted shader
//!
//! The emitted DXIL reads its UAV through the console's root table, so
//! IRConverter's converted function does not take the UAV at
//! `[[buffer(0)]]`. Its AIR takes two implicit buffers:
//!
//! ```text
//! buffer(0) = resource descriptor heap argument buffer, 24-byte descriptors;
//!             the shader loads the buffer's device address from descriptor
//!             index ((user_data[0] + 0x40) >> 5)
//! buffer(2) = top-level global argument buffer; field 0 is the CBV descriptor
//!             whose first dword is the device address of the user-data block
//!             (dword 0: the table's first descriptor byte offset in the heap)
//! ```
//!
//! Reflection (`IRShaderReflectionCopyJSONString`, captured from this exact
//! conversion) reports one entry: `{"EltOffset":0,"Size":24,"Slot":0,
//! "Space":0,"Type":"CBV"}`. The harness builds that layout with
//! [`RootSignatureLayout`], binds the user-data constant block through
//! [`ArgumentBufferEncoder`] (root CBV), fills the descriptor heap entry and
//! binds it at `buffer(0)` itself, and dispatches the console threadgroup.
//!
//! IRConverter's `libmetalirconverter.dylib` is x86_64-only, so on Apple
//! Silicon the conversion runs in the separate x86_64 `victory-irconverter`
//! worker under Rosetta 2 (`ProcessWorker`) and the in-process reflection call
//! cannot load the dylib; the layout then comes from
//! [`RootSignatureLayout::from_parameters`] with the reflected shape above.
//! Build the worker once:
//!
//! ```sh
//! cargo build -p victory-metal --bin victory-irconverter --target x86_64-apple-darwin
//! ```
//!
//! The hardware tests are `#[ignore]`d because the GPU is shared; run them one
//! at a time with `--ignored --test-threads=1`.

#![cfg(target_os = "macos")]

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::PathBuf;
use std::ptr::null_mut;

use victory_metal::VmBufferId;
use victory_metal::macos::bindings::{
    ArgumentBufferEncoder, BufferBinding, EncoderKind, ReflectedStage, ResourceUsage,
    RootParameterKind, RootParameterSpec, RootSignatureLayout,
};
use victory_metal::macos::cache::TranslationOptions;
use victory_metal::macos::irconverter::{
    ProcessWorker, convert_dxil_to_metallib_with_reflection, translate_dxil_with,
};
use victory_shader::emit::{self, Options};
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use victory_shader::{dxbc, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)

/// `s[40:43]`: the raw UAV at table dword 16 (s0 is the table's address).
const PROLOGUE: [u32; 5] = [
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    S_WAITCNT_LGKM,
];

/// `buffer_store_dword v<vdata>, v16, s[40:43], 0 offen offset:<offset>`.
const fn store(vdata: u8, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x800A_0010 | (vdata as u32) << 8]
}

/// An empty DXBC container: `emit` reads only signature parts, and a compute
/// shader has none. The bytes are the minimum `Container::parse` accepts.
fn empty_container() -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

/// Emits the DXIL for `options` (no IRConverter, no GPU work).
fn emitted_dxil(
    words: &[u32],
    threads: u32,
    lds_bytes: u32,
    options: Options,
) -> Result<Vec<u8>, victory_shader::ShaderError> {
    let range = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8),
            range(ResourceType::Srv, 8, 4),
            range(ResourceType::Uav, 16, 4),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | lds_bytes.div_ceil(512) << 15,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let raised = io::raise(&p, &native).expect("io::raise");
    let bytes = empty_container();
    let c = dxbc::Container::parse(&bytes).expect("empty container");
    let host = emit::emit_with(&c, &native, &p, &st, &raised, &res, options)?;
    assert!(
        host.dxil.windows(4).any(|w| w == b"DXIL"),
        "emitted container has no DXIL part"
    );
    Ok(host.dxil)
}

/// A wave64 compute program writing `per_lane` dwords per thread at `l * 32`
/// through the root table's UAV, lowered for a host of 32-lane waves
/// (`host_wave32`, with `narrow_waves` as given).
fn lowered_dxil(words: &[u32], threads: u32, lds_bytes: u32, narrow_waves: bool) -> Vec<u8> {
    emitted_dxil(
        words,
        threads,
        lds_bytes,
        Options {
            host_wave32: true,
            narrow_waves,
            ..Options::default()
        },
    )
    .expect("the wave64 program lowers for a 32-lane host")
}

/// The x86_64 Rosetta converter worker and the x86_64 IRConverter library.
fn converter() -> Result<(PathBuf, PathBuf), String> {
    let worker = std::env::var_os("VICTORY_IRCONVERTER_WORKER")
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            let p = workspace_root()?
                .join("target/x86_64-apple-darwin/debug/victory-irconverter");
            p.is_file().then_some(p)
        })
        .ok_or_else(|| {
            "no x86_64 victory-irconverter worker: build it with \
             `cargo build -p victory-metal --bin victory-irconverter \
             --target x86_64-apple-darwin` or set VICTORY_IRCONVERTER_WORKER"
                .to_string()
        })?;
    let lib = std::env::var_os("VICTORY_METAL_IRCONVERTER")
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            std::env::var_os("D3DMETAL_RES")
                .map(|r| PathBuf::from(r).join("libmetalirconverter.dylib"))
                .filter(|p| p.is_file())
        })
        .or_else(|| {
            let p = PathBuf::from(std::env::var_os("HOME")?).join(
                "Library/Caches/victory-macos/target/wine-cx/lib/external/\
                 D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib",
            );
            p.is_file().then_some(p)
        })
        .ok_or_else(|| {
            "no libmetalirconverter.dylib (set VICTORY_METAL_IRCONVERTER or D3DMETAL_RES)"
                .to_string()
        })?;
    Ok((worker, lib))
}

fn workspace_root() -> Option<PathBuf> {
    // `CARGO_MANIFEST_DIR` is `crates/victory-metal`; the workspace is two up.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
}

/// Converted metallib and the root-signature layout it was built for.
struct Converted {
    metallib: Vec<u8>,
    layout: RootSignatureLayout,
}

/// The layout reflected for these fixtures' converted program.
///
/// Captured with `convert_dxil_to_metallib_with_reflection` on the same
/// emitted DXIL (x86_64): `TopLevelArgumentBuffer` is one entry,
/// `{"EltOffset":0,"Size":24,"Slot":0,"Space":0,"Type":"CBV"}`. The
/// converter reads field 0 as the device address of the user-data block; its
/// 24-byte extent is the converter's descriptor struct, of which only the
/// 8-byte address is used.
fn reflected_layout() -> RootSignatureLayout {
    RootSignatureLayout::from_parameters(
        ReflectedStage::Compute,
        &[RootParameterSpec {
            kind: RootParameterKind::ConstantBufferView,
            size: 0,
            register: 0,
            space: 0,
        }],
    )
    .expect("one root CBV describes the converted program")
}

/// Converts the emitted DXIL and returns its metallib with the root layout.
fn convert(dxil: &[u8]) -> Result<Converted, String> {
    let (worker_path, lib) = converter()?;
    // An x86_64 process can load the converter and read its reflection. On
    // Apple Silicon the dylib is x86_64-only, so the Rosetta worker converts
    // and the layout is the reflected shape captured for this conversion.
    if let Ok(converted) = convert_dxil_to_metallib_with_reflection(dxil, "main", &lib) {
        let layout =
            RootSignatureLayout::from_converter_reflection_json(&converted.reflection_json)
                .map_err(|e| format!("converter reflection: {e}"))?;
        return Ok(Converted {
            metallib: converted.metallib.bytes,
            layout,
        });
    }
    let worker = ProcessWorker::new(worker_path, lib);
    let out = translate_dxil_with(
        None,
        &worker,
        dxil,
        "main",
        8, // compute
        None,
        &TranslationOptions::default(),
        &[],
    )
    .map_err(|e| format!("IRConverter: {e}"))?;
    Ok(Converted {
        metallib: out.metallib,
        layout: reflected_layout(),
    })
}

// ----- minimal Metal / Objective-C glue ------------------------------------

type Id = *mut c_void;
type Sel = *mut c_void;

/// Apple `MTLSize` (`NSUInteger width, height, depth`).
#[repr(C)]
#[derive(Clone, Copy)]
struct MTLSize {
    width: usize,
    height: usize,
    depth: usize,
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_msgSend();
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

#[link(name = "System")]
unsafe extern "C" {
    fn dispatch_data_create(
        buffer: *const c_void,
        size: usize,
        queue: *mut c_void,
        destructor: *mut c_void,
    ) -> Id;
}

fn sel(name: &CStr) -> Sel {
    // SAFETY: `name` is a valid NUL-terminated C string.
    unsafe { sel_registerName(name.as_ptr()) }
}

/// `MTLResourceStorageModeShared` (0): CPU-visible unified memory.
const STORAGE_MODE_SHARED: usize = 0;

/// Descriptor stride of the converted resource descriptor heap.
const DESCRIPTOR_BYTES: usize = 24;

/// Descriptor the converted shader loads: `(user_data[0] + 0x40) >> 5`.
const DESCRIPTOR_INDEX: usize = 2;

unsafe fn send_id0(target: Id, s: Sel) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel) -> Id;
    // SAFETY: caller supplies a live receiver and a selector returning an object.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

unsafe fn send_id1(target: Id, s: Sel, a0: Id) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel, Id) -> Id;
    // SAFETY: caller supplies a live receiver and one object argument.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s, a0)
    }
}

unsafe fn send_id_err(target: Id, s: Sel, a0: Id, err: *mut Id) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id;
    // SAFETY: caller supplies a live receiver and an `NSError **` slot.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s, a0, err)
    }
}

unsafe fn send_id_cstr(target: Id, s: Sel, text: *const c_char) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel, *const c_char) -> Id;
    // SAFETY: caller supplies a live receiver and a NUL-terminated C string.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s, text)
    }
}

unsafe fn send_id_len_options(target: Id, s: Sel, len: usize, options: usize) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel, usize, usize) -> Id;
    // SAFETY: caller supplies a live receiver and two integer arguments.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s, len, options)
    }
}

unsafe fn send_void0(target: Id, s: Sel) {
    type Fn = unsafe extern "C" fn(Id, Sel);
    // SAFETY: caller supplies a live receiver and a void selector.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s);
    }
}

unsafe fn send_void1(target: Id, s: Sel, a0: Id) {
    type Fn = unsafe extern "C" fn(Id, Sel, Id);
    // SAFETY: caller supplies a live receiver and one object argument.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s, a0);
    }
}

unsafe fn send_set_buffer(encoder: Id, buffer: Id, offset: usize, index: usize) {
    type Fn = unsafe extern "C" fn(Id, Sel, Id, usize, usize);
    // SAFETY: caller supplies a live compute encoder and buffer.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(encoder, sel(c"setBuffer:offset:atIndex:"), buffer, offset, index);
    }
}

unsafe fn send_dispatch(encoder: Id, grid: MTLSize, group: MTLSize) {
    type Fn = unsafe extern "C" fn(Id, Sel, MTLSize, MTLSize);
    // SAFETY: caller supplies a live compute encoder and positive sizes.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(encoder, sel(c"dispatchThreads:threadsPerThreadgroup:"), grid, group);
    }
}

unsafe fn send_usize0(target: Id, s: Sel) -> usize {
    type Fn = unsafe extern "C" fn(Id, Sel) -> usize;
    // SAFETY: caller supplies a live receiver and a selector returning NSUInteger.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

unsafe fn send_u64_0(target: Id, s: Sel) -> u64 {
    type Fn = unsafe extern "C" fn(Id, Sel) -> u64;
    // SAFETY: caller supplies a live receiver and a selector returning uint64_t.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

unsafe fn send_ptr0(target: Id, s: Sel) -> *mut c_void {
    type Fn = unsafe extern "C" fn(Id, Sel) -> *mut c_void;
    // SAFETY: caller supplies a live receiver and a selector returning a pointer.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

unsafe fn send_cstr0(target: Id, s: Sel) -> *const c_char {
    type Fn = unsafe extern "C" fn(Id, Sel) -> *const c_char;
    // SAFETY: caller supplies a live receiver and a selector returning a C string.
    unsafe {
        let f: Fn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

struct AutoreleasePool {
    token: *mut c_void,
}

impl AutoreleasePool {
    fn new() -> Self {
        // SAFETY: pushing an autorelease pool is valid on any OS thread.
        let token = unsafe { objc_autoreleasePoolPush() };
        Self { token }
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        if !self.token.is_null() {
            // SAFETY: `token` came from the matching push on this thread.
            unsafe { objc_autoreleasePoolPop(self.token) };
        }
    }
}

fn nsstring(text: &str) -> Id {
    let text = CString::new(text).expect("entry name has no NUL");
    // SAFETY: `objc_getClass` returns the class object for a valid name.
    let cls = unsafe { objc_getClass(c"NSString".as_ptr()) };
    assert!(!cls.is_null(), "no NSString class");
    // SAFETY: `cls` is `NSString` and `text` is a valid C string.
    let string = unsafe { send_id_cstr(cls, sel(c"stringWithUTF8String:"), text.as_ptr()) };
    assert!(!string.is_null(), "NSString creation failed");
    string
}

fn nserror_message(error: Id) -> String {
    if error.is_null() {
        return "unknown Metal error".to_string();
    }
    // SAFETY: `error` is an `NSError *`; `localizedDescription` returns an autoreleased string.
    let description = unsafe { send_id0(error, sel(c"localizedDescription")) };
    if description.is_null() {
        return "unknown Metal error".to_string();
    }
    // SAFETY: `description` is an `NSString *`; `UTF8String` is valid inside the pool.
    let utf8 = unsafe { send_cstr0(description, sel(c"UTF8String")) };
    if utf8.is_null() {
        return "unknown Metal error".to_string();
    }
    // SAFETY: `UTF8String` returns a NUL-terminated C string.
    unsafe { CStr::from_ptr(utf8) }.to_string_lossy().into_owned()
}

/// A Metal device and command queue created for one dispatch.
struct Gpu {
    device: Id,
    queue: Id,
}

impl Gpu {
    fn new() -> Self {
        // SAFETY: `MTLCreateSystemDefaultDevice` has no preconditions.
        let device = unsafe { MTLCreateSystemDefaultDevice() };
        assert!(!device.is_null(), "no Metal device");
        // SAFETY: `device` is a live `MTLDevice`; `newCommandQueue` returns +1.
        let queue = unsafe { send_id0(device, sel(c"newCommandQueue")) };
        assert!(!queue.is_null(), "no Metal command queue");
        Self { device, queue }
    }
}

fn shared_buffer(gpu: &Gpu, len: usize) -> Id {
    // SAFETY: `gpu.device` is live; shared storage is CPU-visible.
    let buffer = unsafe {
        send_id_len_options(
            gpu.device,
            sel(c"newBufferWithLength:options:"),
            len.max(16),
            STORAGE_MODE_SHARED,
        )
    };
    assert!(!buffer.is_null(), "Metal buffer allocation failed");
    buffer
}

fn contents(buffer: Id) -> *mut u8 {
    // SAFETY: `buffer` is a live shared-storage `MTLBuffer`.
    let ptr = unsafe { send_ptr0(buffer, sel(c"contents")) };
    assert!(!ptr.is_null(), "Metal buffer has no CPU-visible contents");
    ptr.cast::<u8>()
}

fn write_u64(buffer: Id, offset: usize, value: u64) {
    // SAFETY: the caller keeps `offset + 8` inside the allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(
            value.to_le_bytes().as_ptr(),
            contents(buffer).add(offset),
            8,
        )
    };
}

fn gpu_address(buffer: Id) -> u64 {
    // SAFETY: `buffer` is a live `MTLBuffer`.
    unsafe { send_u64_0(buffer, sel(c"gpuAddress")) }
}

fn new_library(gpu: &Gpu, metallib: &[u8]) -> Id {
    // SAFETY: `metallib` is valid for its length during the call; a null queue
    // and destructor request the copying path.
    let data = unsafe {
        dispatch_data_create(
            metallib.as_ptr().cast(),
            metallib.len(),
            null_mut(),
            null_mut(),
        )
    };
    assert!(!data.is_null(), "dispatch_data_create failed");
    let mut error: Id = null_mut();
    // SAFETY: `gpu.device` is live, `data` is a dispatch data object and
    // `error` receives an `NSError *`.
    let library = unsafe {
        send_id_err(gpu.device, sel(c"newLibraryWithData:error:"), data, &mut error)
    };
    assert!(
        !library.is_null(),
        "load converted metallib: {}",
        nserror_message(error)
    );
    library
}

fn new_function(library: Id, entry: &str) -> Id {
    let name = nsstring(entry);
    // SAFETY: `library` is a live `MTLLibrary` and `name` an `NSString`.
    let function = unsafe { send_id1(library, sel(c"newFunctionWithName:"), name) };
    assert!(!function.is_null(), "entry point `{entry}` not found");
    function
}

fn new_compute_pipeline(gpu: &Gpu, function: Id) -> Id {
    let mut error: Id = null_mut();
    // SAFETY: `gpu.device` is live and `function` is a live `MTLFunction`.
    let pipeline = unsafe {
        send_id_err(
            gpu.device,
            sel(c"newComputePipelineStateWithFunction:error:"),
            function,
            &mut error,
        )
    };
    assert!(
        !pipeline.is_null(),
        "compute pipeline: {}",
        nserror_message(error)
    );
    pipeline
}

fn check_completion(command_buffer: Id) {
    // SAFETY: `command_buffer` has completed via `waitUntilCompleted`.
    let status = unsafe { send_usize0(command_buffer, sel(c"status")) };
    if status != 4 {
        // SAFETY: `error` returns null or an autoreleased `NSError *`.
        let error = unsafe { send_id0(command_buffer, sel(c"error")) };
        panic!(
            "Metal command buffer failed (status={status}): {}",
            nserror_message(error)
        );
    }
}

/// Lowers, converts, dispatches one console threadgroup, and returns the UAV
/// words.
fn run(words: &[u32], threads: u32, lds_bytes: u32, narrow_waves: bool) -> Vec<u32> {
    let dxil = lowered_dxil(words, threads, lds_bytes, narrow_waves);
    let converted = convert(&dxil).expect("IRConverter conversion");
    dispatch(&converted, threads)
}

/// Binds the converted shader's argument buffers and dispatches `threads`
/// threads as one console threadgroup.
fn dispatch(converted: &Converted, threads: u32) -> Vec<u32> {
    let gpu = Gpu::new();
    let _pool = AutoreleasePool::new();
    let library = new_library(&gpu, &converted.metallib);
    let function = new_function(library, "main");
    let pipeline = new_compute_pipeline(&gpu, function);

    // One 32-byte row per console lane: the table descriptor's base address.
    let uav = shared_buffer(&gpu, threads as usize * 32);
    // The resource descriptor heap argument buffer at `buffer(0)`.
    let heap = shared_buffer(&gpu, DESCRIPTOR_BYTES * (DESCRIPTOR_INDEX + 1));
    // User data: dword 0 is the table's first descriptor byte offset in the
    // heap (0), so the table's 0x40 load names `DESCRIPTOR_INDEX`.
    let user_data = shared_buffer(&gpu, 16);
    // SAFETY: the three buffers are at least the lengths just requested.
    unsafe {
        std::ptr::write_bytes(contents(uav), 0, threads as usize * 32);
        std::ptr::write_bytes(contents(heap), 0, DESCRIPTOR_BYTES * (DESCRIPTOR_INDEX + 1));
        std::ptr::write_bytes(contents(user_data), 0, 16);
    }
    write_u64(heap, DESCRIPTOR_INDEX * DESCRIPTOR_BYTES, gpu_address(uav));
    let user_data_address = gpu_address(user_data);

    // SAFETY: `gpu.queue` is live; the command buffer and encoder are
    // autoreleased into `_pool` and stay alive until completion.
    let command_buffer = unsafe { send_id0(gpu.queue, sel(c"commandBuffer")) };
    assert!(!command_buffer.is_null(), "no command buffer");
    let encoder = unsafe { send_id0(command_buffer, sel(c"computeCommandEncoder")) };
    assert!(!encoder.is_null(), "no compute encoder");
    // SAFETY: `encoder` and `pipeline` are live.
    unsafe {
        send_void1(encoder, sel(c"setComputePipelineState:"), pipeline);
        send_set_buffer(encoder, heap, 0, 0);
    }

    // SAFETY: `gpu.device` and `encoder` are live for this scope.
    let mut binder = unsafe {
        ArgumentBufferEncoder::new(gpu.device, encoder, EncoderKind::Compute, &converted.layout)
    }
    .expect("compute argument-buffer encoder");
    // SAFETY: `user_data` and `uav` are live shared buffers that outlive the
    // dispatch; the bindings carry their device addresses.
    unsafe {
        binder
            .set_root_buffer_view(
                0,
                BufferBinding {
                    buffer: VmBufferId(1),
                    generation: 1,
                    address: user_data_address,
                    offset: 0,
                    length: 16,
                    registered_length: 16,
                },
                user_data,
                ResourceUsage::Read,
            )
            .expect("root CBV: the user-data block");
        binder
            .use_resource(uav, ResourceUsage::ReadWrite)
            .expect("UAV residency");
    }
    let _top_level = binder
        .bind_argument_buffer()
        .expect("top-level argument buffer must upload");

    // SAFETY: `encoder` is live; the console threadgroup is one group.
    unsafe {
        send_dispatch(
            encoder,
            MTLSize {
                width: threads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads as usize,
                height: 1,
                depth: 1,
            },
        );
        send_void0(encoder, sel(c"endEncoding"));
        send_void0(command_buffer, sel(c"commit"));
        send_void0(command_buffer, sel(c"waitUntilCompleted"));
    }
    check_completion(command_buffer);

    // SAFETY: the command buffer completed; `uav` is a live shared buffer of
    // `threads * 32` bytes.
    let out = contents(uav);
    unsafe { std::slice::from_raw_parts(out.cast::<u32>(), threads as usize * 8).to_vec() }
}

/// Per-lane console semantics for the reduction fixture.
fn console_reduction(threads: u32, l: u32, f: impl Fn(u32, u32) -> u32, input: impl Fn(u32) -> u32) -> u32 {
    let wave = l & !63;
    (wave + 1..(wave + 64).min(threads)).fold(input(wave), |a, t| f(a, input(t)))
}

/// Asserts each lane's `width` tagged dwords against the console model.
///
/// The fixtures write at `l * 32` bytes, so lane `l`'s dwords start at
/// `l * 8` in the UAV and `want` packs `width` dwords per lane.
fn assert_lanes(fixture: &str, got: &[u32], want: &[u32], width: usize) {
    assert!(width > 0 && want.len() % width == 0, "{fixture}: want width");
    let lanes = want.len() / width;
    assert!(
        got.len() >= lanes * 8,
        "{fixture}: {} dwords for {lanes} lanes",
        got.len()
    );
    for lane in 0..lanes {
        for dword in 0..width {
            assert_eq!(
                got[lane * 8 + dword],
                want[lane * width + dword],
                "{fixture}: lane {lane}, dword {dword}"
            );
        }
    }
}

// ----- fixtures (the same words as `halves_tests`) -------------------------

const V_ADD_F32: u32 = 0x03;

/// The console compiler's whole-wave reduction by the VOP2 operation `op`,
/// one dword per thread at `l * 32`.
fn reduction(op: u32, ident: u32, input: u32) -> Vec<u32> {
    let vop2 = |word: u32| word & !(0x3F << 25) | op << 25;
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        input,       // v3 from v0
        0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
        0xD501_0001,
        0x0032_0600 | ident, // v_cndmask_b32 v1, ident, v3, s[12:13]
    ]);
    for control in [0x161, 0x162, 0x164, 0x168] {
        w.extend([
            vop2(0x0602_02FA),
            0xFF00_0001 | control << 8, // op v1, v1, v1 row_xmask
        ]);
    }
    w.extend([
        0xD778_0002,
        0x0201_0101,       // v_permlanex16_b32 v2, v1, 0, 0
        vop2(0x0602_0302), // op v1, v2, v1
        0xD760_000E,
        0x0001_4101,       // v_readlane_b32 s14, v1, 32
        vop2(0x0602_020E), // op v1, s14, v1
        0xD760_000F,
        0x0001_0101, // v_readlane_b32 s15, v1, 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_020F, // v_mov_b32 v1, s15
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// `v_readlane_b32` of the thread id at console lanes `k` and `s`, two dwords
/// per thread at `l * 32`.
fn lane_reads(k: u32, s: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD760_000C,
        0x0001_0100 | (128 + k) << 9, // v_readlane_b32 s12, v0, k
        0xBE8D_03FF,
        s, // s_mov_b32 s13, s
        0xD760_000E,
        0x0000_1B00, // v_readlane_b32 s14, v0, s13
        0x7E02_020C, // v_mov_b32 v1, s12
        0x7E04_020E, // v_mov_b32 v2, s14
    ]);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.push(S_ENDPGM);
    w
}

/// The lanes below 40 as data: count, high ballot dword, compaction index, at
/// `l * 32`.
fn lanes_below() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E02_020C, // v_mov_b32 v1, s12
        0x7E04_026B, // v_mov_b32 v2, vcc_hi
        0xD765_0003,
        0x0001_006A, // v_mbcnt_lo_u32_b32 v3, vcc_lo, 0
        0xD766_0003,
        0x0002_066B, // v_mbcnt_hi_u32_b32 v3, vcc_hi, v3
    ]);
    for (vdata, slot) in [(1, 0), (2, 1), (3, 2)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// The console's exclusive prefix sum of the thread id, one dword per thread.
fn exclusive_prefix() -> Vec<u32> {
    const EXEC: u32 = 126;
    let sop1 = |sdst: u32, op: u32, ssrc0: u32| 0xBE80_0000 | sdst << 16 | op << 8 | ssrc0;
    let vop1 = |vdst: u32, src0: u32| 0x7E00_0200 | vdst << 17 | src0;
    let add = |src0: u32| 0x4A02_0200 | src0; // v_add_nc_u32 v1, src0, v1
    let dpp = |ctrl: u32| 0xFF00_0001 | ctrl << 8; // of v1, row and bank masks 0xf
    let readlane = |lane: u32| [0xD760_0002, 0x0000_0101 | (128 + lane) << 9]; // s2, v1
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085,        // v_lshlrev_b32 v16, 5, v0
        sop1(0, 0x25, 193), // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
    ]);
    for shift in [1, 2, 4, 8] {
        w.extend([add(0xFA), dpp(0x110 + shift)]); // v_add_nc_u32 v1, v1, v1 row_shr
    }
    // Each row's sum so far into the rows after it.
    for (last, exec_lo, exec_hi) in [
        (15, 0xFFFF_0000, 0),
        (31, 0, 0x0000_FFFF),
        (47, 0, 0xFFFF_0000),
    ] {
        w.extend(readlane(last));
        w.extend([
            sop1(EXEC, 0x03, 255),
            exec_lo,
            sop1(EXEC + 1, 0x03, 255),
            exec_hi,
        ]);
        w.push(add(2)); // v_add_nc_u32 v1, s2, v1
    }
    w.extend([
        sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
        vop1(3, 257),          // v_mov_b32 v3, v1
        vop1(3, 0xFA),
        dpp(0x111), // v_mov_b32 v3, v1 row_shr:1
    ]);
    // The first lane of each row from the last of the row before.
    for (last, exec_lo, exec_hi) in [(15, 0x0001_0000, 0), (31, 0, 1), (47, 0, 0x0001_0000)] {
        w.extend(readlane(last));
        w.extend([
            sop1(EXEC, 0x03, 255),
            exec_lo,
            sop1(EXEC + 1, 0x03, 255),
            exec_hi,
        ]);
        w.push(vop1(3, 2)); // v_mov_b32 v3, s2
    }
    w.extend([
        sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
        vop1(2, 128),          // v_mov_b32 v2, 0
        sop1(EXEC, 0x04, 194), // s_mov_b64 exec, -2
        vop1(2, 259),          // v_mov_b32 v2, v3
        sop1(EXEC, 0x04, 0),   // s_mov_b64 exec, s[0:1]
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// A one-wave group of 64 writes its `v_mbcnt` to LDS and reads the dword of
/// the lane 32 away (the other half), synced by a group barrier.
fn lds_across_halves() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_WAITCNT_LGKM,
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0300_0004, // ds_read_b32 v3, v4
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(3, 0));
    w.push(S_ENDPGM);
    w
}

/// `v_readfirstlane_b32` of `v0 >> 6` (one value per console wave), at `l * 32`.
///
/// The second dword is the lane id from the two-pass `v_mbcnt` (`lo` then
/// `hi`), the same words as `halves_tests::first_lane_alike`.
fn first_lane_alike() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x2C04_0086, // v_lshrrev_b32 v2, 6, v0
        0x7E18_0502, // v_readfirstlane_b32 s12, v2
        0x7E06_020C, // v_mov_b32 v3, s12
    ]);
    w.extend(store(3, 0));
    w.extend(store(1, 4));
    w.push(S_ENDPGM);
    w
}

/// `v_cmp_gt_u32 vcc, k, v0` (the lanes whose thread id is below `k`).
fn cmp_below(k: u32) -> u32 {
    0x7D88_0000 | (128 + k)
}

/// A lane mask written in a loop whose count is one value per console wave
/// (`v0 >> 6`'s first lane), read after the loop: the console wave's 64-bit
/// mask, counted, at `l * 32`. The write's block is the loop, which the two
/// host halves need not reach the same number of times; the block the read
/// starts in is one every lane runs, so that is where the halves' ballots
/// are exchanged.
fn mask_after_a_per_wave_loop() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x2C04_0086, // v_lshrrev_b32 v2, 6, v0
        0x7E2A_0502, // v_readfirstlane_b32 s21, v2
        0xBE94_0380, // s_mov_b32 s20, 0
        cmp_below(40), // header: v_cmp_gt_u32 vcc, 40, v0 (the mask)
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_1514, // s_cmp_lt_u32 s20, s21
        0xBF85_FFFC, // s_cbranch_scc1 (back to the compare)
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E02_020C, // v_mov_b32 v1, s12
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The live mask (EXEC) of a full console wave read as data, at `l * 32`: 64.
fn live_mask_read() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xBE8C_107E, // s_bcnt1_i32_b64 s12, exec
        0x7E02_020C, // v_mov_b32 v1, s12
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// A literal lane mask (`s[20:21] = -4`, 62 of 64 bits), used as a
/// `v_cndmask` predicate and read as data, at `l * 32`.
fn literal_mask_read() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xBE94_04C4, // s_mov_b64 s[20:21], -4
        0xD501_0006,
        0x0051_0280, // v_cndmask_b32 v6, 0, 1, s[20:21]
        0xBE8C_1014, // s_bcnt1_i32_b64 s12, s[20:21]
        0x7E02_020C, // v_mov_b32 v1, s12
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

// ----- the LDS, reduction, first-lane and lane-read buckets -------------

/// VOP2 opcodes of the integer reductions (see `emit::wave::Op`).
const V_ADD_U32: u32 = 0x25;
const V_MIN_U32: u32 = 0x13;
const V_MAX_U32: u32 = 0x14;
const V_AND_B32: u32 = 0x1b;
const V_OR_B32: u32 = 0x1c;
const V_XOR_B32: u32 = 0x1d;
/// `v_mov_b32 v3, v0`: the thread id as the reduction's input.
const V_MOV_V3_V0: u32 = 0x7E06_0300;
/// `v_add_nc_u32 v3, 1, v0`: 1..64, whose exclusive or is 64.
const V_ADD_V3_1_V0: u32 = 0x4A06_0081;
/// `v_xor_b32 v3, 42, v0`.
const V_XOR_V3_42_V0: u32 = 0x3A06_00AA;
/// `v_not_b32 v3, v0`.
const V_NOT_V3_V0: u32 = 0x7E06_6F00;

/// Four passes writing each lane's running sum to LDS and adding the sum of
/// the lane 32 away (the other half): the barrier each pass needs before its
/// read, and one before the next pass's write.
fn lds_in_a_loop() -> Vec<u32> {
    let body = [
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_WAITCNT_LGKM,
        0xD8D8_0000,
        0x0300_0004, // ds_read_b32 v3, v4
        S_WAITCNT_LGKM,
        0x4A02_0701, // v_add_nc_u32 v1, v1, v3
    ];
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0005,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v5, -1, 0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x7E02_0300, // v_mov_b32 v1, v0
        0xBE94_0380, // s_mov_b32 s20, 0
    ]);
    w.extend(body);
    let back = -(body.len() as i32 + 3);
    w.extend([
        0x8014_8114,                          // s_add_u32 s20, s20, 1
        0xBF0A_8414,                          // s_cmp_lt_u32 s20, 4
        0xBF85_0000 | (back as u32 & 0xFFFF), // s_cbranch_scc1 (back to the write)
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`lds_in_a_loop`]: each pass, a lane's sum and
/// that of the lane 32 away both become their total.
fn console_lds_in_a_loop() -> Vec<u32> {
    let mut sum: Vec<u32> = (0..64).collect();
    for _ in 0..4 {
        sum = (0..64).map(|l| sum[l] + sum[l ^ 32]).collect();
    }
    sum
}

/// `v_readfirstlane_b32` of the thread id (a per-lane value): the console's
/// first lane is 0, in the lower half.
fn first_lane_of_the_thread_id() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0 (console lanes)
        0x7E18_0500, // v_readfirstlane_b32 s12, v0
        0x7E04_020C, // v_mov_b32 v2, s12
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// `v_readlane_b32 s12, v3, s13` of a value every lane holds (`v3` = 1, the
/// lane in `s13` = 33): every lane already holds it, so no exchange.
fn lane_read_of_a_uniform_value() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0 (console lanes)
        0x7E06_0281, // v_mov_b32 v3, 1
        0xBE8D_03FF,
        33, // s_mov_b32 s13, 33
        0xD760_000C,
        0x0001_1B03, // v_readlane_b32 s12, v3, s13
        0x7E02_020C, // v_mov_b32 v1, s12
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The CPU lowering must emit DXIL for every fixture (no translation refusal).
#[test]
fn lowered_wave_fixtures_emit_dxil() {
    let c = empty_container();
    assert_eq!(
        dxbc::Container::parse(&c).expect("empty container").parts.len(),
        0
    );
    for (name, words, threads, lds) in [
        ("reduction", reduction(V_ADD_F32, 0x80, 0x7E06_0D00), 64, 0),
        ("lane_reads", lane_reads(40, 33), 64, 0),
        ("ballot", lanes_below(), 64, 0),
        ("prefix", exclusive_prefix(), 64, 0),
        ("lds", lds_across_halves(), 64, 256),
        ("first_lane", first_lane_alike(), 128, 0),
        ("mask_after_loop", mask_after_a_per_wave_loop(), 128, 0),
        ("live_mask", live_mask_read(), 64, 0),
        ("literal_mask", literal_mask_read(), 64, 0),
        ("lds_in_a_loop", lds_in_a_loop(), 64, 256),
        ("sum_u32", reduction(V_ADD_U32, 0x80, V_MOV_V3_V0), 64, 0),
        ("min_u32", reduction(V_MIN_U32, 0xC1, V_XOR_V3_42_V0), 64, 0),
        ("max_u32", reduction(V_MAX_U32, 0x80, V_XOR_V3_42_V0), 64, 0),
        ("or_u32", reduction(V_OR_B32, 0x80, V_MOV_V3_V0), 64, 0),
        ("and_u32", reduction(V_AND_B32, 0xC1, V_NOT_V3_V0), 64, 0),
        ("xor_u32", reduction(V_XOR_B32, 0x80, V_ADD_V3_1_V0), 64, 0),
        ("first_lane_tid", first_lane_of_the_thread_id(), 64, 0),
        ("lane_read_uniform", lane_read_of_a_uniform_value(), 64, 0),
    ] {
        let dxil = lowered_dxil(&words, threads, lds, lds > 0);
        assert!(dxil.len() > 4, "{name}: no DXIL");
    }
}

/// ADR-025a: `narrow_waves` without `host_wave32` is refused by the emitter
/// itself, before any DXIL exists (so before any metallib or GPU work), and
/// the supported pair still emits.
#[test]
fn narrow_waves_without_host_wave32_is_refused_before_conversion() {
    for words in [lanes_below(), live_mask_read()] {
        let err = emitted_dxil(
            &words,
            64,
            0,
            Options {
                host_wave32: false,
                narrow_waves: true,
                ..Options::default()
            },
        )
        .expect_err("the unsound pair must not lower");
        let text = err.to_string();
        assert!(
            matches!(err, victory_shader::ShaderError::Options(_)),
            "expected the options refusal: {text}"
        );
        assert!(
            text.contains("narrow_waves") && text.contains("host_wave32"),
            "the refusal must name both flags: {text}"
        );
    }
    // The same program with `host_wave32` emits DXIL.
    let dxil = lowered_dxil(&lanes_below(), 64, 0, true);
    assert!(dxil.windows(4).any(|w| w == b"DXIL"));
}

/// Hardware proof: every fixture, converted and run on the Metal GPU, gives
/// the console's per-lane dwords.
#[test]
#[ignore = "hardware: run once on an exclusive GPU with --ignored --test-threads=1"]
fn lowered_wave_patterns_run_like_the_console_on_metal() {
    let _ = converter().expect("IRConverter worker and library");

    // Whole-wave sum of the thread ids (f32 bits).
    let threads = 64;
    let want: Vec<u32> = (0..threads)
        .map(|l| (console_reduction(threads, l, |a, b| a + b, |t| t) as f32).to_bits())
        .collect();
    assert_lanes(
        "reduction",
        &run(&reduction(V_ADD_F32, 0x80, 0x7E06_0D00), threads, 0, false),
        &want,
        1,
    );

    // Two lane reads of the first console wave.
    let want: Vec<u32> = (0..threads).flat_map(|l| [(l & !63) + 40, (l & !63) + 33]).collect();
    assert_lanes(
        "lane reads",
        &run(&lane_reads(40, 33), threads, 0, false),
        &want,
        2,
    );

    // Ballot masks: count, high dword, compaction index.
    let want: Vec<u32> = (0..threads)
        .flat_map(|l| {
            let wave = l & !63;
            let below = |end: u32| 40u32.min(threads).min(end).saturating_sub(wave);
            let high = below(wave + 64).saturating_sub(32);
            [
                below(wave + 64),
                if high == 32 { u32::MAX } else { (1 << high) - 1 },
                below(l),
            ]
        })
        .collect();
    assert_lanes("ballot", &run(&lanes_below(), threads, 0, false), &want, 3);

    // Exclusive prefix sum of the thread id.
    let want: Vec<u32> = (0..threads)
        .map(|l| {
            let (wave, k) = (l & !63, l & 63);
            wave * k + k.saturating_sub(1) * k / 2
        })
        .collect();
    assert_lanes(
        "prefix",
        &run(&exclusive_prefix(), threads, 0, false),
        &want,
        1,
    );

    // LDS after the barrier: the lane 32 away's count.
    let want: Vec<u32> = (0..threads).map(|l| (l ^ 32).min(32)).collect();
    assert_lanes(
        "lds",
        &run(&lds_across_halves(), threads, 256, true),
        &want,
        1,
    );

    // First lane of each console wave, two dwords.
    let threads = 128;
    let want: Vec<u32> = (0..threads).flat_map(|l| [l >> 6, l & 63]).collect();
    assert_lanes(
        "first lane",
        &run(&first_lane_alike(), threads, 0, false),
        &want,
        2,
    );

    // A mask written in a per-wave-count loop, read after it: each console
    // wave's mask as a count. Wave 0 (threads 0-63) has the lanes below 40;
    // wave 1 (64-127) none.
    let threads = 128;
    let want: Vec<u32> = (0..threads)
        .map(|l| if l < 64 { 40 } else { 0 })
        .collect();
    assert_lanes(
        "mask after loop",
        &run(&mask_after_a_per_wave_loop(), threads, 0, false),
        &want,
        1,
    );

    // The live mask of a full console wave, read as data: 64.
    let threads = 64;
    let want: Vec<u32> = vec![64; threads as usize];
    assert_lanes(
        "live mask",
        &run(&live_mask_read(), threads, 0, false),
        &want,
        1,
    );

    // A literal mask (62 of 64 bits) read as its number: 62.
    let want: Vec<u32> = vec![62; threads as usize];
    assert_lanes(
        "literal mask",
        &run(&literal_mask_read(), threads, 0, false),
        &want,
        1,
    );
}

/// ADR-025a's supported pair, on the hardware: a 64-lane wave64 corpus mask
/// program is lowered with `host_wave32: true`, declares `WaveSize(32)` and
/// runs in halves, and every lane of both halves gets the console's dwords —
/// including the ballot's high dword, the bits 32..63 a 32-lane host would
/// lose if the pair were lowered as if it were 64 lanes.
#[test]
#[ignore = "hardware: run once on an exclusive GPU with --ignored --test-threads=1"]
fn a_wave64_corpus_mask_runs_on_both_halves_on_metal() {
    let _ = converter().expect("IRConverter worker and library");
    let threads = 64;

    // The live mask of the whole console wave: 64 in all 64 lanes. A high
    // ballot dword read from a 32-lane host would make it 32.
    let want: Vec<u32> = vec![64; threads as usize];
    assert_lanes(
        "live mask, both halves",
        &run(&live_mask_read(), threads, 0, false),
        &want,
        1,
    );

    // Lanes below 40 as data: the count, the ballot's high dword (bits
    // 32..39 = 0xff) and each lane's compaction index.
    let want: Vec<u32> = (0..threads).flat_map(|l| [40, 0xff, l.min(40)]).collect();
    assert_lanes(
        "ballot masks, both halves",
        &run(&lanes_below(), threads, 0, false),
        &want,
        3,
    );
}

/// The LDS bucket: a group that is one console wave64, whose LDS the console
/// orders by lockstep, gets a group barrier where each host half reads what
/// the other wrote — once in one block, twice in a loop — and every lane of
/// the console threadgroup reads the console's dword.
#[test]
#[ignore = "hardware: run once on an exclusive GPU with --ignored --test-threads=1"]
fn lds_barriers_run_like_the_console_on_metal() {
    let _ = converter().expect("IRConverter worker and library");

    // One barrier between the write and the read of the other half: a lane
    // reads the count the lane 32 away wrote (`v_mbcnt`, at most 32).
    let threads = 64;
    let want: Vec<u32> = (0..threads).map(|l| (l ^ 32).min(32)).collect();
    assert_lanes(
        "lds",
        &run(&lds_across_halves(), threads, 256, true),
        &want,
        1,
    );

    // Two barriers inside the loop: before each pass's read, and before the
    // next pass's write, four passes deep.
    let want = console_lds_in_a_loop();
    assert_lanes(
        "lds in a loop",
        &run(&lds_in_a_loop(), threads, 256, true),
        &want,
        1,
    );
}

/// The reduction bucket: whole-wave sums, min and max, the bit operations
/// and the prefix sum, each taken over both 32-lane halves through the
/// group-shared exchange, give every lane the console wave64's result.
#[test]
#[ignore = "hardware: run once on an exclusive GPU with --ignored --test-threads=1"]
fn reductions_across_halves_run_like_the_console_on_metal() {
    let _ = converter().expect("IRConverter worker and library");
    let threads = 64;

    // An unsigned sum of the thread id.
    let want: Vec<u32> = (0..threads)
        .map(|l| console_reduction(threads, l, |a, b| a + b, |t| t))
        .collect();
    assert_lanes(
        "sum",
        &run(&reduction(V_ADD_U32, 0x80, V_MOV_V3_V0), threads, 0, false),
        &want,
        1,
    );

    // The console's own f32 sum of the converted thread id.
    let want: Vec<u32> = (0..threads)
        .map(|l| (console_reduction(threads, l, |a, b| a + b, |t| t) as f32).to_bits())
        .collect();
    assert_lanes(
        "sum f32",
        &run(&reduction(V_ADD_F32, 0x80, 0x7E06_0D00), threads, 0, false),
        &want,
        1,
    );

    // Unsigned min and max of the thread id xor 42, in both halves' lanes.
    for (name, op, ident, f) in [
        ("min_u32", V_MIN_U32, 0xC1, u32::min as fn(u32, u32) -> u32),
        ("max_u32", V_MAX_U32, 0x80, u32::max as fn(u32, u32) -> u32),
    ] {
        let want: Vec<u32> = (0..threads)
            .map(|l| console_reduction(threads, l, f, |t| t ^ 42))
            .collect();
        assert_lanes(
            name,
            &run(&reduction(op, ident, V_XOR_V3_42_V0), threads, 0, false),
            &want,
            1,
        );
    }

    // The bit operations: or of 0..63, and of their complement, xor of 1..64.
    for (name, op, ident, input, f) in [
        ("or", V_OR_B32, 0x80, V_MOV_V3_V0, (|a, b| a | b) as fn(u32, u32) -> u32),
        ("and", V_AND_B32, 0xC1, V_NOT_V3_V0, |a, b| a & b),
        ("xor", V_XOR_B32, 0x80, V_ADD_V3_1_V0, |a, b| a ^ b),
    ] {
        let input_fn = |t: u32| match name {
            "and" => !t,
            "xor" => t + 1,
            _ => t,
        };
        let want: Vec<u32> = (0..threads)
            .map(|l| console_reduction(threads, l, f, input_fn))
            .collect();
        assert_lanes(
            name,
            &run(&reduction(op, ident, input), threads, 0, false),
            &want,
            1,
        );
    }

    // The exclusive prefix sum of the thread id: the upper half adds the
    // lower half's total before its own.
    let want: Vec<u32> = (0..threads)
        .map(|l| {
            let (wave, k) = (l & !63, l & 63);
            wave * k + k.saturating_sub(1) * k / 2
        })
        .collect();
    assert_lanes(
        "prefix",
        &run(&exclusive_prefix(), threads, 0, false),
        &want,
        1,
    );
}

/// The first-lane and lane-read bucket: a value every lane holds needs no
/// exchange, a per-lane one goes through the halves' exchange, and a lane
/// read of a constant console lane, of a wave-uniform value or of each
/// lane's own index gives every lane the console's dword.
#[test]
#[ignore = "hardware: run once on an exclusive GPU with --ignored --test-threads=1"]
fn first_lane_and_lane_reads_across_halves_run_like_the_console_on_metal() {
    let _ = converter().expect("IRConverter worker and library");

    // One value per console wave (`v0 >> 6`), the same in both its halves:
    // each half keeps its own first lane (no exchange), then its `v_mbcnt`.
    let threads = 128;
    let want: Vec<u32> = (0..threads).flat_map(|l| [l >> 6, l & 63]).collect();
    assert_lanes(
        "first lane alike",
        &run(&first_lane_alike(), threads, 0, false),
        &want,
        2,
    );

    // A per-lane value: the lower half's first lane, exchanged, is 0.
    let threads = 64;
    let want = vec![0u32; threads as usize];
    assert_lanes(
        "first lane",
        &run(&first_lane_of_the_thread_id(), threads, 0, false),
        &want,
        1,
    );

    // Constant console lanes 40 and 33, both in the other half for every
    // lane of the lower one.
    let want: Vec<u32> = (0..threads).flat_map(|_| [40, 33]).collect();
    assert_lanes(
        "lane reads",
        &run(&lane_reads(40, 33), threads, 0, false),
        &want,
        2,
    );

    // A value every lane holds, read at a lane in the other half: kept as
    // it is, no exchange.
    let want = vec![1u32; threads as usize];
    assert_lanes(
        "lane read of a uniform value",
        &run(&lane_read_of_a_uniform_value(), threads, 0, false),
        &want,
        1,
    );
}
