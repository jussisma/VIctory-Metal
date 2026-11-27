//! Native Metal mesh pipeline and mesh dispatch.
//!
//! This module owns the `MTLMeshRenderPipelineDescriptor` path:
//!
//! * create the descriptor with `meshFunction`, `fragmentFunction`, an
//!   optional `objectFunction`, `maxTotalThreadsPerMeshThreadgroup`,
//!   `payloadMemoryLength` and the colour / depth attachment pixel formats of
//!   the pass it draws into;
//! * create an `MTLRenderPipelineState` with
//!   `newRenderPipelineStateWithMeshDescriptor:error:` and a readable error;
//! * record `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`
//!   on a render encoder (the encoder side lives in [`crate::macos::render`]).
//!
//! Nothing here falls back to `MTLRenderPipelineDescriptor` or to D3DMetal: an
//! unsupported request returns an explicit [`MeshError`] before Metal is asked
//! to create a pipeline.
//!
//! # Device budgets and the compute-path failure
//!
//! Metal bounds a mesh shader's declared output at 32 KiB. A declaration over
//! that budget is refused here with [`MeshError::Budget`] *before* the
//! metallib is handed to `MTLDevice`. When the check is skipped, Metal itself
//! rejects the descriptor cleanly (`Total mesh size (33280) exceeds the
//! maximum mesh size allowed (32768)`), so the budget is not the cause of the
//! historical `XPC_ERROR_CONNECTION_INTERRUPTED`.
//!
//! That XPC failure comes from the *compute* path: `load_metallib` resolves
//! the named entry and asks `newComputePipelineStateWithFunction:error:` for a
//! compute pipeline. A mesh-stage function is not a kernel; on this host
//! `MTLCompilerService` crashes compiling it as one
//! (`EXC_BAD_ACCESS` in `MTLCompilerObject::backendCompileModule`) and Metal
//! reports the interrupted XPC connection. The same library loads fine as an
//! `MTLLibrary`, and this module's mesh descriptor path creates the pipeline
//! (or returns Metal's own budget diagnostic) without that crash. Callers with
//! a mesh function must therefore use [`MeshPipeline`], never
//! [`DirectMetalTransport::load_metallib`].

use crate::macos::bindings::{
    DescriptorTableKind, MAX_ARGUMENT_BUFFER_BYTES, RootParameterKind, RootSignatureLayout,
};
use crate::macos::device::{DirectMetalTransport, PipelineAbi};
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::macos::render::{PixelFormat, new_less_depth_state};
use crate::transport::MetalError;
use crate::wire_v3::MAX_COLOR_ATTACHMENTS;

/// Metal's maximum declared mesh output bytes (32 KiB).
pub const METAL_MESH_OUTPUT_BYTES: usize = 32 * 1024;

/// Largest root-constant block a mesh pipeline advertises inline: 64 dwords.
pub const MAX_MESH_ROOT_CONSTANT_DWORDS: usize = 64;

/// One reflected descriptor table and its entry count.
///
/// The converter reflection says a root parameter is a table but not which
/// range kind it stores. The caller supplies the kind, exactly as
/// [`RootSignatureLayout::with_descriptor_table_kinds`] does for graphics and
/// compute, plus the entry count the wire submission will bind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshTableExtent {
    /// Range kind of the table.
    pub kind: DescriptorTableKind,
    /// Entries the submission binds.
    pub count: u32,
}

impl MeshTableExtent {
    /// Converted bytes the table occupies (`count * kind.descriptor_bytes()`).
    pub fn bytes(&self) -> Result<usize, MeshError> {
        (self.count as usize)
            .checked_mul(self.kind.descriptor_bytes())
            .ok_or(MeshError::Budget("descriptor table extent overflows usize"))
    }
}

/// The converted root-signature binding plan of one mesh pipeline.
///
/// A [`MeshPipelineSpec`] decides whether Metal can create the pipeline; this
/// plan decides what the live v3 executor must bind for it. It is built from
/// the three reflected stage layouts (mesh, optional object, fragment) and the
/// descriptor tables the submission will bind:
///
/// * root constants stay inline in the stage's top-level argument buffer and
///   are limited to [`MAX_MESH_ROOT_CONSTANT_DWORDS`] dwords, stricter than the
///   converter's general 4 KiB ceiling;
/// * each stage's argument buffer must fit [`MAX_ARGUMENT_BUFFER_BYTES`];
/// * every descriptor table's converted extent (`count * stride`) must fit the
///   same limit, so a packed table larger than the limit is refused before any
///   `MTLBuffer` is uploaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshAbiPlan {
    mesh: RootSignatureLayout,
    object: Option<RootSignatureLayout>,
    fragment: Option<RootSignatureLayout>,
    tables: Vec<MeshTableExtent>,
    argument_buffer_bytes: usize,
    descriptor_table_bytes: usize,
}

impl MeshAbiPlan {
    /// Builds and validates the plan from reflected stage layouts.
    pub fn new(
        mesh: RootSignatureLayout,
        object: Option<RootSignatureLayout>,
        fragment: Option<RootSignatureLayout>,
        tables: &[MeshTableExtent],
    ) -> Result<Self, MeshError> {
        validate_layout(&mesh)?;
        if let Some(object) = object.as_ref() {
            validate_layout(object)?;
        }
        if let Some(fragment) = fragment.as_ref() {
            validate_layout(fragment)?;
        }
        let argument_buffer_bytes = mesh
            .argument_buffer_len()
            .checked_add(
                object
                    .as_ref()
                    .map_or(0, RootSignatureLayout::argument_buffer_len),
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    fragment
                        .as_ref()
                        .map_or(0, RootSignatureLayout::argument_buffer_len),
                )
            })
            .ok_or(MeshError::Budget(
                "mesh argument-buffer sizes overflow usize",
            ))?;
        let mut plan = Self {
            mesh,
            object,
            fragment,
            tables: Vec::new(),
            argument_buffer_bytes,
            descriptor_table_bytes: 0,
        };
        plan.set_tables(tables)?;
        Ok(plan)
    }

    /// Builds the plan from Apple's `IRShaderReflectionCopyJSONString` output.
    ///
    /// A mesh pipeline always has a fragment stage, but its reflection may be
    /// empty when the stage does not read the root signature.
    pub fn from_reflection_json(
        mesh_json: &str,
        object_json: Option<&str>,
        fragment_json: Option<&str>,
        tables: &[MeshTableExtent],
    ) -> Result<Self, MeshError> {
        let parse = |json: &str, stage: &str| {
            RootSignatureLayout::from_converter_reflection_json(json)
                .map_err(|error| MeshError::Create(format!("{stage} reflection: {error}")))
        };
        let mesh = parse(mesh_json, "mesh")?;
        let object = object_json.map(|json| parse(json, "object")).transpose()?;
        let fragment = fragment_json
            .map(|json| parse(json, "fragment"))
            .transpose()?;
        Self::new(mesh, object, fragment, tables)
    }

    /// Replaces the descriptor tables and revalidates the packed extent.
    pub fn set_tables(&mut self, tables: &[MeshTableExtent]) -> Result<(), MeshError> {
        let mut total = 0usize;
        for table in tables {
            total = total
                .checked_add(table.bytes()?)
                .ok_or(MeshError::Budget("descriptor table extents overflow usize"))?;
        }
        if total > MAX_ARGUMENT_BUFFER_BYTES {
            return Err(MeshError::Budget(
                "descriptor tables exceed the 64 KiB argument-buffer limit",
            ));
        }
        self.tables = tables.to_vec();
        self.descriptor_table_bytes = total;
        Ok(())
    }

    /// Mesh-stage reflected layout.
    pub fn mesh(&self) -> &RootSignatureLayout {
        &self.mesh
    }

    /// Object-stage reflected layout, when the pipeline has an object stage.
    pub fn object(&self) -> Option<&RootSignatureLayout> {
        self.object.as_ref()
    }

    /// Fragment-stage reflected layout.
    pub fn fragment(&self) -> Option<&RootSignatureLayout> {
        self.fragment.as_ref()
    }

    /// Descriptor tables the submission binds.
    pub fn tables(&self) -> &[MeshTableExtent] {
        &self.tables
    }

    /// Sum of the three stage argument-buffer lengths.
    pub fn argument_buffer_bytes(&self) -> usize {
        self.argument_buffer_bytes
    }

    /// Sum of every descriptor table's converted extent.
    pub fn descriptor_table_bytes(&self) -> usize {
        self.descriptor_table_bytes
    }

    /// The ABI the live v3 executor stores with the mesh pipeline handle.
    pub fn pipeline_abi(&self) -> PipelineAbi {
        PipelineAbi::mesh(
            Some(self.mesh.clone()),
            self.object.clone(),
            self.fragment.clone(),
        )
    }
}

/// Rejects a stage layout whose constants or argument buffer exceed a mesh
/// pipeline's advertised budgets.
fn validate_layout(layout: &RootSignatureLayout) -> Result<(), MeshError> {
    if layout.argument_buffer_len() > MAX_ARGUMENT_BUFFER_BYTES {
        return Err(MeshError::Budget(
            "stage argument buffer exceeds the 64 KiB limit",
        ));
    }
    let mut constants = 0usize;
    for parameter in layout.parameters() {
        if parameter.kind() == RootParameterKind::Constants {
            constants = constants
                .checked_add(parameter.size())
                .ok_or(MeshError::Budget("root-constant sizes overflow usize"))?;
        }
    }
    if constants > MAX_MESH_ROOT_CONSTANT_DWORDS * 4 {
        return Err(MeshError::Budget("mesh root constants exceed 64 dwords"));
    }
    Ok(())
}

/// Why a mesh pipeline request could not be satisfied on this device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshError {
    /// The host OS or device does not offer mesh pipeline objects.
    Unsupported(&'static str),
    /// One of the pipeline's declared budgets is outside the device limits.
    Budget(&'static str),
    /// Metal refused to create the pipeline; the message is Metal's own.
    Create(String),
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(why) => write!(f, "mesh pipeline unsupported: {why}"),
            Self::Budget(why) => write!(f, "mesh pipeline budget: {why}"),
            Self::Create(why) => write!(f, "mesh pipeline creation failed: {why}"),
        }
    }
}

impl std::error::Error for MeshError {}

impl From<MeshError> for MetalError {
    fn from(error: MeshError) -> Self {
        match error {
            MeshError::Unsupported(why) => MetalError::NotSupported(why),
            MeshError::Budget(why) => MetalError::InvalidArg(why),
            MeshError::Create(message) => MetalError::CompileFailed(message),
        }
    }
}

/// The pipeline-facing half of a mesh request.
///
/// Only the descriptor's budgets and attachment formats live here. The actual
/// threadgroup sizes used by a draw are carried separately because
/// `DispatchMesh` on the wire names only group counts: the executor takes the
/// per-stage sizes from the registered pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshPipelineSpec {
    /// Colour attachment formats, in slot order (1..=8).
    pub color_formats: Vec<PixelFormat>,
    /// Optional depth (and stencil) attachment format.
    pub depth_format: Option<PixelFormat>,
    /// `maxTotalThreadsPerMeshThreadgroup`; `0` uses the shader's declared
    /// maximum, which is then taken to be the draw's mesh threadgroup product.
    pub max_total_threads_per_mesh_threadgroup: usize,
    /// `maxTotalThreadsPerObjectThreadgroup`; `0` leaves the descriptor default.
    pub max_total_threads_per_object_threadgroup: usize,
    /// `payloadMemoryLength` in bytes.
    pub payload_memory_length: usize,
    /// `maxTotalThreadgroupsPerMeshGrid`; `0` leaves the descriptor default.
    pub max_total_threadgroups_per_mesh_grid: usize,
    /// Threads per object threadgroup used by `DispatchMesh`. `[1, 1, 1]`
    /// when the pipeline has no object function.
    pub threads_per_object_threadgroup: [u32; 3],
    /// Threads per mesh threadgroup used by `DispatchMesh`.
    pub threads_per_mesh_threadgroup: [u32; 3],
    /// The mesh shader's declared output bytes, checked against
    /// [`METAL_MESH_OUTPUT_BYTES`]. `0` skips the check.
    pub mesh_output_bytes: usize,
}

impl MeshPipelineSpec {
    /// A single-mesh-threadgroup spec for `color_formats` and `depth_format`.
    ///
    /// `threads_per_mesh_threadgroup` fixes both the draw size and, unless the
    /// caller later changes it, the descriptor's mesh threadgroup maximum.
    pub fn new(
        color_formats: Vec<PixelFormat>,
        depth_format: Option<PixelFormat>,
        threads_per_mesh_threadgroup: [u32; 3],
    ) -> Self {
        Self {
            color_formats,
            depth_format,
            max_total_threads_per_mesh_threadgroup: threadgroup_product(
                threads_per_mesh_threadgroup,
            )
            .unwrap_or(0),
            max_total_threads_per_object_threadgroup: 0,
            payload_memory_length: 0,
            max_total_threadgroups_per_mesh_grid: 0,
            threads_per_object_threadgroup: [1, 1, 1],
            threads_per_mesh_threadgroup,
            mesh_output_bytes: 0,
        }
    }

    /// Checks every declared budget against the pass formats and the device.
    ///
    /// `max_threads_per_threadgroup` is the product of
    /// `MTLDevice.maxThreadsPerThreadgroup`; `max_threadgroup_memory` is
    /// `MTLDevice.maxThreadgroupMemoryLength`.
    pub(crate) fn validate_limits(
        &self,
        max_threads_per_threadgroup: usize,
        max_threadgroup_memory: usize,
    ) -> Result<(), MeshError> {
        if self.color_formats.is_empty() || self.color_formats.len() > MAX_COLOR_ATTACHMENTS {
            return Err(MeshError::Unsupported(
                "a mesh pipeline needs 1..=8 colour attachment formats",
            ));
        }
        if self.color_formats.iter().any(|format| !format.is_color()) {
            return Err(MeshError::Unsupported(
                "mesh pipeline colour formats must be BGRA8Unorm or RGBA8Unorm",
            ));
        }
        if let Some(format) = self.depth_format
            && !format.is_depth()
        {
            return Err(MeshError::Unsupported(
                "mesh pipeline depth format must be Depth32Float or Depth32Float_Stencil8",
            ));
        }
        if self.mesh_output_bytes > METAL_MESH_OUTPUT_BYTES {
            return Err(MeshError::Budget(
                "mesh output declaration exceeds Metal's 32 KiB budget",
            ));
        }

        let device_max = max_threads_per_threadgroup.max(1);
        let mesh = threadgroup_product(self.threads_per_mesh_threadgroup)
            .ok_or(MeshError::Budget("mesh threadgroup dimensions overflow"))?;
        let object = threadgroup_product(self.threads_per_object_threadgroup)
            .ok_or(MeshError::Budget("object threadgroup dimensions overflow"))?;
        if mesh == 0 || object == 0 {
            return Err(MeshError::Budget(
                "mesh and object threadgroup dimensions must be nonzero",
            ));
        }
        if mesh > device_max || object > device_max {
            return Err(MeshError::Budget(
                "threadgroup size exceeds the device threadgroup maximum",
            ));
        }

        let mesh_max = if self.max_total_threads_per_mesh_threadgroup == 0 {
            mesh
        } else {
            self.max_total_threads_per_mesh_threadgroup
        };
        if mesh_max > device_max {
            return Err(MeshError::Budget(
                "maxTotalThreadsPerMeshThreadgroup exceeds the device threadgroup maximum",
            ));
        }
        if mesh > mesh_max {
            return Err(MeshError::Budget(
                "mesh threadgroup size exceeds maxTotalThreadsPerMeshThreadgroup",
            ));
        }
        if self.max_total_threads_per_object_threadgroup > device_max {
            return Err(MeshError::Budget(
                "maxTotalThreadsPerObjectThreadgroup exceeds the device threadgroup maximum",
            ));
        }
        if self.max_total_threads_per_object_threadgroup != 0
            && object > self.max_total_threads_per_object_threadgroup
        {
            return Err(MeshError::Budget(
                "object threadgroup size exceeds maxTotalThreadsPerObjectThreadgroup",
            ));
        }
        if self.payload_memory_length > max_threadgroup_memory.max(1) {
            return Err(MeshError::Budget(
                "payloadMemoryLength exceeds the device threadgroup memory",
            ));
        }
        Ok(())
    }
}

/// A compiled `MTLRenderPipelineState` created from a mesh descriptor.
pub struct MeshPipeline {
    state: ObjcObject,
    color_formats: Vec<PixelFormat>,
    depth_format: Option<PixelFormat>,
    depth_state: Option<ObjcObject>,
    threads_per_object_threadgroup: MTLSize,
    threads_per_mesh_threadgroup: MTLSize,
    max_total_threadgroups_per_mesh_grid: usize,
}

impl MeshPipeline {
    /// Compiles `source` once and resolves a mesh + fragment pair (plus an
    /// optional object entry) from it.
    ///
    /// The fragment stage is required: a mesh stage without one is not a valid
    /// render pipeline. Use [`Self::from_metallibs`] for precompiled libraries
    /// such as the bytes a Rosetta IRConverter worker produces.
    pub fn compile_msl(
        transport: &DirectMetalTransport,
        source: &str,
        mesh_entry: &str,
        object_entry: Option<&str>,
        fragment_entry: &str,
        spec: &MeshPipelineSpec,
    ) -> Result<Self, MeshError> {
        if source.is_empty() || mesh_entry.is_empty() || fragment_entry.is_empty() {
            return Err(MeshError::Unsupported(
                "MSL source and both mesh and fragment entry names must be non-empty",
            ));
        }
        if let Some(entry) = object_entry
            && entry.is_empty()
        {
            return Err(MeshError::Unsupported(
                "object entry name must be non-empty",
            ));
        }
        device_limits(transport, spec)?;
        let library = compile_library(transport, source)?;
        let mesh = library_function(&library, mesh_entry, "mesh")?;
        let object = match object_entry {
            Some(entry) => Some(library_function(&library, entry, "object")?),
            None => None,
        };
        let fragment = library_function(&library, fragment_entry, "fragment")?;
        build(
            transport,
            &mesh,
            object.as_ref(),
            &fragment,
            spec,
            mesh_entry,
            fragment_entry,
        )
    }

    /// Loads precompiled metallibs and builds the mesh pipeline.
    ///
    /// `object_library` is optional; `fragment_library` is required. Each
    /// library is copied into a `dispatch_data_t` and loaded with
    /// `newLibraryWithData:error:`, so no source or filesystem path is needed
    /// at run time.
    pub fn from_metallibs(
        transport: &DirectMetalTransport,
        mesh_library: &[u8],
        mesh_entry: &str,
        object_library: Option<(&[u8], &str)>,
        fragment_library: &[u8],
        fragment_entry: &str,
        spec: &MeshPipelineSpec,
    ) -> Result<Self, MeshError> {
        if mesh_library.is_empty() || mesh_entry.is_empty() {
            return Err(MeshError::Unsupported("mesh metallib or entry is empty"));
        }
        if fragment_library.is_empty() || fragment_entry.is_empty() {
            return Err(MeshError::Unsupported(
                "mesh stage without a fragment stage is not a render pipeline",
            ));
        }
        if let Some((bytes, entry)) = object_library
            && (bytes.is_empty() || entry.is_empty())
        {
            return Err(MeshError::Unsupported(
                "object metallib and entry must be non-empty when given",
            ));
        }
        device_limits(transport, spec)?;

        let mesh_lib = load_library(transport.device_raw(), mesh_library, "mesh")?;
        let mesh = library_function(&mesh_lib, mesh_entry, "mesh")?;
        let object = match object_library {
            Some((bytes, entry)) => {
                let lib = load_library(transport.device_raw(), bytes, "object")?;
                Some(library_function(&lib, entry, "object")?)
            }
            None => None,
        };
        let fragment_lib = load_library(transport.device_raw(), fragment_library, "fragment")?;
        let fragment = library_function(&fragment_lib, fragment_entry, "fragment")?;
        build(
            transport,
            &mesh,
            object.as_ref(),
            &fragment,
            spec,
            mesh_entry,
            fragment_entry,
        )
    }

    /// Colour attachment formats this pipeline declares, in slot order.
    pub(crate) fn color_formats(&self) -> &[PixelFormat] {
        &self.color_formats
    }

    /// Depth attachment format this pipeline declares.
    pub(crate) fn depth_format(&self) -> Option<PixelFormat> {
        self.depth_format
    }

    /// Raw `MTLRenderPipelineState` handle.
    pub(crate) fn state_raw(&self) -> objc::Id {
        self.state.as_raw()
    }

    /// Raw `MTLDepthStencilState` handle, when the pipeline tests depth.
    pub(crate) fn depth_state_raw(&self) -> Option<objc::Id> {
        self.depth_state.as_ref().map(ObjcObject::as_raw)
    }

    /// Threads per object threadgroup recorded for wire `DispatchMesh`.
    pub(crate) fn threads_per_object_threadgroup(&self) -> MTLSize {
        self.threads_per_object_threadgroup
    }

    /// Threads per mesh threadgroup recorded for wire `DispatchMesh`.
    pub(crate) fn threads_per_mesh_threadgroup(&self) -> MTLSize {
        self.threads_per_mesh_threadgroup
    }

    /// Declared `maxTotalThreadgroupsPerMeshGrid`; `0` means device default.
    pub(crate) fn max_total_threadgroups_per_mesh_grid(&self) -> usize {
        self.max_total_threadgroups_per_mesh_grid
    }
}

/// Reads the device's threadgroup budgets and validates `spec` against them.
fn device_limits(
    transport: &DirectMetalTransport,
    spec: &MeshPipelineSpec,
) -> Result<(), MeshError> {
    let descriptor_class = objc::class(c"MTLMeshRenderPipelineDescriptor");
    if descriptor_class.is_null() {
        return Err(MeshError::Unsupported(
            "MTLMeshRenderPipelineDescriptor is unavailable on this host",
        ));
    }
    let selector = objc::sel(c"newRenderPipelineStateWithMeshDescriptor:error:");
    // SAFETY: `device_raw()` is a live MTLDevice and `selector` is a registered selector.
    if !unsafe { objc::msg_send_responds_to(transport.device_raw(), selector) } {
        return Err(MeshError::Unsupported(
            "MTLDevice does not implement newRenderPipelineStateWithMeshDescriptor:error:",
        ));
    }

    let _pool = AutoreleasePool::new();
    // SAFETY: the device selector returns the documented `MTLSize`; the helper
    // implements the struct-return ABI for both supported architectures.
    let max = unsafe {
        objc::msg_send_mtl_size_0(
            transport.device_raw(),
            objc::sel(c"maxThreadsPerThreadgroup"),
        )
    };
    let max_threads = max
        .width
        .saturating_mul(max.height)
        .saturating_mul(max.depth)
        .max(1);
    // SAFETY: the device selector returns `NSUInteger`.
    let max_memory = unsafe {
        objc::msg_send_usize_0(
            transport.device_raw(),
            objc::sel(c"maxThreadgroupMemoryLength"),
        )
    };
    spec.validate_limits(max_threads, max_memory)?;
    Ok(())
}

/// Copies `bytes` into a dispatch data object and loads it as an `MTLLibrary`.
fn load_library(
    device: objc::Id,
    bytes: &[u8],
    stage: &'static str,
) -> Result<ObjcObject, MeshError> {
    let data = objc::dispatch_data_copy(bytes).ok_or_else(|| {
        MeshError::Create(format!(
            "{stage} metallib bytes are empty or allocation failed"
        ))
    })?;
    let mut ns_err = objc::empty_id();
    // SAFETY: `device` is a live MTLDevice, `data` is a retained dispatch data
    // object, and `ns_err` receives an autoreleased `NSError *` on failure.
    let raw = unsafe {
        objc::msg_send_id_err(
            device,
            objc::sel(c"newLibraryWithData:error:"),
            data.as_raw(),
            &mut ns_err,
        )
    };
    ObjcObject::from_retained(raw).ok_or_else(|| {
        // SAFETY: Metal populated `ns_err` when the library failed to load.
        let message = unsafe { objc::nserror_message(ns_err) };
        MeshError::Create(format!("{stage} metallib: {message}"))
    })
}

/// Compiles one MSL source string into an `MTLLibrary`.
fn compile_library(
    transport: &DirectMetalTransport,
    source: &str,
) -> Result<ObjcObject, MeshError> {
    let source_ns = objc::nsstring(source)
        .ok_or_else(|| MeshError::Create("MSL source string allocation failed".to_string()))?;
    let mut ns_err = objc::empty_id();
    // SAFETY: `device` is a live MTLDevice, `source_ns` is an immutable
    // NSString, `options` is nil, and `ns_err` receives an `NSError *`.
    let raw = unsafe {
        objc::msg_send_id_3_err(
            transport.device_raw(),
            objc::sel(c"newLibraryWithSource:options:error:"),
            source_ns.as_raw(),
            objc::empty_id(),
            &mut ns_err,
        )
    };
    ObjcObject::from_retained(raw).ok_or_else(|| {
        // SAFETY: Metal populated `ns_err` when compilation failed.
        let message = unsafe { objc::nserror_message(ns_err) };
        MeshError::Create(format!("MSL compilation: {message}"))
    })
}

/// Resolves one named entry point from a compiled library.
fn library_function(
    library: &ObjcObject,
    entry: &str,
    stage: &'static str,
) -> Result<ObjcObject, MeshError> {
    let name = objc::nsstring(entry)
        .ok_or_else(|| MeshError::Create(format!("{stage} entry name allocation failed")))?;
    // SAFETY: `library` is a live MTLLibrary and `name` is an NSString.
    let raw = unsafe {
        objc::msg_send_id_1(
            library.as_raw(),
            objc::sel(c"newFunctionWithName:"),
            name.as_raw(),
        )
    };
    ObjcObject::from_retained(raw).ok_or_else(|| {
        MeshError::Create(format!(
            "{stage} function `{entry}` not found in the compiled MTLLibrary"
        ))
    })
}

/// Creates the descriptor, pipeline state and optional depth-stencil state.
fn build(
    transport: &DirectMetalTransport,
    mesh: &ObjcObject,
    object: Option<&ObjcObject>,
    fragment: &ObjcObject,
    spec: &MeshPipelineSpec,
    mesh_entry: &str,
    fragment_entry: &str,
) -> Result<MeshPipeline, MeshError> {
    let _pool = AutoreleasePool::new();
    let descriptor_class = objc::class(c"MTLMeshRenderPipelineDescriptor");
    if descriptor_class.is_null() {
        return Err(MeshError::Unsupported(
            "MTLMeshRenderPipelineDescriptor is unavailable on this host",
        ));
    }
    // SAFETY: `descriptor_class` is the documented class and `new` returns a
    // retained, zero-initialized descriptor.
    let descriptor_raw = unsafe { objc::msg_send_id_0(descriptor_class, objc::sel(c"new")) };
    let descriptor = ObjcObject::from_retained(descriptor_raw).ok_or_else(|| {
        MeshError::Create("MTLMeshRenderPipelineDescriptor creation returned null".to_string())
    })?;

    let mesh_max = if spec.max_total_threads_per_mesh_threadgroup == 0 {
        threadgroup_product(spec.threads_per_mesh_threadgroup).unwrap_or(1)
    } else {
        spec.max_total_threads_per_mesh_threadgroup
    };
    // SAFETY: every receiver is a live Metal object and the selectors take the
    // documented object/`NSUInteger` arguments.
    unsafe {
        objc::msg_send_void_1(
            descriptor.as_raw(),
            objc::sel(c"setMeshFunction:"),
            mesh.as_raw(),
        );
        objc::msg_send_void_1(
            descriptor.as_raw(),
            objc::sel(c"setFragmentFunction:"),
            fragment.as_raw(),
        );
        if let Some(object) = object {
            objc::msg_send_void_1(
                descriptor.as_raw(),
                objc::sel(c"setObjectFunction:"),
                object.as_raw(),
            );
            if spec.max_total_threads_per_object_threadgroup != 0 {
                objc::msg_send_void_usize(
                    descriptor.as_raw(),
                    objc::sel(c"setMaxTotalThreadsPerObjectThreadgroup:"),
                    spec.max_total_threads_per_object_threadgroup,
                );
            }
        }
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setMaxTotalThreadsPerMeshThreadgroup:"),
            mesh_max,
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setPayloadMemoryLength:"),
            spec.payload_memory_length,
        );
        if spec.max_total_threadgroups_per_mesh_grid != 0 {
            objc::msg_send_void_usize(
                descriptor.as_raw(),
                objc::sel(c"setMaxTotalThreadgroupsPerMeshGrid:"),
                spec.max_total_threadgroups_per_mesh_grid,
            );
        }
        let attachments = objc::msg_send_id_0(descriptor.as_raw(), objc::sel(c"colorAttachments"));
        if attachments.is_null() {
            return Err(MeshError::Create(
                "MTLMeshRenderPipelineDescriptor.colorAttachments is nil".to_string(),
            ));
        }
        for (index, format) in spec.color_formats.iter().enumerate() {
            let attachment = objc::msg_send_id_usize(
                attachments,
                objc::sel(c"objectAtIndexedSubscript:"),
                index,
            );
            if attachment.is_null() {
                return Err(MeshError::Create(
                    "mesh colour attachment descriptor is nil".to_string(),
                ));
            }
            objc::msg_send_void_usize(
                attachment,
                objc::sel(c"setPixelFormat:"),
                format.mtl_pixel_format(),
            );
        }
        if let Some(format) = spec.depth_format {
            objc::msg_send_void_usize(
                descriptor.as_raw(),
                objc::sel(c"setDepthAttachmentPixelFormat:"),
                format.mtl_pixel_format(),
            );
            if format.has_stencil() {
                objc::msg_send_void_usize(
                    descriptor.as_raw(),
                    objc::sel(c"setStencilAttachmentPixelFormat:"),
                    format.mtl_pixel_format(),
                );
            }
        }
    }

    let depth_state = match spec.depth_format {
        Some(_) => Some(
            new_less_depth_state(transport)
                .map_err(|error| MeshError::Create(error.to_string()))?,
        ),
        None => None,
    };

    let mut ns_err = objc::empty_id();
    // SAFETY: `descriptor` is fully configured and `ns_err` receives an
    // autoreleased `NSError *` on failure.
    let state_raw = unsafe {
        objc::msg_send_id_err(
            transport.device_raw(),
            objc::sel(c"newRenderPipelineStateWithMeshDescriptor:error:"),
            descriptor.as_raw(),
            &mut ns_err,
        )
    };
    let state = ObjcObject::from_retained(state_raw).ok_or_else(|| {
        // SAFETY: Metal populated `ns_err` when pipeline creation failed.
        let message = unsafe { objc::nserror_message(ns_err) };
        MeshError::Create(format!(
            "mesh `{mesh_entry}` / fragment `{fragment_entry}` pipeline: {message}"
        ))
    })?;

    Ok(MeshPipeline {
        state,
        color_formats: spec.color_formats.clone(),
        depth_format: spec.depth_format,
        depth_state,
        threads_per_object_threadgroup: mtl_size(spec.threads_per_object_threadgroup),
        threads_per_mesh_threadgroup: mtl_size(spec.threads_per_mesh_threadgroup),
        max_total_threadgroups_per_mesh_grid: spec.max_total_threadgroups_per_mesh_grid,
    })
}

fn threadgroup_product(dims: [u32; 3]) -> Option<usize> {
    (dims[0] as usize)
        .checked_mul(dims[1] as usize)?
        .checked_mul(dims[2] as usize)
}

fn mtl_size(dims: [u32; 3]) -> MTLSize {
    MTLSize {
        width: dims[0] as usize,
        height: dims[1] as usize,
        depth: dims[2] as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> MeshPipelineSpec {
        MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [4, 1, 1])
    }

    #[test]
    fn mesh_spec_rejects_an_output_declaration_over_32k() {
        let mut spec = spec();
        spec.mesh_output_bytes = 33_280;
        let error = spec.validate_limits(1024, 32 * 1024).unwrap_err();
        assert_eq!(
            error,
            MeshError::Budget("mesh output declaration exceeds Metal's 32 KiB budget")
        );
        assert_eq!(
            MetalError::from(error).status(),
            crate::abi::VmStatus::InvalidArg
        );
    }

    #[test]
    fn mesh_spec_accepts_the_exact_32k_boundary() {
        let mut spec = spec();
        spec.mesh_output_bytes = METAL_MESH_OUTPUT_BYTES;
        spec.validate_limits(1024, 32 * 1024)
            .expect("the exact budget is allowed");
    }

    #[test]
    fn mesh_spec_rejects_threadgroups_over_the_device_limit() {
        let spec = MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [64, 1, 1]);
        let error = spec.validate_limits(32, 32 * 1024).unwrap_err();
        assert_eq!(
            error,
            MeshError::Budget("threadgroup size exceeds the device threadgroup maximum")
        );
    }

    #[test]
    fn mesh_spec_rejects_a_mesh_stage_without_colour_attachments() {
        let spec = MeshPipelineSpec::new(Vec::new(), None, [1, 1, 1]);
        assert!(matches!(
            spec.validate_limits(1024, 32 * 1024),
            Err(MeshError::Unsupported(_))
        ));
    }

    #[test]
    fn mesh_spec_accepts_a_depth_stencil_pipeline() {
        let spec = MeshPipelineSpec::new(
            vec![PixelFormat::Bgra8Unorm],
            Some(PixelFormat::Depth32FloatStencil8),
            [32, 1, 1],
        );
        spec.validate_limits(1024, 32 * 1024)
            .expect("stencil format is a valid depth attachment");
    }

    #[test]
    fn mesh_abi_plan_counts_inline_constants_and_tables() {
        use crate::macos::bindings::{DescriptorTableKind, RootParameterSpec};

        let mesh = RootSignatureLayout::from_parameters(
            crate::macos::bindings::ReflectedStage::Other,
            &[
                RootParameterSpec {
                    kind: RootParameterKind::DescriptorTable(
                        DescriptorTableKind::ShaderResourceView,
                    ),
                    size: 0,
                    register: 0,
                    space: 0,
                },
                RootParameterSpec {
                    kind: RootParameterKind::Constants,
                    size: 16,
                    register: 0,
                    space: 0,
                },
                RootParameterSpec {
                    kind: RootParameterKind::ConstantBufferView,
                    size: 8,
                    register: 0,
                    space: 1,
                },
            ],
        )
        .expect("mesh layout");
        let tables = [MeshTableExtent {
            kind: DescriptorTableKind::ShaderResourceView,
            count: 8,
        }];
        let plan = MeshAbiPlan::new(mesh.clone(), None, Some(mesh), &tables).expect("plan builds");
        assert_eq!(plan.mesh().argument_buffer_len(), 32);
        assert_eq!(plan.argument_buffer_bytes(), 64);
        assert_eq!(plan.descriptor_table_bytes(), 64);
        assert_eq!(plan.tables().len(), 1);
        assert!(plan.pipeline_abi().mesh.is_some());
        assert!(plan.pipeline_abi().fragment.is_some());
    }

    #[test]
    fn mesh_abi_plan_refuses_root_constants_over_64_dwords() {
        use crate::macos::bindings::RootParameterSpec;

        let mesh = RootSignatureLayout::from_parameters(
            crate::macos::bindings::ReflectedStage::Other,
            &[RootParameterSpec {
                kind: RootParameterKind::Constants,
                size: (MAX_MESH_ROOT_CONSTANT_DWORDS + 4) * 4,
                register: 0,
                space: 0,
            }],
        )
        .expect("a 68-dword layout is constructible");
        let error = MeshAbiPlan::new(mesh, None, None, &[]).unwrap_err();
        assert_eq!(
            error,
            MeshError::Budget("mesh root constants exceed 64 dwords")
        );
    }

    #[test]
    fn mesh_abi_plan_refuses_a_packed_table_over_64k() {
        use crate::macos::bindings::{DescriptorTableKind, RootParameterSpec};

        let mesh = RootSignatureLayout::from_parameters(
            crate::macos::bindings::ReflectedStage::Other,
            &[RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::ShaderResourceView),
                size: 0,
                register: 0,
                space: 0,
            }],
        )
        .expect("mesh layout");
        // 8,193 eight-byte addresses = 65,544 B, just over the 64 KiB ceiling.
        let tables = [MeshTableExtent {
            kind: DescriptorTableKind::ShaderResourceView,
            count: 8_193,
        }];
        let error = MeshAbiPlan::new(mesh, None, None, &tables).unwrap_err();
        assert_eq!(
            error,
            MeshError::Budget("descriptor tables exceed the 64 KiB argument-buffer limit")
        );
    }

    #[test]
    fn mesh_abi_plan_parses_converter_reflection() {
        let json = r#"{
            "EntryPoint": "main",
            "ShaderType": "Mesh",
            "TopLevelArgumentBuffer": [
                {"EltOffset": 0, "Size": 8, "Slot": 0, "Space": 0, "Type": "Table"},
                {"EltOffset": 8, "Size": 16, "Slot": 0, "Space": 1, "Type": "Constant"}
            ]
        }"#;
        let plan = MeshAbiPlan::from_reflection_json(
            json,
            None,
            None,
            &[MeshTableExtent {
                kind: DescriptorTableKind::ShaderResourceView,
                count: 4,
            }],
        )
        .expect("the reflected plan builds");
        assert_eq!(plan.mesh().parameters().len(), 2);
        assert_eq!(plan.mesh().argument_buffer_len(), 24);
        assert_eq!(plan.descriptor_table_bytes(), 32);
    }
}
