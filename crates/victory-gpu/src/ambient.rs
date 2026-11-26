//! Indirect-light floor for the missing ray-traced probes, from
//! Belobug/VIctory-Belobug-AMD (06031237). Build the helper with DXC because
//! D3DMetal needs DXIL. The original lighting remains the default.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};

/// The title compute shader that resolves indirect diffuse light.
const PROGRAM: &[u8] = b"CS_ScreenProbes_TemporalResolve";

/// Root parameter and table offset (dwords) of its packed R9G9B9E5 output.
pub(crate) const LIGHT: (u8, u32) = (1, 176);

pub(crate) fn level() -> Option<f32> {
    static LEVEL: OnceLock<Option<f32>> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("VICTORY_AMBIENT")
            .ok()?
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|level| *level > 0.0 && *level <= 65408.0)
    })
}

pub(crate) fn resolve_shader(container: &[u8]) -> bool {
    level().is_some() && container.windows(PROGRAM.len()).any(|name| name == PROGRAM)
}

struct Pass {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/ambient-floor.dxil"));
    // SAFETY: D3D12 validates the bytecode and its embedded root signature.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        let pipeline = device.CreateComputePipelineState(&desc);
        std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
        Ok(Pass { root, pipeline: pipeline? })
    }
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        make(&host()?.device)
            .inspect_err(|error| tracing::warn!(%error, "could not create the indirect-light floor"))
            .ok()
    })
    .as_ref()
}

/// Records a helper dispatch using the resolve shader's existing UAV.
pub(crate) fn raise(
    list: &HostList,
    light: D3D12_GPU_DESCRIPTOR_HANDLE,
    width: u32,
    height: u32,
    level: f32,
) -> bool {
    let Some(pass) = pass() else {
        return false;
    };
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        tracing::info!(width, height, level, "using the indirect-light floor");
    }
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        ..Default::default()
    };
    // SAFETY: the title has just dispatched on this list with the heap
    // holding this UAV, so the view and its resource are ready to use.
    unsafe {
        list.list.ResourceBarrier(std::slice::from_ref(&barrier));
        list.list.SetPipelineState(&pass.pipeline);
        list.list.SetComputeRootSignature(&pass.root);
        list.list.SetComputeRoot32BitConstant(0, level.to_bits(), 0);
        list.list.SetComputeRootDescriptorTable(1, light);
        list.list.Dispatch(width.div_ceil(8), height.div_ceil(8), 1);
        list.list.ResourceBarrier(std::slice::from_ref(&barrier));
    }
    true
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires D3D12 hardware and a build with the DXC helper"]
    fn creates_the_indirect_light_pipeline() {
        let host = super::host().expect("D3D12 host");
        super::make(&host.device).expect("indirect-light compute pipeline");
    }
}
