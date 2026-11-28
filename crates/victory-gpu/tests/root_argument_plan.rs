//! ADR-057 guest proofs: every submission carries a `VRAB` root-argument plan,
//! including one that writes no root constant, and the encoded plan round-trips
//! through the host decoder.
//!
//! These are CPU proofs; the queue is driven through a recording [`V3Sink`], so
//! no socket, title or GPU is involved.

use victory_gpu::native_metal::bindings;
use victory_gpu::native_metal::producer::{MetalProducer, RootSignatureStage};
use victory_gpu::native_metal::queue::{NativeQueue, NativeQueueError, V3Sink};
use victory_metal::root_arguments::{decode_root_plan, PlanArgumentKind};
use victory_metal::wire::ReplyStatus;
use victory_shader::emit::USER_DATA_SPACE;

/// A sink that records the frame and the plan section it was handed.
#[derive(Debug, Default)]
struct RecordingSink {
    frames: Vec<Vec<u8>>,
    plans: Vec<Option<Vec<u8>>>,
}

impl V3Sink for RecordingSink {
    fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError> {
        self.frames.push(bytes.to_vec());
        self.plans.push(None);
        Ok(ReplyStatus::Ok)
    }

    fn send_with_plan(
        &mut self,
        bytes: &[u8],
        plan: Option<&[u8]>,
    ) -> Result<ReplyStatus, NativeQueueError> {
        self.frames.push(bytes.to_vec());
        self.plans.push(plan.map(<[u8]>::to_vec));
        Ok(ReplyStatus::Ok)
    }
}

/// One compute submission that dispatches but never calls
/// `SetComputeRoot32BitConstants`.
fn dispatch_without_root_constants(list_id: u64) -> MetalProducer {
    let mut producer = MetalProducer::new(list_id, 1).expect("a list generation");
    // The selected pipeline declares a four-dword constants block. No
    // `Set*Root32BitConstants` follows, so the plan can only be non-empty if
    // the declaration itself is carried.
    producer.declare_root_constant_block(
        RootSignatureStage::Compute,
        0,
        USER_DATA_SPACE as u16,
        4,
    );
    producer
        .set_compute_pipeline(0x700)
        .expect("compute pipeline");
    producer
        .set_compute_threadgroup_size([64, 1, 1])
        .expect("threadgroup size");
    producer.dispatch(1, 1, 1).expect("dispatch");
    producer
}

/// A batch with no root-constant op still carries a non-empty plan for a
/// pipeline declaring root arguments, and the encoded plan decodes through the
/// host's `VRAB` reader.
#[test]
fn a_batch_without_root_constant_ops_still_carries_its_declared_plan() {
    let producer = dispatch_without_root_constants(11);
    let plan = producer.plan_for_submission();
    assert!(
        !plan.is_empty(),
        "the declared constants block must reach the plan"
    );
    let constants = plan
        .iter()
        .find(|argument| argument.kind == bindings::RootArgumentKind::Constants)
        .expect("the declared block is a constants entry");
    assert_eq!(constants.root_parameter, 0);
    assert_eq!(constants.space, USER_DATA_SPACE as u16);
    assert_eq!(constants.size, 16, "four dwords");

    let encoded =
        bindings::encode_root_arguments(&plan).expect("the guest plan encodes");
    let decoded = decode_root_plan(&encoded).expect("the host decoder accepts it");
    assert_eq!(decoded.len(), 1);
    assert_eq!(
        decoded.entry(0).map(|entry| entry.kind),
        Some(PlanArgumentKind::Constants)
    );
    assert_eq!(decoded.entry(0).map(|entry| entry.space), Some(USER_DATA_SPACE as u16));
}

/// The queue attaches a plan section to every frame, including an empty plan
/// (the versioned header alone) and a submission with no root-constant op.
#[test]
fn every_frame_carries_a_plan_section() {
    let producer = dispatch_without_root_constants(12);
    let plan = producer.plan_for_submission();
    let mut producer = producer;
    let submission = producer.flush().expect("the dispatch flushes");

    let mut queue = NativeQueue::new(RecordingSink::default());
    queue
        .submit_with_root_arguments(std::slice::from_ref(&submission), &[plan.as_slice()])
        .expect("the declared plan carries");
    let planned = queue.sink().plans[0]
        .as_ref()
        .expect("a plan section rode with the frame");
    assert!(!planned.is_empty(), "the declared plan is not the header alone");
    let decoded = decode_root_plan(planned).expect("the host decoder accepts the plan");
    assert_eq!(decoded.entry(0).map(|entry| entry.kind), Some(PlanArgumentKind::Constants));

    // A submission that passes no plan at all still carries the versioned
    // header, which the host decodes as "plan present, no entries".
    queue
        .submit_with_root_arguments(std::slice::from_ref(&submission), &[&[]])
        .expect("an empty plan carries");
    let empty = queue.sink().plans[1]
        .as_ref()
        .expect("the empty plan section is carried");
    assert_eq!(empty.len(), 7, "the empty plan is the seven-byte header");
    assert!(
        decode_root_plan(empty)
            .expect("the empty plan decodes")
            .is_empty()
    );

    // Both frames are still canonical v3 submissions; the plan never enters
    // the frame bytes.
    for frame in &queue.sink().frames {
        let decoded = victory_metal::wire_v3::decode_submission_v3(frame)
            .expect("the frame is still a v3 submission");
        assert_eq!(decoded.commands.len(), submission.commands.len());
    }
}

/// ADR-060: a graphics `SetGraphicsRoot32BitConstants` recorded before a draw
/// is forwarded as a `SetRootConstants` command at that draw instead of being
/// dropped, and the emitted bytes are exactly the title's.
///
/// This is the producer half of the guest audit. The D3D12 vtable
/// (`list::graphics_constant(s)`) calls this method directly and the native
/// telemetry counter next to it; a draw that encodes with no such command means
/// the title never made the call, not that the producer discarded it.
#[test]
fn a_graphics_root_constant_reaches_the_encoded_draw() {
    use victory_gpu::native_metal::producer::{RenderTarget, ResourceId};
    use victory_metal::wire_v3::{
        CommandV3, LoadOp, PrimitiveTopology, StoreOp,
    };

    let mut producer = MetalProducer::new(21, 1).expect("a list generation");
    let target = RenderTarget {
        resource: ResourceId::new(0x1000, 1).expect("target identity"),
        mip_level: 0,
        slice: 0,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        clear_color: [0.0; 4],
    };
    producer
        .set_render_targets(&[target], None)
        .expect("render target");
    producer
        .set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0)
        .expect("viewport");
    producer.set_scissor_edges(0, 0, 64, 64).expect("scissor");
    producer
        .set_topology(PrimitiveTopology::TriangleList)
        .expect("topology");
    producer
        .declare_root_constant_block(
            RootSignatureStage::Graphics,
            0,
            USER_DATA_SPACE as u16,
            4,
        );
    producer.set_pipeline(0x800).expect("graphics pipeline");
    producer
        .set_graphics_root_constants(0, &[9, 8, 7, 6], 0)
        .expect("root constants recorded");
    producer.draw(3, 1, 0, 0).expect("draw records");

    let emitted = producer
        .commands()
        .iter()
        .find_map(|command| match command {
            CommandV3::SetRootConstants(constants) => Some(constants.clone()),
            _ => None,
        })
        .expect("the title's constants reach the wire at the draw");
    assert_eq!(emitted.slot, 0);
    let expected: Vec<u8> = [9u32, 8, 7, 6]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    assert_eq!(emitted.bytes, expected);

    // The plan names the same translated identity, so the host can resolve the
    // wire slot through the title's root-argument plan as well.
    let plan = producer.plan_for_submission();
    assert!(
        plan.iter().any(|argument| argument.kind
            == bindings::RootArgumentKind::Constants
            && argument.root_parameter == 0),
        "the recorded constants are in the plan"
    );
}
