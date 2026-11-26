//! Synthetic end-to-end proof: real XMem file-backed pages, the opt-in
//! guest-range resolver, the Win64 guest IPC client and the native ARM64
//! Metal host.
//!
//! The test is inert unless the launcher (`tools/macos/metal-poc/xmem-ipc.sh`)
//! provides a session:
//!
//! | variable | meaning |
//! | --- | --- |
//! | `VICTORY_METAL_SHARE_DIR` | private Win-visible XMem share directory |
//! | `VICTORY_IPC_PORT` | loopback port of the native host |
//! | `VICTORY_IPC_PIPELINE_ID` | pipeline the host compiled and published |
//! | `VICTORY_IPC_PIPELINE_TICKET` | numeric ticket of the offline `.metallib` |
//! | `VICTORY_IPC_PIPELINE_ENTRY` | entry point inside that library |
//! | `VICTORY_IPC_SECRET_FILE` | Win-visible file with the 32-byte secret |
//!
//! The test allocates actual physical pages through
//! [`super::XMemAllocatePhysicalPages`], maps a two-page run at a nonzero
//! section offset into a title placeholder through
//! [`super::XMemMapPhysicalPages`], aliases it through [`super::map_alias`],
//! resolves the work page with [`super::share::resolve_guest_range`], registers
//! that real backing file with the host and submits two ordered dispatches of
//! the one bootstrap pipeline the host publishes. Two sequential applications
//! of that kernel are deliberate: the final value can only arise if the second
//! dispatch observed the first dispatch's write, so the order is proven by the
//! data. The result is read back through the guest alias, which is a plain CPU
//! read of the file-backed page the GPU wrote.
//!
//! The second test reuses the same wiring for pipeline registration: the
//! launcher compiles a synthetic compute kernel with Apple's offline Metal
//! toolchain and drops it in the share directory as
//! `{ticket:016x}.metallib`; the guest asks the host to register it by ticket
//! and entry name only, and the returned native id is dispatched over the
//! same real alias. An explicitly rejected ticket must leave the session open
//! and the alias untouched.
//!
//! The harness owns the secret file and the share directory; nothing here
//! prints the secret. No title is launched.

use super::{PAGE, PAGES_64K};

use crate::modules::xmem::share;
use victory_metal::guest_ipc::SECRET_LEN;
use victory_metal::wire::{
    BufferBinding, Command, Dispatch1d, DrawIndexed, ListIdentity, ReplyStatus, RootConstants,
    SetPipeline, Submission,
};
use victory_metal::{GuestIpcClient, GuestIpcError};

use windows_sys::Win32::System::Memory::{FlushViewOfFile, MEM_RESERVE, PAGE_READWRITE};

/// One 64 KiB page of dwords: the dispatched range.
const WORDS: usize = PAGE / 4;
/// The untouched page in front of the bound range.
const PREFIX_SENTINEL: u32 = 0xDEAD_BEEF;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// Reads the 64 hex characters the harness wrote into `VICTORY_IPC_SECRET_FILE`.
fn secret() -> Option<[u8; SECRET_LEN]> {
    let path = env("VICTORY_IPC_SECRET_FILE")?;
    let text = std::fs::read_to_string(&path).expect("the harness secret file");
    let text = text.trim();
    assert_eq!(text.len(), 64, "the secret file must be 64 hex characters");
    let mut secret = [0u8; SECRET_LEN];
    for index in 0..SECRET_LEN {
        let pair = &text[index * 2..index * 2 + 2];
        secret[index] = u8::from_str_radix(pair, 16).expect("hex digits");
    }
    Some(secret)
}

#[test]
fn two_ordered_host_dispatches_write_the_real_guest_alias() {
    let Some(directory) = env("VICTORY_METAL_SHARE_DIR") else {
        eprintln!("skipped: VICTORY_METAL_SHARE_DIR is not set");
        return;
    };
    let Some(port) = env("VICTORY_IPC_PORT") else {
        eprintln!("skipped: VICTORY_IPC_PORT is not set");
        return;
    };
    let Some(pipeline) = env("VICTORY_IPC_PIPELINE_ID") else {
        eprintln!("skipped: VICTORY_IPC_PIPELINE_ID is not set");
        return;
    };
    let Some(secret) = secret() else {
        eprintln!("skipped: VICTORY_IPC_SECRET_FILE is not set");
        return;
    };
    let port: u16 = port.parse().expect("a numeric IPC port");
    let pipeline: u64 = pipeline.parse().expect("a numeric pipeline id");

    // Actual physical pages: three of them, so the mapped run starts at a
    // nonzero section offset (page 1) and has a page in front of it.
    let mut pages = [0usize; 3];
    let mut count = pages.len();
    // SAFETY: `count` and `pages` are valid for three page numbers.
    let allocated = super::XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr());
    assert_eq!(allocated, 1, "XMemAllocatePhysicalPages failed");
    assert_eq!(count, pages.len(), "expected three physical pages");

    // A title placeholder, and the two-page alias the GPU maps into.
    // SAFETY: a plain reservation request; the returned range is only used
    // through the mapping calls below.
    let alias = super::XMemVirtualAlloc(
        std::ptr::null_mut(),
        2 * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!alias.is_null(), "XMemVirtualAlloc failed");

    // Map section pages 1 and 2: the work page is at section offset 128 KiB,
    // behind a fully initialised prefix page at offset 64 KiB.
    // SAFETY: `pages[1..]` is two valid page numbers for the reservation.
    assert_eq!(
        super::XMemMapPhysicalPages(alias, 2, pages[1..].as_ptr()),
        alias,
        "XMemMapPhysicalPages failed"
    );

    // SAFETY: the reservation is mapped read-write for two pages.
    let words = unsafe { std::slice::from_raw_parts_mut(alias.cast::<u32>(), 2 * PAGE / 4) };
    let (prefix, work) = words.split_at_mut(WORDS);
    prefix.fill(PREFIX_SENTINEL);
    for (index, value) in work.iter_mut().enumerate() {
        *value = index as u32 + 1;
    }

    // The GPU alias of the exact same physical page. `map_alias` is the
    // existing GPU page-mapping copy path, not a test shortcut.
    // SAFETY: a plain reservation request, as above.
    let gpu = super::XMemVirtualAlloc(
        std::ptr::null_mut(),
        2 * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!gpu.is_null(), "the GPU alias reservation failed");
    assert!(
        super::map_alias(gpu as u64, alias as u64, (2 * PAGE) as u64),
        "map_alias failed"
    );

    // Publish the page before the host reads it through its own mapping.
    assert_ne!(
        // SAFETY: `alias` is a live two-page view of the file-backed section.
        unsafe { FlushViewOfFile(alias.cast(), 2 * PAGE) },
        0,
        "FlushViewOfFile failed"
    );

    // Resolve the work page only: id, generation, nonzero 128 KiB offset.
    let work_address = gpu.cast::<u8>().wrapping_add(PAGE) as usize;
    let lease = share::resolve_guest_range(work_address, PAGE)
        .expect("the real guest alias must resolve to host-visible backing");
    let range = lease.range;
    assert_ne!(range.id, 0, "the resolved id is zero");
    assert_eq!(
        range.generation,
        share::GENERATION,
        "unexpected generation"
    );
    assert_eq!(
        range.offset,
        (2 * PAGE) as u64,
        "unexpected section offset"
    );
    assert_eq!(range.length, PAGE as u64, "unexpected resolved length");
    assert_eq!(
        lease.path().parent(),
        Some(std::path::Path::new(&directory)),
        "the backing file is outside the harness share directory"
    );

    // The host derives the backing path from the id, so register the exact
    // file length the provider created (three pages).
    let file_len = std::fs::metadata(lease.path())
        .expect("the backing file metadata")
        .len();
    assert_eq!(
        file_len,
        (3 * PAGE) as u64,
        "unexpected backing file length"
    );

    let mut client =
        GuestIpcClient::connect(port, &secret).expect("the native host must accept the session");
    client
        .register(range.id, file_len)
        .expect("the real backing must register");

    let generation = u32::try_from(range.generation).expect("a u32 generation");
    let binding = || BufferBinding {
        allocation_id: range.id,
        generation,
        offset: range.offset,
        length: range.length,
    };
    let dispatch = |pipeline_id: u64| Dispatch1d {
        pipeline_id,
        threads: WORDS as u32,
        threads_per_group: 64,
        bindings: vec![binding()],
    };
    let submission = Submission {
        sequence: 1,
        commands: vec![
            Command::Dispatch1d(dispatch(pipeline)),
            Command::Dispatch1d(dispatch(pipeline)),
        ],
    };

    let reply = client
        .submit(&submission)
        .expect("the two ordered dispatches must complete");
    assert_eq!(reply.sequence, 1, "unexpected reply sequence");
    assert_eq!(reply.status, ReplyStatus::Ok, "unexpected reply status");

    // Read the result through the guest alias, not the source view: this is
    // the exact page the host wrote through its own file mapping.
    // SAFETY: `gpu` is the live two-page alias created above.
    let through_alias = unsafe { std::slice::from_raw_parts(gpu.cast::<u32>(), 2 * PAGE / 4) };
    let (alias_prefix, alias_work) = through_alias.split_at(WORDS);
    for (index, &value) in alias_work.iter().enumerate() {
        // Two sequential applications of the published *3+7 kernel.
        let expected = ((index as u32 + 1) * 3 + 7) * 3 + 7;
        assert_eq!(
            value, expected,
            "alias dword {index} does not match the ordered dispatches"
        );
    }
    assert!(
        alias_prefix.iter().all(|&value| value == PREFIX_SENTINEL),
        "the ordered batch modified the prefix page before the bound range"
    );

    // A stale generation must be rejected before any GPU work and leave the
    // real alias byte-for-byte unchanged.
    let mut stale = submission.clone();
    stale.sequence = 2;
    for command in &mut stale.commands {
        let Command::Dispatch1d(dispatch) = command else {
            panic!("the fixture submits only 1D dispatches");
        };
        dispatch.bindings[0].generation += 1;
    }
    let reply = client
        .submit(&stale)
        .expect("a stale generation must still return a reply");
    assert_eq!(reply.sequence, 2, "unexpected stale reply sequence");
    assert_eq!(
        reply.status,
        ReplyStatus::UnknownHandle,
        "unexpected stale-generation reply"
    );
    for (index, &value) in alias_work.iter().enumerate() {
        // Two sequential applications of the published *3+7 kernel.
        let expected = ((index as u32 + 1) * 3 + 7) * 3 + 7;
        assert_eq!(
            value, expected,
            "stale generation changed alias dword {index}"
        );
    }
    assert!(
        alias_prefix.iter().all(|&value| value == PREFIX_SENTINEL),
        "the stale generation modified the prefix page"
    );

    client.close();
    // SAFETY: unmapping the two views we made; the pages are freed after.
    super::XMemMapPhysicalPages(gpu, 2, std::ptr::null());
    super::XMemMapPhysicalPages(alias, 2, std::ptr::null());
    // SAFETY: `pages` holds the three allocated page numbers.
    assert_eq!(
        super::XMemFreePhysicalPages(pages.len(), pages.as_ptr()),
        1,
        "XMemFreePhysicalPages failed"
    );
    println!(
        "PASS xmem-ipc id={} generation={} offset={} length={} words={} sentinel={:#010x}",
        range.id, range.generation, range.offset, range.length, WORDS, PREFIX_SENTINEL
    );
    drop(lease);
}

/// Registers a launcher-compiled `.metallib` over the real guest transport,
/// dispatches the returned native pipeline on the real alias, and proves that
/// an explicit rejection neither closes the session nor dispatches.
///
/// The first dispatch uses the bootstrap pipeline the worker compiled from
/// MSL; the second uses the pipeline the host loaded from the offline
/// `.metallib`, selected by entry name. Both run in one ordered submission, so
/// the alias only holds the expected values if registration returned a live
/// native id for the requested entry point.
#[test]
fn registered_metallib_pipeline_dispatches_and_rejection_stays_inert() {
    let Some(directory) = env("VICTORY_METAL_SHARE_DIR") else {
        eprintln!("skipped: VICTORY_METAL_SHARE_DIR is not set");
        return;
    };
    let Some(port) = env("VICTORY_IPC_PORT") else {
        eprintln!("skipped: VICTORY_IPC_PORT is not set");
        return;
    };
    let Some(bootstrap) = env("VICTORY_IPC_PIPELINE_ID") else {
        eprintln!("skipped: VICTORY_IPC_PIPELINE_ID is not set");
        return;
    };
    let Some(ticket) = env("VICTORY_IPC_PIPELINE_TICKET") else {
        eprintln!("skipped: VICTORY_IPC_PIPELINE_TICKET is not set");
        return;
    };
    let Some(entry) = env("VICTORY_IPC_PIPELINE_ENTRY") else {
        eprintln!("skipped: VICTORY_IPC_PIPELINE_ENTRY is not set");
        return;
    };
    let Some(secret) = secret() else {
        eprintln!("skipped: VICTORY_IPC_SECRET_FILE is not set");
        return;
    };
    let port: u16 = port.parse().expect("a numeric IPC port");
    let bootstrap: u64 = bootstrap.parse().expect("a numeric bootstrap pipeline id");
    let ticket: u64 = ticket.parse().expect("a numeric pipeline ticket");

    // The launcher's offline compiler artifact, seen through the guest's own
    // Win-visible share directory. The guest only ever names it by ticket.
    let artifact = std::path::Path::new(&directory).join(format!("{ticket:016x}.metallib"));
    let artifact_len = std::fs::metadata(&artifact)
        .unwrap_or_else(|error| panic!("missing {}: {error}", artifact.display()))
        .len();
    assert!(
        artifact_len > 0,
        "the offline .metallib is empty: {}",
        artifact.display()
    );

    // Actual physical pages: three, so the mapped work page sits at a nonzero
    // section offset behind an untouched sentinel page.
    let mut pages = [0usize; 3];
    let mut count = pages.len();
    // SAFETY: `count` and `pages` are valid for three page numbers.
    let allocated = super::XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr());
    assert_eq!(allocated, 1, "XMemAllocatePhysicalPages failed");
    assert_eq!(count, pages.len(), "expected three physical pages");

    // SAFETY: a plain reservation request; the range is used only through the
    // existing mapping calls below.
    let alias = super::XMemVirtualAlloc(
        std::ptr::null_mut(),
        2 * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!alias.is_null(), "XMemVirtualAlloc failed");
    // SAFETY: `pages[1..]` is two valid page numbers for the reservation.
    assert_eq!(
        super::XMemMapPhysicalPages(alias, 2, pages[1..].as_ptr()),
        alias,
        "XMemMapPhysicalPages failed"
    );

    // SAFETY: the reservation is mapped read-write for two pages.
    let words = unsafe { std::slice::from_raw_parts_mut(alias.cast::<u32>(), 2 * PAGE / 4) };
    let (prefix, work) = words.split_at_mut(WORDS);
    prefix.fill(PREFIX_SENTINEL);
    for (index, value) in work.iter_mut().enumerate() {
        *value = index as u32 + 1;
    }

    // The GPU alias of the same physical page, through the real mapping path.
    // SAFETY: a plain reservation request, as above.
    let gpu = super::XMemVirtualAlloc(
        std::ptr::null_mut(),
        2 * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!gpu.is_null(), "the GPU alias reservation failed");
    assert!(
        super::map_alias(gpu as u64, alias as u64, (2 * PAGE) as u64),
        "map_alias failed"
    );
    assert_ne!(
        // SAFETY: `alias` is a live two-page view of the file-backed section.
        unsafe { FlushViewOfFile(alias.cast(), 2 * PAGE) },
        0,
        "FlushViewOfFile failed"
    );

    let work_address = gpu.cast::<u8>().wrapping_add(PAGE) as usize;
    let lease = share::resolve_guest_range(work_address, PAGE)
        .expect("the real guest alias must resolve to host-visible backing");
    let range = lease.range;
    assert_eq!(
        (range.offset, range.length),
        ((2 * PAGE) as u64, PAGE as u64),
        "unexpected resolved work range"
    );

    let file_len = std::fs::metadata(lease.path())
        .expect("the backing file metadata")
        .len();
    assert_eq!(file_len, (3 * PAGE) as u64, "unexpected backing length");

    let mut client =
        GuestIpcClient::connect(port, &secret).expect("the native host must accept the session");
    client
        .register(range.id, file_len)
        .expect("the real backing must register");

    // The contract under test: a ticket and an entry name yield a live native
    // pipeline id that is not the bootstrap pipeline.
    let registered = client
        .register_pipeline(ticket, &entry)
        .expect("the offline .metallib must register");
    assert_ne!(registered.0, 0, "the host returned the null pipeline id");
    assert_ne!(
        registered.0, bootstrap,
        "pipeline registration must not reuse the bootstrap pipeline"
    );

    let generation = u32::try_from(range.generation).expect("a u32 generation");
    let dispatch = |pipeline_id: u64| Dispatch1d {
        pipeline_id,
        threads: WORDS as u32,
        threads_per_group: 64,
        bindings: vec![BufferBinding {
            allocation_id: range.id,
            generation,
            offset: range.offset,
            length: range.length,
        }],
    };
    let submission = Submission {
        sequence: 1,
        commands: vec![
            // The bootstrap kernel: x -> 3x + 7.
            Command::Dispatch1d(dispatch(bootstrap)),
            // The offline-compiled kernel: x -> 5x + 3.
            Command::Dispatch1d(dispatch(registered.0)),
        ],
    };
    let reply = client
        .submit(&submission)
        .expect("both pipelines must complete in one ordered submission");
    assert_eq!(reply.sequence, 1, "unexpected reply sequence");
    assert_eq!(reply.status, ReplyStatus::Ok, "unexpected reply status");

    // SAFETY: `gpu` is the live two-page alias created above.
    let through_alias = unsafe { std::slice::from_raw_parts(gpu.cast::<u32>(), 2 * PAGE / 4) };
    let (alias_prefix, alias_work) = through_alias.split_at(WORDS);
    let expected_at = |index: usize| ((index as u32 + 1) * 3 + 7) * 5 + 3;
    for (index, &value) in alias_work.iter().enumerate() {
        let expected = expected_at(index);
        assert_eq!(
            value, expected,
            "alias dword {index} does not match bootstrap *3+7 then metallib *5+3"
        );
    }
    assert!(
        alias_prefix.iter().all(|&value| value == PREFIX_SENTINEL),
        "the dispatch modified the prefix page"
    );

    // A ticket with no launcher file must be an explicit rejection: the
    // session stays open and the alias stays exactly as the valid dispatch
    // left it.
    let rejected_ticket = ticket.checked_add(1).expect("ticket space exhausted");
    let error = client
        .register_pipeline(rejected_ticket, &entry)
        .expect_err("a ticket without a backing library must be rejected");
    assert!(
        matches!(
            error,
            GuestIpcError::Rejected {
                operation: "pipeline registration"
            }
        ),
        "unexpected rejection: {error}"
    );
    assert!(
        !client.is_closed(),
        "an explicit rejection must keep the session open"
    );
    for (index, &value) in alias_work.iter().enumerate() {
        assert_eq!(
            value,
            expected_at(index),
            "a rejected registration dispatched and changed alias dword {index}"
        );
    }
    assert!(
        alias_prefix.iter().all(|&value| value == PREFIX_SENTINEL),
        "a rejected registration modified the prefix page"
    );

    // A locally invalid entry never reaches the wire either.
    assert!(matches!(
        client.register_pipeline(rejected_ticket, "../vm_entry"),
        Err(GuestIpcError::InvalidPipelineEntry)
    ));
    assert!(
        !client.is_closed(),
        "a locally invalid entry must keep the session open"
    );

    // An id no registration produced is rejected before GPU work, proving the
    // rejection did not leave a pipeline behind.
    let unknown = Submission {
        sequence: 2,
        commands: vec![Command::Dispatch1d(dispatch(registered.0 + 1))],
    };
    let reply = client
        .submit(&unknown)
        .expect("an unknown pipeline must still return a reply");
    assert_eq!(reply.sequence, 2, "unexpected unknown-pipeline sequence");
    assert_eq!(
        reply.status,
        ReplyStatus::UnknownHandle,
        "unexpected unknown-pipeline reply"
    );
    for (index, &value) in alias_work.iter().enumerate() {
        assert_eq!(
            value,
            expected_at(index),
            "an unknown pipeline changed alias dword {index}"
        );
    }
    assert!(
        alias_prefix.iter().all(|&value| value == PREFIX_SENTINEL),
        "an unknown pipeline modified the prefix page"
    );

    client.close();
    // SAFETY: unmapping the two views we made; the pages are freed after.
    super::XMemMapPhysicalPages(gpu, 2, std::ptr::null());
    super::XMemMapPhysicalPages(alias, 2, std::ptr::null());
    // SAFETY: `pages` holds the three allocated page numbers.
    assert_eq!(
        super::XMemFreePhysicalPages(pages.len(), pages.as_ptr()),
        1,
        "XMemFreePhysicalPages failed"
    );
    println!(
        "PASS xmem-ipc-pipeline ticket={} entry={} artifact_bytes={} bootstrap={} registered={} rejected_ticket={} words={}",
        ticket, entry, artifact_len, bootstrap, registered.0, rejected_ticket, WORDS
    );
    drop(lease);
}

/// Input dword the guest writes for the published bootstrap kernel (`*3+7`).
const FRAME_COMPUTE_INPUT: u32 = 6;
/// Result the published bootstrap kernel must produce from that input.
const FRAME_COMPUTE_OUTPUT: u32 = FRAME_COMPUTE_INPUT * 3 + 7;

/// Guest half of the full Wine-to-native frame gate.
///
/// The guest writes a three-index/triangle pair into its real XMem allocation,
/// registers that single backing, then sends the recorded graphics list over
/// loopback IPC with a [`ListIdentity`]. The native host executes
/// `SetPipeline`, `RootConstants` and `DrawIndexed` through `device.submit_batch`
/// into its 64x64 host target. A second submission exercises the published
/// compute kernel on a different page of the same allocation. The guest then
/// replays the graphics list with the same identity and a new sequence: the
/// host must refuse it as [`ReplyStatus::AlreadyExecuted`] without touching the
/// guest's alias.
///
/// The host launcher (`tools/macos/metal-poc/synthetic-frame.sh`) checks the
/// drawn centre pixel and the computed constant from the host's own observations
/// (`kind=draw ...`, `kind=compute value=...`). This test proves the guest side:
/// replies, the alias-visible compute result and the unchanged alias after the
/// refused replay.
#[test]
fn wine_guest_draws_and_computes_through_native_frame_gate() {
    let Some(directory) = env("VICTORY_METAL_SHARE_DIR") else {
        eprintln!("skipped: VICTORY_METAL_SHARE_DIR is not set");
        return;
    };
    let Some(port) = env("VICTORY_IPC_PORT") else {
        eprintln!("skipped: VICTORY_IPC_PORT is not set");
        return;
    };
    let Some(compute_pipeline) = env("VICTORY_IPC_PIPELINE_ID") else {
        eprintln!("skipped: VICTORY_IPC_PIPELINE_ID is not set");
        return;
    };
    let Some(graphics_pipeline) = env("VICTORY_IPC_GRAPHICS_PIPELINE_ID") else {
        eprintln!("skipped: VICTORY_IPC_GRAPHICS_PIPELINE_ID is not set");
        return;
    };
    let Some(secret) = secret() else {
        eprintln!("skipped: VICTORY_IPC_SECRET_FILE is not set");
        return;
    };
    let port: u16 = port.parse().expect("a numeric IPC port");
    let compute_pipeline: u64 = compute_pipeline.parse().expect("a numeric compute pipeline id");
    let graphics_pipeline: u64 = graphics_pipeline
        .parse()
        .expect("a numeric graphics pipeline id");

    // Three physical pages; two are mapped so the graphics page sits behind
    // the compute page at a nonzero section offset.
    let mut pages = [0usize; 3];
    let mut count = pages.len();
    // SAFETY: `count` and `pages` are valid for three page numbers.
    let allocated = super::XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr());
    assert_eq!(allocated, 1, "XMemAllocatePhysicalPages failed");
    assert_eq!(count, pages.len(), "expected three physical pages");

    // SAFETY: a plain reservation request; the range is used only through the
    // existing mapping calls below.
    let alias = super::XMemVirtualAlloc(
        std::ptr::null_mut(),
        2 * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!alias.is_null(), "XMemVirtualAlloc failed");
    // SAFETY: `pages[1..]` is two valid page numbers for the reservation.
    assert_eq!(
        super::XMemMapPhysicalPages(alias, 2, pages[1..].as_ptr()),
        alias,
        "XMemMapPhysicalPages failed"
    );

    // Page 0 of the alias is the compute dword; page 1 holds the indexed
    // triangle the guest records. Both live in the same guest allocation and
    // therefore the same registered backing file.
    // SAFETY: the reservation is mapped read-write for two pages.
    let bytes = unsafe { std::slice::from_raw_parts_mut(alias.cast::<u8>(), 2 * PAGE) };
    bytes.fill(0);
    bytes[..4].copy_from_slice(&FRAME_COMPUTE_INPUT.to_le_bytes());
    let graphics = &mut bytes[PAGE..];
    for (index, value) in [0u32, 1, 2].iter().enumerate() {
        let at = index * 4;
        graphics[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    // A full-viewport triangle: positions at byte 16 (float2 per vertex).
    let positions: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
    for (index, value) in positions.iter().enumerate() {
        let at = 16 + index * 4;
        graphics[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    // The GPU alias of the exact same physical pages.
    // SAFETY: a plain reservation request, as above.
    let gpu = super::XMemVirtualAlloc(
        std::ptr::null_mut(),
        2 * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!gpu.is_null(), "the GPU alias reservation failed");
    assert!(
        super::map_alias(gpu as u64, alias as u64, (2 * PAGE) as u64),
        "map_alias failed"
    );
    assert_ne!(
        // SAFETY: `alias` is a live two-page view of the file-backed section.
        unsafe { FlushViewOfFile(alias.cast(), 2 * PAGE) },
        0,
        "FlushViewOfFile failed"
    );

    let compute_address = gpu as usize;
    let graphics_address = compute_address + PAGE;
    let compute_lease = share::resolve_guest_range(compute_address, PAGE)
        .expect("the compute page must resolve to host-visible backing");
    let graphics_lease = share::resolve_guest_range(graphics_address, PAGE)
        .expect("the graphics page must resolve to host-visible backing");
    let range = compute_lease.range;
    assert_ne!(range.id, 0, "the resolved id is zero");
    assert_eq!(range.generation, share::GENERATION, "unexpected generation");
    assert_eq!(
        graphics_lease.range.id, range.id,
        "the two pages must share one allocation"
    );
    assert_eq!(
        graphics_lease.path(),
        compute_lease.path(),
        "the two pages must share one backing file"
    );
    assert_ne!(
        graphics_lease.range.offset, range.offset,
        "the graphics page must not overlap the compute page"
    );

    let file_len = std::fs::metadata(compute_lease.path())
        .expect("the backing file metadata")
        .len();
    assert_eq!(file_len, (3 * PAGE) as u64, "unexpected backing file length");
    assert_eq!(
        compute_lease.path().parent(),
        Some(std::path::Path::new(&directory)),
        "the backing file is outside the harness share directory"
    );

    let mut client =
        GuestIpcClient::connect(port, &secret).expect("the native host must accept the session");
    client
        .register(range.id, file_len)
        .expect("the real backing must register");

    let generation = u32::try_from(range.generation).expect("a u32 generation");
    let compute_binding = BufferBinding {
        allocation_id: range.id,
        generation,
        offset: compute_lease.range.offset,
        length: 4,
    };
    let index_binding = BufferBinding {
        allocation_id: range.id,
        generation,
        offset: graphics_lease.range.offset,
        length: (3 * 4) as u64,
    };
    let vertex_binding = BufferBinding {
        allocation_id: range.id,
        generation,
        offset: graphics_lease.range.offset + 16,
        length: (6 * 4) as u64,
    };

    let identity = ListIdentity::new(1, 1).expect("a valid list identity");
    let mut green = Vec::new();
    for value in [0.0f32, 1.0, 0.0, 1.0] {
        green.extend_from_slice(&value.to_le_bytes());
    }
    let graphics_commands = vec![
        Command::SetPipeline(SetPipeline {
            pipeline_id: graphics_pipeline,
        }),
        Command::RootConstants(RootConstants {
            slot: 1,
            bytes: green,
        }),
        Command::DrawIndexed(DrawIndexed {
            pipeline_id: graphics_pipeline,
            index_count: 3,
            instance_count: 1,
            start_index: 0,
            base_vertex: 0,
            start_instance: 0,
            list_id: identity.list_id,
            list_generation: identity.list_generation,
            bindings: vec![index_binding, vertex_binding],
        }),
    ];

    // Submission 1: the graphics list, claimed by its list identity.
    let reply = client
        .submit_with_list(graphics_commands.clone(), identity)
        .expect("the graphics list must complete on the native host");
    assert_eq!(reply.sequence, 1, "unexpected graphics reply sequence");
    assert_eq!(reply.status, ReplyStatus::Ok, "unexpected graphics reply");

    // Submission 2: the published compute kernel over the other page.
    let compute_commands = vec![Command::Dispatch1d(Dispatch1d {
        pipeline_id: compute_pipeline,
        threads: 1,
        threads_per_group: 1,
        bindings: vec![compute_binding],
    })];
    let reply = client
        .submit_commands(compute_commands)
        .expect("the compute list must complete on the native host");
    assert_eq!(reply.sequence, 2, "unexpected compute reply sequence");
    assert_eq!(reply.status, ReplyStatus::Ok, "unexpected compute reply");

    // The compute result is visible through the guest alias, not merely in a
    // host-side buffer.
    // SAFETY: `gpu` is the live two-page alias created above.
    let through_alias = unsafe { std::slice::from_raw_parts(gpu.cast::<u32>(), 2 * PAGE / 4) };
    assert_eq!(
        through_alias[0], FRAME_COMPUTE_OUTPUT,
        "the compute constant is not visible through the guest alias"
    );
    // The triangle bytes the guest wrote must still be the bytes the host read.
    let gfx_words = &through_alias[PAGE / 4..];
    assert_eq!(&gfx_words[..3], &[0u32, 1, 2], "graphics indices changed");
    let gfx_positions: Vec<f32> = gfx_words[4..10]
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect();
    assert_eq!(
        gfx_positions,
        positions.to_vec(),
        "graphics positions changed in the guest alias"
    );

    // Snapshot the whole alias after the compute result, then replay the same
    // graphics list with the same identity and the next sequence.
    // SAFETY: `gpu` is the live two-page alias created above.
    let snapshot = unsafe { std::slice::from_raw_parts(gpu.cast::<u8>(), 2 * PAGE) }.to_vec();
    let reply = client
        .submit_with_list(graphics_commands, identity)
        .expect("a replay must still return a reply");
    assert_eq!(reply.sequence, 3, "unexpected replay reply sequence");
    assert_eq!(
        reply.status,
        ReplyStatus::AlreadyExecuted,
        "a replay of the executed graphics list must be refused"
    );

    // SAFETY: `gpu` is the live two-page alias created above.
    let after_replay = unsafe { std::slice::from_raw_parts(gpu.cast::<u8>(), 2 * PAGE) };
    assert_eq!(
        after_replay, snapshot.as_slice(),
        "the refused replay modified the guest alias"
    );

    client.close();
    // SAFETY: unmapping the two views we made; the pages are freed after.
    super::XMemMapPhysicalPages(gpu, 2, std::ptr::null());
    super::XMemMapPhysicalPages(alias, 2, std::ptr::null());
    // SAFETY: `pages` holds the three allocated page numbers.
    assert_eq!(
        super::XMemFreePhysicalPages(pages.len(), pages.as_ptr()),
        1,
        "XMemFreePhysicalPages failed"
    );
    println!(
        "PASS xmem-ipc-frame draw=triangle compute_input={FRAME_COMPUTE_INPUT} \
         compute={FRAME_COMPUTE_OUTPUT} replay=AlreadyExecuted alias_unchanged=1 \
         graphics_pipeline={graphics_pipeline} compute_pipeline={compute_pipeline} \
         list_id={} list_generation={} id={} generation={} offset={} length={}",
        identity.list_id,
        identity.list_generation,
        range.id,
        range.generation,
        graphics_lease.range.offset,
        graphics_lease.range.length
    );
    drop(compute_lease);
    drop(graphics_lease);
}
