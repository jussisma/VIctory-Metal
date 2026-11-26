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
//! The harness owns the secret file and the share directory; nothing here
//! prints the secret. No title is launched.

use super::{PAGE, PAGES_64K};

use crate::modules::xmem::share;
use victory_metal::guest_ipc::SECRET_LEN;
use victory_metal::wire::{BufferBinding, Command, Dispatch1d, ReplyStatus, Submission};
use victory_metal::GuestIpcClient;

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
        let Command::Dispatch1d(dispatch) = command;
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
