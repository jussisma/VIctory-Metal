//! Path-in/path-out and persistent stdio DXIL to serialized metallib converter.
//!
//! Build and run this binary as `x86_64-apple-darwin` (Rosetta 2 on Apple
//! Silicon), because Apple's `libmetalirconverter.dylib` is x86_64-only. The
//! converter does not create an `MTLDevice`, an `MTLLibrary`, or any GPU work;
//! it only serializes the metallib bytes.
//!
//! Usage:
//! ```text
//! victory-irconverter --converter <libmetalirconverter.dylib> \
//!     --entry <entry-point> [--reflection <path>] <input.dxil> <output.metallib>
//! victory-irconverter --stdio --converter <libmetalirconverter.dylib> \
//!     [--reflection-dir <dir>]
//! ```
//!
//! `--converter` is required in both modes; `--entry` is required for the
//! path-in/path-out mode and is taken from each job frame in `--stdio` mode.
//! The root signature must already be embedded in the DXIL; `--root-signature`
//! only carries the caller's raw layout bytes into the reflection payload and
//! cache key, it never overrides the embedded signature. DXIL input is copied
//! by IRConverter (`IRObjectCreateFromDXIL` ownership `1`) before the Rust
//! buffer is released.
//!
//! `--reflection` writes the binary reflection payload consumed by
//! `victory-metal`: converter stage, metallib hash and size, DXIL/root-layout
//! hashes, translation options and any `--binding` metadata.
//! `--converter-version`, `--target`, `--min-os`, `--flag` and `--binding` are
//! supplied by the renderer so key and reflection agree.
//!
//! # Persistent stdio protocol
//!
//! `--stdio` keeps one process alive and serves framed jobs on stdin/stdout
//! using the protocol [`victory_metal::compiler_pool`] already speaks. Each
//! request is a text header followed by the raw payload:
//!
//! ```text
//! VMJOB <seq> <job-id> <stage> <entry-len> <dxil-len>\n<entry><dxil>
//! ```
//!
//! and each response echoes `seq` and `job-id`. Every job is converted through
//! the reflection-aware converter API; the *response frame* is selected by
//! `VICTORY_IRCONVERTER_STDIO_REFLECTION`, which the pool sets on every child
//! it starts:
//!
//! ```text
//! VMRES2 <seq> <job-id> <status> <metallib-len> <reflection-len>\n\
//!     <metallib><reflection-json>
//! VMRES  <seq> <job-id> <status> <payload-len>\n<payload>   (no env switch)
//! ```
//!
//! `status == 0` carries the serialized metallib plus Apple's reflection JSON
//! (`IRShaderReflectionCopyJSONString`) for the linked program; any other
//! status carries a UTF-8 diagnostic and an empty reflection. The reflection is
//! what the pool hands to the guest so it can build the per-stage
//! `PipelineAbi`; the stdio path therefore uses
//! [`convert_dxil_to_metallib_with_reflection`], not the metallib-only
//! conversion. A converter build without the reflection symbols is an explicit
//! error, never a metallib with a fabricated layout. Because the switch only
//! changes the response, an unchanged driver such as the live gate's own DXIL
//! pool stage keeps reading `VMRES` and is not broken by the new channel.
//!
//! A malformed header whose lengths still parse is rejected with a nonzero
//! status after its payload is drained, so the loop stays alive. The process
//! exits 0 on a clean stdin EOF. `--reflection-dir` is a daemon option that
//! writes `<seq>-<job-id>.reflection` (the binary [`ReflectionPayload`]) for
//! each successful job; the pool frame carries no per-job reflection path, so
//! the pool itself derives reflection metadata from the job instead of asking
//! the worker.
//!
//! Input limits are fixed by `victory-metal`: at most
//! `MAX_DXIL_INPUT_BYTES` (16 MiB) of DXIL, at most `MAX_ENTRY_POINT_BYTES`
//! (256) bytes in the entry name, and at most `MAX_METALLIB_OUTPUT_BYTES`
//! (64 MiB) of output. There is no environment-based converter fallback: the
//! caller names exactly one converter file in both modes.
//!
//! On non-macOS targets this binary exists only so workspace `--all-targets`
//! checks keep compiling; it exits with a usage error.

#[cfg(target_os = "macos")]
fn main() {
    std::process::exit(cli::run());
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("victory-irconverter: only the macOS x86_64 build is supported");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod cli {
    use std::env;
    use std::ffi::OsString;
    use std::fs::{self, File};
    use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
    use std::path::{Path, PathBuf};

    use victory_metal::macos::cache::{
        BindingKind, BindingMetadata, MAX_CACHED_ROOT_SIGNATURE_BYTES, ReflectionPayload,
        TranslationOptions,
    };
    use victory_metal::macos::irconverter::{
        DEFAULT_CONVERTER_VERSION, MAX_DXIL_INPUT_BYTES, MAX_ENTRY_POINT_BYTES,
        MAX_METALLIB_OUTPUT_BYTES, convert_dxil_to_metallib,
        convert_dxil_to_metallib_with_reflection,
    };

    /// Environment switch that makes a worker answer the `VMRES2` frame.
    ///
    /// The pool sets it on every child it starts. A caller that drives the
    /// worker directly (the live gate's own DXIL pool stage) leaves it unset
    /// and keeps receiving the original `VMRES` frame, so an unchanged driver
    /// is not broken by the reflection channel. This is still a *response*
    /// extension: the request frame and the one-shot CLI are untouched.
    const REFLECTION_FRAMES_ENV: &str = "VICTORY_IRCONVERTER_STDIO_REFLECTION";

    /// Largest accepted stdio request header, matching the pool's own bound.
    const MAX_STDIO_HEADER_BYTES: usize = 4096;
    /// Largest accepted stdio payload (entry plus DXIL) before it is rejected
    /// as unrecoverable framing.
    const MAX_STDIO_PAYLOAD_BYTES: usize = MAX_DXIL_INPUT_BYTES + MAX_ENTRY_POINT_BYTES;

    const USAGE: &str = "\
usage: victory-irconverter --converter <libmetalirconverter.dylib> \\
           --entry <entry-point> [options] <input.dxil> <output.metallib>
       victory-irconverter --stdio --converter <libmetalirconverter.dylib> \\
           [--reflection-dir <dir>] [options]

  --converter, -c <path>  explicitly configured x86_64 libmetalirconverter.dylib
  --entry, -e <name>      DXIL entry point (embedded root signature is used)
  --stdio                 serve VMJOB/VMRES2 frames on stdin/stdout instead of
                          converting one input/output pair
  --reflection <path>     write the reflection payload (metallib hash, stage,
                          root layout, options and binding metadata)
  --reflection-dir <dir>  with --stdio, write <seq>-<job-id>.reflection for each
                          successful job (the pool frame has no per-job path)
  --root-signature <path> raw root signature bytes to carry into the reflection
                          and cache key (conversion still uses the embedded one)
  --converter-version <s> converter/protocol version string
  --target <string>       translation target (default apple-macos)
  --min-os <string>       minimum OS version
  --flag <string>         converter flag (repeatable)
  --binding <spec>        binding metadata uav:<register>:<metal-index>:<offset>
  --help, -h              show this help

The converter is deterministic and never opens a Metal device or runs GPU work.
Limits: DXIL <= 16777216 bytes, entry <= 256 bytes, metallib <= 67108864 bytes,
root signature <= 1048576 bytes.
";

    struct Args {
        converter: PathBuf,
        /// One-shot entry point. Empty in stdio mode, where each job names it.
        entry: String,
        /// One-shot input path. Empty in stdio mode.
        input: PathBuf,
        /// One-shot output path. Empty in stdio mode.
        output: PathBuf,
        reflection: Option<PathBuf>,
        reflection_dir: Option<PathBuf>,
        root_signature: Option<PathBuf>,
        converter_version: String,
        options: TranslationOptions,
        bindings: Vec<BindingMetadata>,
        stdio: bool,
    }

    pub(super) fn run() -> i32 {
        match parse_args(env::args_os().skip(1)) {
            Ok(None) => {
                print!("{USAGE}");
                0
            }
            Err(message) => {
                eprintln!("victory-irconverter: {message}\n\n{USAGE}");
                2
            }
            Ok(Some(args)) if args.stdio => run_stdio(&args),
            Ok(Some(args)) => match convert(&args) {
                Ok(()) => 0,
                Err(message) => {
                    eprintln!("victory-irconverter: {message}");
                    1
                }
            },
        }
    }

    fn parse_args(mut raw: impl Iterator<Item = OsString>) -> Result<Option<Args>, String> {
        let mut converter = None;
        let mut entry = None;
        let mut reflection = None;
        let mut reflection_dir = None;
        let mut root_signature = None;
        let mut converter_version = DEFAULT_CONVERTER_VERSION.to_string();
        let mut options = TranslationOptions::default();
        let mut bindings = Vec::new();
        let mut positional = Vec::new();
        let mut only_positional = false;
        let mut stdio = false;

        while let Some(arg) = raw.next() {
            if !only_positional && arg == "--" {
                only_positional = true;
                continue;
            }
            let text = arg.to_str();
            match text {
                Some("--help" | "-h") => return Ok(None),
                Some("--stdio") if !only_positional => {
                    if stdio {
                        return Err("--stdio was given more than once".to_string());
                    }
                    stdio = true;
                }
                Some("--converter" | "-c") if !only_positional => {
                    if converter.is_some() {
                        return Err("--converter was given more than once".to_string());
                    }
                    converter = Some(PathBuf::from(next_value(&mut raw, "--converter")?));
                }
                Some("--entry" | "-e") if !only_positional => {
                    if entry.is_some() {
                        return Err("--entry was given more than once".to_string());
                    }
                    let value = next_value(&mut raw, "--entry")?;
                    entry = Some(
                        value
                            .to_str()
                            .ok_or_else(|| "--entry must be valid UTF-8".to_string())?
                            .to_string(),
                    );
                }
                Some("--reflection") if !only_positional => {
                    if reflection.is_some() {
                        return Err("--reflection was given more than once".to_string());
                    }
                    reflection = Some(PathBuf::from(next_value(&mut raw, "--reflection")?));
                }
                Some("--reflection-dir") if !only_positional => {
                    if reflection_dir.is_some() {
                        return Err("--reflection-dir was given more than once".to_string());
                    }
                    reflection_dir = Some(PathBuf::from(next_value(&mut raw, "--reflection-dir")?));
                }
                Some("--root-signature") if !only_positional => {
                    if root_signature.is_some() {
                        return Err("--root-signature was given more than once".to_string());
                    }
                    root_signature = Some(PathBuf::from(next_value(&mut raw, "--root-signature")?));
                }
                Some("--converter-version") if !only_positional => {
                    converter_version = string_value(&mut raw, "--converter-version")?;
                }
                Some("--target") if !only_positional => {
                    options.target = string_value(&mut raw, "--target")?;
                }
                Some("--min-os") if !only_positional => {
                    options.min_os = string_value(&mut raw, "--min-os")?;
                }
                Some("--flag") if !only_positional => {
                    options.flags.push(string_value(&mut raw, "--flag")?);
                }
                Some("--binding") if !only_positional => {
                    let spec = string_value(&mut raw, "--binding")?;
                    bindings.push(parse_binding(&spec)?);
                }
                Some(value) if !only_positional && value.starts_with('-') && value != "-" => {
                    return Err(format!("unknown option `{value}`"));
                }
                _ => positional.push(arg),
            }
        }

        if stdio {
            if !positional.is_empty() {
                return Err(format!(
                    "--stdio serves alternating frames; it does not take input or output paths \
                     (got {})",
                    positional.len()
                ));
            }
            if entry.is_some() {
                return Err(
                    "--stdio takes the entry point from each job frame; drop --entry".to_string(),
                );
            }
            if reflection.is_some() {
                return Err(
                    "--stdio takes no per-job reflection path; use --reflection-dir".to_string(),
                );
            }
            if root_signature.is_some() {
                return Err(
                    "--stdio cannot carry a per-job root signature; the DXIL's embedded \
                     signature is used"
                        .to_string(),
                );
            }
            let converter =
                converter.ok_or_else(|| "--converter is required with --stdio".to_string())?;
            return Ok(Some(Args {
                converter,
                entry: String::new(),
                input: PathBuf::new(),
                output: PathBuf::new(),
                reflection: None,
                reflection_dir,
                root_signature: None,
                converter_version,
                options,
                bindings,
                stdio: true,
            }));
        }

        if reflection_dir.is_some() {
            return Err("--reflection-dir is only valid with --stdio".to_string());
        }
        if positional.len() != 2 {
            return Err(format!(
                "expected exactly one input and one output path, got {}",
                positional.len()
            ));
        }
        let output = PathBuf::from(positional.pop().unwrap());
        let input = PathBuf::from(positional.pop().unwrap());
        let converter = converter.ok_or_else(|| "--converter is required".to_string())?;
        let entry = entry.ok_or_else(|| "--entry is required".to_string())?;
        Ok(Some(Args {
            converter,
            entry,
            input,
            output,
            reflection,
            reflection_dir: None,
            root_signature,
            converter_version,
            options,
            bindings,
            stdio: false,
        }))
    }

    fn string_value(
        raw: &mut impl Iterator<Item = OsString>,
        option: &str,
    ) -> Result<String, String> {
        next_value(raw, option)?
            .into_string()
            .map_err(|_| format!("{option} must be valid UTF-8"))
    }

    fn parse_binding(spec: &str) -> Result<BindingMetadata, String> {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 4 {
            return Err(format!(
                "invalid --binding `{spec}`; expected uav:<register>:<metal-index>:<offset>"
            ));
        }
        let kind = BindingKind::from_str(parts[0])
            .ok_or_else(|| format!("unknown binding kind `{}`", parts[0]))?;
        let number = |text: &str, label: &str| -> Result<u32, String> {
            text.parse::<u32>()
                .map_err(|_| format!("invalid {label} `{text}` in --binding"))
        };
        Ok(BindingMetadata {
            kind,
            register: number(parts[1], "register")?,
            metal_index: number(parts[2], "metal index")?,
            byte_offset: number(parts[3], "byte offset")?,
        })
    }

    fn next_value(
        raw: &mut impl Iterator<Item = OsString>,
        option: &str,
    ) -> Result<OsString, String> {
        raw.next()
            .ok_or_else(|| format!("{option} requires a value"))
    }

    fn convert(args: &Args) -> Result<(), String> {
        let dxil = read_limited(&args.input, MAX_DXIL_INPUT_BYTES, "DXIL input")?;
        if args.entry.len() > MAX_ENTRY_POINT_BYTES {
            return Err(format!(
                "entry point is {} bytes; limit is {MAX_ENTRY_POINT_BYTES}",
                args.entry.len()
            ));
        }
        let root_signature = match &args.root_signature {
            Some(path) => Some(read_root_signature(path)?),
            None => None,
        };
        let output = convert_dxil_to_metallib(&dxil, &args.entry, &args.converter)
            .map_err(|err| format!("conversion failed: {err}"))?;
        if output.bytes.is_empty() || output.bytes.len() > MAX_METALLIB_OUTPUT_BYTES {
            return Err(format!(
                "converted metallib has invalid size {}",
                output.bytes.len()
            ));
        }
        fs::write(&args.output, &output.bytes).map_err(|err| {
            format!(
                "failed to write output {}: {err}",
                args.output.display()
            )
        })?;

        if let Some(path) = &args.reflection {
            let reflection = ReflectionPayload::from_translation(
                &output.bytes,
                output.stage,
                &args.entry,
                &dxil,
                root_signature.as_deref(),
                &args.converter_version,
                &args.options,
                &args.bindings,
            );
            fs::write(path, reflection.encode()).map_err(|err| {
                format!("failed to write reflection {}: {err}", path.display())
            })?;
        }

        println!(
            "IRCONVERT_OK entry={:?} stage={} dxil_bytes={} metallib_bytes={}",
            args.entry,
            output.stage,
            dxil.len(),
            output.bytes.len()
        );
        Ok(())
    }

    /// Serves the pool's persistent `VMJOB`/`VMRES2` protocol until stdin EOF.
    ///
    /// The frame layout is owned by [`victory_metal::compiler_pool`]; this
    /// function must not invent another one. Every job goes through
    /// [`convert_dxil_to_metallib_with_reflection`] so the answer carries both
    /// the metallib and the converter's reflection JSON; the one-shot mode
    /// keeps the metallib-only [`convert_dxil_to_metallib`] path.
    ///
    /// [`REFLECTION_FRAMES_ENV`] selects the response frame. The pool sets it
    /// and receives `VMRES2`; a direct driver that does not set it keeps
    /// receiving the original `VMRES` frame (metallib only).
    fn run_stdio(args: &Args) -> i32 {
        if !args.converter.is_file() {
            eprintln!(
                "victory-irconverter: --converter {} is not a regular file",
                args.converter.display()
            );
            return 2;
        }
        let reflection_frames = env::var_os(REFLECTION_FRAMES_ENV)
            .is_some_and(|value| !value.is_empty() && value != "0");
        if let Some(dir) = &args.reflection_dir {
            if let Err(err) = fs::create_dir_all(dir) {
                eprintln!(
                    "victory-irconverter: cannot create reflection directory {}: {err}",
                    dir.display()
                );
                return 2;
            }
        }

        let stdin = io::stdin();
        let mut reader = BufReader::new(stdin.lock());
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());
        let mut served: u64 = 0;

        loop {
            let mut line = Vec::new();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) => {
                    eprintln!("victory-irconverter: stdio EOF after {served} jobs");
                    return 0;
                }
                Ok(_) => {}
                Err(err) => {
                    eprintln!("victory-irconverter: failed to read stdio header: {err}");
                    return 2;
                }
            }
            if line.len() > MAX_STDIO_HEADER_BYTES {
                eprintln!(
                    "victory-irconverter: stdio header is {} bytes; limit is {}",
                    line.len(),
                    MAX_STDIO_HEADER_BYTES
                );
                return 2;
            }
            while matches!(line.last(), Some(b'\n' | b'\r')) {
                line.pop();
            }
            let Ok(text) = std::str::from_utf8(&line) else {
                eprintln!("victory-irconverter: stdio header is not UTF-8");
                return 2;
            };
            let fields: Vec<&str> = text.split_ascii_whitespace().collect();
            let seq = fields.get(1).and_then(|value| value.parse::<u64>().ok());
            let id = fields.get(2).and_then(|value| value.parse::<u64>().ok());
            let entry_len = fields.get(4).and_then(|value| value.parse::<usize>().ok());
            let dxil_len = fields.get(5).and_then(|value| value.parse::<usize>().ok());

            let (Some(entry_len), Some(dxil_len)) = (entry_len, dxil_len) else {
                // Without lengths there is no way to find the next frame; answer
                // with the ids we could parse and end the loop instead of
                // silently misparsing binary payload as headers.
                let _ = answer(
                    &mut writer,
                    reflection_frames,
                    seq.unwrap_or(0),
                    id.unwrap_or(0),
                    2,
                    b"malformed VMJOB header without byte lengths",
                    &[],
                );
                return 2;
            };
            let Some(payload_len) = entry_len.checked_add(dxil_len) else {
                eprintln!("victory-irconverter: stdio frame length overflow");
                return 2;
            };
            if payload_len > MAX_STDIO_PAYLOAD_BYTES {
                eprintln!(
                    "victory-irconverter: stdio frame is {payload_len} bytes; limit is \
                     {MAX_STDIO_PAYLOAD_BYTES}"
                );
                return 2;
            }
            let mut payload = vec![0u8; payload_len];
            if let Err(err) = reader.read_exact(&mut payload) {
                eprintln!("victory-irconverter: truncated stdio frame: {err}");
                return 2;
            }

            let header_ok = fields.len() == 6
                && fields[0] == "VMJOB"
                && seq.is_some()
                && id.is_some()
                && fields[3].parse::<u32>().is_ok();
            if !header_ok {
                if answer(
                    &mut writer,
                    reflection_frames,
                    seq.unwrap_or(0),
                    id.unwrap_or(0),
                    2,
                    b"malformed VMJOB header",
                    &[],
                )
                .is_err()
                {
                    return 0;
                }
                continue;
            }
            let seq = seq.expect("checked above");
            let id = id.expect("checked above");
            let stage = fields[3].parse::<u32>().expect("checked above");
            let (entry_bytes, dxil) = payload.split_at(entry_len);
            let Ok(entry) = std::str::from_utf8(entry_bytes) else {
                if answer(
                    &mut writer,
                    reflection_frames,
                    seq,
                    id,
                    2,
                    b"entry point is not UTF-8",
                    &[],
                )
                .is_err()
                {
                    return 0;
                }
                continue;
            };

            match convert_stdio_job(args, seq, id, stage, entry, dxil) {
                Ok(job) => {
                    served += 1;
                    if answer(
                        &mut writer,
                        reflection_frames,
                        seq,
                        id,
                        0,
                        &job.metallib,
                        &job.reflection,
                    )
                    .is_err()
                    {
                        return 0;
                    }
                }
                Err((status, message)) => {
                    if answer(
                        &mut writer,
                        reflection_frames,
                        seq,
                        id,
                        status,
                        message.as_bytes(),
                        &[],
                    )
                    .is_err()
                    {
                        return 0;
                    }
                }
            }
        }
    }

    /// One stdio answer: the metallib and the converter reflection JSON.
    struct StdioJob {
        metallib: Vec<u8>,
        reflection: Vec<u8>,
    }

    /// Runs one stdio job through the reflection-aware conversion core.
    ///
    /// The stdio path always asks the converter for reflection: the pool's
    /// `compile` ignores it, but `compile_with_reflection` needs it to build a
    /// pipeline ABI, so emitting it unconditionally keeps one worker binary
    /// able to serve both callers.
    fn convert_stdio_job(
        args: &Args,
        seq: u64,
        id: u64,
        stage: u32,
        entry: &str,
        dxil: &[u8],
    ) -> Result<StdioJob, (u32, String)> {
        if entry.is_empty() || entry.len() > MAX_ENTRY_POINT_BYTES {
            return Err((
                2,
                format!(
                    "entry point is {} bytes; limit is {MAX_ENTRY_POINT_BYTES}",
                    entry.len()
                ),
            ));
        }
        if dxil.is_empty() || dxil.len() > MAX_DXIL_INPUT_BYTES {
            return Err((
                2,
                format!(
                    "DXIL input is {} bytes; limit is {MAX_DXIL_INPUT_BYTES}",
                    dxil.len()
                ),
            ));
        }
        let output = convert_dxil_to_metallib_with_reflection(dxil, entry, &args.converter)
            .map_err(|err| (1, format!("conversion failed: {err}")))?;
        let metallib = output.metallib;
        if metallib.bytes.is_empty() || metallib.bytes.len() > MAX_METALLIB_OUTPUT_BYTES {
            return Err((
                1,
                format!("converted metallib has invalid size {}", metallib.bytes.len()),
            ));
        }
        if stage != 0 && metallib.stage != stage {
            return Err((
                3,
                format!(
                    "frame declared stage {stage}, converter reported stage {}",
                    metallib.stage
                ),
            ));
        }
        let reflection = output.reflection_json.into_bytes();
        if reflection.is_empty() {
            return Err((
                1,
                "converter returned an empty reflection JSON".to_string(),
            ));
        }
        if let Some(dir) = &args.reflection_dir {
            let payload = ReflectionPayload::from_translation(
                &metallib.bytes,
                metallib.stage,
                entry,
                dxil,
                None,
                &args.converter_version,
                &args.options,
                &args.bindings,
            );
            let path = dir.join(format!("{seq}-{id}.reflection"));
            if let Err(err) = fs::write(&path, payload.encode()) {
                return Err((
                    4,
                    format!("failed to write reflection {}: {err}", path.display()),
                ));
            }
        }
        Ok(StdioJob {
            metallib: metallib.bytes,
            reflection,
        })
    }

    /// Writes the response in the frame the caller selected.
    ///
    /// `VMRES2` carries the reflection next to the metallib; `VMRES` is the
    /// original metallib-only frame a direct driver keeps receiving.
    fn answer(
        writer: &mut impl Write,
        reflection_frames: bool,
        seq: u64,
        id: u64,
        status: u32,
        metallib: &[u8],
        reflection: &[u8],
    ) -> io::Result<()> {
        if reflection_frames {
            write_response_with_reflection(writer, seq, id, status, metallib, reflection)
        } else {
            write_response(writer, seq, id, status, metallib)
        }
    }

    fn write_response(
        writer: &mut impl Write,
        seq: u64,
        id: u64,
        status: u32,
        payload: &[u8],
    ) -> io::Result<()> {
        write!(writer, "VMRES {seq} {id} {status} {}\n", payload.len())?;
        writer.write_all(payload)?;
        writer.flush()
    }

    fn write_response_with_reflection(
        writer: &mut impl Write,
        seq: u64,
        id: u64,
        status: u32,
        metallib: &[u8],
        reflection: &[u8],
    ) -> io::Result<()> {
        write!(
            writer,
            "VMRES2 {seq} {id} {status} {} {}\n",
            metallib.len(),
            reflection.len()
        )?;
        writer.write_all(metallib)?;
        writer.write_all(reflection)?;
        writer.flush()
    }

    fn read_root_signature(path: &Path) -> Result<Vec<u8>, String> {
        let metadata = fs::metadata(path)
            .map_err(|err| format!("failed to stat root signature {}: {err}", path.display()))?;
        if !metadata.is_file() {
            return Err(format!(
                "root signature is not a regular file: {}",
                path.display()
            ));
        }
        let reported = metadata.len();
        if reported > MAX_CACHED_ROOT_SIGNATURE_BYTES as u64 {
            return Err(format!(
                "root signature is {reported} bytes; limit is {MAX_CACHED_ROOT_SIGNATURE_BYTES}"
            ));
        }
        let mut bytes = Vec::with_capacity(reported as usize);
        File::open(path)
            .and_then(|mut file| file.read_to_end(&mut bytes))
            .map_err(|err| {
                format!(
                    "failed to read root signature {}: {err}",
                    path.display()
                )
            })?;
        if bytes.len() > MAX_CACHED_ROOT_SIGNATURE_BYTES {
            return Err(format!(
                "root signature grew past the {MAX_CACHED_ROOT_SIGNATURE_BYTES}-byte limit"
            ));
        }
        Ok(bytes)
    }

    fn read_limited(path: &Path, limit: usize, label: &str) -> Result<Vec<u8>, String> {
        let metadata = fs::metadata(path)
            .map_err(|err| format!("failed to stat {label} {}: {err}", path.display()))?;
        if !metadata.is_file() {
            return Err(format!("{label} is not a regular file: {}", path.display()));
        }
        let reported = metadata.len();
        if reported == 0 {
            return Err(format!("{label} is empty: {}", path.display()));
        }
        if reported > limit as u64 {
            return Err(format!(
                "{label} is {reported} bytes; limit is {limit}: {}",
                path.display()
            ));
        }

        let file = File::open(path)
            .map_err(|err| format!("failed to open {label} {}: {err}", path.display()))?;
        let mut bytes = Vec::with_capacity(reported as usize);
        file.take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|err| format!("failed to read {label} {}: {err}", path.display()))?;
        if bytes.is_empty() {
            return Err(format!("{label} is empty: {}", path.display()));
        }
        if bytes.len() > limit {
            return Err(format!(
                "{label} grew past the {limit}-byte limit while reading: {}",
                path.display()
            ));
        }
        Ok(bytes)
    }
}
