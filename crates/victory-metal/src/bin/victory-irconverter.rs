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
//! # Per-job isolation
//!
//! The `--stdio` process never runs a conversion itself. For every job it
//! re-execs this binary in the private `--convert-one` mode (DXIL on stdin,
//! one private frame on stdout) and waits for it. A SIGSEGV/SIGABRT inside
//! `libmetalirconverter.dylib`, or a Rust panic, therefore kills only that
//! short-lived child; the supervisor reports `converter child died: signal
//! <n> (<NAME>)`, its scrubbed stderr tail, or its exit code in a nonzero
//! `VMRES2` frame and keeps serving the next job. The private child frame is
//! an implementation detail and never crosses the pool boundary.
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
    use std::process::{Command, Stdio};
    use std::sync::mpsc::{self, Receiver};
    use std::thread;
    use std::time::Duration;

    use victory_metal::compiler_pool::MAX_REFLECTION_OUTPUT_BYTES;
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

    /// Internal mode: convert one job from stdin and answer on stdout.
    ///
    /// The `--stdio` supervisor starts this as a short-lived child per job so
    /// a SIGSEGV/SIGABRT inside the x86_64 converter (or a Rust panic) kills
    /// only that child, never the process holding the protocol loop.
    const CONVERT_ONE_ARG: &str = "--convert-one";
    /// Internal supervisor option: program to re-exec per job (defaults to the
    /// running worker). Tests use a stub; production leaves it unset.
    const JOB_RUNNER_ARG: &str = "--job-runner";
    /// Magic of the private supervisor <-> conversion-child frame.
    ///
    /// This never crosses the pool boundary: the supervisor validates it and
    /// re-emits the pool's `VMRES`/`VMRES2` frame with the request's seq/id.
    const JOB_CHILD_MAGIC: &str = "VMCONV";
    /// Largest diagnostic the supervisor puts into a failed `VMRES2` answer.
    const MAX_JOB_REASON_BYTES: usize = 1024;
    /// Raw child stderr retained by the supervisor before scrubbing.
    const MAX_CHILD_STDERR_CAPTURE_BYTES: usize = 8192;
    /// Scrubbed child stderr appended to a failure reason.
    const MAX_CHILD_STDERR_TAIL_BYTES: usize = 512;

    /// Where the caller is expected to name the converter library.
    ///
    /// The pool resolves `--converter` from the same two variables and passes
    /// it explicitly, so a missing flag is an installation error rather than a
    /// worker default. The sentence is part of the worker's stderr, which the
    /// pool captures and redacts.
    const CONVERTER_HINT: &str =
        "set VICTORY_METAL_IRCONVERTER or D3DMETAL_RES to libmetalirconverter.dylib";

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
        /// One-shot input path. Empty in stdio mode and in `--convert-one`.
        input: PathBuf,
        /// One-shot output path. Empty in stdio mode and in `--convert-one`.
        output: PathBuf,
        reflection: Option<PathBuf>,
        reflection_dir: Option<PathBuf>,
        root_signature: Option<PathBuf>,
        converter_version: String,
        options: TranslationOptions,
        bindings: Vec<BindingMetadata>,
        stdio: bool,
        /// Internal per-job conversion child mode (stdin DXIL, stdout frame).
        convert_one: bool,
        /// Declared shader stage of this `--convert-one` job, when the
        /// supervisor named one.
        stage: Option<u32>,
        /// Internal supervisor override of the per-job child program.
        job_runner: Option<PathBuf>,
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
            Ok(Some(args)) if args.convert_one => run_convert_one(&args),
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
        let mut convert_one = false;
        let mut stage = None;
        let mut job_runner = None;

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
                Some(CONVERT_ONE_ARG) if !only_positional => {
                    if convert_one {
                        return Err(format!("{CONVERT_ONE_ARG} was given more than once"));
                    }
                    convert_one = true;
                }
                Some(JOB_RUNNER_ARG) if !only_positional => {
                    if job_runner.is_some() {
                        return Err(format!("{JOB_RUNNER_ARG} was given more than once"));
                    }
                    job_runner = Some(PathBuf::from(next_value(&mut raw, JOB_RUNNER_ARG)?));
                }
                Some("--stage") if !only_positional => {
                    if stage.is_some() {
                        return Err("--stage was given more than once".to_string());
                    }
                    let value = string_value(&mut raw, "--stage")?;
                    stage = Some(
                        value
                            .parse::<u32>()
                            .map_err(|_| format!("invalid --stage `{value}`"))?,
                    );
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

        if convert_one && stdio {
            return Err(format!(
                "{CONVERT_ONE_ARG} and --stdio are two different transports; pick one"
            ));
        }

        if convert_one {
            if !positional.is_empty() {
                return Err(format!(
                    "{CONVERT_ONE_ARG} reads one DXIL module from stdin; it takes no paths \
                     (got {})",
                    positional.len()
                ));
            }
            if reflection.is_some() {
                return Err(format!(
                    "{CONVERT_ONE_ARG} answers on stdout; it takes no --reflection path"
                ));
            }
            if reflection_dir.is_some() {
                return Err(format!(
                    "{CONVERT_ONE_ARG} is the supervisor's private child; --reflection-dir is \
                     handled by the supervisor"
                ));
            }
            if root_signature.is_some() {
                return Err(format!(
                    "{CONVERT_ONE_ARG} cannot carry a per-job root signature; the DXIL's \
                     embedded signature is used"
                ));
            }
            if job_runner.is_some() {
                return Err(format!(
                    "{JOB_RUNNER_ARG} is a supervisor option, not a {CONVERT_ONE_ARG} one"
                ));
            }
            let converter = converter.ok_or_else(|| {
                format!("--converter is required with {CONVERT_ONE_ARG} ({CONVERTER_HINT})")
            })?;
            let entry = entry.ok_or_else(|| {
                format!("--entry is required with {CONVERT_ONE_ARG}")
            })?;
            return Ok(Some(Args {
                converter,
                entry,
                input: PathBuf::new(),
                output: PathBuf::new(),
                reflection: None,
                reflection_dir: None,
                root_signature: None,
                converter_version,
                options,
                bindings,
                stdio: false,
                convert_one: true,
                stage,
                job_runner: None,
            }));
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
            if stage.is_some() {
                return Err(
                    "--stdio takes the stage from each job frame; drop --stage".to_string(),
                );
            }
            let converter = converter.ok_or_else(|| {
                format!("--converter is required with --stdio ({CONVERTER_HINT})")
            })?;
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
                convert_one: false,
                stage: None,
                job_runner,
            }));
        }

        if reflection_dir.is_some() {
            return Err("--reflection-dir is only valid with --stdio".to_string());
        }
        if job_runner.is_some() {
            return Err(format!("{JOB_RUNNER_ARG} is only valid with --stdio"));
        }
        if stage.is_some() {
            return Err("--stage is only valid with --convert-one".to_string());
        }
        if positional.len() != 2 {
            return Err(format!(
                "expected exactly one input and one output path, got {}",
                positional.len()
            ));
        }
        let output = PathBuf::from(positional.pop().unwrap());
        let input = PathBuf::from(positional.pop().unwrap());
        let converter =
            converter.ok_or_else(|| format!("--converter is required ({CONVERTER_HINT})"))?;
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
            convert_one: false,
            stage: None,
            job_runner: None,
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

    /// Converts exactly one job and writes the private child frame.
    ///
    /// This is the process the supervisor re-execs per job: if the converter
    /// (or this code) dies by signal, only this short-lived child is lost. The
    /// answer is written to stdout as
    ///
    /// ```text
    /// VMCONV <status> <stage> <payload-len> <reflection-len>\n<payload><reflection>
    /// ```
    ///
    /// with `status == 0` carrying metallib bytes plus reflection JSON and a
    /// nonzero status carrying a UTF-8 diagnostic. The private frame never
    /// crosses the pool boundary; the supervisor re-emits `VMRES`/`VMRES2`.
    /// Conversion failures are reported in the frame and the process exits 0,
    /// so the supervisor can tell "converted and refused" from "child died".
    fn run_convert_one(args: &Args) -> i32 {
        let dxil = match read_stdin_limited(MAX_DXIL_INPUT_BYTES, "DXIL input") {
            Ok(bytes) => bytes,
            Err(message) => return answer_child_error(1, &message),
        };
        if let Err(message) = validate_job_inputs(&args.entry, &dxil) {
            return answer_child_error(2, &message);
        }
        let output = match convert_dxil_to_metallib_with_reflection(
            &dxil,
            &args.entry,
            &args.converter,
        ) {
            Ok(output) => output,
            Err(err) => return answer_child_error(1, &format!("conversion failed: {err}")),
        };
        let metallib = output.metallib;
        if metallib.bytes.is_empty() || metallib.bytes.len() > MAX_METALLIB_OUTPUT_BYTES {
            return answer_child_error(
                1,
                &format!("converted metallib has invalid size {}", metallib.bytes.len()),
            );
        }
        if let Some(stage) = args.stage {
            if stage != 0 && metallib.stage != stage {
                return answer_child_error(
                    3,
                    &format!(
                        "frame declared stage {stage}, converter reported stage {}",
                        metallib.stage
                    ),
                );
            }
        }
        if output.reflection_json.is_empty() {
            return answer_child_error(1, "converter returned an empty reflection JSON");
        }
        let mut stdout = io::stdout();
        match write_child_frame(
            &mut stdout,
            0,
            metallib.stage,
            &metallib.bytes,
            output.reflection_json.as_bytes(),
        ) {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("victory-irconverter: cannot write the conversion frame: {err}");
                1
            }
        }
    }

    /// Validates the entry/DXIL pair shared by the stdio path and its child.
    fn validate_job_inputs(entry: &str, dxil: &[u8]) -> Result<(), String> {
        if entry.is_empty() || entry.len() > MAX_ENTRY_POINT_BYTES {
            return Err(format!(
                "entry point is {} bytes; limit is {MAX_ENTRY_POINT_BYTES}",
                entry.len()
            ));
        }
        if dxil.is_empty() || dxil.len() > MAX_DXIL_INPUT_BYTES {
            return Err(format!(
                "DXIL input is {} bytes; limit is {MAX_DXIL_INPUT_BYTES}",
                dxil.len()
            ));
        }
        Ok(())
    }

    /// Reads one bounded stdin stream for `--convert-one`.
    fn read_stdin_limited(limit: usize, label: &str) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        io::stdin()
            .lock()
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|err| format!("failed to read {label} from stdin: {err}"))?;
        if bytes.is_empty() {
            return Err(format!("{label} is empty"));
        }
        if bytes.len() > limit {
            return Err(format!("{label} exceeds the {limit}-byte limit"));
        }
        Ok(bytes)
    }

    /// Writes a failed child frame and returns the process status to exit with.
    fn answer_child_error(status: u32, message: &str) -> i32 {
        let message = sanitize_reason(message, MAX_JOB_REASON_BYTES);
        let mut stdout = io::stdout();
        match write_child_frame(&mut stdout, status, 0, message.as_bytes(), &[]) {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("victory-irconverter: cannot write the conversion frame: {err}");
                1
            }
        }
    }

    /// Writes the private supervisor <-> conversion-child frame.
    fn write_child_frame(
        writer: &mut impl Write,
        status: u32,
        stage: u32,
        payload: &[u8],
        reflection: &[u8],
    ) -> io::Result<()> {
        write!(
            writer,
            "{JOB_CHILD_MAGIC} {status} {stage} {} {}\n",
            payload.len(),
            reflection.len()
        )?;
        writer.write_all(payload)?;
        writer.write_all(reflection)?;
        writer.flush()
    }

    /// Serves the pool's persistent `VMJOB`/`VMRES2` protocol until stdin EOF.
    ///
    /// The frame layout is owned by [`victory_metal::compiler_pool`]; this
    /// function must not invent another one. Every job runs in a short-lived
    /// [`run_convert_one`] child (a re-exec of this binary), so a
    /// SIGSEGV/SIGABRT/panic inside the x86_64 converter kills only that child.
    /// The supervisor reports the child's signal/exit code and scrubbed stderr
    /// tail in a nonzero `VMRES2` frame and keeps reading stdin, so a failed
    /// job can never desync the protocol.
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
        // One line on stderr names the converter this child was actually given
        // and the frame it will answer. The pool keeps a bounded tail of
        // stderr, so a later exit can be explained instead of reported as a
        // silent `WorkerExited`.
        eprintln!(
            "victory-irconverter: serving --stdio with converter {} (frame {})",
            args.converter.display(),
            if reflection_frames { "VMRES2" } else { "VMRES" }
        );
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
                    // Put the bounded reason on the supervisor's own stderr as
                    // well: if this process is ever lost, the pool's captured
                    // stderr tail still explains the last failed job instead of
                    // stopping at the startup banner.
                    eprintln!(
                        "victory-irconverter: job {seq}/{id} failed with status {status}: {message}"
                    );
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

    /// One stdio answer: the metallib, its reflection JSON and the stage the
    /// conversion child reported.
    struct StdioJob {
        metallib: Vec<u8>,
        reflection: Vec<u8>,
        stage: u32,
    }

    /// Runs one stdio job in an isolated conversion child.
    ///
    /// Nothing the converter can abort on runs in the supervisor: this builds
    /// a `--convert-one` command (normally a re-exec of `current_exe`, or the
    /// internal `--job-runner` used by tests), sends the DXIL on its stdin and
    /// reads the private child frame from its stdout. A signal, a nonzero exit
    /// or a malformed/absent frame becomes a bounded nonzero answer; the
    /// supervisor itself never dies and never loses its place on stdin.
    fn convert_stdio_job(
        args: &Args,
        seq: u64,
        id: u64,
        stage: u32,
        entry: &str,
        dxil: &[u8],
    ) -> Result<StdioJob, (u32, String)> {
        if let Err(message) = validate_job_inputs(entry, dxil) {
            return Err((2, message));
        }
        let job = run_conversion_child(args, stage, entry, dxil)?;
        if let Some(dir) = &args.reflection_dir {
            let payload = ReflectionPayload::from_translation(
                &job.metallib,
                job.stage,
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
        Ok(job)
    }

    /// Spawns the short-lived conversion child and turns its outcome into a
    /// [`StdioJob`] or a bounded `(status, reason)` answer.
    fn run_conversion_child(
        args: &Args,
        stage: u32,
        entry: &str,
        dxil: &[u8],
    ) -> Result<StdioJob, (u32, String)> {
        let program = match &args.job_runner {
            Some(path) => path.clone(),
            None => env::current_exe().map_err(|err| {
                (
                    1,
                    format!("cannot locate the conversion child program: {err}"),
                )
            })?,
        };
        let mut command = Command::new(&program);
        command
            .arg(CONVERT_ONE_ARG)
            .arg("--converter")
            .arg(&args.converter)
            .arg("--entry")
            .arg(entry)
            .arg("--converter-version")
            .arg(&args.converter_version)
            .arg("--target")
            .arg(&args.options.target)
            .arg("--min-os")
            .arg(&args.options.min_os);
        for flag in &args.options.flags {
            command.arg("--flag").arg(flag);
        }
        for binding in &args.bindings {
            command.arg("--binding").arg(format!(
                "{}:{}:{}:{}",
                binding.kind.as_str(),
                binding.register,
                binding.metal_index,
                binding.byte_offset
            ));
        }
        if stage != 0 {
            command.arg("--stage").arg(stage.to_string());
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // The child inherits the supervisor's process group on purpose: the
        // pool kills that group on a job timeout, and an independent group
        // would let a hung converter outlive the supervisor.
        let mut child = command.spawn().map_err(|err| {
            (
                1,
                format!(
                    "failed to start conversion child {}: {err}",
                    program.display()
                ),
            )
        })?;
        let stderr = child.stderr.take().map(spawn_child_stderr_reader);
        let stdout = child.stdout.take();
        let stdin = child.stdin.take();
        if let Some(mut stdin) = stdin {
            // A child that crashes before reading is expected here; its exit
            // status and stderr, not this write error, explain the failure.
            let _ = stdin.write_all(dxil);
        }
        let frame = match stdout {
            Some(stdout) => read_child_frame(BufReader::new(stdout)),
            None => Err(io::Error::other("conversion child stdout was not piped")),
        };
        let waited = child.wait();
        let stderr_tail = stderr.and_then(|rx| rx.recv_timeout(Duration::from_millis(500)).ok());
        match (frame, waited) {
            (Ok(frame), Ok(_)) if frame.status == 0 => {
                if frame.payload.is_empty() {
                    return Err((
                        1,
                        "conversion child returned an empty metallib".to_string(),
                    ));
                }
                if stage != 0 && frame.stage != stage {
                    return Err((
                        3,
                        format!(
                            "frame declared stage {stage}, converter reported stage {}",
                            frame.stage
                        ),
                    ));
                }
                Ok(StdioJob {
                    metallib: frame.payload,
                    reflection: frame.reflection,
                    stage: frame.stage,
                })
            }
            // The child completed a *conversion* failure: its reason is the
            // payload, plus the child's stderr tail when it said more there.
            (Ok(frame), Ok(_)) => Err((
                frame.status,
                merge_reason(&frame.payload, stderr_tail.as_deref()),
            )),
            (Ok(_), Err(err)) => Err((
                1,
                format!("could not wait for the conversion child: {err}"),
            )),
            (Err(err), waited) => Err((
                1,
                child_death_reason(waited.ok().as_ref(), stderr_tail.as_deref(), &err),
            )),
        }
    }

    /// Combines a child's diagnostic payload with its stderr tail, bounded.
    fn merge_reason(payload: &[u8], stderr: Option<&str>) -> String {
        let mut reason = sanitize_reason(&String::from_utf8_lossy(payload), MAX_JOB_REASON_BYTES);
        if reason.is_empty() {
            reason = "conversion child reported a failure without a reason".to_string();
        }
        if let Some(stderr) = stderr {
            let tail = sanitize_reason(stderr, MAX_CHILD_STDERR_TAIL_BYTES);
            if !tail.is_empty() && !reason.contains(&tail) {
                reason.push_str("; converter stderr: ");
                reason.push_str(&tail);
            }
        }
        sanitize_reason(&reason, MAX_JOB_REASON_BYTES)
    }

    /// What the supervisor answers when the conversion child is gone.
    fn child_death_reason(
        status: Option<&std::process::ExitStatus>,
        stderr: Option<&str>,
        read_error: &io::Error,
    ) -> String {
        let how = match status {
            Some(status) => describe_exit_status(status),
            None => format!("could not wait for the conversion child: {read_error}"),
        };
        let mut reason = format!("converter child died: {how}");
        if let Some(stderr) = stderr {
            let tail = sanitize_reason(stderr, MAX_CHILD_STDERR_TAIL_BYTES);
            if !tail.is_empty() {
                reason.push_str("; converter stderr: ");
                reason.push_str(&tail);
            }
        }
        sanitize_reason(&reason, MAX_JOB_REASON_BYTES)
    }

    /// `signal 11 (SIGSEGV)` / `exit code 9`.
    fn describe_exit_status(status: &std::process::ExitStatus) -> String {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            if let Some(signal) = status.signal() {
                return format!("signal {signal}{}", signal_suffix(signal));
            }
        }
        match status.code() {
            Some(code) => format!("exit code {code}"),
            None => "unknown exit status".to_string(),
        }
    }

    fn signal_suffix(signal: i32) -> &'static str {
        match signal {
            4 => " (SIGILL)",
            5 => " (SIGTRAP)",
            6 => " (SIGABRT)",
            8 => " (SIGFPE)",
            9 => " (SIGKILL)",
            10 => " (SIGBUS)",
            11 => " (SIGSEGV)",
            12 => " (SIGSYS)",
            13 => " (SIGPIPE)",
            15 => " (SIGTERM)",
            _ => "",
        }
    }

    /// One parsed private conversion frame.
    struct ChildFrame {
        status: u32,
        stage: u32,
        payload: Vec<u8>,
        reflection: Vec<u8>,
    }

    /// Reads and validates a private child frame with fixed bounds.
    ///
    /// A partial frame (the child died mid-write) or any out-of-bounds length
    /// is an IO error; the supervisor then reports the child's exit status and
    /// stderr instead of trying to resynchronise on the child's pipe.
    fn read_child_frame(mut reader: impl BufRead) -> io::Result<ChildFrame> {
        let mut line = Vec::new();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "conversion child wrote no response",
            ));
        }
        if line.len() > MAX_STDIO_HEADER_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "conversion child response header is too long",
            ));
        }
        let text = std::str::from_utf8(&line)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 child header"))?;
        let fields: Vec<&str> = text.split_ascii_whitespace().collect();
        if fields.len() != 5 || fields[0] != JOB_CHILD_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "conversion child answered a malformed header",
            ));
        }
        let status = fields[1]
            .parse::<u32>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid child status"))?;
        let stage = fields[2]
            .parse::<u32>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid child stage"))?;
        let payload_len = fields[3]
            .parse::<usize>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid child payload length"))?;
        let reflection_len = fields[4].parse::<usize>().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid child reflection length")
        })?;
        if payload_len > MAX_METALLIB_OUTPUT_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "conversion child payload exceeds the metallib limit",
            ));
        }
        if reflection_len > MAX_REFLECTION_OUTPUT_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "conversion child reflection exceeds the reflection limit",
            ));
        }
        if status != 0 && reflection_len != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "failed conversion child frame carries a reflection",
            ));
        }
        let mut payload = vec![0u8; payload_len];
        reader.read_exact(&mut payload)?;
        let mut reflection = vec![0u8; reflection_len];
        reader.read_exact(&mut reflection)?;
        Ok(ChildFrame {
            status,
            stage,
            payload,
            reflection,
        })
    }

    /// Drains a conversion child's stderr into a bounded tail, off the
    /// supervisor thread so a chatty converter cannot deadlock on a full pipe.
    fn spawn_child_stderr_reader(mut pipe: impl Read + Send + 'static) -> Receiver<String> {
        let (tx, rx) = mpsc::channel();
        let spawned = thread::Builder::new()
            .name("victory-irconverter-child-stderr".to_string())
            .spawn(move || {
                let mut ring: Vec<u8> = Vec::new();
                let mut chunk = [0u8; 1024];
                loop {
                    match pipe.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(read) => {
                            ring.extend_from_slice(&chunk[..read]);
                            if ring.len() > MAX_CHILD_STDERR_CAPTURE_BYTES {
                                let overflow = ring.len() - MAX_CHILD_STDERR_CAPTURE_BYTES;
                                ring.drain(..overflow);
                            }
                        }
                    }
                }
                let text = String::from_utf8_lossy(&ring).into_owned();
                let _ = tx.send(text);
            });
        if let Err(err) = spawned {
            tracing::warn!(error = %err, "victory-irconverter could not read child stderr");
        }
        rx
    }

    /// Flattens a diagnostic into one bounded, path-scrubbed line.
    ///
    /// Worker stderr and converter errors can name absolute cache/framework
    /// paths; the `VMRES2` reason is forwarded to the guest and logged, so it
    /// keeps the pool's `<path>` convention. Newlines/tabs collapse to spaces,
    /// other control characters become `?`, and the *tail* is kept when the
    /// text is over `max`, because the last line is the crash reason.
    fn sanitize_reason(text: &str, max: usize) -> String {
        let mut cleaned = String::with_capacity(text.len());
        let mut last_space = false;
        for ch in text.chars() {
            let mapped = match ch {
                '\n' | '\r' | '\t' => ' ',
                ch if ch.is_control() => '?',
                ch => ch,
            };
            if mapped == ' ' {
                if !last_space && !cleaned.is_empty() {
                    cleaned.push(' ');
                }
                last_space = true;
            } else {
                cleaned.push(mapped);
                last_space = false;
            }
        }
        let redacted: Vec<String> = cleaned
            .trim()
            .split(' ')
            .map(|token| {
                // Match the pool's convention: redact an absolute path either
                // as a bare token or after a `key=` prefix, so a converter
                // error naming its framework path never leaks a host path.
                let (prefix, value) = match token.split_once('=') {
                    Some((prefix, value)) if !prefix.is_empty() => (Some(prefix), value),
                    _ => (None, token),
                };
                let core = value.trim_matches(|ch: char| {
                    matches!(
                        ch,
                        '\'' | '"' | '`' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '<'
                            | '>' | '.'
                    )
                });
                if core.starts_with('/') || core.starts_with("~/") {
                    match prefix {
                        Some(prefix) => format!("{prefix}=<path>"),
                        None => "<path>".to_string(),
                    }
                } else {
                    token.to_string()
                }
            })
            .collect();
        let joined = redacted.join(" ");
        if joined.len() <= max {
            return joined;
        }
        let mut start = joined.len() - max;
        while start < joined.len() && !joined.is_char_boundary(start) {
            start += 1;
        }
        format!("{} [truncated]", &joined[start..])
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
