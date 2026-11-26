//! Path-in/path-out DXIL to serialized metallib converter.
//!
//! Build and run this binary as `x86_64-apple-darwin` (Rosetta 2 on Apple
//! Silicon), because Apple's `libmetalirconverter.dylib` is x86_64-only. The
//! converter does not create an `MTLDevice`, an `MTLLibrary`, or any GPU work;
//! it only serializes the metallib bytes to the output path.
//!
//! Usage:
//! ```text
//! victory-irconverter --converter <libmetalirconverter.dylib> \
//!     --entry <entry-point> <input.dxil> <output.metallib>
//! ```
//!
//! `--converter` and `--entry` are required. The root signature must already be
//! embedded in the DXIL; no global/explicit root signature is passed. DXIL input
//! is copied by IRConverter (`IRObjectCreateFromDXIL` ownership `1`) before the
//! Rust buffer is released.
//!
//! Input limits are fixed by `victory-metal`: at most
//! `MAX_DXIL_INPUT_BYTES` (16 MiB) of DXIL, at most `MAX_ENTRY_POINT_BYTES`
//! (256) bytes in the entry name, and at most `MAX_METALLIB_OUTPUT_BYTES`
//! (64 MiB) of output. There is no stdin/stdout mode and no environment-based
//! converter fallback: the caller names exactly one converter file.
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
    use std::io::Read;
    use std::path::{Path, PathBuf};

    use victory_metal::macos::irconverter::{
        MAX_DXIL_INPUT_BYTES, MAX_ENTRY_POINT_BYTES, MAX_METALLIB_OUTPUT_BYTES,
        convert_dxil_to_metallib,
    };

    const USAGE: &str = "\
usage: victory-irconverter --converter <libmetalirconverter.dylib> \\
           --entry <entry-point> <input.dxil> <output.metallib>

  --converter, -c <path>  explicitly configured x86_64 libmetalirconverter.dylib
  --entry, -e <name>      DXIL entry point (embedded root signature is used)
  --help, -h              show this help

The converter is deterministic and never opens a Metal device or runs GPU work.
Limits: DXIL <= 16777216 bytes, entry <= 256 bytes, metallib <= 67108864 bytes.
";

    struct Args {
        converter: PathBuf,
        entry: String,
        input: PathBuf,
        output: PathBuf,
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
        let mut positional = Vec::new();
        let mut only_positional = false;

        while let Some(arg) = raw.next() {
            if !only_positional && arg == "--" {
                only_positional = true;
                continue;
            }
            let text = arg.to_str();
            match text {
                Some("--help" | "-h") => return Ok(None),
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
                    let value = value
                        .to_str()
                        .ok_or_else(|| "--entry must be valid UTF-8".to_string())?;
                    entry = Some(value.to_string());
                }
                Some(value) if !only_positional && value.starts_with('-') && value != "-" => {
                    return Err(format!("unknown option `{value}`"));
                }
                _ => positional.push(arg),
            }
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
        }))
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
        println!(
            "IRCONVERT_OK entry={:?} stage={} dxil_bytes={} metallib_bytes={}",
            args.entry,
            output.stage,
            dxil.len(),
            output.bytes.len()
        );
        Ok(())
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
