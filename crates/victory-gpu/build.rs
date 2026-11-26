use std::{env, path::PathBuf, process::Command};

// Only a build with VICTORY_HELPER_DXC set runs the compiler. CI never sets it,
// so the DXC path is not built or tested there: it is checked by hand with the
// ignored test `hardware_uses_precompiled_clamp` (see docs/macos.md).
fn main() {
    println!("cargo:rerun-if-env-changed=VICTORY_HELPER_DXC");
    println!("cargo:rerun-if-env-changed=DXC_EXE");
    let helpers = [
        ("indirect-clamp", "victory_dxc_clamp"),
        ("ambient-floor", "victory_dxc_ambient"),
    ];
    for (name, cfg) in helpers {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
        println!("cargo:rerun-if-changed=src/{name}.hlsl");
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let Some(compiler) = env::var_os("VICTORY_HELPER_DXC") else {
        return;
    };
    let compiler = PathBuf::from(compiler);
    if compiler.is_file() {
        println!("cargo:rerun-if-changed={}", compiler.display());
    }
    if let Some(exe) = env::var_os("DXC_EXE") {
        println!("cargo:rerun-if-changed={}", PathBuf::from(exe).display());
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    for (name, cfg) in helpers {
        let status = Command::new(&compiler)
            .args(["-T", "cs_6_0", "-E", "main", "-Fo"])
            .arg(out.join(format!("{name}.dxil")))
            .arg(format!("src/{name}.hlsl"))
            .status()
            .expect("start the configured DXC compiler");
        assert!(status.success(), "DXC failed for {name}");
        println!("cargo:rustc-cfg={cfg}");
    }
}
