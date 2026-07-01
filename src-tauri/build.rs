use std::path::Path;

fn main() {
    // Bundled Tinfoil key (friends-and-family, zero-setup). Resolved at
    // COMPILE TIME, in priority order, and handed to the crate via rustc-env
    // so lib.rs reads it with env!(). NOT a tracked file — the key never
    // enters the public source tree. Empty => keyless build.
    //   1. BUNDLED_TINFOIL_KEY env  (release CI, from the GH Actions secret)
    //   2. ../.env  (repo-root dotenv — same file evals use), key TINFOIL_API_KEY
    //   3. empty
    //
    // NOTE: this keeps the key out of the SOURCE, not the shipped binary —
    // it stays strings-extractable from the .app. It is a shared low-trust
    // free-tier credential, not a secret. See lib.rs::BUNDLED_TINFOIL_KEY.
    let key = std::env::var("BUNDLED_TINFOIL_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())
        .or_else(|| read_dotenv_key(Path::new("../.env"), "TINFOIL_API_KEY"))
        .unwrap_or_default();
    println!("cargo:rustc-env=BUNDLED_TINFOIL_KEY={}", key.trim());
    println!("cargo:rerun-if-env-changed=BUNDLED_TINFOIL_KEY");
    println!("cargo:rerun-if-changed=../.env");

    tauri_build::build()
}

/// Minimal `KEY=value` reader for a dotenv file. Returns the first match,
/// trimmed, with surrounding quotes stripped. None if file/key absent.
fn read_dotenv_key(path: &Path, name: &str) -> Option<String> {
    let body = std::fs::read_to_string(path).ok()?;
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == name {
                let v = v.trim().trim_matches('"').trim_matches('\'').trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}
