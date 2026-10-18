use crate::{common::*, launcher};
use anyhow::{ensure, Context, Result};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File, OpenOptions},
    io::Read,
    path::Path,
    process::Command,
};
pub const MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_MANIFEST: u64 = 2 * 1024 * 1024;
pub const ROOT_FILES: [&str; 5] = [
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    "rust-toolchain.toml",
    "run_vortex.sh",
];
const FIXTURE: &str = "simulation_tests/reference/dim_noisy_tag.png";
const CONTROLLER: &str = "tools/vortexctl";
pub fn payload(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && !name.starts_with('/')
            && !name.contains(['\\', ':'])
            && !name.chars().any(char::is_control),
        "Invalid archive path"
    );
    ensure!(
        name.split('/').all(|s| !s.is_empty()
            && ![
                ".",
                "..",
                ".git",
                "target",
                "node_modules",
                ".venv",
                "generated",
                "__pycache__"
            ]
            .contains(&s)),
        "Unsafe archive path"
    );
    let extension = Path::new(name)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    ensure!(
        ROOT_FILES.contains(&name)
            || [
                FIXTURE,
                CONTROLLER,
                "config/device.json",
                "config/config.json",
                "config/apriltag_map.json",
                "config/2026-rebuilt-welded.json",
                "models/rockpaperscizzors.engine",
                "scripts/run_vortex.py",
                "scripts/deploy_release.py",
                "scripts/activate_release.py",
                "scripts/initialize_jetson.py"
            ]
            .contains(&name)
            || (name.starts_with("src/") && ["rs", "c", "cpp", "h", "cu"].contains(&extension)),
        "Not a deployment resource: {name}"
    );
    tar::Header::new_ustar()
        .set_path(name)
        .context("Deployment paths must fit USTAR headers")?;
    Ok(())
}
// Python's original manifest used sorted JSON with spaces. Preserve that wire format
// so already staged releases remain available for rollback.
pub fn canonical(v: &Value) -> String {
    fn quoted(s: &str) -> String {
        let encoded = serde_json::to_string(s).unwrap();
        let mut ascii = String::new();
        for c in encoded.chars() {
            if c.is_ascii() {
                ascii.push(c);
            } else {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    ascii.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
        ascii
    }
    match v {
        Value::Object(m) => {
            let sorted: BTreeMap<_, _> = m.iter().collect();
            format!(
                "{{{}}}",
                sorted
                    .iter()
                    .map(|(k, v)| format!("{}: {}", quoted(k), canonical(v)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(canonical).collect::<Vec<_>>().join(", ")
        ),
        Value::String(s) => quoted(s),
        _ => v.to_string(),
    }
}
pub fn validate_manifest(m: &Value, identity: &str) -> Result<()> {
    id(identity)?;
    ensure!(
        m["schema_version"].as_u64() == Some(1) && m["release_id"] == identity,
        "Manifest identity mismatch"
    );
    let f = launcher::features(string(m, "build_profile")?)?;
    ensure!(m["features"] == json!(f), "Manifest features mismatch");
    let files = m["files"].as_object().context("Invalid manifest files")?;
    ensure!(
        !files.is_empty() && files.len() <= 10000,
        "Invalid file count"
    );
    for n in ROOT_FILES
        .into_iter()
        .chain([FIXTURE, "config/device.json", "config/config.json"])
    {
        ensure!(files.contains_key(n), "Missing required file: {n}");
    }
    ensure!(
        files.contains_key(CONTROLLER)
            || [
                "scripts/run_vortex.py",
                "scripts/deploy_release.py",
                "scripts/activate_release.py",
                "scripts/initialize_jetson.py"
            ]
            .iter()
            .all(|n| files.contains_key(*n)),
        "Missing release controller"
    );
    let mut total = 0u64;
    for (n, info) in files {
        payload(n)?;
        total = total
            .checked_add(info["size"].as_u64().context("Invalid file size")?)
            .context("Size overflow")?;
        checksum(string(info, "sha256")?)?;
        ensure!(
            [Some(0o644), Some(0o755)].contains(&info["mode"].as_u64()),
            "Invalid file mode"
        );
    }
    ensure!(total <= MAX_BYTES, "Release exceeds limits");
    ensure!(
        m["snapshot_sha256"] == hash_bytes(canonical(&m["files"]).as_bytes()),
        "Manifest snapshot checksum mismatch"
    );
    Ok(())
}
pub fn verify(root: &Path, m: &Value) -> Result<()> {
    for (n, info) in m["files"].as_object().context("Invalid manifest")? {
        let p = child(root, n)?;
        regular(&p)?;
        ensure!(
            fs::metadata(&p)?.len() == info["size"].as_u64().unwrap()
                && hash(&p)? == string(info, "sha256")?,
            "Release payload changed: {n}"
        );
    }
    Ok(())
}
pub fn elf(path: &Path, arch: Option<&str>) -> Result<String> {
    regular(path)?;
    let mut header = [0; 20];
    File::open(path)?.read_exact(&mut header)?;
    ensure!(
        &header[..4] == b"\x7fELF" && header[4] == 2 && [1, 2].contains(&header[5]),
        "Expected 64-bit Linux ELF controller/executable"
    );
    let machine = if header[5] == 1 {
        u16::from_le_bytes([header[18], header[19]])
    } else {
        u16::from_be_bytes([header[18], header[19]])
    };
    let name = match machine {
        183 => "aarch64",
        62 => "x86_64",
        _ => anyhow::bail!("Unsupported ELF architecture"),
    };
    if let Some(arch) = arch {
        ensure!(
            arch == name,
            "Executable architecture does not match device"
        );
    }
    Ok(name.into())
}
pub fn extract(archive: &Path, dest: &Path, identity: &str) -> Result<Value> {
    regular(archive)?;
    ensure!(
        fs::metadata(archive)?.len() <= MAX_BYTES,
        "Archive exceeds extraction limit"
    );
    let mut tar = tar::Archive::new(GzDecoder::new(File::open(archive)?));
    let mut entries = tar.entries()?.raw(true);
    let mut first = entries.next().context("Missing manifest")??;
    ensure!(
        first.path_bytes().as_ref() == b"manifest.json"
            && first.header().entry_type().is_file()
            && first.size() <= MAX_MANIFEST,
        "Missing or oversized manifest"
    );
    let mut bytes = Vec::new();
    first.read_to_end(&mut bytes)?;
    let manifest: Value = serde_json::from_slice(&bytes)?;
    validate_manifest(&manifest, identity)?;
    let needed: u64 = manifest["files"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v["size"].as_u64().unwrap())
        .sum();
    ensure!(
        fs2::available_space(dest.parent().context("Extraction parent")?)?
            >= needed + 512 * 1024 * 1024,
        "Insufficient disk space"
    );
    fs::create_dir(dest)?;
    let mut seen = HashSet::new();
    for entry in entries {
        let mut entry = entry?;
        let name = std::str::from_utf8(&entry.path_bytes())?.to_owned();
        payload(&name)?;
        ensure!(seen.insert(name.clone()), "Duplicate archive entry");
        let info = manifest["files"]
            .get(&name)
            .context("Unexpected archive entry")?;
        ensure!(
            entry.header().entry_type() == tar::EntryType::Regular
                && entry.size() == info["size"].as_u64().unwrap()
                && entry.header().mode()? as u64 == info["mode"].as_u64().unwrap(),
            "Archive metadata mismatch"
        );
        let path = child(dest, &name)?;
        fs::create_dir_all(path.parent().unwrap())?;
        let mut f = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        std::io::copy(&mut entry, &mut f)?;
        ensure!(
            hash(&path)? == string(info, "sha256")?,
            "Payload checksum mismatch: {name}"
        );
        mode(&path, info["mode"].as_u64().unwrap() as u32)?;
    }
    ensure!(
        seen.len() == manifest["files"].as_object().unwrap().len(),
        "Incomplete archive"
    );
    atomic(dest.join("manifest.json"), &manifest)?;
    Ok(manifest)
}
fn git(root: &Path) -> Value {
    let commit = output(
        Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "HEAD"]),
        10,
    );
    let status = output(
        Command::new("git").arg("-C").arg(root).args([
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ]),
        10,
    );
    match (commit, status) {
        (Ok(c), Ok(s)) if c.status.success() && s.status.success() => {
            json!({"commit":String::from_utf8_lossy(&c.stdout).trim(),"dirty":!s.stdout.is_empty()})
        }
        _ => json!({"commit":null,"dirty":null}),
    }
}
pub fn pack(
    source: &Path,
    output: &Path,
    device: &Path,
    runtime: Option<&Path>,
    map: Option<&Path>,
    controller: &Path,
) -> Result<Value> {
    let source = fs::canonicalize(source)?;
    let output = root(output)?;
    ensure!(
        source != fs::canonicalize(&output)?,
        "Separate source and output directories required"
    );
    elf(controller, None)?;
    let c = read(device)?;
    launcher::config(&c)?;
    let keys = c.as_object().context("Device object")?;
    ensure!(
        keys.len() == 8
            && [
                "schema_version",
                "runtime",
                "build_profile",
                "cameras",
                "runtime_config",
                "tag_map",
                "networktables",
                "udp"
            ]
            .iter()
            .all(|k| keys.contains_key(*k)),
        "Unknown device fields"
    );
    ensure!(
        c["runtime_config"] == "config/config.json"
            && [
                "config/apriltag_map.json",
                "config/2026-rebuilt-welded.json"
            ]
            .contains(&string(&c, "tag_map")?),
        "Nonportable device resources"
    );
    let mut provenance = git(&source);
    let identity = format!("release-{}-{}", now() as u64, &token()[..16]);
    let archive = output.join("release.tar.gz");
    ensure!(!archive.exists(), "Output archive already exists");
    let snapshot = tempfile::Builder::new()
        .prefix("snapshot-")
        .tempdir_in(&output)?;
    let s = snapshot.path();
    for n in ROOT_FILES.into_iter().chain([FIXTURE]) {
        let src = child(&source, n)?;
        copy(
            &src,
            &s.join(n),
            if n.ends_with(".sh") { 0o755 } else { 0o644 },
        )?;
    }
    for e in walkdir::WalkDir::new(source.join("src")) {
        let e = e?;
        ensure!(
            !e.file_type().is_symlink(),
            "Source symlinks are not packaged"
        );
        if e.file_type().is_file() {
            let n = e
                .path()
                .strip_prefix(&source)?
                .to_string_lossy()
                .replace('\\', "/");
            payload(&n)?;
            copy(e.path(), &s.join(n), 0o644)?;
        }
    }
    ensure!(s.join("src/main.rs").is_file(), "No production Rust source");
    copy(controller, &s.join(CONTROLLER), 0o755)?;
    atomic(s.join("config/device.json"), &c)?;
    copy(
        runtime.unwrap_or(&source.join("config/config.json")),
        &s.join("config/config.json"),
        0o644,
    )?;
    let map_name = string(&c, "tag_map")?;
    copy(
        map.unwrap_or(&source.join(map_name)),
        &s.join(map_name),
        0o644,
    )?;
    let r = launcher::resources(s, &c)?;
    if r["object_detection"]["use_nn"] != false
        && launcher::features(string(&c, "build_profile")?)?.contains(&"tensorrt")
    {
        copy(
            &child(&source, "models/rockpaperscizzors.engine")?,
            &s.join("models/rockpaperscizzors.engine"),
            0o644,
        )?;
    }
    if git(&source) != provenance {
        provenance["dirty"] = json!(true);
    }
    let mut files = BTreeMap::new();
    for e in walkdir::WalkDir::new(s) {
        let e = e?;
        if e.file_type().is_file() {
            let n = e
                .path()
                .strip_prefix(s)?
                .to_string_lossy()
                .replace('\\', "/");
            payload(&n)?;
            let bits = if n == CONTROLLER || n.ends_with(".sh") {
                0o755
            } else {
                0o644
            };
            files.insert(
                n,
                json!({"size":e.metadata()?.len(),"sha256":hash(e.path())?,"mode":bits}),
            );
        }
    }
    let mut m = json!({"schema_version":1,"release_id":identity,"created_at":now(),"git":provenance,"build_profile":c["build_profile"],"features":launcher::features(string(&c,"build_profile")?)?,"files":files});
    m["snapshot_sha256"] = json!(hash_bytes(canonical(&m["files"]).as_bytes()));
    validate_manifest(&m, &identity)?;
    let data = serde_json::to_vec(&m)?;
    ensure!(data.len() as u64 <= MAX_MANIFEST, "Manifest too large");
    let temp = output.join("release.tar.gz.part");
    let mut tar = tar::Builder::new(GzEncoder::new(File::create(&temp)?, Compression::default()));
    fn header(size: u64, bits: u32) -> tar::Header {
        let mut h = tar::Header::new_ustar();
        h.set_size(size);
        h.set_mode(bits);
        h.set_entry_type(tar::EntryType::Regular);
        h.set_cksum();
        h
    }
    tar.append_data(
        &mut header(data.len() as u64, 0o644),
        "manifest.json",
        data.as_slice(),
    )?;
    for (n, info) in m["files"].as_object().unwrap() {
        tar.append_data(
            &mut header(
                info["size"].as_u64().unwrap(),
                info["mode"].as_u64().unwrap() as u32,
            ),
            n,
            File::open(s.join(n))?,
        )?;
    }
    tar.into_inner()?.finish()?.sync_all()?;
    fs::rename(temp, &archive)?;
    copy(controller, &output.join("vortexctl"), 0o755)?;
    let metadata = json!({"schema_version":1,"release_id":identity,"archive_sha256":hash(&archive)?,"archive_bytes":fs::metadata(archive)?.len(),"controller_sha256":hash(output.join("vortexctl"))?,"controller_architecture":elf(controller,None)?,"build_profile":c["build_profile"],"git":provenance});
    atomic(output.join("package.json"), &metadata)?;
    Ok(metadata)
}
pub fn stage(root: &Path, archive: &Path, sha: &str, identity: &str, debug: bool) -> Result<Value> {
    id(identity)?;
    checksum(sha)?;
    let root = root.to_path_buf();
    for name in ["staging", "releases", "jobs", "build-cache"] {
        fs::create_dir_all(child(&root, name)?)?;
    }
    claim(&root, identity, "claim")?;
    let job_path = root.join("jobs").join(format!("{identity}.json"));
    let mut job = json!({"schema_version":1,"release_id":identity,"state":"queued","activated":false,"started_at":now()});
    atomic(&job_path, &job)?;
    let result = (|| -> Result<()> {
        let _lock = Lock::acquire(&lock_path()?)?;
        let log_path = root.join("jobs").join(format!("{identity}.log"));
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;
        job["log_path"] = json!(log_path);
        job["state"] = json!("verifying");
        atomic(&job_path, &job)?;
        regular(archive)?;
        ensure!(
            fs::metadata(archive)?.len() <= MAX_BYTES && hash(archive)? == sha,
            "Uploaded archive checksum/size mismatch"
        );
        let staging = child(&root, &format!("staging/{identity}"))?;
        let final_path = child(&root, &format!("releases/{identity}"))?;
        ensure!(
            !staging.exists() && !final_path.exists(),
            "Release already exists; create a new package"
        );
        job["state"] = json!("extracting");
        atomic(&job_path, &job)?;
        let manifest = extract(archive, &staging, identity)?;
        let c = read(staging.join("config/device.json"))?;
        ensure!(
            c["build_profile"] == manifest["build_profile"]
                && c["runtime_config"] == "config/config.json"
                && manifest["files"].get(string(&c, "tag_map")?).is_some(),
            "Manifest/device mismatch"
        );
        job["profile"] = c["build_profile"].clone();
        job["state"] = json!("validating");
        atomic(&job_path, &job)?;
        launcher::resources(&staging, &c)?;
        job["state"] = json!("building");
        atomic(&job_path, &job)?;
        launcher::build(
            &staging,
            &c,
            false,
            debug,
            Some(&root.join("build-cache").join(string(&c, "build_profile")?)),
            Some(&log),
        )?;
        job["state"] = json!("checking");
        atomic(&job_path, &job)?;
        launcher::checked_binary(&staging, &c, launcher::MAIN)?;
        if debug {
            launcher::checked_binary(&staging, &c, launcher::BRIDGE)?;
        }
        verify(&staging, &manifest)?;
        atomic(
            staging.join("release.json"),
            &json!({"schema_version":1,"release_id":identity,"profile":manifest["build_profile"],"archive_sha256":sha,"activated":false}),
        )?;
        fs::rename(&staging, &final_path)?;
        sync_dir(&root.join("releases"))?;
        job["state"] = json!("ready");
        job["release_path"] = json!(final_path);
        job["message"] = json!("Release built and verified; not activated");
        atomic(&job_path, &job)?;
        Ok(())
    })();
    if let Err(e) = result {
        job["state"] = json!("failed");
        job["error"] = json!(format!("{e:#}"));
        atomic(&job_path, &job)?;
        return Err(e);
    }
    Ok(job)
}
