//! Portable provisioning inputs built on a matching Ubuntu/architecture machine.
//! Credentials and arbitrary Cargo configuration are deliberately excluded.
use crate::{common::*, launcher};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};
const LIMIT: u64 = 12 * 1024 * 1024 * 1024;
fn path(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && !name.contains(['\\', ':'])
            && !name.chars().any(char::is_control)
            && !name.starts_with('/')
            && name
                .split('/')
                .all(|p| !p.is_empty() && p != "." && p != ".."),
        "Unsafe offline bundle path"
    );
    ensure!(
        name == "rustup"
            || name == "Cargo.lock"
            || name == "Packages"
            || (name.starts_with("packages/")
                && name.ends_with(".deb")
                && name.split('/').count() == 2)
            || name.starts_with("toolchain/")
            || name.starts_with("cargo/registry/cache/")
            || name.starts_with("cargo/registry/index/"),
        "Unexpected offline bundle resource: {name}"
    );
    tar::Header::new_ustar().set_path(name)?;
    Ok(())
}
pub(crate) fn host() -> Result<Value> {
    let text = fs::read_to_string("/etc/os-release")?;
    let mut v = json!({"architecture":std::env::consts::ARCH});
    for l in text.lines() {
        if let Some((k, s)) = l.split_once('=') {
            if k == "ID" {
                v["os"] = json!(s.trim_matches('"'));
            }
            if k == "VERSION_ID" {
                v["ubuntu"] = json!(s.trim_matches('"'));
            }
        }
    }
    ensure!(
        v["os"] == "ubuntu" && ["20.04", "22.04", "24.04"].contains(&string(&v, "ubuntu")?),
        "Offline provisioning requires supported Ubuntu"
    );
    Ok(v)
}
pub fn manifest_valid(m: &Value) -> Result<()> {
    ensure!(
        m["schema_version"] == 1 && m["kind"] == "vortex-offline",
        "Invalid offline manifest"
    );
    launcher::features(string(m, "profile")?)?;
    checksum(string(m, "cargo_lock_sha256")?)?;
    let pin = string(m, "toolchain")?;
    ensure!(
        pin.split('.').count() == 3
            && pin
                .split('.')
                .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())),
        "Invalid bundled toolchain"
    );
    ensure!(
        ["aarch64", "x86_64"].contains(&string(&m["host"], "architecture")?)
            && m["host"]["os"] == "ubuntu"
            && ["20.04", "22.04", "24.04"].contains(&string(&m["host"], "ubuntu")?),
        "Invalid bundle host"
    );
    let files = m["files"]
        .as_object()
        .context("Missing offline file records")?;
    ensure!(
        !files.is_empty() && files.len() <= 100000,
        "Invalid offline file count"
    );
    let mut total = 0u64;
    for (n, record) in files {
        path(n)?;
        checksum(string(record, "sha256")?)?;
        ensure!(
            [Some(0o644), Some(0o755)].contains(&record["mode"].as_u64()),
            "Invalid offline file mode"
        );
        total = total
            .checked_add(
                record["size"]
                    .as_u64()
                    .context("Invalid offline file size")?,
            )
            .context("Offline size overflow")?;
    }
    ensure!(total <= LIMIT, "Offline bundle exceeds size limit");
    ensure!(
        files.contains_key("rustup")
            && files.contains_key("Cargo.lock")
            && files.contains_key("Packages")
            && files.contains_key("toolchain/bin/rustc")
            && files.contains_key("toolchain/bin/cargo")
            && files.keys().any(|n| n.starts_with("packages/")),
        "Offline bundle is incomplete"
    );
    Ok(())
}
fn tree(source: &Path, dest: &Path, allow_internal_links: bool) -> Result<()> {
    let base = fs::canonicalize(source)?;
    for e in walkdir::WalkDir::new(&base).min_depth(1) {
        let e = e?;
        let relative = e.path().strip_prefix(&base)?;
        let target = dest.join(relative);
        if e.file_type().is_dir() {
            fs::create_dir_all(&target)?;
        } else {
            let actual = if e.file_type().is_symlink() {
                ensure!(allow_internal_links, "Symlink in offline cache");
                let actual = fs::canonicalize(e.path())?;
                ensure!(
                    actual.starts_with(&base) && actual.is_file(),
                    "Offline symlink escapes toolchain"
                );
                actual
            } else {
                e.path().to_path_buf()
            };
            #[cfg(unix)]
            let bits = {
                use std::os::unix::fs::PermissionsExt;
                if fs::metadata(&actual)?.permissions().mode() & 0o111 != 0 {
                    0o755
                } else {
                    0o644
                }
            };
            #[cfg(not(unix))]
            let bits = 0o644;
            copy(&actual, &target, bits)?;
        }
    }
    Ok(())
}
pub fn create(source: &Path, output: &Path, packages: &Path, profile: &str) -> Result<Value> {
    linux()?;
    launcher::features(profile)?;
    let source = fs::canonicalize(source)?;
    let output = root(output)?;
    ensure!(
        fs::canonicalize(&output)? != source,
        "Separate offline output required"
    );
    let lock = fs::read_to_string(source.join("Cargo.lock"))?;
    ensure!(
        !lock.contains("git+")
            && lock
                .lines()
                .filter_map(|l| l.trim().strip_prefix("source = "))
                .all(|s| s == "\"registry+https://github.com/rust-lang/crates.io-index\""),
        "Offline export currently supports crates.io dependencies only"
    );
    let pin = launcher::pin(&source)?;
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or(home()?.join(".cargo"));
    let rustup = home()?.join(".cargo/bin/rustup");
    checked(
        Command::new(&rustup)
            .args(["run", &pin, "cargo", "fetch", "--locked", "--manifest-path"])
            .arg(source.join("Cargo.toml")),
        7200,
    )?;
    let sysroot = checked(
        Command::new(&rustup).args(["run", &pin, "rustc", "--print", "sysroot"]),
        30,
    )?;
    let sysroot = PathBuf::from(String::from_utf8(sysroot.stdout)?.trim());
    let snapshot = tempfile::tempdir_in(&output)?;
    let s = snapshot.path();
    copy(&rustup, &s.join("rustup"), 0o755)?;
    copy(&source.join("Cargo.lock"), &s.join("Cargo.lock"), 0o644)?;
    tree(&sysroot, &s.join("toolchain"), true)?;
    for category in ["cache", "index"] {
        let dir = cargo_home.join("registry").join(category);
        ensure!(dir.is_dir(), "Missing Cargo registry {category}");
        for entry in fs::read_dir(dir)? {
            let e = entry?;
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with("index.crates.io-") || n == "github.com-1ecc6299db9ec823" {
                tree(
                    &e.path(),
                    &s.join("cargo/registry").join(category).join(n),
                    false,
                )?;
            }
        }
    }
    for entry in fs::read_dir(packages)? {
        let e = entry?;
        let n = e.file_name().to_string_lossy().to_string();
        if n.ends_with(".deb") {
            copy(&e.path(), &s.join("packages").join(n), 0o644)?;
        }
    }
    let mut index = String::new();
    let mut debs = fs::read_dir(s.join("packages"))?.collect::<std::result::Result<Vec<_>, _>>()?;
    debs.sort_by_key(|e| e.file_name());
    for e in debs {
        let control = checked(Command::new("dpkg-deb").arg("--field").arg(e.path()), 30)?;
        index.push_str(String::from_utf8(control.stdout)?.trim_end());
        index.push_str(&format!(
            "\nFilename: packages/{}\nSize: {}\nSHA256: {}\n\n",
            e.file_name().to_string_lossy(),
            e.metadata()?.len(),
            hash(e.path())?
        ));
    }
    fs::write(s.join("Packages"), index)?;
    let mut files = serde_json::Map::new();
    for e in walkdir::WalkDir::new(s) {
        let e = e?;
        if e.file_type().is_file() {
            let n = e
                .path()
                .strip_prefix(s)?
                .to_string_lossy()
                .replace('\\', "/");
            path(&n)?;
            #[cfg(unix)]
            let mode = {
                use std::os::unix::fs::PermissionsExt;
                fs::metadata(e.path())?.permissions().mode() & 0o777
            };
            #[cfg(not(unix))]
            let mode = 0o644;
            files.insert(
                n,
                json!({"size":e.metadata()?.len(),"sha256":hash(e.path())?,"mode":mode}),
            );
        }
    }
    let manifest = json!({"schema_version":1,"kind":"vortex-offline","host":host()?,"profile":profile,"toolchain":pin,"cargo_lock_sha256":hash(source.join("Cargo.lock"))?,"files":files});
    manifest_valid(&manifest)?;
    let archive = output.join("offline.tar.gz");
    ensure!(!archive.exists(), "Offline bundle already exists");
    let mut temp = tempfile::NamedTempFile::new_in(&output)?;
    {
        let mut tar = tar::Builder::new(flate2::write::GzEncoder::new(
            temp.as_file_mut(),
            flate2::Compression::fast(),
        ));
        let bytes = serde_json::to_vec(&manifest)?;
        let mut h = tar::Header::new_ustar();
        h.set_size(bytes.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        tar.append_data(&mut h, "offline-manifest.json", bytes.as_slice())?;
        for (n, r) in manifest["files"].as_object().unwrap() {
            let mut h = tar::Header::new_ustar();
            h.set_size(r["size"].as_u64().unwrap());
            h.set_mode(r["mode"].as_u64().unwrap() as u32);
            h.set_cksum();
            tar.append_data(&mut h, n, File::open(s.join(n))?)?;
        }
        tar.into_inner()?.finish()?;
    }
    temp.as_file().sync_all()?;
    temp.persist(&archive).map_err(|e| e.error)?;
    let metadata = json!({"schema_version":1,"kind":"vortex-offline","host":manifest["host"],"profile":profile,"toolchain":pin,"cargo_lock_sha256":manifest["cargo_lock_sha256"],"archive_sha256":hash(&archive)?,"archive_bytes":fs::metadata(&archive)?.len()});
    atomic(output.join("offline.json"), &metadata)?;
    Ok(metadata)
}
pub fn extract(
    archive: &Path,
    sha: &str,
    dest: &Path,
    source: &Path,
    profile: &str,
) -> Result<Value> {
    checksum(sha)?;
    regular(archive)?;
    ensure!(
        fs::metadata(archive)?.len() <= LIMIT && hash(archive)? == sha,
        "Offline archive checksum/size mismatch"
    );
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(File::open(archive)?));
    let mut entries = tar.entries()?.raw(true);
    let mut first = entries.next().context("Missing offline manifest")??;
    ensure!(
        first.path_bytes().as_ref() == b"offline-manifest.json"
            && first.header().entry_type() == tar::EntryType::Regular
            && first.size() <= 32 * 1024 * 1024,
        "Invalid offline manifest header"
    );
    let mut bytes = Vec::new();
    first.read_to_end(&mut bytes)?;
    let m: Value = serde_json::from_slice(&bytes)?;
    manifest_valid(&m)?;
    ensure!(
        m["host"] == host()?
            && m["toolchain"] == launcher::pin(source)?
            && m["profile"] == profile
            && m["cargo_lock_sha256"] == hash(source.join("Cargo.lock"))?,
        "Offline bundle does not match Ubuntu, architecture, profile, toolchain or Cargo.lock"
    );
    let needed: u64 = m["files"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v["size"].as_u64().unwrap())
        .sum();
    ensure!(
        fs2::available_space(dest.parent().context("Offline parent")?)?
            >= needed + 6 * 1024 * 1024 * 1024,
        "Insufficient space for offline provisioning"
    );
    fs::create_dir(dest)?;
    let mut seen = std::collections::HashSet::new();
    for entry in entries {
        let mut e = entry?;
        let n = std::str::from_utf8(&e.path_bytes())?.to_owned();
        path(&n)?;
        ensure!(seen.insert(n.clone()), "Duplicate offline resource");
        let r = m["files"].get(&n).context("Unexpected offline resource")?;
        ensure!(
            e.header().entry_type() == tar::EntryType::Regular
                && e.size() == r["size"].as_u64().unwrap()
                && e.header().mode()? as u64 == r["mode"].as_u64().unwrap(),
            "Offline resource metadata mismatch"
        );
        let p = child(dest, &n)?;
        fs::create_dir_all(p.parent().unwrap())?;
        let mut f = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&p)?;
        std::io::copy(&mut e, &mut f)?;
        ensure!(
            hash(&p)? == string(r, "sha256")?,
            "Offline resource checksum mismatch"
        );
        mode(&p, r["mode"].as_u64().unwrap() as u32)?;
    }
    ensure!(
        seen.len() == m["files"].as_object().unwrap().len(),
        "Incomplete offline archive"
    );
    ensure!(
        hash(dest.join("Cargo.lock"))? == string(&m, "cargo_lock_sha256")?,
        "Offline Cargo.lock mismatch"
    );
    atomic(dest.join("offline-manifest.json"), &m)?;
    Ok(m)
}
pub fn package_commands(bundle: &Path, simulate: bool) -> Result<(Command, Command)> {
    simple(bundle.to_str().context("Offline repository path")?)?;
    let state = bundle.join("apt-state");
    fs::create_dir_all(state.join("lists/partial"))?;
    fs::create_dir_all(state.join("archives/partial"))?;
    let sources = state.join("sources.list");
    fs::write(
        &sources,
        format!("deb [trusted=yes] file:{} ./\n", bundle.display()),
    )?;
    for e in fs::read_dir(bundle.join("packages"))? {
        let e = e?;
        regular(&e.path())?;
        let dest = state.join("archives").join(e.file_name());
        if !dest.exists() {
            fs::hard_link(e.path(), &dest)?;
        }
    }
    let make = || {
        let mut c = Command::new("unshare");
        c.args([
            "--net",
            "--",
            "apt-get",
            "-o",
            "DPkg::Lock::Timeout=120",
            "-o",
            "Acquire::Languages=none",
            "-o",
            "Dir::Etc::sourceparts=-",
            "-o",
            "Dir::Etc::parts=-",
        ])
        .arg("-o")
        .arg(format!("Dir::Etc::sourcelist={}", sources.display()))
        .arg("-o")
        .arg(format!(
            "Dir::State::lists={}/",
            state.join("lists").display()
        ))
        .arg("-o")
        .arg(format!(
            "Dir::Cache::archives={}/",
            state.join("archives").display()
        ))
        .env("DEBIAN_FRONTEND", "noninteractive");
        c
    };
    let mut update = make();
    update.arg("update");
    let mut install = make();
    if simulate {
        install.arg("--simulate");
    }
    install
        .args([
            "--no-download",
            "--no-remove",
            "--no-install-recommends",
            "install",
            "-y",
        ])
        .args(crate::initialize::PACKAGES);
    Ok((update, install))
}
pub fn install_packages(bundle: &Path, log: &File) -> Result<()> {
    let (mut update, mut install) = package_commands(bundle, false)?;
    logged(&mut update, log, 120)?;
    logged(&mut install, log, 7200)
}
pub fn install_user(bundle: &Path, user: &str, home: &Path, pin: &str) -> Result<()> {
    let sysroot = home.join(".rustup/toolchains").join(format!(
        "{pin}-{}-unknown-linux-gnu",
        std::env::consts::ARCH
    ));
    if !sysroot.exists() {
        tree(&bundle.join("toolchain"), &sysroot, false)?;
    }
    for n in ["rustup", "cargo", "rustc"] {
        let dest = home.join(".cargo/bin").join(n);
        if !dest.exists() {
            copy(&bundle.join("rustup"), &dest, 0o755)?;
        }
    }
    tree(
        &bundle.join("cargo/registry"),
        &home.join(".cargo/registry"),
        false,
    )?;
    // Change only new/merged controller inputs, never arbitrary paths from a manifest.
    let a = crate::service::account(user)?;
    for p in [home.join(".cargo"), home.join(".rustup")] {
        for e in walkdir::WalkDir::new(&p) {
            let e = e?;
            if !e.file_type().is_symlink() {
                crate::service::chown(e.path(), &a)?;
            }
        }
    }
    let o = checked(
        Command::new("runuser")
            .args(["-u", user, "--", "env"])
            .arg(format!("HOME={}", home.display()))
            .arg(format!("RUSTUP_HOME={}/.rustup", home.display()))
            .arg(home.join(".cargo/bin/rustup"))
            .args(["run", pin, "rustc", "--version"]),
        30,
    )?;
    ensure!(
        String::from_utf8_lossy(&o.stdout).starts_with(&format!("rustc {pin} ")),
        "Installed toolchain version mismatch"
    );
    Ok(())
}
pub fn download(output: &Path) -> Result<Value> {
    admin()?;
    host()?;
    let output = root(output)?;
    let packages = output.join("packages");
    fs::create_dir_all(packages.join("partial"))?;
    let status = output.join("empty-status");
    fs::write(&status, "")?;
    checked(
        Command::new("apt-get").args(["-o", "DPkg::Lock::Timeout=120", "update"]),
        7200,
    )?;
    checked(
        Command::new("apt-get")
            .arg("-o")
            .arg(format!("Dir::State::status={}", status.display()))
            .arg("-o")
            .arg(format!("Dir::Cache::archives={}/", packages.display()))
            .args([
                "--download-only",
                "--no-install-recommends",
                "install",
                "-y",
            ])
            .args(crate::initialize::PACKAGES)
            .env("DEBIAN_FRONTEND", "noninteractive"),
        7200,
    )?;
    Ok(
        json!({"packages":packages,"host":host()?,"message":"Complete package dependency closure downloaded; prepare Rust and run offline-create as the runtime user"}),
    )
}
