use crate::{common::*, jobs, launcher, offline, package, service};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
pub const PACKAGES: &[&str] = &[
    "ca-certificates",
    "curl",
    "git",
    "build-essential",
    "pkg-config",
    "cmake",
    "nasm",
    "clang",
    "libclang-dev",
    "libssl-dev",
    "libudev-dev",
    "libturbojpeg0-dev",
    "libv4l-dev",
    "v4l-utils",
    "dbus-user-session",
];
pub struct Options<'a> {
    pub root: &'a Path,
    pub legacy: &'a Path,
    pub user: &'a str,
    pub service: &'a str,
    pub job: &'a str,
    pub release: &'a str,
    pub sha: &'a str,
    pub offline: Option<(&'a Path, &'a str)>,
}
impl Options<'_> {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.root.to_string_lossy() == format!("{}.deployment", self.legacy.display()),
            "Managed root must be the legacy directory's sibling"
        );
        for s in [
            self.root.to_str().unwrap_or(""),
            self.legacy.to_str().unwrap_or(""),
            self.user,
        ] {
            simple(s)?;
        }
        ensure!(
            self.user != "root"
                && self
                    .user
                    .as_bytes()
                    .first()
                    .is_some_and(|b| b.is_ascii_lowercase() || *b == b'_')
                && self
                    .user
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-".contains(&b)),
            "Select a non-root runtime account"
        );
        service(self.service)?;
        id(self.job)?;
        id(self.release)?;
        checksum(self.sha)?;
        if let Some((path, sha)) = self.offline {
            ensure!(path.is_absolute(), "Offline bundle path must be absolute");
            checksum(sha)?;
        }
        Ok(())
    }
    fn args(&self, action: &str) -> Vec<String> {
        let mut args: Vec<String> = [
            action,
            "--root",
            self.root.to_str().unwrap(),
            "--legacy-root",
            self.legacy.to_str().unwrap(),
            "--user",
            self.user,
            "--service",
            self.service,
            "--job-id",
            self.job,
            "--release-id",
            self.release,
            "--sha256",
            self.sha,
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        if let Some((path, sha)) = self.offline {
            args.extend([
                "--offline-bundle".into(),
                path.display().to_string(),
                "--offline-sha256".into(),
                sha.into(),
            ]);
        }
        args
    }
}
pub fn submit(o: &Options) -> Result<Value> {
    o.validate()?;
    admin()?;
    ensure!(
        std::env::var("SUDO_USER").unwrap_or(o.user.into()) == o.user,
        "Connect as the runtime account when initializing"
    );
    jobs::dispatch(
        o.root,
        o.job,
        "initialize",
        true,
        &executable()?,
        &o.args("initialize-run"),
    )
}
pub fn compatibility(
    profile: &str,
    arch: &str,
    etc: &Path,
    exists: &dyn Fn(&Path) -> bool,
) -> Result<Value> {
    let release = fs::read_to_string(etc.join("os-release"))?;
    let mut values = std::collections::HashMap::new();
    for l in release.lines() {
        if let Some((k, v)) = l.split_once('=') {
            values.insert(k, v.trim_matches('"'));
        }
    }
    ensure!(
        ["aarch64", "x86_64"].contains(&arch)
            && values.get("ID") == Some(&"ubuntu")
            && [Some(&"20.04"), Some(&"22.04"), Some(&"24.04")].contains(&values.get("VERSION_ID")),
        "Supported targets are Ubuntu 20.04/22.04/24.04 on aarch64/x86_64"
    );
    ensure!(
        exists(Path::new("/run/systemd/system")),
        "Device must boot with systemd"
    );
    let f = launcher::features(profile)?;
    if f.contains(&"gpu") {
        ensure!(arch=="aarch64" && [2,3].iter().any(|v|exists(Path::new(&format!("/opt/nvidia/vpi{v}/include/vpi/VPI.h"))) && ["lib64","lib/aarch64-linux-gnu"].iter().any(|d|exists(Path::new(&format!("/opt/nvidia/vpi{v}/{d}/libnvvpi.so"))))),"GPU needs matching JetPack VPI 2/3 headers and libraries; install JetPack or select CPU");
    }
    if f.contains(&"tensorrt") {
        ensure!(
            arch == "aarch64"
                && exists(Path::new("/usr/local/cuda/include/cuda_runtime_api.h"))
                && [
                    "/usr/include/aarch64-linux-gnu/NvInfer.h",
                    "/usr/include/NvInfer.h"
                ]
                .iter()
                .any(|p| exists(Path::new(p)))
                && [
                    "/usr/lib/aarch64-linux-gnu/libnvinfer.so",
                    "/opt/nvidia/tensorrt/lib/libnvinfer.so"
                ]
                .iter()
                .any(|p| exists(Path::new(p))),
            "TensorRT needs matching JetPack CUDA/TensorRT development packages"
        );
    }
    Ok(json!({"architecture":arch,"ubuntu":values["VERSION_ID"],"profile":profile}))
}
fn run_user(cmd: &[String], user: &str, home: &Path, log: &fs::File, seconds: u64) -> Result<()> {
    let mut c = Command::new("runuser");
    c.args(["-u", user, "--", "env"])
        .arg(format!("HOME={}", home.display()))
        .arg(format!("CARGO_HOME={}/.cargo", home.display()))
        .arg(format!("RUSTUP_HOME={}/.rustup", home.display()))
        .arg(format!(
            "PATH={}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            home.display()
        ))
        .args(cmd)
        .env("DEBIAN_FRONTEND", "noninteractive")
        .env("CARGO_BUILD_JOBS", "2")
        .env("RUSTUP_AUTO_INSTALL", "0");
    logged(&mut c, log, seconds)
}
pub fn initialize(o: &Options) -> Result<Value> {
    o.validate()?;
    admin()?;
    let a = service::account(o.user)?;
    ensure!(a.uid != 0, "Runtime account must not be root");
    let lock = a.home.join(".local/state/vortex/deployment.lock");
    fs::create_dir_all(lock.parent().unwrap())?;
    fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock)?;
    for p in [
        a.home.join(".local"),
        a.home.join(".local/state"),
        lock.parent().unwrap().into(),
        lock.clone(),
    ] {
        service::chown(&p, &a)?;
    }
    claim(o.root, o.job, "claim")?;
    let job_path = o.root.join("jobs").join(format!("{}.json", o.job));
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(o.root.join("jobs").join(format!("{}.log", o.job)))?;
    mode(&o.root.join("jobs").join(format!("{}.log", o.job)), 0o644)?;
    let mut job = json!({"schema_version":1,"operation_id":o.job,"action":"initialize","state":"queued","release_id":o.release,"activated":false});
    atomic(&job_path, &job)?;
    let mut update = |state: &str, extra: Value| -> Result<()> {
        job["state"] = json!(state);
        job.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        atomic(&job_path, &job)
    };
    let result = (|| -> Result<()> {
        let archive = child(o.root, &format!("incoming/{}/release.tar.gz", o.release))?;
        let work: PathBuf;
        {
            let _lock = Lock::acquire(&lock)?;
            // The privileged submitter creates jobs before the runtime account
            // stages a release. Shared directories must remain writable by that
            // account, including on a completely new installation.
            service::chown(o.root, &a)?;
            for name in ["jobs", "staging", "releases", "build-cache"] {
                let directory = child(o.root, name)?;
                fs::create_dir_all(&directory)?;
                service::chown(&directory, &a)?;
            }
            update("verifying", json!({}))?;
            regular(&archive)?;
            ensure!(
                hash(&archive)? == o.sha,
                "Initialization archive checksum mismatch"
            );
            let parent = child(o.root, "initialization")?;
            fs::create_dir_all(&parent)?;
            mode(&parent, 0o755)?;
            work = child(o.root, &format!("initialization/{}", o.job))?;
            let manifest = package::extract(&archive, &work, o.release)?;
            for e in walkdir::WalkDir::new(&work) {
                service::chown(e?.path(), &a)?;
            }
            update(
                "checking",
                json!({"compatibility":compatibility(string(&manifest,"build_profile")?,std::env::consts::ARCH,Path::new("/etc"),&Path::exists)?}),
            )?;
            ensure!(
                fs2::available_space(o.root)? >= 6 * 1024 * 1024 * 1024,
                "Initialization needs 6 GiB free space"
            );
            let bundle = if let Some((archive, sha)) = o.offline {
                let dest = work.join("offline-inputs");
                offline::extract(
                    archive,
                    sha,
                    &dest,
                    &work,
                    string(&manifest, "build_profile")?,
                )?;
                Some(dest)
            } else {
                None
            };
            update("packages", json!({"offline":bundle.is_some()}))?;
            if let Some(bundle) = &bundle {
                offline::install_packages(bundle, &log)?;
            } else {
                logged(
                    Command::new("apt-get")
                        .args(["-o", "DPkg::Lock::Timeout=120", "update"])
                        .env("DEBIAN_FRONTEND", "noninteractive"),
                    &log,
                    7200,
                )?;
                logged(
                    Command::new("apt-get")
                        .args(["-o", "DPkg::Lock::Timeout=120", "install", "-y"])
                        .args(PACKAGES)
                        .env("DEBIAN_FRONTEND", "noninteractive"),
                    &log,
                    7200,
                )?;
            }
            for (exe, args) in [
                ("usermod", vec!["-aG".into(), "video".into(), o.user.into()]),
                ("loginctl", vec!["enable-linger".into(), o.user.into()]),
                (
                    "systemctl",
                    vec!["start".into(), format!("user@{}.service", a.uid)],
                ),
            ] {
                logged(Command::new(exe).args(args), &log, 30)?;
            }
            // Capture manager environment privately; it may contain credentials.
            let mut manager = Command::new("runuser");
            manager
                .args(["-u", o.user, "--", "env"])
                .arg(format!("XDG_RUNTIME_DIR=/run/user/{}", a.uid))
                .arg(format!(
                    "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{}/bus",
                    a.uid
                ))
                .args(["systemctl", "--user", "show-environment"]);
            ensure!(
                output(&mut manager, 30)?.status.success(),
                "Runtime user manager is unavailable"
            );
            update("toolchain", json!({}))?;
            let rustup = a.home.join(".cargo/bin/rustup");
            let pin = launcher::pin(&work)?;
            if let Some(bundle) = &bundle {
                offline::install_user(bundle, o.user, &a.home, &pin)?;
            } else {
                if !rustup.is_file() {
                    let installer = work.join("rustup-init.sh");
                    logged(
                        Command::new("curl")
                            .args([
                                "--proto",
                                "=https",
                                "--tlsv1.2",
                                "-fsSL",
                                "https://sh.rustup.rs",
                                "-o",
                            ])
                            .arg(&installer),
                        &log,
                        300,
                    )?;
                    service::chown(&installer, &a)?;
                    run_user(
                        &[
                            "sh".into(),
                            installer.display().to_string(),
                            "-y".into(),
                            "--profile".into(),
                            "minimal".into(),
                            "--default-toolchain".into(),
                            "none".into(),
                        ],
                        o.user,
                        &a.home,
                        &log,
                        7200,
                    )?;
                }
                run_user(
                    &[
                        rustup.display().to_string(),
                        "toolchain".into(),
                        "install".into(),
                        pin.clone(),
                        "--profile".into(),
                        "minimal".into(),
                    ],
                    o.user,
                    &a.home,
                    &log,
                    7200,
                )?;
            }
            update("dependencies", json!({}))?;
            let mut fetch = vec![
                a.home.join(".cargo/bin/cargo").display().to_string(),
                format!("+{pin}"),
                "fetch".into(),
                "--locked".into(),
                "--manifest-path".into(),
                work.join("Cargo.toml").display().to_string(),
            ];
            if bundle.is_some() {
                fetch.push("--offline".into());
            }
            run_user(&fetch, o.user, &a.home, &log, 7200)?;
        }
        // The staging worker acquires the normal user's lock. Never build or run
        // the vision executable as root, including the saved-image smoke test.
        let controller = work.join("tools/vortexctl");
        update("building", json!({}))?;
        run_user(
            &[
                controller.display().to_string(),
                "stage".into(),
                "--root".into(),
                o.root.display().to_string(),
                "--archive".into(),
                archive.display().to_string(),
                "--sha256".into(),
                o.sha.into(),
                "--release-id".into(),
                o.release.into(),
                "--with-debug-bridge".into(),
            ],
            o.user,
            &a.home,
            &log,
            7200,
        )?;
        let ready = child(o.root, &format!("releases/{}", o.release))?;
        update("validating", json!({}))?;
        run_user(
            &[
                controller.display().to_string(),
                "preflight".into(),
                "--root".into(),
                o.root.display().to_string(),
                "--release-id".into(),
                o.release.into(),
            ],
            o.user,
            &a.home,
            &log,
            120,
        )?;
        update("service", json!({}))?;
        let installed = service::install(o.root, o.legacy, o.service, o.user)?;
        ensure!(
            installed["installed"] == true && installed["controller_update_deferred"] != true,
            "Recover unfinished activation before updating setup"
        );
        let c = read(ready.join("config/device.json"))?;
        update(
            "ready",
            json!({"release_path":ready,"profile":c["build_profile"],"verification":"dependencies_build_smoke_service","message":"Jetson initialized; release ready for Activate"}),
        )?;
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
