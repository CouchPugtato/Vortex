"""Provision an existing Linux account and prepare Vortex without activating it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import deploy_release as deployment

PACKAGES = ["ca-certificates", "curl", "git", "build-essential", "pkg-config", "cmake", "clang",
            "libclang-dev", "libssl-dev", "libudev-dev", "libturbojpeg0-dev", "libv4l-dev",
            "v4l-utils", "python3", "dbus-user-session"]


def compatibility(profile, system=None, machine=None, etc=Path("/etc"), exists=lambda path: Path(path).exists()):
    deployment.require((system or platform.system()) == "Linux", "Initialization requires Linux with systemd")
    machine = machine or platform.machine()
    deployment.require(machine in ("aarch64", "x86_64"), "Unsupported architecture: "+machine)
    release = dict(line.split("=", 1) for line in (etc/"os-release").read_text().splitlines() if "=" in line)
    release = {key: value.strip('"') for key, value in release.items()}
    deployment.require(release.get("ID") == "ubuntu" and release.get("VERSION_ID") in ("20.04", "22.04", "24.04"),
                       "Supported initialization targets are Ubuntu 20.04, 22.04 and 24.04")
    deployment.require(exists("/run/systemd/system"), "The device must be booted with systemd")
    deployment.require(profile in deployment.PROFILES, "Unknown build profile")
    if "gpu" in deployment.PROFILES[profile]:
        deployment.require(machine == "aarch64" and any(exists(f"/opt/nvidia/vpi{version}/include/vpi/VPI.h") and
                           any(exists(f"/opt/nvidia/vpi{version}/{folder}/libnvvpi.so") for folder in
                               ("lib64", "lib/aarch64-linux-gnu")) for version in (2, 3)),
                           "GPU profile needs JetPack VPI 2 or 3 development headers/libraries. Install matching JetPack or select CPU.")
    if "tensorrt" in deployment.PROFILES[profile]:
        deployment.require(machine == "aarch64" and exists("/usr/local/cuda/include/cuda_runtime_api.h") and
                           any(exists(path) for path in ("/usr/include/aarch64-linux-gnu/NvInfer.h", "/usr/include/NvInfer.h")) and
                           any(exists(path) for path in ("/usr/lib/aarch64-linux-gnu/libnvinfer.so", "/opt/nvidia/tensorrt/lib/libnvinfer.so")),
                           "TensorRT profile needs matching JetPack CUDA/TensorRT development packages. Install those or select CPU.")
    return dict(architecture=machine, ubuntu=release["VERSION_ID"], profile=profile)


def validate_options(args):
    args.root = args.root.resolve()
    args.legacy_root = args.legacy_root.resolve()
    deployment.require(str(args.root) == str(args.legacy_root)+".deployment", "Managed root must be the legacy directory's sibling")
    for value in (str(args.root), str(args.legacy_root), args.user):
        deployment.require(re.fullmatch(r"[A-Za-z0-9_./:\\-]+", value), "Use simple account names and deployment paths")
    deployment.require(re.fullmatch(r"[a-z_][a-z0-9_-]*", args.user) and args.user != "root", "Select a non-root runtime account")
    deployment.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.service", args.service), "Invalid service name")
    deployment.release_id(args.job_id)
    deployment.release_id(args.release_id)
    deployment.require(re.fullmatch("[a-f0-9]{64}", args.sha256), "Invalid archive checksum")


def submit(args, execute=subprocess.run):
    validate_options(args)
    deployment.require(sys.platform == "linux" and os.geteuid() == 0, "Initialization needs administrator privileges")
    deployment.require(os.environ.get("SUDO_USER", args.user) == args.user, "Connect as the runtime account when initializing")
    jobs = deployment.child(args.root, "jobs")
    jobs.mkdir(parents=True, exist_ok=True)
    unit = "vortex-job-"+hashlib.sha256((str(args.root)+"/"+args.job_id).encode()).hexdigest()[:32]+".service"
    with (jobs/(args.job_id+".dispatch")).open("x") as stream:
        stream.write(unit)
    descriptor = dict(schema_version=1, job_id=args.job_id, kind="initialize", manager="system",
                      unit=unit, submitted_at=time.time(), dispatch="submitting")
    path = jobs/(args.job_id+".unit.json")
    deployment.atomic_json(path, descriptor, mode=0o644)
    log_path = jobs/(args.job_id+".controller.log")
    log_path.touch(exist_ok=False)
    log_path.chmod(0o644)
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--root", str(args.root),
               "--legacy-root", str(args.legacy_root), "--user", args.user, "--service", args.service,
               "--job-id", args.job_id, "--release-id", args.release_id, "--sha256", args.sha256]
    execute(["systemd-run", "--unit="+unit, "--property=Type=exec", "--property=Restart=no",
             "--property=RuntimeMaxSec=10800", "--property=TimeoutStopSec=20", "--property=KillMode=control-group",
             "--property=StandardInput=null", "--property=StandardOutput=append:"+str(jobs/(args.job_id+".controller.log")),
             "--property=StandardError=inherit", "--", *command],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True, timeout=30)
    descriptor["dispatch"] = "submitted"
    deployment.atomic_json(path, descriptor, mode=0o644)
    return descriptor


def initialize(args, execute=subprocess.run):
    validate_options(args)
    deployment.require(sys.platform == "linux" and os.geteuid() == 0, "Initialization needs administrator privileges")
    import pwd
    account = pwd.getpwnam(args.user)
    deployment.require(account.pw_uid != 0, "Runtime account cannot be root")
    home = Path(account.pw_dir)
    lock = home/".local/state/vortex/deployment.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch(exist_ok=True)
    # A newly provisioned lock must remain usable by subsequent unprivileged jobs.
    for path in (home/".local", home/".local/state", lock.parent, lock):
        os.chown(path, account.pw_uid, account.pw_gid)
    job_path = deployment.child(args.root, "jobs", args.job_id+".json")
    with job_path.with_suffix(".claim").open("x") as stream:
        stream.write(str(os.getpid()))
    job = dict(schema_version=1, operation_id=args.job_id, action="initialize", state="queued", release_id=args.release_id, activated=False)
    def update(state, **values):
        job.update(state=state, **values)
        deployment.atomic_json(job_path, job, mode=0o644)
        print("Initialization: "+state, flush=True)
    def run(command, user=False, timeout=7200, quiet=False):
        if user:
            command = ["runuser", "-u", args.user, "--", "env", "HOME="+str(home),
                       "CARGO_HOME="+str(home/".cargo"), "RUSTUP_HOME="+str(home/".rustup"),
                       "PATH="+str(home/".cargo/bin")+":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", *command]
        execute(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL if quiet else None, check=True, timeout=timeout,
                env=dict(os.environ, DEBIAN_FRONTEND="noninteractive", CARGO_BUILD_JOBS="2", RUSTUP_AUTO_INSTALL="0"))
    try:
        with deployment.deployment_lock(lock):
            update("verifying")
            archive = deployment.child(args.root, "incoming", args.release_id, "release.tar.gz")
            deployment.require(archive.is_file() and deployment.sha256(archive) == args.sha256, "Initialization archive checksum mismatch")
            parent = deployment.child(args.root, "initialization")
            parent.mkdir(exist_ok=True)
            parent.chmod(0o755)
            work = parent/args.job_id
            manifest = deployment.extract_verified(archive, work, args.release_id)
            for path in [work, *work.rglob("*")]:
                os.chown(path, account.pw_uid, account.pw_gid)
            update("checking", compatibility=compatibility(manifest["build_profile"]))
            deployment.require(shutil.disk_usage(args.root).free >= 6*1024**3, "Initialization needs at least 6 GiB of free disk space")
            update("packages")
            run(["apt-get", "-o", "DPkg::Lock::Timeout=120", "update"])
            run(["apt-get", "-o", "DPkg::Lock::Timeout=120", "install", "-y", *PACKAGES])
            run(["usermod", "-aG", "video", args.user], timeout=30)
            run(["loginctl", "enable-linger", args.user], timeout=30)
            # Verify the account's user manager is available for ordinary jobs.
            run(["systemctl", "start", f"user@{account.pw_uid}.service"], timeout=30)
            run(["env", f"XDG_RUNTIME_DIR=/run/user/{account.pw_uid}",
                 f"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{account.pw_uid}/bus",
                 "systemctl", "--user", "show-environment"], user=True, timeout=30, quiet=True)
            update("toolchain")
            rustup = home/".cargo/bin/rustup"
            if not rustup.is_file():
                installer = work/"rustup-init.sh"
                run(["curl", "--proto", "=https", "--tlsv1.2", "-fsSL", "https://sh.rustup.rs", "-o", str(installer)])
                os.chown(installer, account.pw_uid, account.pw_gid)
                run(["sh", str(installer), "-y", "--profile", "minimal", "--default-toolchain", "none"], user=True)
            pin = re.search(r'channel\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', (work/"rust-toolchain.toml").read_text())
            deployment.require(pin is not None, "The source must pin a Rust release")
            run([str(rustup), "toolchain", "install", pin[1], "--profile", "minimal"], user=True)
            update("dependencies")
            run([str(home/".cargo/bin/cargo"), "+"+pin[1], "fetch", "--locked", "--manifest-path", str(work/"Cargo.toml")], user=True)
        # The normal stage worker owns the same lock while building as the runtime user.
        update("building")
        run([sys.executable, str(work/"scripts/deploy_release.py"), "stage", "--root", str(args.root),
             "--archive", str(archive), "--sha256", args.sha256, "--release-id", args.release_id, "--with-debug-bridge"], user=True)
        ready = deployment.child(args.root, "releases", args.release_id)
        update("validating")
        sys.path.insert(0, str(work/"scripts"))
        import activate_release
        run([sys.executable, str(ready/"scripts/activate_release.py"), "preflight", "--root", str(args.root),
             "--release-id", args.release_id], user=True, timeout=120)
        update("service")
        result = activate_release.install_service(args.root, args.legacy_root, args.service, args.user)
        deployment.require(result.get("installed") and not result.get("controller_update_deferred"), "Recover the unfinished activation before updating setup")
        # Validate the installed launcher metadata without opening cameras or selecting it.
        deployment.require(activate_release.status(args.root)["service_configured"], "Managed service was not installed")
        update("ready", release_path=str(ready), profile=manifest["build_profile"],
               verification="dependencies_build_smoke_service", message="Jetson initialized; release ready for Activate")
        return job
    except Exception as error:
        update("failed", error=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "run", "status"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--user")
    parser.add_argument("--service", default="vortex-vision.service")
    parser.add_argument("--release-id")
    parser.add_argument("--sha256")
    args = parser.parse_args(argv)
    try:
        if args.action == "status":
            path = deployment.child(args.root.resolve(), "jobs", deployment.release_id(args.job_id)+".json")
            job = deployment.read_json(path) if path.exists() else dict(operation_id=args.job_id, state="starting")
            result = deployment.supervised_status(args.root, args.job_id, job)
            if job.get("state") == "building" and job.get("release_id"):
                stage_log = deployment.child(args.root.resolve(), "jobs", job["release_id"]+".log")
                if stage_log.is_file():
                    with stage_log.open("rb") as stream:
                        stream.seek(max(0, stage_log.stat().st_size-16384))
                        result["log_tail"] = stream.read(16384).decode("utf-8", errors="replace")
        else:
            deployment.require(args.legacy_root is not None and args.user and args.release_id and args.sha256, "Initialization options missing")
            result = submit(args) if args.action == "submit" else initialize(args)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print("Initialization failed: "+str(error), file=sys.stderr)
        return 1


if __name__ == "__main__": sys.exit(main())
