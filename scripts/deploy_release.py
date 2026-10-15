"""Package, verify, and build isolated Vortex releases. Never activates a release."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid


MAX_FILES = 10000
MAX_BYTES = 2 * 1024**3
MAX_MANIFEST = 2 * 1024**2
ROOT_FILES = {"Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml", "run_vortex.sh"}
SCRIPT_FILES = {"scripts/run_vortex.py", "scripts/deploy_release.py", "scripts/activate_release.py",
                "scripts/initialize_jetson.py",
                "simulation_tests/reference/dim_noisy_tag.png"}
RESOURCE_FILES = {"config/device.json", "config/config.json", "config/apriltag_map.json",
                  "config/2026-rebuilt-welded.json", "models/rockpaperscizzors.engine"}
PROFILES = {"cpu": [], "gpu": ["gpu"], "tensorrt": ["tensorrt"], "gpu-tensorrt": ["gpu", "tensorrt"]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            if mode is not None:
                os.chmod(temporary, mode)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value), "Invalid release ID")
    return value


def payload_path(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value and
            not any(ord(c) < 32 or ord(c) == 127 for c in value), "Invalid archive path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value and
            not any(part in (".", "..", ".git", "target", "node_modules", ".venv", "generated", "__pycache__")
                    for part in path.parts), f"Unsafe archive path: {value}")
    allowed = value in ROOT_FILES | SCRIPT_FILES | RESOURCE_FILES or (
        len(path.parts) > 1 and path.parts[0] == "src" and path.suffix in (".rs", ".c", ".h", ".cpp", ".cu"))
    require(allowed, f"Not a deployment resource: {value}")
    return path


def child(root, *parts):
    result = root.joinpath(*parts)
    require(result.resolve() == result and root in result.parents, f"Unsafe managed path: {result}")
    return result


def copy_regular(source, destination, source_root=None):
    source = Path(source).absolute()
    require(source.is_file() and not source.is_symlink(), f"Required regular file missing: {source}")
    if source_root is not None:
        require(source.resolve() == source and source_root in source.parents,
                f"Source symlinks are not packaged: {source}")
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and
            sha256(source) == sha256(destination), f"Source changed while packaging: {source}")


def git_snapshot(root):
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                                capture_output=True, text=True, check=True).stdout
        return dict(commit=commit, dirty=bool(status))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)


def package_release(source, output, device_config, runtime_config=None, tag_map=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(source.is_dir() and output != source, "Select a source directory and separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = read_json(device_config)
    require(isinstance(device, dict) and isinstance(device.get("build_profile"), str) and
            device["build_profile"] in PROFILES, "Invalid device configuration")
    require(device.get("runtime_config") == "config/config.json" and
            device.get("tag_map") in ("config/apriltag_map.json", "config/2026-rebuilt-welded.json"), "Nonportable device resources")
    require(set(device) == {"schema_version", "runtime", "build_profile", "cameras", "runtime_config", "tag_map", "networktables", "udp"},
            "Device payload contains unknown fields")
    provenance = git_snapshot(source)
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:16]
    archive = output/"release.tar.gz"
    require(not archive.exists(), "Output archive already exists")
    with tempfile.TemporaryDirectory(dir=output, prefix="snapshot-") as folder:
        snapshot = Path(folder)
        for relative in sorted(ROOT_FILES | SCRIPT_FILES):
            copy_regular(source/relative, snapshot/relative, source)
        require((source/"src").is_dir() and not (source/"src").is_symlink(), "Missing regular src directory")
        for path in sorted((source/"src").rglob("*")):
            require(not path.is_symlink(), f"Source symlinks are not packaged: {path}")
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                payload_path(relative)
                copy_regular(path, snapshot/relative, source)
        require(any((snapshot/"src").rglob("*.rs")), "No Rust sources found")
        atomic_json(snapshot/"config/device.json", device)
        copy_regular(runtime_config or source/device["runtime_config"], snapshot/device["runtime_config"])
        copy_regular(tag_map or source/device["tag_map"], snapshot/device["tag_map"])
        runtime = read_json(snapshot/device["runtime_config"])
        use_nn = runtime.get("object_detection", {}).get("use_nn", True)
        if use_nn and "tensorrt" in PROFILES[device["build_profile"]]:
            copy_regular(source/"models/rockpaperscizzors.engine", snapshot/"models/rockpaperscizzors.engine", source)
        validation = subprocess.run([sys.executable, str(snapshot/"scripts/run_vortex.py"), "--validate-config"],
                                    capture_output=True, text=True)
        require(validation.returncode == 0, f"Invalid release configuration: {validation.stderr.strip()}")
        after = git_snapshot(source)
        if after != provenance:
            provenance["dirty"] = True
        files = {}
        for path in sorted(snapshot.rglob("*")):
            if path.is_file():
                relative = path.relative_to(snapshot).as_posix()
                payload_path(relative)
                files[relative] = dict(size=path.stat().st_size, sha256=sha256(path), mode=0o755 if relative.endswith(".sh") else 0o644)
        require(len(files) <= MAX_FILES and sum(f["size"] for f in files.values()) <= MAX_BYTES, "Release exceeds package limits")
        manifest = dict(schema_version=1, release_id=identity, created_at=datetime.now(timezone.utc).isoformat(),
                        git=provenance, build_profile=device["build_profile"], features=PROFILES[device["build_profile"]], files=files)
        manifest["snapshot_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        data = json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
        require(len(data) <= MAX_MANIFEST, "Manifest too large")
        temporary = output/"release.tar.gz.part"
        try:
            with tarfile.open(temporary, "w:gz", format=tarfile.USTAR_FORMAT) as bundle:
                header = tarfile.TarInfo("manifest.json")
                header.size, header.mode = len(data), 0o644
                bundle.addfile(header, io.BytesIO(data))
                for relative, info in files.items():
                    header = tarfile.TarInfo(relative)
                    header.size, header.mode = info["size"], info["mode"]
                    with (snapshot/relative).open("rb") as stream:
                        bundle.addfile(header, stream)
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
        copy_regular(snapshot/"scripts/deploy_release.py", output/"controller.py")
        copy_regular(snapshot/"scripts/initialize_jetson.py", output/"initialize_jetson.py")
    metadata = dict(schema_version=1, release_id=identity, archive_sha256=sha256(archive),
                    archive_bytes=archive.stat().st_size, controller_sha256=sha256(output/"controller.py"),
                    build_profile=device["build_profile"], git=provenance,
                    initializer_sha256=sha256(output/"initialize_jetson.py"))
    atomic_json(output/"package.json", metadata)
    return metadata


class LockBusy(RuntimeError):
    pass


@contextmanager
def deployment_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise LockBusy("Another Vortex deployment is in progress") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)


def validate_manifest(manifest, identity):
    require(isinstance(manifest, dict) and type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1 and
            manifest.get("release_id") == identity, "Manifest release identity mismatch")
    profile = manifest.get("build_profile")
    require(isinstance(profile, str) and profile in PROFILES and manifest.get("features") == PROFILES[profile], "Invalid manifest profile")
    files = manifest.get("files")
    require(isinstance(files, dict) and 0 < len(files) <= MAX_FILES, "Invalid manifest file count")
    require(ROOT_FILES | SCRIPT_FILES | {"config/device.json", "config/config.json"} <= set(files), "Required release files missing")
    for relative, info in files.items():
        payload_path(relative)
        require(isinstance(info, dict) and type(info.get("size")) is int and info["size"] >= 0 and
                isinstance(info.get("sha256"), str) and re.fullmatch("[a-f0-9]{64}", info["sha256"]) and
                info.get("mode") in (0o644, 0o755), "Invalid manifest file record")
    require(sum(info["size"] for info in files.values()) <= MAX_BYTES, "Release exceeds extraction limit")
    require(hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == manifest.get("snapshot_sha256"),
            "Manifest snapshot checksum mismatch")
    return manifest


def extract_verified(archive, destination, identity):
    with tarfile.open(archive, "r:gz") as bundle:
        first = next(iter(bundle), None)
        require(first is not None and first.name == "manifest.json" and first.isfile() and
                0 <= first.size <= MAX_MANIFEST, "Missing or oversized release manifest")
        manifest = validate_manifest(json.loads(bundle.extractfile(first).read()), identity)
        needed = sum(info["size"] for info in manifest["files"].values()) + 512*1024**2
        require(shutil.disk_usage(destination.parent).free >= needed, "Insufficient free space for staged release")
        destination.mkdir()
        seen = {"manifest.json"}
        for member in bundle:
            # tarfile iteration starts at the cached first member.
            if member is first:
                continue
            require(member.name not in seen, "Duplicate archive entry")
            seen.add(member.name)
            payload_path(member.name)
            require(member.isfile() and not member.issparse() and member.name in manifest["files"], "Unexpected or nonregular archive entry")
            info = manifest["files"][member.name]
            require(member.size == info["size"] and member.mode == info["mode"], "Archive file metadata mismatch")
            target = child(destination, *PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                shutil.copyfileobj(bundle.extractfile(member), stream)
            require(sha256(target) == info["sha256"], f"Payload checksum mismatch: {member.name}")
            target.chmod(info["mode"])
        require(seen == set(manifest["files"]) | {"manifest.json"}, "Archive is incomplete")
        atomic_json(destination/"manifest.json", manifest)
        return manifest


def verify_payload(directory, manifest):
    for relative, info in manifest["files"].items():
        target = child(directory, *PurePosixPath(relative).parts)
        require(target.is_file() and target.stat().st_size == info["size"] and sha256(target) == info["sha256"],
                f"Release payload changed: {relative}")


def run_command(command, log, env):
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)


def stage_release(root, archive, archive_sha256, identity, with_debug_bridge=False, runner=run_command, lock_path=None):
    root = Path(root).resolve()
    identity = release_id(identity)
    require(isinstance(archive_sha256, str) and re.fullmatch("[a-f0-9]{64}", archive_sha256), "Invalid archive checksum")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("staging", "releases", "jobs", "build-cache"):
        child(root, name).mkdir(exist_ok=True)
    job_path = child(root, "jobs", identity+".json")
    job = dict(schema_version=1, release_id=identity, state="queued", activated=False,
               started_at=datetime.now(timezone.utc).isoformat())
    # Exclusive creation prevents another invocation from replacing a live job record.
    with child(root, "jobs", identity+".claim").open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    atomic_json(job_path, job)
    def update(state, **fields):
        job.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        atomic_json(job_path, job)
    log_path = child(root, "jobs", identity+".log")
    try:
        lock = lock_path or Path.home()/".local/state/vortex/deployment.lock"
        with deployment_lock(lock), log_path.open("a", encoding="utf-8") as log:
            update("verifying", log_path=str(log_path))
            archive = Path(archive).resolve()
            require(archive.is_file() and archive.stat().st_size <= MAX_BYTES and sha256(archive) == archive_sha256,
                    "Uploaded archive checksum mismatch or size limit exceeded")
            staging = child(root, "staging", identity)
            final = child(root, "releases", identity)
            require(not staging.exists() and not final.exists(), "Release already exists; create a new package")
            update("extracting")
            manifest = extract_verified(archive, staging, identity)
            device = read_json(staging/"config/device.json")
            require(device.get("build_profile") == manifest["build_profile"] and
                    device.get("runtime_config") == "config/config.json" and
                    device.get("tag_map") in manifest["files"], "Manifest/device resource mismatch")
            launcher = [sys.executable, str(staging/"scripts/run_vortex.py")]
            env = dict(os.environ, CARGO_BUILD_JOBS="2", RUSTUP_AUTO_INSTALL="0")
            update("validating", profile=manifest["build_profile"])
            runner(launcher+["--validate-config"], log, env)
            update("building")
            command = launcher+["--build", "--build-cache", str(child(root, "build-cache", manifest["build_profile"]))]
            if with_debug_bridge:
                command.append("--with-debug-bridge")
            runner(command, log, env)
            update("checking")
            runner(launcher+["--check"], log, env)
            if with_debug_bridge:
                runner(launcher+["--check-debug-bridge"], log, env)
            verify_payload(staging, manifest)
            atomic_json(staging/"release.json", dict(schema_version=1, release_id=identity, profile=manifest["build_profile"],
                                                     archive_sha256=archive_sha256, activated=False))
            staging.rename(final)
            update("ready", release_path=str(final), message="Release built and verified; not activated")
        return job
    except Exception as error:
        update("failed", error=str(error), log_path=str(log_path))
        raise


TERMINAL_STATES = {"ready", "active", "failed", "rolled_back", "rollback_failed", "recovered", "interrupted"}


def user_systemd_env():
    require(sys.platform == "linux", "Managed deployment jobs require Linux systemd")
    runtime = f"/run/user/{os.getuid()}"
    return dict(os.environ, XDG_RUNTIME_DIR=runtime,
                DBUS_SESSION_BUS_ADDRESS=f"unix:path={runtime}/bus")


def start_job(root, identity, kind, release=None, checksum=None, execute=subprocess.run):
    """Submit once to the persistent user manager; never restart an interrupted job."""
    root = Path(root).resolve()
    identity = release_id(identity)
    require(kind in ("stage", "activate", "rollback"), "Invalid managed job kind")
    require(re.fullmatch(r"[A-Za-z0-9_./:\\-]+", str(root)), "Managed job directory must use simple absolute Linux paths")
    env = user_systemd_env()
    linger = execute(["loginctl", "show-user", str(os.getuid()), "-p", "Linger", "--value"],
                     env=env, capture_output=True, text=True, check=True, timeout=15)
    require(linger.stdout.strip() == "yes", "Persistent deployment jobs need user lingering. Use Initialize Jetson in the deployment app to enable it.")
    execute(["systemctl", "--user", "show-environment"], env=env, capture_output=True,
            text=True, check=True, timeout=15)
    jobs = child(root, "jobs")
    jobs.mkdir(parents=True, exist_ok=True)
    unit = "vortex-job-"+hashlib.sha256((str(root)+"/"+identity).encode()).hexdigest()[:32]+".service"
    controller = Path(__file__).resolve()
    if kind == "stage":
        require(release == identity, "Stage identity mismatch")
        require(isinstance(checksum, str) and re.fullmatch("[a-f0-9]{64}", checksum), "Invalid archive checksum")
        incoming = child(root, "incoming", identity)
        require(controller == incoming/"controller.py", "Stage must use its verified uploaded controller")
        command = [sys.executable, str(controller), "stage", "--root", str(root), "--archive",
                   str(incoming/"release.tar.gz"), "--sha256", checksum, "--release-id", identity, "--with-debug-bridge"]
    else:
        command = [sys.executable, str(root/"activate_release.py"), kind, "--root", str(root), "--operation-id", identity]
        if kind == "activate":
            command += ["--release-id", release_id(release)]
    descriptor = jobs/(identity+".unit.json")
    # An exclusive dispatch claim also covers a lost SSH reply during submission.
    with (jobs/(identity+".dispatch")).open("x") as stream:
        stream.write(unit)
    data = dict(schema_version=1, job_id=identity, kind=kind, unit=unit, submitted_at=time.time(), dispatch="submitting")
    atomic_json(descriptor, data)
    try:
        execute(["systemd-run", "--user", "--unit="+unit, "--description=Vortex "+kind,
                 "--property=Type=exec", "--property=Restart=no", "--property=KillMode=control-group",
                 "--property=TimeoutStopSec=15", "--property=RuntimeMaxSec="+("7200" if kind == "stage" else "300"),
                 "--property=Nice=10", "--property=WorkingDirectory="+str(root),
                 "--property=StandardOutput=append:"+str(jobs/(identity+".controller.log")),
                 "--property=StandardError=inherit", "--", *command],
                env=env, capture_output=True, text=True, check=True, timeout=30)
        data["dispatch"] = "submitted"
        atomic_json(descriptor, data)
    except Exception:
        # Submission may have succeeded before SSH/systemd-run lost its reply.
        # Preserve the descriptor; status reconciles with the manager.
        raise
    return data


def supervised_status(root, identity, job, execute=subprocess.run):
    root = Path(root).resolve()
    descriptor = child(root, "jobs", release_id(identity)+".unit.json")
    if descriptor.exists():
        data = read_json(descriptor)
        expected = "vortex-job-"+hashlib.sha256((str(root)+"/"+identity).encode()).hexdigest()[:32]+".service"
        require(data["unit"] == expected and data["job_id"] == identity, "Invalid managed job descriptor")
        manager = data.get("manager", "user")
        require(manager in ("user", "system") and (manager != "system" or data.get("kind") == "initialize"), "Invalid job manager")
        response = execute(["systemctl", *(["--user"] if manager == "user" else []), "show", expected, "-p", "ActiveState", "-p", "Result",
                            "-p", "ExecMainStatus", "-p", "LoadState"], env=user_systemd_env() if manager == "user" else None,
                           capture_output=True, text=True, timeout=15)
        values = dict(line.split("=", 1) for line in response.stdout.splitlines() if "=" in line)
        require(response.returncode == 0 or values.get("LoadState") == "not-found",
                "Cannot inspect deployment job manager; reconnect before retrying")
        job["unit"] = expected
        job["unit_state"] = values.get("ActiveState")
        if job["state"] not in TERMINAL_STATES and values.get("ActiveState") not in ("active", "activating", "reloading", "deactivating"):
            if data["dispatch"] == "submitted" or time.time()-data["submitted_at"] > 30:
                job = dict(job, state="interrupted", error="Deployment worker stopped before completing; inspect logs and recovery state",
                           unit_result=values.get("Result"), exit_status=values.get("ExecMainStatus"))
    # The compiler writes a private log; otherwise show the controller's output.
    for suffix in (".log", ".controller.log"):
        log = child(root, "jobs", identity+suffix)
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size-16384))
                job["log_tail"] = stream.read(16384).decode("utf-8", errors="replace")
            break
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    pack = actions.add_parser("pack")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--device-config", type=Path, required=True)
    pack.add_argument("--runtime-config", type=Path)
    pack.add_argument("--tag-map", type=Path)
    stage = actions.add_parser("stage")
    stage.add_argument("--root", type=Path, required=True)
    stage.add_argument("--archive", type=Path, required=True)
    stage.add_argument("--sha256", required=True)
    stage.add_argument("--release-id", required=True)
    stage.add_argument("--with-debug-bridge", action="store_true")
    status = actions.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    status.add_argument("--release-id", required=True)
    start = actions.add_parser("start-job")
    start.add_argument("--root", type=Path, required=True)
    start.add_argument("--job-id", required=True)
    start.add_argument("--kind", choices=("stage", "activate", "rollback"), required=True)
    start.add_argument("--release-id")
    start.add_argument("--sha256")
    args = parser.parse_args(argv)
    try:
        if args.action == "pack":
            result = package_release(args.source, args.output, args.device_config, args.runtime_config, args.tag_map)
        elif args.action == "stage":
            if hasattr(os, "nice"):
                os.nice(10)
            result = stage_release(args.root, args.archive, args.sha256, args.release_id, args.with_debug_bridge)
        elif args.action == "start-job":
            result = start_job(args.root, args.job_id, args.kind, args.release_id, args.sha256)
        else:
            identity = release_id(args.release_id)
            job_path = child(args.root.resolve(), "jobs", identity+".json")
            result = read_json(job_path) if job_path.exists() else dict(release_id=identity, state="starting", activated=False)
            result = supervised_status(args.root, identity, result)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, EOFError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(f"Vortex release operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
