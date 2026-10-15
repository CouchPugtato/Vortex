"""Transactional release activation, systemd entry point, and interrupted-switch recovery."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import uuid

import deploy_release as deployment


require = deployment.require
atomic_json = deployment.atomic_json
read_json = deployment.read_json
child = deployment.child


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_identity(pid):
    try:
        # comm may contain spaces; field 22 is the process start tick.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except OSError:
        return None


def owner_alive(journal):
    return journal.get("boot_id") == boot_id() and process_identity(journal.get("pid")) == journal.get("process_start")


def lock_path():
    return Path.home()/".local/state/vortex/deployment.lock"


def optional_json(path, default=None):
    return read_json(path) if path.exists() else default


def service_name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]*\.service", value), "Invalid service name")
    return value


def ready_release(root, identity):
    identity = deployment.release_id(identity)
    directory = child(root, "releases", identity)
    manifest = deployment.validate_manifest(read_json(directory/"manifest.json"), identity)
    metadata = read_json(directory/"release.json")
    require(metadata.get("release_id") == identity and metadata.get("profile") == manifest["build_profile"], "Invalid ready release")
    return directory, manifest


def native_preflight(root, identity, execute=subprocess.run, machine=None):
    directory, manifest = ready_release(root, identity)
    deployment.verify_payload(directory, manifest)
    binary = directory/"target/release/dumapril-taglocalization"
    require(binary.is_file(), "Missing Linux production executable")
    with binary.open("rb") as stream:
        header = stream.read(20)
    machine = machine or platform.machine().lower()
    expected = {"aarch64": 183, "arm64": 183, "x86_64": 62, "amd64": 62}.get(machine)
    require(expected is not None and len(header) == 20 and header[:4] == b"\x7fELF" and header[4] == 2 and
            header[5] in (1, 2), "Expected a supported 64-bit Linux ELF executable")
    require(struct.unpack("<H" if header[5] == 1 else ">H", header[18:20])[0] == expected,
            "Executable architecture does not match this device")
    def run(command, timeout=30):
        return execute(command, cwd=directory, capture_output=True, text=True, timeout=timeout)
    checked = run([sys.executable, str(directory/"scripts/run_vortex.py"), "--check"])
    require(checked.returncode == 0, f"Build verification failed: {checked.stderr}")
    libraries = run(["ldd", str(binary)])
    library_text = libraries.stdout+libraries.stderr
    require("not found" not in library_text and (libraries.returncode == 0 or
            "not a dynamic executable" in library_text or "statically linked" in library_text),
            f"Runtime libraries unavailable: {library_text.strip()}")
    smoke = run([str(binary), "--deployment-smoke", str(directory/"simulation_tests/reference/dim_noisy_tag.png")])
    reports = [line[len("VORTEX_SMOKE "):] for line in smoke.stdout.splitlines() if line.startswith("VORTEX_SMOKE ")]
    require(smoke.returncode == 0 and len(reports) == 1, f"Saved-image smoke test failed: {smoke.stderr}")
    result = json.loads(reports[0])
    require(result.get("passed") is True and result.get("tag_id") == 1 and
            type(result.get("camera_error_m")) in (int, float) and 0 <= result["camera_error_m"] < .2,
            "Saved-image pose check failed")
    # Make the known-good release durable before publishing its pointer.
    if os.name != "nt":
        for relative in list(manifest["files"])+["target/release/dumapril-taglocalization", "target/release/vortex-build.json", "release.json"]:
            with (directory/relative).open("rb") as stream:
                os.fsync(stream.fileno())
        for path in sorted((p for p in directory.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            deployment.sync_directory(path)
        deployment.sync_directory(directory)
    return result


class Systemd:
    def __init__(self, root):
        self.root = root
        self.config = read_json(root/"managed-service.json")
        require(self.config.get("installed") is True, "Managed service preparation is incomplete")
        self.name = service_name(self.config["service"])

    def command(self, *args):
        prefix = [] if os.name != "nt" and os.geteuid() == 0 else ["sudo", "-n"]
        return subprocess.run(prefix+["systemctl", *args, self.name], capture_output=True, text=True, check=True, timeout=30)

    def info(self):
        result = subprocess.run(["systemctl", "show", self.name, "-p", "ActiveState", "-p", "MainPID"],
                                capture_output=True, text=True, check=True, timeout=10)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        return dict(active=values.get("ActiveState") == "active", pid=int(values.get("MainPID", "0")))

    def stop(self):
        self.command("stop")
        require(not self.info()["active"] and self.info()["pid"] == 0, "Service did not stop")

    def start(self):
        self.command("reset-failed")
        self.command("start")


def validate_selection(root, value):
    require(isinstance(value, dict) and value.get("kind") in ("release", "legacy", "none"), "Invalid release selection")
    if value["kind"] == "release":
        ready_release(root, value["release_id"])
    elif value["kind"] == "legacy":
        config = read_json(root/"managed-service.json")
        require(value.get("path") == config.get("legacy_root") and Path(value["path"]).is_absolute(), "Invalid legacy selection")
    return value


def selected(root):
    return validate_selection(root, optional_json(root/"selection.json", {"kind": "none"}))


def publish_pointer(root, value):
    pointer = root/"current"
    require(not pointer.exists() or pointer.is_symlink(), "Current pointer is not a symlink")
    if value["kind"] == "release":
        temporary = root/("current-"+uuid.uuid4().hex+".tmp")
        try:
            temporary.symlink_to(child(root, "releases", value["release_id"]), target_is_directory=True)
            os.replace(temporary, pointer)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        pointer.unlink(missing_ok=True)
    deployment.sync_directory(root)


def switch_selection(root, value):
    validate_selection(root, value)
    publish_pointer(root, value)
    atomic_json(root/"selection.json", value)


def health_sample(root, selection, manager):
    report = read_json(root/"health.json")
    info = manager.info()
    require(info["active"] and info["pid"] > 0 and report.get("pid") == info["pid"], "Health process is not the active service")
    require(type(report.get("schema_version")) is int and report["schema_version"] == 1 and report.get("release_id") == selection["release_id"] and
            report.get("launch_token") == selection["launch_token"], "Health identity is stale")
    directory, _ = ready_release(root, selection["release_id"])
    device = read_json(directory/"config/device.json")
    runtime = read_json(directory/device["runtime_config"])
    expected = {"apriltag-gpu" if "gpu" in deployment.PROFILES[device["build_profile"]] else "apriltag-cpu"}
    if "tensorrt" in deployment.PROFILES[device["build_profile"]] and runtime.get("object_detection", {}).get("use_nn", True):
        expected.add("yolo-tensorrt")
    cameras = report.get("cameras")
    require(isinstance(cameras, list) and len(cameras) == len(device["cameras"]), "Missing configured cameras")
    require(all(type(camera.get("index")) is int for camera in cameras) and
            {camera.get("index") for camera in cameras} == set(device["cameras"]), "Wrong configured cameras")
    for camera in cameras:
        require(type(camera.get("frames")) is int and camera["frames"] > 0 and camera.get("pipeline_ok") is True and
                type(camera.get("last_frame_age_ms")) in (int, float) and 0 <= camera["last_frame_age_ms"] <= 1500 and
                expected <= set(camera.get("backends", [])), "Camera is stale or required detector is unavailable")
    require(type(report.get("uptime_ms")) in (int, float) and report["uptime_ms"] >= 0, "Invalid health clock")
    return report


def wait_healthy(root, selection, manager, timeout=30, stable_seconds=3, clock=time.monotonic, sleep=time.sleep):
    if selection["kind"] == "none":
        require(not manager.info()["active"], "First-install failure left a running service")
        return {"verification": "stopped_no_previous_release"}
    deadline = clock()+timeout
    baseline = None
    last_uptime = None
    published = clock()
    error = "No health report"
    while clock() < deadline:
        now = clock()
        try:
            if selection["kind"] == "legacy":
                info = manager.info()
                require(info["active"] and info["pid"] > 0, "Legacy process did not restart")
                if baseline is None or baseline["pid"] != info["pid"]:
                    baseline, started = info, now
                if now-started >= stable_seconds:
                    return {"verification": "legacy_process_only"}
                sleep(.25)
                continue
            report = health_sample(root, selection, manager)
            if baseline is None or report["pid"] != baseline["pid"] or report["uptime_ms"] < (last_uptime or 0):
                baseline, started, published = report, now, now
            if report["uptime_ms"] != last_uptime:
                last_uptime, published = report["uptime_ms"], now
            require(now-published < 1.25, "Health reporter stopped advancing")
            old_counts = {c["index"]: c["frames"] for c in baseline["cameras"]}
            if now-started >= stable_seconds and report["uptime_ms"] > baseline["uptime_ms"] and all(
                    c["frames"] > old_counts[c["index"]] for c in report["cameras"]):
                return {"verification": "all_cameras_advancing", "pid": report["pid"]}
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as problem:
            baseline, last_uptime, error = None, None, str(problem)
        sleep(.25)
    raise RuntimeError(f"Runtime health verification timed out: {error}")


def restore_metadata(root, old, previous):
    switch_selection(root, old)
    atomic_json(root/"active.json", old)
    atomic_json(root/"previous.json", previous)


def clear_journal(root):
    (root/"activation.json").unlink(missing_ok=True)
    deployment.sync_directory(root)


def recover_locked(root, manager=None, owner_check=True):
    journal = optional_json(root/"activation.json")
    if journal is None:
        return {"state": "no_recovery_needed"}
    require(journal.get("schema_version") == 1, "Invalid activation journal")
    require(journal.get("phase") in ("prepared", "verifying", "restoring", "confirmed"), "Invalid activation phase")
    for name in ("old", "new", "previous_before"):
        validate_selection(root, journal[name])
    if owner_check and owner_alive(journal):
        raise deployment.LockBusy("Activation is still running")
    if manager:
        manager.stop()
    if journal["phase"] == "confirmed":
        switch_selection(root, journal["new"])
        atomic_json(root/"active.json", journal["new"])
        atomic_json(root/"previous.json", journal["old"])
        state = "completed_confirmed_activation"
    else:
        restore_metadata(root, journal["old"], journal["previous_before"])
        state = "restored_previous_selection"
    result = dict(state=state, operation_id=journal["operation_id"], runtime_health_verified=False)
    atomic_json(root/"recovery.json", result)
    clear_journal(root)
    if manager and selected(root)["kind"] != "none":
        manager.start()
        result.update(wait_healthy(root, selected(root), manager))
        result["runtime_health_verified"] = result.get("verification") == "all_cameras_advancing"
        atomic_json(root/"recovery.json", result)
    return result


def activate(root, identity=None, rollback=False, operation_id=None, manager=None, preflight=native_preflight,
             health=wait_healthy, lock=None):
    root = Path(root).resolve()
    manager = manager or Systemd(root)
    operation_id = deployment.release_id(operation_id or "activation-"+uuid.uuid4().hex)
    (root/"jobs").mkdir(exist_ok=True)
    job_path = child(root, "jobs", operation_id+".json")
    with child(root, "jobs", operation_id+".claim").open("x") as stream:
        stream.write(str(os.getpid()))
    job = dict(operation_id=operation_id, state="queued", action="rollback" if rollback else "activate")
    def update(state, **values):
        job.update(state=state, **values)
        atomic_json(job_path, job)
    update("queued")
    try:
        with deployment.deployment_lock(lock or lock_path()):
            if (root/"activation.json").exists():
                recover_locked(root, manager)
            old = selected(root)
            committed = optional_json(root/"active.json", {"kind": "none"})
            require(old == committed, "Selection differs from committed state; recover before activation")
            previous = validate_selection(root, optional_json(root/"previous.json", {"kind": "none"}))
            if rollback:
                new = validate_selection(root, dict(previous))
                require(new["kind"] != "none", "No previous release is available")
            else:
                new = {"kind": "release", "release_id": deployment.release_id(identity)}
            if new["kind"] == "release":
                require(old.get("release_id") != new["release_id"], "Release is already selected")
                update("validating", release_id=new["release_id"])
                preflight(root, new["release_id"])
            new["launch_token"] = uuid.uuid4().hex
            journal = dict(schema_version=1, operation_id=operation_id, phase="prepared", old=old,
                           new=new, previous_before=previous, boot_id=boot_id(), pid=os.getpid(),
                           process_start=process_identity(os.getpid()))
            atomic_json(root/"activation.json", journal)
            committed = False
            try:
                update("stopping")
                manager.stop()
                update("switching")
                switch_selection(root, new)
                journal["phase"] = "verifying"
                atomic_json(root/"activation.json", journal)
                manager.start()
                update("verifying")
                verified = health(root, new, manager)
                atomic_json(root/"previous.json", old)
                atomic_json(root/"active.json", new)
                journal["phase"] = "confirmed"
                atomic_json(root/"activation.json", journal)
                committed = True
                update("active", selection=new, **verified)
                clear_journal(root)
                return job
            except Exception as failure:
                if committed:
                    # A confirmed, durable activation is not undone for a log/job
                    # write failure. The confirmed journal can be finalized later.
                    job.update(state="active", selection=new, warning=str(failure), **verified)
                    return job
                try:
                    # If stop failed before the switch, preserve the running old process.
                    unchanged = selected(root) == old and manager.info()["active"]
                    restored = dict(old)
                    if not unchanged:
                        manager.stop()
                        restored["launch_token"] = uuid.uuid4().hex
                    restore_metadata(root, restored, previous)
                    journal.update(old=restored, phase="restoring")
                    atomic_json(root/"activation.json", journal)
                    if not unchanged and restored["kind"] != "none":
                        manager.start()
                    checked = health(root, restored, manager)
                    clear_journal(root)
                    update("rolled_back", error=str(failure), selection=restored, **checked)
                except Exception as rollback_error:
                    update("rollback_failed", error=str(failure), rollback_error=str(rollback_error))
                return job
    except Exception as error:
        update("failed", error=str(error))
        raise


def verify_current_pointer(root, directory):
    require((root/"current").is_symlink() and (root/"current").resolve() == directory, "Release pointer mismatch")


def service_run(root):
    journal = optional_json(root/"activation.json")
    if journal and not owner_alive(journal):
        with deployment.deployment_lock(lock_path()):
            recover_locked(root)
    selection = selected(root)
    require(selection["kind"] != "none", "No confirmed or pending release selected")
    if not (root/"activation.json").exists():
        require(selection == read_json(root/"active.json"), "Uncommitted release selection")
    if selection["kind"] == "release":
        directory, _ = ready_release(root, selection["release_id"])
        verify_current_pointer(root, directory)
        env = dict(os.environ, VORTEX_HEALTH_PATH=str(root/"health.json"),
                   VORTEX_RELEASE_ID=selection["release_id"], VORTEX_LAUNCH_TOKEN=selection["launch_token"])
        os.execve(sys.executable, [sys.executable, str(directory/"scripts/run_vortex.py")], env)
    else:
        script = Path(selection["path"])/"run_vortex.sh"
        os.chdir(script.parent)
        os.execve("/bin/bash", ["/bin/bash", str(script)], dict(os.environ))


def install_service(root, legacy_root, name, user, execute=subprocess.run, unit_dir=Path("/etc/systemd/system")):
    require(os.name != "nt" and os.geteuid() == 0, "Service installation requires sudo/root on Linux")
    import pwd
    account = pwd.getpwnam(user)
    root, legacy_root = Path(root).resolve(), Path(legacy_root).resolve()
    name = service_name(name)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.service", name), "Managed service must use a simple unit name")
    for value in (str(root), str(legacy_root), user):
        require(re.fullmatch(r"[A-Za-z0-9_./-]+", value), "Service paths/user must not contain spaces or shell/specifier characters")
    user_lock = Path(account.pw_dir)/".local/state/vortex/deployment.lock"
    require(user_lock.exists(), "Stage a release as the runtime user before installing the managed service")
    with deployment.deployment_lock(user_lock):
        configured = optional_json(root/"managed-service.json")
        require(configured is None or (configured["service"] == name and configured["user"] == user and
                                      configured["legacy_root"] == str(legacy_root)), "Managed service identity cannot change during activation")
        if (root/"activation.json").exists():
            require(configured is not None, "Recover the unfinished activation before installing its controller")
            return dict(configured, controller_update_deferred=True)
        dropins = execute(["systemctl", "show", name, "-p", "DropInPaths", "--value"], capture_output=True, text=True, timeout=15)
        require(not dropins.stdout.strip(), "Existing service has drop-ins; migrate them explicitly before activation")
        baseline = {"kind": "none"}
        if configured is None:
            running = execute(["systemctl", "is-active", "--quiet", name], capture_output=True, text=True, timeout=15).returncode == 0
            if running:
                command = execute(["systemctl", "show", name, "-p", "ExecStart", "--value"], capture_output=True, text=True, check=True, timeout=15)
                require(f"path={legacy_root}/run_vortex.sh ;" in command.stdout, "Existing service is not the expected legacy Vortex launcher")
                baseline = dict(kind="legacy", path=str(legacy_root))
        unit_path = unit_dir/name
        original = unit_path.read_bytes() if unit_path.exists() else None
        if configured is None and original is not None:
            (root/"legacy-service.unit").write_bytes(original)
        for filename in ("activate_release.py", "deploy_release.py"):
            source = Path(__file__).resolve().parent/filename
            destination = root/filename
            temporary = root/(filename+".new")
            shutil.copyfile(source, temporary)
            temporary.chmod(0o644)
            os.replace(temporary, destination)
        config = dict(schema_version=1, service=name, user=user, legacy_root=str(legacy_root), installed=False)
        atomic_json(root/"managed-service.json", config)
        if configured is None:
            atomic_json(root/"selection.json", baseline)
            atomic_json(root/"active.json", baseline)
            atomic_json(root/"previous.json", {"kind": "none"})
        # Metadata is read/written by the unprivileged activation worker.
        for filename in ("managed-service.json", "selection.json", "active.json", "previous.json"):
            os.chown(root/filename, account.pw_uid, account.pw_gid)
        unit = f"""[Unit]
Description=Vortex managed vision release
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User={user}
Group={account.pw_gid}
ExecStart=/usr/bin/python3 {root}/activate_release.py service-run --root {root}
Restart=on-failure
RestartSec=2
KillMode=control-group
KillSignal=SIGINT
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
"""
        temporary = unit_path.with_suffix(".service.new")
        temporary.write_text(unit, encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, unit_path)
        deployment.sync_directory(unit_dir)
        try:
            # Narrow service control permission; no arbitrary root command access.
            sudoers = Path("/etc/sudoers.d")/("vortex-"+re.sub(r"[^A-Za-z0-9_-]", "_", name[:-8]))
            sudoers_temp = sudoers.with_suffix(".new")
            executable = shutil.which("systemctl") or "/usr/bin/systemctl"
            sudoers_temp.write_text(f"{user} ALL=(root) NOPASSWD: "+", ".join(
                f"{executable} {action} {name}" for action in ("start", "stop", "reset-failed"))+"\n")
            sudoers_temp.chmod(0o440)
            try:
                execute(["visudo", "-cf", str(sudoers_temp)], capture_output=True, text=True, check=True, timeout=15)
                os.replace(sudoers_temp, sudoers)
            finally:
                sudoers_temp.unlink(missing_ok=True)
            execute(["systemctl", "daemon-reload"], check=True, timeout=20)
            execute(["systemctl", "enable", name], check=True, timeout=20)
            execute(["loginctl", "enable-linger", user], check=True, timeout=20)
        except Exception:
            if original is None:
                unit_path.unlink(missing_ok=True)
            else:
                unit_path.write_bytes(original)
            execute(["systemctl", "daemon-reload"], check=True, timeout=20)
            raise
        config["installed"] = True
        atomic_json(root/"managed-service.json", config)
        os.chown(root/"managed-service.json", account.pw_uid, account.pw_gid)
        return config


def status(root, operation_id=None):
    if operation_id:
        path = child(root, "jobs", deployment.release_id(operation_id)+".json")
        job = optional_json(path, dict(operation_id=operation_id, state="starting"))
        recovered = optional_json(root/"recovery.json")
        if recovered and recovered.get("operation_id") == operation_id and job["state"] in (
                "queued", "stopping", "switching", "verifying", "rollback_failed"):
            return dict(job, state="recovered", recovery=recovered,
                        error="Interrupted activation recovered; refresh the selected release and inspect runtime health")
        return job
    ready = []
    for path in sorted((root/"releases").glob("*/release.json"), reverse=True):
        try:
            directory, _ = ready_release(root, path.parent.name)
            ready.append(directory.name)
        except (OSError, ValueError, KeyError):
            continue
    installed = root/"activate_release.py"
    return dict(ready=ready, active=optional_json(root/"active.json", {"kind": "none"}),
                previous=optional_json(root/"previous.json", {"kind": "none"}),
                pending=optional_json(root/"activation.json"), service_configured=bool(optional_json(root/"managed-service.json", {}).get("installed")),
                controller_current=installed.is_file() and deployment.sha256(installed) == deployment.sha256(Path(__file__)) and
                (root/"deploy_release.py").is_file() and deployment.sha256(root/"deploy_release.py") == deployment.sha256(Path(deployment.__file__)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("activate", "rollback", "recover", "service-run", "install-service", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release-id")
    parser.add_argument("--operation-id")
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--service", default="vortex-vision.service")
    parser.add_argument("--user", default="vortex")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.action == "install-service":
            require(args.legacy_root is not None, "--legacy-root is required")
            result = install_service(root, args.legacy_root, args.service, args.user)
        elif args.action == "service-run":
            service_run(root)
            return 0
        elif args.action == "recover":
            with deployment.deployment_lock(lock_path()):
                result = recover_locked(root, Systemd(root))
        elif args.action == "status":
            result = status(root, args.operation_id)
            if args.operation_id:
                result = deployment.supervised_status(root, args.operation_id, result)
        else:
            result = activate(root, args.release_id, args.action == "rollback", args.operation_id)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Vortex activation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
