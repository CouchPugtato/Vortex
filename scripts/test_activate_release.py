"""Activation failure injection; service and symlink publication are mocked on Windows."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

import activate_release as activation
import deploy_release as deployment
import run_vortex as runtime


class SimulatedPowerLoss(BaseException):
    pass


class Service:
    def __init__(self):
        self.active, self.pid = True, 50
        self.events = []
        self.fail_stop = False

    def info(self):
        return dict(active=self.active, pid=self.pid if self.active else 0)

    def stop(self):
        self.events.append("stop")
        if self.fail_stop:
            raise RuntimeError("stop failed")
        self.active = False

    def start(self):
        self.events.append("start")
        self.active = True
        self.pid += 1


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.lock = self.root/"test.lock"
        self.service = Service()
        self.pointer_history = []
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(activation, "boot_id", return_value="boot-1").start()
        mock.patch.object(activation, "process_identity", return_value="start-1").start()
        # Real POSIX atomic symlink replacement is not available to this Windows
        # account. All journal/file writes and recovery decisions remain real.
        mock.patch.object(activation, "publish_pointer", side_effect=lambda root, value: self.pointer_history.append(copy.deepcopy(value))).start()
        self.make_release("old")
        self.make_release("new")
        deployment.atomic_json(self.root/"managed-service.json", dict(service="vortex-vision.service", legacy_root=str(self.root/"legacy"), installed=True))
        self.old = dict(kind="release", release_id="old", launch_token="old-token")
        deployment.atomic_json(self.root/"selection.json", self.old)
        deployment.atomic_json(self.root/"active.json", self.old)
        deployment.atomic_json(self.root/"previous.json", {"kind": "none"})

    def make_release(self, identity):
        directory = self.root/"releases"/identity
        directory.mkdir(parents=True)
        files = {}
        paths = deployment.ROOT_FILES | deployment.SCRIPT_FILES | {"src/main.rs", "config/device.json", "config/config.json", "config/2026-rebuilt-welded.json"}
        for relative in paths:
            target = directory/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (runtime.ROOT/relative).read_bytes() if relative.startswith("config/") else b"fixture"
            target.write_bytes(data)
            files[relative] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest(), mode=0o644)
        manifest = dict(schema_version=1, release_id=identity, build_profile="cpu", features=[], files=files,
                        snapshot_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
        deployment.atomic_json(directory/"manifest.json", manifest)
        deployment.atomic_json(directory/"release.json", dict(release_id=identity, profile="cpu"))
        binary = directory/"target/release/dumapril-taglocalization"
        binary.parent.mkdir(parents=True)
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", header, 18, 183)
        binary.write_bytes(header)
        deployment.atomic_json(binary.parent/"vortex-build.json", {})
        return directory

    def activate(self, **kwargs):
        return activation.activate(self.root, kwargs.pop("identity", "new"), manager=self.service, lock=self.lock,
                                   preflight=kwargs.pop("preflight", lambda *args: {}),
                                   health=kwargs.pop("health", lambda *args: {"verification": "all_cameras_advancing"}), **kwargs)

    def health_report(self, selection=None, frames=1, uptime=1000):
        selection = selection or self.old
        return dict(schema_version=1, release_id=selection["release_id"], launch_token=selection["launch_token"],
                    pid=self.service.pid, uptime_ms=uptime,
                    cameras=[dict(index=i, frames=frames, last_frame_age_ms=0, pipeline_ok=True,
                                  backends=["apriltag-cpu"]) for i in (0, 2)])

    def test_success_commits_selection_and_previous_after_health(self):
        def healthy(root, selected, manager):
            self.assertEqual(deployment.read_json(root/"active.json"), self.old)
            self.assertTrue((root/"activation.json").exists())
            return dict(verification="all_cameras_advancing")
        result = self.activate(health=healthy)
        self.assertEqual(result["state"], "active")
        self.assertEqual(deployment.read_json(self.root/"active.json")["release_id"], "new")
        self.assertEqual(deployment.read_json(self.root/"previous.json"), self.old)
        self.assertFalse((self.root/"activation.json").exists())
        self.assertEqual(self.service.events, ["stop", "start"])

    def test_preflight_failure_never_stops_service(self):
        with self.assertRaisesRegex(ValueError, "bad libraries"):
            self.activate(preflight=lambda *args: (_ for _ in ()).throw(ValueError("bad libraries")))
        self.assertFalse(self.service.events)
        self.assertEqual(deployment.read_json(self.root/"active.json"), self.old)
        self.assertFalse((self.root/"activation.json").exists())

    def test_failed_health_restores_old_and_verifies_rollback(self):
        seen = []
        def healthy(root, selection, manager):
            seen.append(selection["release_id"])
            if selection["release_id"] == "new":
                raise RuntimeError("camera 2 missing")
            return dict(verification="all_cameras_advancing")
        result = self.activate(health=healthy)
        self.assertEqual(result["state"], "rolled_back")
        self.assertEqual(seen, ["new", "old"])
        self.assertEqual(activation.selected(self.root)["release_id"], "old")
        self.assertNotEqual(activation.selected(self.root)["launch_token"], self.old["launch_token"])
        self.assertFalse((self.root/"activation.json").exists())

    def test_failed_rollback_remains_explicit_and_recoverable(self):
        result = self.activate(health=lambda *args: (_ for _ in ()).throw(RuntimeError("no frames")))
        self.assertEqual(result["state"], "rollback_failed")
        self.assertTrue((self.root/"activation.json").exists())
        self.assertEqual(activation.selected(self.root)["release_id"], "old")

    def test_failed_stop_preserves_existing_process(self):
        self.service.fail_stop = True
        result = self.activate()
        self.assertEqual(result["state"], "rolled_back")
        self.assertEqual(self.service.events, ["stop"])
        self.assertEqual(activation.selected(self.root), self.old)

    def test_manual_rollback_swaps_current_and_previous(self):
        self.activate()
        result = self.activate(rollback=True)
        self.assertEqual(result["state"], "active")
        self.assertEqual(activation.selected(self.root)["release_id"], "old")
        self.assertEqual(deployment.read_json(self.root/"previous.json")["release_id"], "new")

    def test_first_install_failure_stops_without_inventing_previous(self):
        deployment.atomic_json(self.root/"selection.json", {"kind": "none"})
        deployment.atomic_json(self.root/"active.json", {"kind": "none"})
        def health(root, value, manager):
            if value["kind"] == "release":
                raise RuntimeError("camera missing")
            self.assertFalse(manager.info()["active"])
            return dict(verification="stopped_no_previous_release")
        result = self.activate(health=health)
        self.assertEqual(result["state"], "rolled_back")
        self.assertEqual(activation.selected(self.root)["kind"], "none")
        self.assertFalse(self.service.active)

    def test_no_previous_release_is_an_error_before_downtime(self):
        with self.assertRaisesRegex(ValueError, "No previous"):
            self.activate(rollback=True)
        self.assertFalse(self.service.events)

    def test_legacy_previous_is_restored_without_claiming_camera_health(self):
        legacy = dict(kind="legacy", path=str(self.root/"legacy"))
        deployment.atomic_json(self.root/"selection.json", legacy)
        deployment.atomic_json(self.root/"active.json", legacy)
        def healthy(root, value, manager):
            if value["kind"] == "release": raise RuntimeError("new failed")
            return activation.wait_healthy(root, value, manager)
        result = self.activate(health=healthy)
        self.assertEqual(result["verification"], "legacy_process_only")
        self.assertEqual(activation.selected(self.root)["kind"], "legacy")

    def test_deployment_lock_blocks_activation_before_service_control(self):
        with deployment.deployment_lock(self.lock), self.assertRaises(deployment.LockBusy):
            self.activate()
        self.assertFalse(self.service.events)

    def test_power_loss_during_switch_restores_previous_selection(self):
        def crash(*args): raise SimulatedPowerLoss()
        with self.assertRaises(SimulatedPowerLoss): self.activate(health=crash)
        self.assertEqual(activation.selected(self.root)["release_id"], "new")
        with mock.patch.object(activation, "owner_alive", return_value=False):
            result = activation.recover_locked(self.root)
        self.assertEqual(result["state"], "restored_previous_selection")
        self.assertEqual(activation.selected(self.root), self.old)
        self.assertFalse(result["runtime_health_verified"])
        self.assertFalse((self.root/"activation.json").exists())

    def test_power_loss_before_confirmation_restores_previous_history(self):
        prior = dict(kind="legacy", path=str(self.root/"legacy"))
        deployment.atomic_json(self.root/"previous.json", prior)
        original = activation.atomic_json
        def write(path, data):
            original(path, data)
            if path.name == "active.json" and data.get("release_id") == "new": raise SimulatedPowerLoss()
        with mock.patch.object(activation, "atomic_json", side_effect=write), self.assertRaises(SimulatedPowerLoss):
            self.activate()
        activation.recover_locked(self.root, owner_check=False)
        self.assertEqual(deployment.read_json(self.root/"previous.json"), prior)
        self.assertEqual(activation.selected(self.root), self.old)

    def test_power_loss_after_confirmation_keeps_verified_new_release(self):
        original = activation.atomic_json
        def write(path, data):
            original(path, data)
            if path.name == "activation.json" and data.get("phase") == "confirmed": raise SimulatedPowerLoss()
        with mock.patch.object(activation, "atomic_json", side_effect=write), self.assertRaises(SimulatedPowerLoss):
            self.activate()
        result = activation.recover_locked(self.root, owner_check=False)
        self.assertEqual(result["state"], "completed_confirmed_activation")
        self.assertEqual(activation.selected(self.root)["release_id"], "new")
        self.assertEqual(deployment.read_json(self.root/"previous.json"), self.old)

    def test_job_write_failure_after_commit_does_not_undo_healthy_release(self):
        original = activation.atomic_json
        def write(path, data):
            if path.parent.name == "jobs" and data.get("state") == "active": raise OSError("disk full")
            original(path, data)
        with mock.patch.object(activation, "atomic_json", side_effect=write):
            result = self.activate()
        self.assertEqual(result["state"], "active")
        self.assertIn("disk full", result["warning"])
        self.assertEqual(activation.selected(self.root)["release_id"], "new")
        self.assertEqual(deployment.read_json(self.root/"activation.json")["phase"], "confirmed")
        self.assertEqual(self.service.events, ["stop", "start"])

    def test_recovered_job_is_reported_without_claiming_current_camera_health(self):
        with self.assertRaises(SimulatedPowerLoss):
            self.activate(operation_id="interrupted", health=lambda *args: (_ for _ in ()).throw(SimulatedPowerLoss()))
        activation.recover_locked(self.root, owner_check=False)
        result = activation.status(self.root, "interrupted")
        self.assertEqual(result["state"], "recovered")
        self.assertFalse(result["recovery"]["runtime_health_verified"])

    def test_service_entry_recovers_interrupted_switch_before_executing_old_runtime(self):
        with self.assertRaises(SimulatedPowerLoss):
            self.activate(health=lambda *args: (_ for _ in ()).throw(SimulatedPowerLoss()))
        with mock.patch.object(activation, "owner_alive", return_value=False), \
                mock.patch.object(activation, "lock_path", return_value=self.lock), \
                mock.patch.object(activation, "verify_current_pointer") as pointer, \
                mock.patch.object(activation.os, "execve") as execute:
            activation.service_run(self.root)
        self.assertEqual(activation.selected(self.root)["release_id"], "old")
        pointer.assert_called_once_with(self.root, self.root/"releases/old")
        self.assertIn(str(self.root/"releases/old/scripts/run_vortex.py"), execute.call_args.args[1])
        self.assertEqual(execute.call_args.args[2]["VORTEX_RELEASE_ID"], "old")

    def test_live_activation_cannot_be_recovered_out_from_under_worker(self):
        with self.assertRaises(SimulatedPowerLoss):
            self.activate(health=lambda *args: (_ for _ in ()).throw(SimulatedPowerLoss()))
        with mock.patch.object(activation, "owner_alive", return_value=True), self.assertRaises(deployment.LockBusy):
            activation.recover_locked(self.root)

    def test_invalid_recovery_journal_does_not_stop_current_process(self):
        deployment.atomic_json(self.root/"activation.json", dict(schema_version=1, phase="prepared", old={"kind": "unknown"}))
        with self.assertRaises(ValueError): activation.recover_locked(self.root, self.service, owner_check=False)
        self.assertFalse(self.service.events)

    def test_incomplete_service_install_is_not_accepted_for_activation(self):
        config = deployment.read_json(self.root/"managed-service.json")
        deployment.atomic_json(self.root/"managed-service.json", dict(config, installed=False))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            activation.Systemd(self.root)
        self.assertFalse(activation.status(self.root)["service_configured"])

    def test_health_accepts_empty_scene_and_rejects_wrong_pid_token_and_backend(self):
        report = self.health_report()
        deployment.atomic_json(self.root/"health.json", report)
        self.assertEqual(activation.health_sample(self.root, self.old, self.service)["cameras"][0]["frames"], 1)
        for change in (dict(pid=999), dict(launch_token="stale"), dict(release_id="new"), dict(cameras=[])):
            deployment.atomic_json(self.root/"health.json", dict(report, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                activation.health_sample(self.root, self.old, self.service)
        for field, value in (("last_frame_age_ms", 2000), ("pipeline_ok", False), ("backends", ["wrong"]), ("frames", 0)):
            data = copy.deepcopy(report)
            data["cameras"][1][field] = value
            deployment.atomic_json(self.root/"health.json", data)
            with self.subTest(field=field), self.assertRaises(ValueError):
                activation.health_sample(self.root, self.old, self.service)

    def test_health_window_requires_advancing_frames_and_publications(self):
        now = [0.]
        def clock(): return now[0]
        def sleep(dt): now[0] += dt
        def advancing(*args): return self.health_report(frames=1+int(now[0]*20), uptime=1000+int(now[0]*1000))
        with mock.patch.object(activation, "health_sample", side_effect=advancing):
            result = activation.wait_healthy(self.root, self.old, self.service, timeout=4, stable_seconds=1, clock=clock, sleep=sleep)
        self.assertEqual(result["verification"], "all_cameras_advancing")
        now[0] = 0
        with mock.patch.object(activation, "health_sample", return_value=self.health_report()):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                activation.wait_healthy(self.root, self.old, self.service, timeout=4, stable_seconds=1, clock=clock, sleep=sleep)

    def execute_native(self, command, **kwargs):
        if command[0] == "ldd": return subprocess.CompletedProcess(command, 0, "libc.so => /lib/libc.so", "")
        if "--deployment-smoke" in command:
            return subprocess.CompletedProcess(command, 0, 'VORTEX_SMOKE {"passed":true,"tag_id":1,"camera_error_m":0.01}\n', "")
        return subprocess.CompletedProcess(command, 0, "verified", "")

    def test_native_preflight_checks_elf_libraries_and_smoke_without_service_control(self):
        result = activation.native_preflight(self.root, "new", execute=self.execute_native, machine="aarch64")
        self.assertTrue(result["passed"])
        self.assertFalse(self.service.events)
        with self.assertRaisesRegex(ValueError, "architecture"):
            activation.native_preflight(self.root, "new", execute=self.execute_native, machine="x86_64")

    def test_missing_runtime_libraries_and_failed_smoke_are_rejected(self):
        for failure in ("ldd", "--deployment-smoke"):
            def execute(command, **kwargs):
                if failure in command:
                    return subprocess.CompletedProcess(command, 1, "libmissing.so => not found", "failed")
                return self.execute_native(command, **kwargs)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                activation.native_preflight(self.root, "new", execute=execute, machine="aarch64")


if __name__ == "__main__":
    unittest.main()
