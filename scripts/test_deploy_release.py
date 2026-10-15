"""Release round trips and failure injection without cameras, Rust builds, or SSH."""
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import deploy_release as deployment
import run_vortex as runtime


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.source = self.directory/"project"
        for relative in deployment.ROOT_FILES | deployment.SCRIPT_FILES | {
                "config/device.json", "config/config.json", "config/2026-rebuilt-welded.json"}:
            destination = self.source/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runtime.ROOT/relative, destination)
        (self.source/"src").mkdir()
        (self.source/"src/main.rs").write_text("fn main() {}")
        self.output = self.directory/"package"
        self.root = self.directory/"managed"
        self.lock = self.directory/"device.lock"
        self.calls = []

    def package(self, **kwargs):
        self.metadata = deployment.package_release(self.source, self.output, self.source/"config/device.json", **kwargs)
        self.identity = self.metadata["release_id"]
        self.archive = self.output/"release.tar.gz"
        return self.metadata

    def stage(self, **kwargs):
        return deployment.stage_release(self.root, self.archive, deployment.sha256(self.archive), self.identity,
                                        runner=kwargs.pop("runner", self.fake_runner), lock_path=self.lock, **kwargs)

    def fake_runner(self, command, log, env):
        self.calls.append((command, env))
        if "--build" in command:
            stage_root = Path(command[1]).parents[1]
            config = runtime.validate_config(runtime.read_json(stage_root/"config/device.json"))
            cache = Path(command[command.index("--build-cache")+1])
            def cargo_build(args, **kwargs):
                for name in (runtime.MAIN, runtime.BRIDGE):
                    binary = cache/"release"/runtime.binary_path(stage_root, name).name
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(b"mock native executable")
                    binary.chmod(0o755)
            with mock.patch.object(runtime.subprocess, "run", side_effect=cargo_build):
                runtime.build(stage_root, config, cache_dir=cache, with_debug_bridge="--with-debug-bridge" in command)
        else:
            deployment.run_command(command, log, env)

    def rewrite(self, mutate):
        entries = []
        with tarfile.open(self.archive, "r:gz") as bundle:
            for member in bundle:
                entries.append((copy.copy(member), bundle.extractfile(member).read() if member.isfile() else None))
        mutate(entries)
        with tarfile.open(self.archive, "w:gz") as bundle:
            for member, data in entries:
                if data is not None:
                    member.size = len(data)
                bundle.addfile(member, io.BytesIO(data) if data is not None else None)

    def read_job(self):
        return deployment.read_json(self.root/"jobs"/(self.identity+".json"))

    def test_actual_repository_round_trip(self):
        metadata = deployment.package_release(runtime.ROOT, self.output, runtime.ROOT/"config/device.json")
        manifest = deployment.extract_verified(self.output/"release.tar.gz", self.directory/"extracted", metadata["release_id"])
        deployment.verify_payload(self.directory/"extracted", manifest)
        self.assertTrue(manifest["files"])
        self.assertLess(metadata["archive_bytes"], 2*1024**2)

    def test_only_required_payload_is_packaged(self):
        for relative in ("docs/demo.gif", "chassis_sim/generated/capture.png", "config/deployment_tool_config.json",
                         "target/old-binary", ".venv/secret", ".git/config", "node_modules/large.js"):
            path = self.source/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("excluded")
        self.package()
        with tarfile.open(self.archive) as bundle:
            names = bundle.getnames()
        self.assertIn("src/main.rs", names)
        self.assertIn("config/device.json", names)
        self.assertFalse(any(name.startswith(("docs/", ".venv/", "target/", "chassis_sim/", "node_modules/")) for name in names))
        self.assertNotIn("config/deployment_tool_config.json", names)

    def test_manifest_records_snapshot_revision_profile_and_checksums(self):
        with mock.patch.object(deployment, "git_snapshot", return_value=dict(commit="abc123", dirty=True)):
            metadata = self.package()
        with tarfile.open(self.archive) as bundle:
            manifest = json.load(bundle.extractfile("manifest.json"))
        self.assertEqual(manifest["git"], dict(commit="abc123", dirty=True))
        self.assertEqual(manifest["build_profile"], "cpu")
        self.assertEqual(manifest["files"]["src/main.rs"]["sha256"], deployment.sha256(self.source/"src/main.rs"))
        self.assertEqual(metadata["controller_sha256"], deployment.sha256(self.output/"controller.py"))
        self.assertEqual(metadata["archive_sha256"], deployment.sha256(self.archive))

    def test_selected_external_runtime_and_map_are_snapshotted(self):
        runtime_file = self.directory/"selected.json"
        config = runtime.read_json(self.source/"config/config.json")
        config["camera"]["fx"] = 1234.
        runtime.atomic_json(runtime_file, config)
        self.package(runtime_config=runtime_file, tag_map=self.source/"config/2026-rebuilt-welded.json")
        with tarfile.open(self.archive) as bundle:
            self.assertEqual(json.load(bundle.extractfile("config/config.json"))["camera"]["fx"], 1234.)

    def test_invalid_local_resources_fail_before_upload(self):
        (self.source/"config/config.json").write_text('{"camera": {}}')
        with self.assertRaisesRegex(ValueError, "Invalid release configuration"):
            self.package()
        self.assertFalse((self.output/"release.tar.gz").exists())

    def test_device_paths_and_unknown_secret_fields_are_rejected(self):
        for change in (dict(runtime_config="../secret.json"), dict(passphrase="secret")):
            config = runtime.read_json(self.source/"config/device.json")
            runtime.atomic_json(self.source/"config/device.json", dict(config, **change))
            with self.assertRaises(ValueError):
                self.package()
            shutil.copyfile(runtime.ROOT/"config/device.json", self.source/"config/device.json")

    def test_new_release_is_ready_without_touching_current(self):
        self.package()
        current = self.root/"current"
        current.mkdir(parents=True)
        (current/"binary").write_bytes(b"old running installation")
        result = self.stage(with_debug_bridge=True)
        self.assertEqual(result["state"], "ready")
        self.assertFalse(result["activated"])
        self.assertEqual((current/"binary").read_bytes(), b"old running installation")
        final = Path(result["release_path"])
        config = runtime.read_json(final/"config/device.json")
        runtime.checked_binary(final, config)
        runtime.checked_binary(final, config, runtime.BRIDGE)
        self.assertFalse((self.root/"staging"/self.identity).exists())
        self.assertTrue((final/"release.json").exists())
        self.assertEqual(self.read_job()["state"], "ready")
        self.assertTrue(all(env["RUSTUP_AUTO_INSTALL"] == "0" and env["CARGO_BUILD_JOBS"] == "2" for _, env in self.calls))

    def test_later_cache_changes_cannot_modify_release_binary(self):
        self.package()
        job = self.stage()
        final = Path(job["release_path"])
        binary = runtime.binary_path(final, runtime.MAIN)
        before = binary.read_bytes()
        (self.root/"build-cache/cpu/release"/binary.name).write_bytes(b"later build")
        self.assertEqual(binary.read_bytes(), before)
        runtime.checked_binary(final, runtime.read_json(final/"config/device.json"))

    def test_corrupted_transport_hash_fails_without_extraction(self):
        self.package()
        with self.assertRaisesRegex(ValueError, "archive checksum"):
            deployment.stage_release(self.root, self.archive, "0"*64, self.identity, lock_path=self.lock)
        self.assertEqual(self.read_job()["state"], "failed")
        self.assertFalse((self.root/"staging"/self.identity).exists())
        self.assertFalse(self.calls)

    def test_changed_payload_with_valid_outer_hash_is_rejected(self):
        self.package()
        def mutate(entries):
            index = next(i for i, (info, _) in enumerate(entries) if info.name == "src/main.rs")
            info, data = entries[index]
            entries[index] = info, data.replace(b"main", b"evil")
        self.rewrite(mutate)
        with self.assertRaisesRegex(ValueError, "Payload checksum"):
            self.stage()
        self.assertEqual(self.read_job()["state"], "failed")
        self.assertFalse(self.calls)

    def test_path_traversal_absolute_and_windows_paths_are_rejected(self):
        for name in ("../escape.rs", "/tmp/escape", "C:/escape.rs", "src/../escape.rs", "src\\escape.rs", "docs/image.png"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                deployment.payload_path(name)

    def test_links_and_duplicate_entries_are_rejected(self):
        self.package()
        original = self.archive.read_bytes()
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, "duplicate"):
            self.archive.write_bytes(original)
            def mutate(entries):
                index = next(i for i, (info, _) in enumerate(entries) if info.name == "src/main.rs")
                if kind == "duplicate":
                    entries.append(copy.deepcopy(entries[index]))
                else:
                    info, _ = entries[index]
                    info.type, info.linkname, info.size = kind, "/tmp/escape", 0
                    entries[index] = info, None
            self.rewrite(mutate)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                deployment.extract_verified(self.archive, self.directory/str(kind), self.identity)

    def test_missing_or_undeclared_archive_members_are_rejected(self):
        self.package()
        original = self.archive.read_bytes()
        for extra in (False, True):
            self.archive.write_bytes(original)
            def mutate(entries):
                if extra:
                    info = tarfile.TarInfo("src/extra.rs")
                    info.mode = 0o644
                    entries.append((info, b"extra"))
                else:
                    entries.pop()
            self.rewrite(mutate)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                deployment.extract_verified(self.archive, self.directory/("extra" if extra else "missing"), self.identity)

    def test_disk_full_is_reported_before_extraction(self):
        self.package()
        with mock.patch.object(deployment.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError, "free space"):
                self.stage()
        self.assertFalse((self.root/"staging"/self.identity).exists())

    def test_manifest_identity_profile_modes_and_digest_are_checked(self):
        self.package()
        with tarfile.open(self.archive) as bundle:
            original = json.load(bundle.extractfile("manifest.json"))
        for change in (dict(release_id="other"), dict(features=["gpu"]), dict(schema_version=True),
                       dict(snapshot_sha256="0"*64)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                deployment.validate_manifest(dict(original, **change), self.identity)
        invalid = copy.deepcopy(original)
        invalid["files"]["src/main.rs"]["mode"] = 0o4777
        with self.assertRaises(ValueError):
            deployment.validate_manifest(invalid, self.identity)

    def test_truncated_archive_never_publishes_release(self):
        self.package()
        data = self.archive.read_bytes()
        self.archive.write_bytes(data[:len(data)//2])
        with self.assertRaises((EOFError, tarfile.TarError, ValueError)):
            self.stage()
        self.assertEqual(self.read_job()["state"], "failed")
        self.assertFalse((self.root/"releases"/self.identity).exists())

    def test_failed_build_retains_diagnostics_without_publishing_release(self):
        self.package()
        def failed(command, log, env):
            if "--build" in command:
                log.write("simulated compiler failure\n")
                log.flush()
                raise subprocess.CalledProcessError(1, command)
            self.fake_runner(command, log, env)
        with self.assertRaises(subprocess.CalledProcessError):
            self.stage(runner=failed)
        self.assertEqual(self.read_job()["state"], "failed")
        self.assertFalse((self.root/"releases"/self.identity).exists())
        self.assertIn("compiler failure", (self.root/"jobs"/(self.identity+".log")).read_text())

    def test_mutation_during_build_cannot_be_published(self):
        self.package()
        def mutate(command, log, env):
            self.fake_runner(command, log, env)
            if "--build" in command:
                path = Path(command[1]).parents[1]/"config/config.json"
                value = runtime.read_json(path)
                value["camera"]["fx"] += 1
                runtime.atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, "payload changed"):
            self.stage(runner=mutate)
        self.assertFalse((self.root/"releases"/self.identity).exists())

    def test_existing_ready_job_and_release_cannot_be_overwritten(self):
        self.package()
        self.stage()
        before = self.read_job()
        with self.assertRaises(FileExistsError):
            self.stage()
        self.assertEqual(self.read_job(), before)

    def test_busy_device_lock_rejects_other_release_without_build(self):
        self.package()
        with deployment.deployment_lock(self.lock):
            with self.assertRaises(deployment.LockBusy):
                self.stage()
        self.assertEqual(self.read_job()["state"], "failed")
        self.assertFalse(self.calls)

    def test_lock_works_between_processes_and_releases_after_exit(self):
        program = """import sys
sys.path.insert(0, sys.argv[1])
from deploy_release import deployment_lock, LockBusy
try:
    with deployment_lock(sys.argv[2]): print('acquired')
except LockBusy: print('busy')
"""
        command = [sys.executable, "-c", program, str(runtime.ROOT/"scripts"), str(self.lock)]
        with deployment.deployment_lock(self.lock):
            self.assertEqual(subprocess.check_output(command, text=True).strip(), "busy")
        self.assertEqual(subprocess.check_output(command, text=True).strip(), "acquired")

    def test_os_releases_lock_after_controller_crash(self):
        program = """import sys
sys.path.insert(0, sys.argv[1])
from deploy_release import deployment_lock
with deployment_lock(sys.argv[2]):
    print('ready', flush=True)
    sys.stdin.read()
"""
        process = subprocess.Popen([sys.executable, "-c", program, str(runtime.ROOT/"scripts"), str(self.lock)],
                                   stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            process.terminate()
            process.wait(timeout=5)
            with deployment.deployment_lock(self.lock):
                pass
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stdin.close()


if __name__ == "__main__":
    unittest.main()
