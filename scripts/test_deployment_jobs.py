"""Supervisor submission and reconnect reconciliation without a Linux manager."""
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import deploy_release as jobs


class JobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()/"managed"
        self.identity = "release-test"
        self.calls = []
        self.linger = "yes"
        self.state = "active"
        self.result = "success"
        self.code = 0
        self.load = "loaded"
        patch = mock.patch.object(jobs, "user_systemd_env", return_value={"TEST_MANAGER": "1"})
        patch.start()
        self.addCleanup(patch.stop)
        uid = mock.patch.object(jobs.os, "getuid", return_value=1000, create=True)
        uid.start()
        self.addCleanup(uid.stop)

    def execute(self, command, **kwargs):
        self.calls.append((command, kwargs))
        output = ""
        if command[0] == "loginctl": output = self.linger+"\n"
        if command[:3] == ["systemctl", "--user", "show"]:
            output = f"ActiveState={self.state}\nResult={self.result}\nExecMainStatus=9\nLoadState={self.load}\n"
        return subprocess.CompletedProcess(command, self.code, output, "")

    def start(self, kind="activate", **kwargs):
        return jobs.start_job(self.root, self.identity, kind, release="candidate", execute=self.execute, **kwargs)

    def report(self, state="verifying"):
        return jobs.supervised_status(self.root, self.identity, dict(state=state), execute=self.execute)

    def test_submission_owns_control_group_logs_and_timeout_without_ssh(self):
        submitted = self.start()
        command, options = self.calls[-1]
        self.assertEqual(command[0], "systemd-run")
        self.assertIn("--user", command)
        self.assertIn("--property=Type=exec", command)
        self.assertIn("--property=Restart=no", command)
        self.assertIn("--property=KillMode=control-group", command)
        self.assertIn("--property=RuntimeMaxSec=300", command)
        self.assertIn("--operation-id", command)
        self.assertEqual(options["env"], {"TEST_MANAGER": "1"})
        self.assertEqual(submitted["dispatch"], "submitted")
        self.assertEqual(self.report()["unit_state"], "active")

    def test_duplicate_submission_cannot_restart_or_replace_a_job(self):
        self.start()
        count = len([c for c, _ in self.calls if c[0] == "systemd-run"])
        with self.assertRaises(FileExistsError): self.start()
        self.assertEqual(len([c for c, _ in self.calls if c[0] == "systemd-run"]), count)

    def test_lingering_required_before_recording_or_starting_a_job(self):
        self.linger = "no"
        with self.assertRaisesRegex(ValueError, "enable-linger"): self.start()
        self.assertFalse((self.root/"jobs").exists())
        self.assertFalse(any(c[0] == "systemd-run" for c, _ in self.calls))

    def test_lost_submission_reply_preserves_identity_and_does_not_retry(self):
        execute = self.execute
        def lost(command, **kwargs):
            if command[0] == "systemd-run": raise subprocess.TimeoutExpired(command, 30)
            return execute(command, **kwargs)
        with self.assertRaises(subprocess.TimeoutExpired):
            jobs.start_job(self.root, self.identity, "activate", release="candidate", execute=lost)
        self.assertEqual(self.report()["state"], "verifying")
        with self.assertRaises(FileExistsError): self.start()

    def test_worker_crash_timeout_and_reboot_are_interrupted_without_false_success(self):
        self.start()
        for state, load, code in (("failed", "loaded", 0), ("inactive", "not-found", 1)):
            self.state, self.load, self.code = state, load, code
            self.result = "timeout"
            report = self.report()
            self.assertEqual(report["state"], "interrupted")
            self.assertEqual(report["unit_result"], "timeout")
            self.assertFalse((self.root/"activation.json").exists())

    def test_saved_terminal_result_survives_manager_cleanup(self):
        self.start()
        self.state, self.load, self.code = "inactive", "not-found", 1
        for state in ("ready", "active", "rolled_back", "rollback_failed", "recovered", "failed"):
            self.assertEqual(self.report(state)["state"], state)

    def test_manager_connection_failure_does_not_label_live_job_interrupted(self):
        self.start()
        def unavailable(command, **kwargs): return subprocess.CompletedProcess(command, 1, "", "bus unavailable")
        with self.assertRaisesRegex(ValueError, "Cannot inspect"):
            jobs.supervised_status(self.root, self.identity, {"state": "verifying"}, execute=unavailable)

    def test_bounded_compiler_log_is_restored_after_reopening(self):
        self.start()
        (self.root/"jobs"/(self.identity+".log")).write_bytes(b"a"*20000+b"compiler failed\n")
        report = self.report()
        self.assertLessEqual(len(report["log_tail"].encode()), 16384)
        self.assertTrue(report["log_tail"].endswith("compiler failed\n"))

    def test_different_roots_get_different_units(self):
        first = self.start()["unit"]
        self.root = self.root/"other"
        self.assertNotEqual(first, self.start()["unit"])

    def test_invalid_kind_release_and_unsafe_root_never_submit(self):
        with self.assertRaises(ValueError): self.start("delete")
        with self.assertRaises(ValueError):
            jobs.start_job(self.root, self.identity, "activate", release="../bad", execute=self.execute)
        self.root = self.root/"unsafe%h"
        with self.assertRaises(ValueError): self.start()
        self.assertFalse(any(c[0] == "systemd-run" for c, _ in self.calls))

    def test_stage_uses_verified_controller_and_archive_not_an_arbitrary_command(self):
        incoming = self.root/"incoming"/self.identity
        incoming.mkdir(parents=True)
        controller = incoming/"controller.py"
        with mock.patch.object(jobs, "__file__", str(controller)):
            jobs.start_job(self.root, self.identity, "stage", release=self.identity,
                           checksum="a"*64, execute=self.execute)
        command = self.calls[-1][0]
        self.assertIn(str(incoming/"release.tar.gz"), command)
        self.assertIn("--property=RuntimeMaxSec=7200", command)
        self.assertIn("--with-debug-bridge", command)

    def test_crash_during_dispatch_only_becomes_interrupted_after_submission_grace(self):
        self.start()
        path = self.root/"jobs"/(self.identity+".unit.json")
        descriptor = jobs.read_json(path)
        descriptor.update(dispatch="submitting", submitted_at=time.time())
        jobs.atomic_json(path, descriptor)
        self.state = "inactive"
        self.assertEqual(self.report()["state"], "verifying")
        descriptor["submitted_at"] -= 31
        jobs.atomic_json(path, descriptor)
        self.assertEqual(self.report()["state"], "interrupted")


if __name__ == "__main__": unittest.main()
