"""Read production snapshots without starting/stopping a service or a camera."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import activate_release as activation
import deploy_release as deployment


class ReadOnlyService:
    def __init__(self): self.active, self.pid = True, 44
    def info(self): return dict(active=self.active, pid=self.pid, state="active", restarts=2)
    def start(self): raise AssertionError("Diagnostics must not start vision")
    def stop(self): raise AssertionError("Diagnostics must not stop vision")


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        directory = self.root/"releases/active"
        directory.mkdir(parents=True)
        files = {key: dict(size=0, sha256="a"*64, mode=0o644) for key in
                 deployment.ROOT_FILES | deployment.SCRIPT_FILES | {"config/device.json", "config/config.json"}}
        deployment.atomic_json(directory/"manifest.json", dict(schema_version=1, release_id="active", build_profile="cpu", features=[], files=files,
            snapshot_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()))
        deployment.atomic_json(directory/"release.json", dict(release_id="active", profile="cpu"))
        self.selection = dict(kind="release", release_id="active", launch_token="nonce")
        deployment.atomic_json(self.root/"selection.json", self.selection)
        self.report = dict(schema_version=1, diagnostics_version=1, timestamp_ms=100000, pid=44, release_id="active", launch_token="nonce",
                           watchdog="healthy", cameras=[dict(index=0, fps=50, frames=100, pipeline_ok=True), dict(index=2, fps=40)])
        deployment.atomic_json(self.root/"health.json", self.report)
        self.directory = self.root/"diagnostics"
        self.directory.mkdir()
        self.frame = dict(schema_version=1, timestamp_ms=100000, latency_ms=30, camera_index=0, client_id="client-one",
                          pid=44, release_id="active", launch_token="nonce", frame_id="frame-44-7.jpg",
                          apriltags=[dict(id=7)], objects=[])
        self.write_frame(self.frame)
        self.jpeg = b"\xff\xd8\xff\xe0paired-frame"
        (self.directory/"frame-44-7.jpg").write_bytes(self.jpeg)
        self.service = ReadOnlyService()

    def write_frame(self, value): deployment.atomic_json(self.directory/"preview.json", value)
    def snapshot(self, camera=0, client="client-one", **kwargs):
        return activation.diagnostics(self.root, camera, client, manager=self.service, clock=lambda: 100., **kwargs)

    def test_frame_and_results_are_paired_and_only_a_short_lease_is_written(self):
        result = self.snapshot()
        self.assertTrue(result["ready"])
        self.assertEqual(result["state"]["apriltags"][0]["id"], 7)
        self.assertEqual(result["state"]["frame_id"], "frame-44-7.jpg")
        self.assertEqual(result["state"]["fps"], 50)
        self.assertEqual(deployment.read_json(self.directory/"request.json")["expires_ms"], 103000)
        self.assertEqual(deployment.read_json(self.root/"selection.json"), self.selection)

    def test_restart_wrong_camera_wrong_client_and_old_frames_are_never_displayed(self):
        for change in (dict(pid=9), dict(launch_token="old"), dict(release_id="old"), dict(camera_index=2),
                       dict(client_id="other"), dict(timestamp_ms=98000), dict(latency_ms=2000), dict(schema_version=2)):
            self.write_frame(dict(self.frame, **change))
            with self.subTest(change=change): self.assertFalse(self.snapshot()["ready"])

    def test_total_preview_age_includes_queue_processing_and_time_since_publication(self):
        self.write_frame(dict(self.frame, timestamp_ms=99000, latency_ms=700))
        self.assertFalse(self.snapshot()["ready"])

    def test_stale_health_or_older_runtime_does_not_enable_preview(self):
        for change in (dict(pid=1), dict(timestamp_ms=98000), dict(diagnostics_version=0), dict(schema_version=2)):
            deployment.atomic_json(self.root/"health.json", dict(self.report, **change))
            self.assertFalse(self.snapshot()["ready"])
            self.assertFalse((self.directory/"request.json").exists())

    def test_service_restart_during_read_discards_the_image(self):
        count = [0]
        def info():
            count[0] += 1
            return dict(active=True, pid=44 if count[0] == 1 else 45)
        self.service.info = info
        self.assertFalse(self.snapshot()["ready"])

    def test_missing_rotated_file_is_a_temporary_waiting_state(self):
        (self.directory/"frame-44-7.jpg").unlink()
        self.assertFalse(self.snapshot()["ready"])

    def test_unconfigured_camera_and_unsafe_filename_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not configured"): self.snapshot(camera=63)
        self.write_frame(dict(self.frame, frame_id="../../secret.jpg"))
        with self.assertRaisesRegex(ValueError, "frame path"): self.snapshot()

    def test_preview_image_size_limit_is_enforced(self):
        (self.directory/"frame-44-7.jpg").write_bytes(b"\xff\xd8"+b"x"*524288)
        with self.assertRaisesRegex(ValueError, "limits"): self.snapshot()

    def test_only_lease_owner_can_detach_and_other_clients_wait_until_expiry(self):
        self.snapshot()
        result = self.snapshot(client="second")
        self.assertFalse(result["ready"])
        self.assertIn("Another", result["reason"])
        self.snapshot(client="second", stop=True)
        self.assertTrue((self.directory/"request.json").exists())
        self.snapshot(stop=True)
        self.assertFalse((self.directory/"request.json").exists())

    def test_dead_service_does_not_show_old_frame_or_rewrite_lease(self):
        self.service.active = False
        self.assertFalse(self.snapshot()["ready"])
        self.assertFalse((self.directory/"request.json").exists())

    def test_empty_detection_list_is_a_valid_processed_preview(self):
        self.write_frame(dict(self.frame, apriltags=[]))
        self.assertTrue(self.snapshot()["ready"])


if __name__ == "__main__": unittest.main()
