"""Launcher contract tests; never open cameras, build Rust, or contact a Jetson."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import run_vortex as runtime


CONFIG = json.loads((runtime.ROOT/"config/device.json").read_text())


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(CONFIG)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/"config").mkdir()
        (self.root/"src").mkdir()
        (self.root/"src/main.rs").write_text("fn main() {}")
        for name in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml"):
            (self.root/name).write_text("test")
        (self.root/"rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.96.0"\n')
        (self.root/"config/device.json").write_text(json.dumps(self.config))
        self.runtime = {"camera": dict(fx=500., fy=500., cx=320., cy=240., k1=0., k2=0.,
                                       k3=0., p1=0., p2=0., tag_size_m=.16)}
        self.field = {"tags": [{"ID": 1, "pose": {"translation": dict(x=0., y=0., z=1.),
                                                "rotation": {"quaternion": dict(W=1., X=0., Y=0., Z=0.)}}}]}
        (self.root/self.config["runtime_config"]).write_text(json.dumps(self.runtime))
        (self.root/self.config["tag_map"]).write_text(json.dumps(self.field))

    def fake_build(self, command, **kwargs):
        for name in (runtime.MAIN, runtime.BRIDGE):
            binary = runtime.binary_path(self.root, name)
            if "--target-dir" in command:
                binary = Path(command[command.index("--target-dir")+1])/"release"/binary.name
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b"executable")
            binary.chmod(0o755)

    def build(self, **kwargs):
        with mock.patch.object(runtime.subprocess, "run", side_effect=self.fake_build) as run:
            runtime.build(self.root, self.config, **kwargs)
        return run.call_args

    def test_team_nt_and_udp_follow_settings_not_old_environment(self):
        self.config["networktables"].update(enabled=True, mode="team", team=1234)
        self.config["udp"].update(target="robot.local", port=5092)
        env = runtime.environment(self.root, self.config,
                                  {"VORTEX_NT_SERVER": "wrong", "VORTEX_UDP_PORT": "1", "KEEP": "yes"})
        self.assertEqual(env["VORTEX_NT_TEAM"], "1234")
        self.assertNotIn("VORTEX_NT_SERVER", env)
        self.assertEqual((env["VORTEX_UDP_TARGET"], env["VORTEX_UDP_PORT"]), ("robot.local", "5092"))
        self.assertEqual(env["KEEP"], "yes")

    def test_custom_and_local_nt_do_not_inherit_team(self):
        for mode, server in (("custom", "roborio.local"), ("local", "127.0.0.1")):
            with self.subTest(mode=mode):
                self.config["networktables"].update(enabled=True, mode=mode, server=server)
                env = runtime.environment(self.root, self.config, {"VORTEX_NT_TEAM": "999"})
                self.assertNotIn("VORTEX_NT_TEAM", env)
                self.assertEqual(env["VORTEX_NT_SERVER"], server)

    def test_disabled_networking_clears_inherited_flags(self):
        self.config["networktables"]["enabled"] = False
        self.config["udp"]["enabled"] = False
        env = runtime.environment(self.root, self.config, {"VORTEX_NT_TEAM": "999", "VORTEX_NT_SERVER": "wrong"})
        self.assertEqual(env["VORTEX_NT_ENABLE"], "0")
        self.assertEqual(env["VORTEX_UDP_ENABLE"], "0")
        self.assertNotIn("VORTEX_NT_TEAM", env)
        self.assertNotIn("VORTEX_NT_SERVER", env)

    def test_invalid_settings_are_rejected_without_coercion(self):
        changes = [dict(schema_version=True), dict(schema_version=2), dict(runtime="orin_bridge"),
                   dict(build_profile="automatic"), dict(build_profile=[]), dict(cameras=[]),
                   dict(cameras=[0, 0]), dict(cameras=[True]), dict(cameras=[64]), dict(cameras=[0.5]),
                   dict(cameras=["0"]), dict(runtime_config="x\ncommand")]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runtime.validate_config(dict(self.config, **change))

    def test_invalid_networking_is_rejected(self):
        for section, change in [("udp", dict(port=0)), ("udp", dict(port=65536)),
                                ("udp", dict(enabled="false")), ("udp", dict(target="")),
                                ("networktables", dict(team=True)), ("networktables", dict(table="relative")),
                                ("networktables", dict(enabled=True, mode="custom", server=""))]:
            value = copy.deepcopy(self.config)
            value[section].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                runtime.validate_config(value)

    def test_resources_and_nonfinite_json_fail_before_launch(self):
        runtime.validate_resources(self.root, self.config)
        for data in ('{"camera": {"fx": NaN}}', '{"camera": {"fx": 1e999}}', '{"not_camera": {}}'):
            (self.root/self.config["runtime_config"]).write_text(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                runtime.validate_resources(self.root, self.config)

    def test_default_build_is_pinned_locked_offline_and_explicit_cpu(self):
        call = self.build()
        command = call.args[0]
        self.assertEqual(command[1], "+1.96.0")
        for flag in ("--release", "--locked", "--offline", "--no-default-features"):
            self.assertIn(flag, command)
        self.assertNotIn("--features", command)
        self.assertEqual(call.kwargs["cwd"], self.root)
        self.assertEqual(call.kwargs["env"]["RUSTUP_AUTO_INSTALL"], "0")
        for name in ("CARGO_TARGET_DIR", "CARGO_BUILD_TARGET", "RUSTUP_TOOLCHAIN"):
            self.assertNotIn(name, call.kwargs["env"])
        self.assertEqual(runtime.checked_binary(self.root, self.config), runtime.binary_path(self.root, runtime.MAIN))

    def test_invalid_camera_and_object_settings_are_rejected(self):
        for value in ({"camera": {}}, {"camera": dict(self.runtime["camera"], fx=0)},
                      dict(self.runtime, processing={"sensor_gain": "invalid"}),
                      dict(self.runtime, object_detection={"use_nn": "false"})):
            (self.root/self.config["runtime_config"]).write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.validate_resources(self.root, self.config)

    def test_duplicate_tags_and_invalid_map_geometry_are_rejected(self):
        for value in ({"tags": []}, {"tags": self.field["tags"]*2}, {"tags": [{"ID": 1}]},
                      {"tags": [{"ID": 1, "pose": {"translation": dict(x=0, y=0, z=0),
                                 "rotation": {"quaternion": dict(W=0, X=0, Y=0, Z=0)}}}]}):
            (self.root/self.config["tag_map"]).write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.validate_resources(self.root, self.config)

    def test_build_profiles_never_silently_fall_back(self):
        for profile, features in runtime.PROFILES.items():
            self.config["build_profile"] = profile
            command = self.build().args[0]
            if features:
                self.assertEqual(command[command.index("--features")+1], ",".join(features))
            else:
                self.assertNotIn("--features", command)

    def test_online_and_debug_build_require_explicit_flags(self):
        call = self.build(online=True, with_debug_bridge=True)
        command = call.args[0]
        self.assertNotIn("--offline", command)
        self.assertEqual(call.kwargs["env"]["RUSTUP_AUTO_INSTALL"], "1")
        self.assertIn(runtime.BRIDGE, command)
        runtime.checked_binary(self.root, self.config, runtime.BRIDGE)

    def test_failed_build_invalidates_old_stamp_and_does_not_retry(self):
        self.build()
        with mock.patch.object(runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "cargo")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.build(self.root, self.config)
            self.assertEqual(run.call_count, 1)
        with self.assertRaisesRegex(ValueError, "Verified executable missing"):
            runtime.checked_binary(self.root, self.config)

    def test_changed_sources_profile_or_binary_are_rejected(self):
        self.build()
        different = dict(self.config, build_profile="gpu")
        with self.assertRaisesRegex(ValueError, "stale"):
            runtime.checked_binary(self.root, different)
        binary = runtime.binary_path(self.root, runtime.MAIN)
        binary.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "stale"):
            runtime.checked_binary(self.root, self.config)
        self.build()
        (self.root/"src/main.rs").write_text("changed")
        with self.assertRaisesRegex(ValueError, "stale"):
            runtime.checked_binary(self.root, self.config)

    def test_config_edits_do_not_force_recompile(self):
        self.build()
        self.config["cameras"] = [1, 3]
        self.config["udp"]["port"] = 9999
        runtime.checked_binary(self.root, self.config)

    def test_cache_build_copies_private_artifacts_into_release(self):
        cache = self.root/"external-cache"
        call = self.build(cache_dir=cache)
        self.assertEqual(call.args[0][call.args[0].index("--target-dir")+1], str(cache.resolve()))
        binary = runtime.checked_binary(self.root, self.config)
        (cache/"release"/binary.name).write_bytes(b"next build")
        self.assertEqual(binary.read_bytes(), b"executable")
        runtime.checked_binary(self.root, self.config)

    def test_changed_sources_during_build_are_not_stamped(self):
        def change_source(command, **kwargs):
            self.fake_build(command, **kwargs)
            (self.root/"src/main.rs").write_text("changed during build")
        with mock.patch.object(runtime.subprocess, "run", side_effect=change_source):
            with self.assertRaisesRegex(ValueError, "during build"):
                runtime.build(self.root, self.config)
        self.assertFalse((self.root/"target/release/vortex-build.json").exists())

    def test_production_launch_uses_configured_cameras_and_never_builds(self):
        self.build()
        with mock.patch.object(runtime, "ROOT", self.root), mock.patch.object(runtime.os, "chdir"), \
                mock.patch.object(runtime.os, "execve") as execute, mock.patch.object(runtime.subprocess, "run") as build:
            runtime.main([])
        self.assertEqual(execute.call_args.args[1][1], "0,2")
        self.assertEqual(execute.call_args.args[2]["VORTEX_UDP_PORT"], "5091")
        build.assert_not_called()

    def test_missing_verified_build_is_an_error_not_implicit_compilation(self):
        with mock.patch.object(runtime, "ROOT", self.root), mock.patch.object(runtime.os, "execve") as execute, \
                mock.patch.object(runtime.subprocess, "run") as build:
            self.assertEqual(runtime.main([]), 1)
        execute.assert_not_called()
        build.assert_not_called()

    def test_check_debug_build_does_not_open_cameras(self):
        self.build(with_debug_bridge=True)
        with mock.patch.object(runtime, "ROOT", self.root), mock.patch.object(runtime.os, "execve") as execute:
            self.assertEqual(runtime.main(["--check-debug-bridge"]), 0)
        execute.assert_not_called()

    def test_legacy_bridge_is_only_launched_explicitly(self):
        self.build(with_debug_bridge=True)
        with mock.patch.object(runtime, "ROOT", self.root), mock.patch.object(runtime.os, "chdir"), \
                mock.patch.object(runtime.os, "execve") as execute:
            runtime.main(["--debug-bridge", "--camera", "2"])
        self.assertEqual(Path(execute.call_args.args[0]).stem, runtime.BRIDGE)
        self.assertEqual(execute.call_args.args[2]["VORTEX_BRIDGE_CAMERA"], "2")
        self.assertEqual(execute.call_args.args[2]["VORTEX_BRIDGE_CONFIG"], str(self.root/"config/config.json"))

    def test_atomic_json_replaces_config_without_leaking_temp_files(self):
        path = self.root/"config/device.json"
        runtime.atomic_json(path, dict(version=2))
        self.assertEqual(json.loads(path.read_text()), dict(version=2))
        self.assertFalse(list(path.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
