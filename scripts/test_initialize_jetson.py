"""Initialization failure injection; never contacts a device or installs packages."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import deploy_release as deployment
import initialize_jetson as initialization


class InitializationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.legacy = self.directory/"Vortex"
        self.root = self.directory/"Vortex.deployment"
        self.home = self.directory/"account"
        self.home.mkdir()
        self.etc = self.directory/"etc"
        self.etc.mkdir()
        (self.etc/"os-release").write_text('ID=ubuntu\nVERSION_ID="22.04"\n')
        self.account = SimpleNamespace(pw_uid=1000, pw_gid=1000, pw_dir=str(self.home))
        self.calls = []
        self.native = mock.Mock(return_value={"passed": True})
        self.install = mock.Mock(return_value={"installed": True})
        self.activation = SimpleNamespace(native_preflight=self.native, install_service=self.install,
                                          status=lambda root: {"service_configured": True})
        for patch in (mock.patch.object(initialization.sys, "platform", "linux"),
                      mock.patch.object(initialization.os, "geteuid", return_value=0, create=True),
                      mock.patch.object(initialization.os, "chown", create=True),
                      mock.patch.dict(sys.modules, {"pwd": SimpleNamespace(getpwnam=lambda user: self.account),
                                                   "activate_release": self.activation})):
            patch.start()
            self.addCleanup(patch.stop)

    def package(self):
        source = self.directory/"source"
        repo = Path(__file__).resolve().parent.parent
        for relative in deployment.ROOT_FILES | deployment.SCRIPT_FILES | {
                "config/device.json", "config/config.json", "config/2026-rebuilt-welded.json"}:
            target = source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/relative, target)
        (source/"src").mkdir()
        (source/"src/main.rs").write_text("fn main() {}")
        output = self.directory/"package"
        metadata = deployment.package_release(source, output, source/"config/device.json")
        incoming = self.root/"incoming"/metadata["release_id"]
        incoming.mkdir(parents=True)
        shutil.copyfile(output/"release.tar.gz", incoming/"release.tar.gz")
        (self.root/"jobs").mkdir()
        return SimpleNamespace(root=self.root, legacy_root=self.legacy, user="vortex", service="vortex-vision.service",
                               job_id="initialize-test", release_id=metadata["release_id"], sha256=metadata["archive_sha256"])

    def execute(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if "stage" in command:
            (self.root/"releases"/self.args.release_id).mkdir(parents=True)
        if "preflight" in command:
            self.native(self.root, self.args.release_id)
        return subprocess.CompletedProcess(command, 0, "", "")

    def run_initialization(self, args):
        self.args = args
        with mock.patch.object(initialization, "compatibility", return_value={"architecture": "aarch64"}), \
                mock.patch.object(initialization.shutil, "disk_usage", return_value=SimpleNamespace(free=8*1024**3)):
            return initialization.initialize(args, execute=self.execute)

    def test_cpu_compatibility_requires_supported_ubuntu_architecture_and_systemd(self):
        kwargs = dict(system="Linux", machine="aarch64", etc=self.etc, exists=lambda path: path == "/run/systemd/system")
        self.assertEqual(initialization.compatibility("cpu", **kwargs)["ubuntu"], "22.04")
        for change in (dict(system="Windows"), dict(machine="armv7l"), dict(exists=lambda path: False)):
            with self.assertRaises(ValueError): initialization.compatibility("cpu", **dict(kwargs, **change))
        (self.etc/"os-release").write_text('ID=ubuntu\nVERSION_ID="18.04"\n')
        with self.assertRaisesRegex(ValueError, "Ubuntu"): initialization.compatibility("cpu", **kwargs)

    def test_gpu_and_tensorrt_require_preinstalled_matching_jetpack_components(self):
        base = dict(system="Linux", machine="aarch64", etc=self.etc)
        for profile in ("gpu", "tensorrt", "gpu-tensorrt"):
            with self.assertRaisesRegex(ValueError, "JetPack"):
                initialization.compatibility(profile, **base, exists=lambda path: path == "/run/systemd/system")
            self.assertEqual(initialization.compatibility(profile, **base, exists=lambda path: True)["profile"], profile)

    def test_fresh_device_is_provisioned_built_and_verified_without_activation(self):
        args = self.package()
        result = self.run_initialization(args)
        self.assertEqual(result["state"], "ready")
        self.assertFalse(result["activated"])
        self.assertEqual(result["verification"], "dependencies_build_smoke_service")
        commands = [command for command, _ in self.calls]
        self.assertTrue(any(command[0] == "apt-get" for command in commands))
        self.assertTrue(any(command[:2] == ["loginctl", "enable-linger"] for command in commands))
        self.assertTrue(any("fetch" in command and "--locked" in command for command in commands))
        stage = next(command for command in commands if "stage" in command)
        self.assertEqual(stage[:3], ["runuser", "-u", "vortex"])
        self.assertFalse(any("activate" in command or "restart" in command or "reboot" in command for command in commands))
        self.native.assert_called_once_with(self.root, args.release_id)
        self.install.assert_called_once()

    def test_bad_archive_cannot_install_dependencies_or_modify_service(self):
        args = self.package()
        args.sha256 = "f"*64
        with self.assertRaisesRegex(ValueError, "checksum"): self.run_initialization(args)
        self.assertEqual(self.calls, [])
        self.install.assert_not_called()
        self.assertEqual(deployment.read_json(self.root/"jobs/initialize-test.json")["state"], "failed")

    def test_package_or_build_failure_keeps_runtime_selection_and_service_untouched(self):
        for failure in ("apt-get", "stage"):
            with self.subTest(failure=failure):
                args = self.package() if failure == "apt-get" else self.args
                if failure == "stage":
                    args.job_id = "initialize-second"
                deployment.atomic_json(self.root/"active.json", {"kind": "legacy", "path": str(self.legacy)})
                original = self.execute
                def fail(command, **kwargs):
                    if failure in command: raise subprocess.CalledProcessError(1, command)
                    return original(command, **kwargs)
                with mock.patch.object(self, "execute", side_effect=fail), self.assertRaises(subprocess.CalledProcessError):
                    self.run_initialization(args)
                self.assertEqual(deployment.read_json(self.root/"active.json")["kind"], "legacy")
                self.install.assert_not_called()

    def test_native_smoke_failure_never_installs_or_claims_ready(self):
        args = self.package()
        self.native.side_effect = ValueError("native smoke failed")
        with self.assertRaisesRegex(ValueError, "smoke"): self.run_initialization(args)
        self.install.assert_not_called()
        self.assertEqual(deployment.read_json(self.root/"jobs/initialize-test.json")["state"], "failed")

    def test_privileged_submission_is_system_managed_without_password_or_ssh_input(self):
        args = self.package()
        with mock.patch.dict(initialization.os.environ, {"SUDO_USER": "vortex"}):
            result = initialization.submit(args, execute=self.execute)
        command, options = self.calls[-1]
        self.assertEqual(command[0], "systemd-run")
        self.assertNotIn("--user", command[:command.index("--")])
        self.assertIn("--property=StandardInput=null", command)
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertEqual(result["manager"], "system")
        with self.assertRaises(FileExistsError): initialization.submit(args, execute=self.execute)

    def test_setup_status_needs_no_lingering_user_manager_or_root_privileges(self):
        args = self.package()
        initialization.submit(args, execute=self.execute)
        def inspect(command, **kwargs):
            self.assertEqual(command[:2], ["systemctl", "show"])
            self.assertNotIn("--user", command)
            return subprocess.CompletedProcess(command, 0, "ActiveState=active\nLoadState=loaded\n", "")
        with mock.patch.object(deployment, "user_systemd_env", side_effect=AssertionError("user manager must not be used")):
            result = deployment.supervised_status(self.root, args.job_id, {"state": "packages"}, execute=inspect)
        self.assertEqual(result["state"], "packages")

    def test_busy_device_rejects_initialization_before_installing(self):
        args = self.package()
        lock = self.home/".local/state/vortex/deployment.lock"
        lock.parent.mkdir(parents=True)
        with deployment.deployment_lock(lock), self.assertRaises(deployment.LockBusy): self.run_initialization(args)
        self.assertEqual(self.calls, [])
        self.install.assert_not_called()

    def test_low_disk_space_is_reported_before_downloads(self):
        args = self.package()
        self.args = args
        with mock.patch.object(initialization, "compatibility", return_value={}), \
                mock.patch.object(initialization.shutil, "disk_usage", return_value=SimpleNamespace(free=3*1024**3)):
            with self.assertRaisesRegex(ValueError, "6 GiB"): initialization.initialize(args, execute=self.execute)
        self.assertFalse(self.calls)


if __name__ == "__main__": unittest.main()
