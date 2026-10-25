# Deployment rehearsal without robot hardware

Run from Ubuntu 24.04 x64, including WSL2. The runner creates a fresh Ubuntu cloud
VM, blocks its outbound network access, and runs the production controller.
The host needs internet for the image and offline bundle preparation.

```sh
sudo apt-get install qemu-system-x86 qemu-utils cloud-image-utils jq openssh-client curl acl
rustup toolchain install 1.96.0 --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
# Optional hardware acceleration; otherwise QEMU uses slower software emulation.
test ! -e /dev/kvm || sudo setfacl -m u:$(id -un):rw /dev/kvm
bash deployment/rehearsal/run.sh
```

The host must run the same Ubuntu version and architecture as the guest. It needs
about 8 GiB available memory and 15 GiB disk space. Host package preparation uses
sudo to download dependencies, without installing them. The guest installs them.
To reuse an existing matching CPU bundle, set
`VORTEX_VM_BUNDLE=/absolute/path/offline.tar.gz`; keep `offline.json` beside it.

The rehearsal verifies:

- Real detached initialization on an empty account: offline package installation,
  compiler/cache restore, actual CPU vision and debug bridge build, saved-image
  smoke test, and managed service installation.
- Service permissions, detached activation jobs, manual rollback, automatic
  rollback after a failed candidate, and watchdog restart using the native fixture.
- Service recovery after a real guest reboot, followed by interrupted-activation
  recovery across another real reboot. Both phases require a changed boot ID and
  fresh process health.
- Removal of the dedicated test accounts and services.

Reports, serial output, guest build/job logs and VM disks remain in
`~/.cache/vortex-vm/run-*`. `result.json` appears only after success. Failed runs
retain evidence; remove their directories when no longer needed. The VM is stopped
on exit. Override the cache and SSH port with `VORTEX_VM_CACHE` and
`VORTEX_VM_PORT` when needed. Do not run two rehearsals on the same port.

GitHub's **Clean Ubuntu deployment rehearsal** workflow runs this on pull requests
or manual dispatch and uploads logs and the result. No VM disk, SSH key, compiler
cache or offline bundle is uploaded as an artifact.

This exercises x64 CPU deployment. Camera health in lifecycle/reboot tests comes
from a fixture; actual camera capture, Jetson ARM64, GPU/TensorRT and physical
power-loss durability still require hardware. Guest reboots are orderly reboots,
with a deliberately written interrupted-activation journal for the recovery case.
