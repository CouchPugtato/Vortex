# Deployment validation

The opt-in Linux test uses actual systemd, loginctl, sudoers and native processes.
It creates a unique `vortex-test-*` account and service and removes its resources
on completion. Run it on a disposable Ubuntu VM or WSL systemd distribution as
root; it never opens cameras or changes the production Vortex service.

```sh
cd deployment
cargo build --example service_fixture --locked
cargo test --test systemd --no-run --message-format=json > target/systemd-build.json
TEST_BINARY=$(jq -r 'select(.reason == "compiler-artifact" and .target.name == "systemd") | .executable // empty' target/systemd-build.json)
sudo env VORTEX_SERVICE_FIXTURE="$PWD/target/debug/examples/service_fixture" "$TEST_BINARY" --ignored --nocapture
```

This covers installation, restricted service control, jobs surviving the submitting
process, activation, manual rollback, failed candidate rollback, watchdog restart
and recovery when service startup finds an interrupted activation. It simulates
an old boot identity; it does not reboot the host. CI runs this test on Ubuntu.

To additionally test an actual offline bundle, pass
`VORTEX_OFFLINE_BUNDLE=/absolute/path/offline.tar.gz` in the sudo environment above.
The test checks package dependency resolution without installing host packages,
restores Rust and Cargo into the new account, and runs a locked offline Cargo fetch
inside a network namespace with no network access. `unshare` support is required.

## On the Jetson

Use **Initialize Jetson**, **Activate**, detach diagnostics, then choose
**Deployment maintenance → Run Device Check**. The 150-second check runs the
saved-image smoke test, verifies fresh health and required CPU/GPU/TensorRT backends,
checks every configured camera advances, and requires a fresh processed preview
from each camera. It fails if the release or main PID changes. The result includes
the device model, boot ID and maximum frame/preview ages and is printed in the log.

For operators using a terminal, the equivalent controller command is:

```sh
/path/to/Vortex.deployment/vortexctl validate-device --root /path/to/Vortex.deployment --duration-seconds 150
```

Hardware acceptance still requires these observations:

1. Run the check for each intended camera count, build profile and JetPack image.
2. Repeat while staging another release; compare FPS and preview age under load.
3. Unplug a camera, confirm watchdog restart and useful diagnostics, then reconnect.
4. Activate a deliberately invalid configuration and verify the previous release recovers.
5. Reboot and power-cycle during a test deployment, then verify selected release,
   recovery state and camera health after reconnection.

Perform fault injection on a test installation. A passing device check establishes
runtime and preview liveness; it does not certify pose accuracy, GPU equivalence,
physical power-loss durability or camera-to-desktop display latency.
