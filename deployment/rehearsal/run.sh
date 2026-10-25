#!/usr/bin/env bash
# Linux/WSL host. Every package installation below runs inside a disposable guest.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cache=${VORTEX_VM_CACHE:-$HOME/.cache/vortex-vm}
target=${CARGO_TARGET_DIR:-$HOME/.cache/vortexctl-check}
mkdir -p "$cache"
for tool in cargo qemu-system-x86_64 qemu-img cloud-localds ssh scp ssh-keygen curl jq sha256sum; do
  command -v "$tool" >/dev/null || { echo "Missing host tool: $tool" >&2; exit 1; }
done
image=noble-server-cloudimg-amd64.img
url=https://cloud-images.ubuntu.com/noble/current
if [[ ! -f "$cache/$image" ]]; then
  curl -fL --retry 3 "$url/$image" -o "$cache/$image.partial"
  mv "$cache/$image.partial" "$cache/$image"
fi
curl -fsSL "$url/SHA256SUMS" -o "$cache/SHA256SUMS"
(cd "$cache"; awk '$2 ~ /^\*?noble-server-cloudimg-amd64[.]img$/ {print}' SHA256SUMS > image.sha256; test -s image.sha256; sha256sum -c image.sha256)
run=$(mktemp -d "$cache/run-XXXXXXXX")
exec > >(tee "$run/rehearsal.log") 2>&1
echo "Rehearsal artifacts: $run"
cargo +1.96.0 build --manifest-path "$repo/deployment/Cargo.toml" --target-dir "$target" --locked --release --example service_fixture --bin vortexctl
cargo +1.96.0 test --manifest-path "$repo/deployment/Cargo.toml" --target-dir "$target" --locked --release --test systemd --no-run --message-format=json > "$run/build.json"
testbin=$(jq -r 'select(.reason == "compiler-artifact" and .target.name == "systemd") | .executable // empty' "$run/build.json")
test -x "$testbin"
cargo +1.96.0 test --manifest-path "$repo/deployment/Cargo.toml" --target-dir "$target" --locked --release --test network --no-run --message-format=json > "$run/network-build.json"
networktest=$(jq -r 'select(.reason == "compiler-artifact" and .target.name == "network") | .executable // empty' "$run/network-build.json")
bundle=${VORTEX_VM_BUNDLE:-}
if [[ -z "$bundle" ]]; then
  # APT download does not install host packages. Offline export uses the current lockfile.
  sudo "$target/release/vortexctl" offline-download --output "$run/prerequisites"
  "$target/release/vortexctl" offline-create --source "$repo" --packages "$run/prerequisites/packages" --output "$run/bundle" --profile cpu
  bundle=$run/bundle/offline.tar.gz
fi
mkdir "$run/payload"
cp "$testbin" "$run/payload/systemd-test"
cp "$networktest" "$run/payload/network-test"
cp "$target/release/vortexctl" "$run/payload/vortexctl"
cp "$target/release/examples/service_fixture" "$run/payload/service_fixture"
cp "$bundle" "$run/payload/offline.tar.gz"
cp "$(dirname "$bundle")/offline.json" "$run/payload/offline.json"
tar -C "$repo" -cf "$run/payload/repository.tar" src config Cargo.toml Cargo.lock build.rs rust-toolchain.toml run_vortex.sh simulation_tests/reference/dim_noisy_tag.png
ssh-keygen -q -t ed25519 -N '' -f "$run/key"
cat > "$run/user-data" <<EOF
#cloud-config
users:
  - name: rehearsal
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat "$run/key.pub")
ssh_pwauth: false
write_files:
  - path: /etc/vortex-rehearsal-guest
    content: disposable
EOF
printf 'instance-id: vortex-%s\nlocal-hostname: vortex-rehearsal\n' "$(basename "$run")" > "$run/meta-data"
cat > "$run/network-config" <<EOF
version: 2
ethernets:
  management:
    match: {macaddress: '52:54:00:65:43:21'}
    set-name: eth0
    dhcp4: true
  radio:
    match: {macaddress: '52:54:00:12:34:56'}
    set-name: eth1
    dhcp4: false
    optional: true
EOF
cloud-localds --network-config="$run/network-config" "$run/seed.img" "$run/user-data" "$run/meta-data"
qemu-img create -q -f qcow2 -F qcow2 -b "$cache/$image" "$run/disk.qcow2" 24G
port=${VORTEX_VM_PORT:-22229}
accel=tcg
[[ -r /dev/kvm && -w /dev/kvm ]] && accel=kvm
echo "Starting fresh Ubuntu 24.04 guest ($accel); outbound networking blocked by QEMU"
qemu-system-x86_64 -accel "$accel" -m 6144 -smp 4 -display none -serial "file:$run/serial.log" \
  -drive "file=$run/disk.qcow2,if=virtio" -drive "file=$run/seed.img,format=raw,if=virtio" \
  -netdev "user,id=management,restrict=on,hostfwd=tcp:127.0.0.1:$port-:22" \
  -device virtio-net-pci,netdev=management,mac=52:54:00:65:43:21 \
  -netdev user,id=radio,restrict=on,net=10.5.9.0/24,host=10.5.9.1,dhcpstart=10.5.9.200 \
  -device virtio-net-pci,netdev=radio,mac=52:54:00:12:34:56 > "$run/qemu.log" 2>&1 &
vm_pid=$!
trap 'kill "$vm_pid" 2>/dev/null || true; wait "$vm_pid" 2>/dev/null || true' EXIT
opts=(-i "$run/key" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$run/known_hosts" -o ConnectTimeout=3)
# First contact is an isolated local VM with a freshly generated disk and key.
ready() {
  for _ in $(seq 1 180); do
    kill -0 "$vm_pid" || { cat "$run/qemu.log"; return 1; }
    if ssh "${opts[@]}" -p "$port" rehearsal@127.0.0.1 true 2>/dev/null; then return 0; fi
    sleep 2
  done
  tail -80 "$run/serial.log"; return 1
}
remote() { ssh "${opts[@]}" -p "$port" rehearsal@127.0.0.1 "$@"; }
cleanup() {
  remote 'sudo tar -C /home/rehearsal -cf - results' > "$run/guest-results.tar" 2>/dev/null || true
  remote 'sudo sync' 2>/dev/null || true
  kill "$vm_pid" 2>/dev/null || true
  wait "$vm_pid" 2>/dev/null || true
}
trap cleanup EXIT
ready
remote 'sudo cloud-init status --wait'
scp "${opts[@]}" -P "$port" "$run/payload/"* rehearsal@127.0.0.1:.
remote 'mkdir repository; tar -xf repository.tar -C repository; chmod +x systemd-test network-test vortexctl service_fixture'
environment='sudo env VORTEX_TEST_REPOSITORY=/home/rehearsal/repository VORTEX_TEST_CONTROLLER=/home/rehearsal/vortexctl VORTEX_SERVICE_FIXTURE=/home/rehearsal/service_fixture'
remote "$environment VORTEX_OFFLINE_BUNDLE=/home/rehearsal/offline.tar.gz ./systemd-test --ignored --exact actual_offline_initialization --nocapture"
remote "$environment ./network-test --ignored --exact real_network_confirmation_and_rollback --nocapture"
remote "$environment VORTEX_OFFLINE_BUNDLE=/home/rehearsal/offline.tar.gz ./systemd-test --ignored --exact actual_service_lifecycle --nocapture"
remote "$environment VORTEX_REBOOT_PHASE=prepare ./systemd-test --ignored --exact actual_reboot_recovery --nocapture"
reboot_guest() {
  local before after
  before=$(remote 'cat /proc/sys/kernel/random/boot_id')
  remote 'sudo systemctl reboot' || true
  for _ in $(seq 1 180); do
    sleep 2
    if after=$(remote 'cat /proc/sys/kernel/random/boot_id' 2>/dev/null) && [[ "$after" != "$before" ]]; then ready; return; fi
  done
  return 1
}
reboot_guest
remote "$environment ./network-test --ignored --exact real_network_survives_reboot --nocapture"
remote "$environment VORTEX_REBOOT_PHASE=interrupt ./systemd-test --ignored --exact actual_reboot_recovery --nocapture"
reboot_guest
remote "$environment VORTEX_REBOOT_PHASE=recover ./systemd-test --ignored --exact actual_reboot_recovery --nocapture"
# Expand the account lookup inside the guest, not on the host.
# shellcheck disable=SC2016
remote 'test -z "$(getent passwd | cut -d: -f1 | grep ^vortex-test- || true)"'
printf '{"passed":true,"guest":"Ubuntu 24.04 x64","network":"outbound blocked","actual_initialization":true,"real_cpu_build_and_smoke":true,"actual_package_install":true,"network_confirmation_and_rollback":true,"network_reboot":true,"actual_reboots":2,"service_runtime":"fixture"}\n' > "$run/result.json"
echo "PASS: actual offline initialization and CPU build/smoke, lifecycle, rollback, watchdog, two real reboots. Report: $run/result.json"
