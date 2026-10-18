"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { LAUNCHER, PROFILES, buildDeviceConfig } = require("../launch-config");

test("checked-in launcher is identical to the deployed launcher", () => {
  const checkedIn = fs.readFileSync(path.join(__dirname, "../../run_vortex.sh"), "utf8").replace(/\r\n/g, "\n");
  assert.equal(LAUNCHER, checkedIn);
  assert(!LAUNCHER.includes("cargo"));
  assert(!LAUNCHER.includes("5091"));
  assert(!LAUNCHER.includes("5092"));
});

test("UI settings control cameras, networking, and feature profile", () => {
  const config = buildDeviceConfig({ startup_camera_indices: "1, 3", build_profile: "gpu-tensorrt",
    vortex_nt_enable: true, vortex_nt_mode: "custom", vortex_nt_server: "robot.local",
    vortex_nt_table: "/Team/Vision", vortex_udp_enable: false, vortex_udp_target: "", vortex_udp_port: 12345 });
  assert.deepEqual(config.cameras, [1, 3]);
  assert.equal(config.build_profile, "gpu-tensorrt");
  assert.deepEqual(config.networktables, { enabled: true, mode: "custom", team: 509,
    server: "robot.local", table: "/Team/Vision" });
  assert.deepEqual(config.udp, { enabled: false, target: "", port: 12345 });
});

test("device payload excludes desktop paths and SSH credentials", () => {
  const config = buildDeviceConfig({ pass: "secret", user: "someone", host: "device", local_path: "C:/source",
    tag_map_path: "C:/maps/custom.json" });
  assert.equal(config.tag_map, "config/apriltag_map.json");
  for (const excluded of ["secret", "someone", "C:/", '"host"', '"local_path"']) {
    assert(!JSON.stringify(config).includes(excluded));
  }
});

test("unsafe or ambiguous settings fail rather than silently becoming defaults", () => {
  for (const change of [{ startup_camera_indices: "0,0" }, { startup_camera_indices: "0,2.5" },
    { startup_camera_indices: "-1" }, { startup_camera_indices: "64" }, { startup_camera_indices: "" },
    { startup_camera_indices: "0," }, { build_profile: "auto" }, { vortex_udp_port: 0 },
    { vortex_udp_port: true }, { vortex_udp_enable: "false" }, { vortex_nt_table: "relative" },
    { vortex_nt_enable: true, vortex_nt_mode: "custom", vortex_nt_server: "" },
    { vortex_udp_target: "robot\ncommand" }]) {
    assert.throws(() => buildDeviceConfig(change), undefined, JSON.stringify(change));
  }
});

test("every generated profile and NT mode is accepted by the native launcher", async () => {
  const samples = PROFILES.flatMap((profile) => ["team", "custom", "local"].map((mode) =>
    buildDeviceConfig({ build_profile: profile, vortex_nt_enable: true,
      vortex_nt_mode: mode, vortex_nt_server: "robot.local" })));
  const { nativeController } = require("../native-controller");
  const executable = await nativeController(path.resolve(__dirname, ".."), process.platform, process.arch);
  const os = require("node:os");
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "vortex-config-test-"));
  try {
    for (const sample of samples) {
      const filename = path.join(directory, "device.json");
      fs.writeFileSync(filename, JSON.stringify(sample));
      const result = spawnSync(executable, ["launch", "--source", path.resolve(__dirname, "../.."),
        "--device-config", filename, "--validate-config"], { encoding: "utf8" });
      assert.equal(result.status, 0, result.stderr || result.error?.message);
    }
  } finally { fs.rmSync(directory, { recursive: true, force: true }); }
});

test("packaged app includes the launch configuration module", () => {
  const pkg = require("../package.json");
  assert(pkg.build.files.includes("launch-config.js"));
  assert(pkg.build.files.includes("staging-deploy.js"));
});
