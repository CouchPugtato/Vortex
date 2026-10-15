"use strict";

const PROFILES = Object.freeze(["cpu", "gpu", "tensorrt", "gpu-tensorrt"]);
const LAUNCHER = "#!/usr/bin/env bash\nset -euo pipefail\n\n" +
  'APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"\n' +
  'exec python3 "$APP_DIR/scripts/run_vortex.py" "$@"\n';

function ensure(condition, message) {
  if (!condition) throw new Error(message);
}

function integer(value, name, low, high) {
  ensure(typeof value === "number" || typeof value === "string", `Invalid ${name}`);
  ensure(String(value).trim() !== "", `Missing ${name}`);
  const number = Number(value);
  ensure(Number.isInteger(number) && number >= low && number <= high, `Invalid ${name}`);
  return number;
}

function text(value, name, allowEmpty = false) {
  ensure(typeof value === "string" && !/[\x00-\x1f\x7f]/.test(value) &&
    (allowEmpty || value.trim().length > 0), `Invalid ${name}`);
  return value.trim();
}

function enabled(value, fallback, name) {
  const result = value === undefined ? fallback : value;
  ensure(typeof result === "boolean", `${name} must be a boolean`);
  return result;
}

function buildDeviceConfig(settings) {
  const profile = settings.build_profile ?? "cpu";
  ensure(PROFILES.includes(profile), "Unknown build profile");
  const cameras = text(settings.startup_camera_indices ?? "0,1", "startup cameras")
    .split(",").map((part) => {
      ensure(/^\s*\d+\s*$/.test(part), "Camera indices must be comma-separated integers");
      return integer(part, "camera index", 0, 63);
    });
  ensure(new Set(cameras).size === cameras.length, "Camera indices must be unique");
  const ntEnabled = enabled(settings.vortex_nt_enable, false, "NT enabled");
  const udpEnabled = enabled(settings.vortex_udp_enable, true, "UDP enabled");
  const mode = settings.vortex_nt_mode ?? "team";
  ensure(["team", "custom", "local"].includes(mode), "Invalid NT mode");
  const table = text(settings.vortex_nt_table ?? "/Vortex/Vision", "NT table");
  ensure(table.startsWith("/"), "NT table must start with /");
  return {
    schema_version: 1,
    runtime: "main",
    build_profile: profile,
    cameras,
    runtime_config: "config/config.json",
    tag_map: settings.tag_map_path ? "config/apriltag_map.json" : "config/2026-rebuilt-welded.json",
    networktables: {
      enabled: ntEnabled,
      mode,
      team: integer(settings.vortex_nt_team ?? 509, "NT team", 1, 99999),
      server: text(settings.vortex_nt_server ?? "", "NT server", !(ntEnabled && mode === "custom")),
      table
    },
    udp: {
      enabled: udpEnabled,
      target: text(settings.vortex_udp_target ?? "192.168.1.24", "UDP target", !udpEnabled),
      port: integer(settings.vortex_udp_port ?? 5809, "UDP port", 1, 65535)
    }
  };
}

module.exports = { PROFILES, LAUNCHER, buildDeviceConfig };
