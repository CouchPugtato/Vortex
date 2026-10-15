"""Validated production launcher and explicit native build entry point (stdlib only)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"cpu": [], "gpu": ["gpu"], "tensorrt": ["tensorrt"],
            "gpu-tensorrt": ["gpu", "tensorrt"]}
MAIN = "dumapril-taglocalization"
BRIDGE = "orin_bridge"
MANAGED_ENV = ("VORTEX_RUNTIME_CONFIG", "VORTEX_TAG_MAP", "VORTEX_NT_ENABLE",
               "VORTEX_NT_TEAM", "VORTEX_NT_SERVER", "VORTEX_NT_TABLE",
               "VORTEX_UDP_ENABLE", "VORTEX_UDP_TARGET", "VORTEX_UDP_PORT",
               "VORTEX_BRIDGE_CONFIG", "VORTEX_BRIDGE_CAMERA", "VORTEX_BRIDGE_FRAME",
               "VORTEX_BRIDGE_STATE")


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    data = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite JSON number")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(data)
    return data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, name, allow_empty=False):
    require(isinstance(value, str) and (allow_empty or bool(value.strip())) and
            not any(ord(c) < 32 or ord(c) == 127 for c in value), f"Invalid {name}")
    return value


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def validate_config(config):
    require(isinstance(config, dict), "Device settings must be an object")
    require(type(config.get("schema_version")) is int and config["schema_version"] == 1,
            "Unsupported device schema_version")
    require(config.get("runtime") == "main", "Production runtime must be 'main'")
    require(isinstance(config.get("build_profile"), str) and config["build_profile"] in PROFILES,
            "Unknown build_profile")
    cameras = config.get("cameras")
    require(isinstance(cameras, list) and cameras and
            all(integer(i, 0, 63) for i in cameras) and len(set(cameras)) == len(cameras),
            "cameras must contain unique integer indices from 0 to 63")
    text(config.get("runtime_config"), "runtime_config")
    text(config.get("tag_map"), "tag_map")
    nt, udp = config.get("networktables"), config.get("udp")
    require(isinstance(nt, dict) and isinstance(udp, dict), "Missing networking settings")
    require(type(nt.get("enabled")) is bool and type(udp.get("enabled")) is bool,
            "Networking enable flags must be booleans")
    require(nt.get("mode") in ("team", "custom", "local"), "Invalid NT mode")
    require(integer(nt.get("team"), 1, 99999), "Invalid NT team number")
    text(nt.get("server"), "NT server", allow_empty=not (nt["enabled"] and nt["mode"] == "custom"))
    require(text(nt.get("table"), "NT table").startswith("/"), "NT table must start with /")
    text(udp.get("target"), "UDP target", allow_empty=not udp["enabled"])
    require(integer(udp.get("port"), 1, 65535), "UDP port must be from 1 to 65535")
    return config


def absolute(root, value):
    path = Path(value)
    return path if path.is_absolute() else root/path


def validate_resources(root, config):
    runtime = read_json(absolute(root, config["runtime_config"]))
    require(isinstance(runtime, dict) and isinstance(runtime.get("camera"), dict),
            "Runtime config must contain a camera object")
    profiles = runtime.get("camera_profiles", {})
    require(isinstance(profiles, dict), "camera_profiles must be an object")
    for name, camera in [("camera", runtime["camera"]), *profiles.items()]:
        require(isinstance(camera, dict), f"Invalid camera profile: {name}")
        for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "tag_size_m"):
            value = camera.get(key)
            require(type(value) in (int, float) and math.isfinite(value), f"Invalid {name}.{key}")
        require(min(camera["fx"], camera["fy"], camera["tag_size_m"]) > 0,
                f"Focal lengths and tag size must be positive: {name}")
        for key in ("x_offset", "y_offset", "z_offset", "roll_deg", "pitch_deg", "yaw_deg"):
            require(type(camera.get(key, 0)) in (int, float), f"Invalid {name}.{key}")
    for section in ("processing", "object_detection"):
        values = runtime.get(section, {})
        require(isinstance(values, dict), f"Invalid {section}")
        for key, value in values.items():
            if key == "use_nn" and section == "object_detection":
                require(type(value) is bool, "use_nn must be a boolean")
            elif key in ("smoothing_alpha", "resolution_scale_factor", "gpu_scale_factor", "black_level_offset",
                         "sensor_gain", "red_balance", "blue_balance", "yolo_obj_width_m", "yolo_obj_height_m",
                         "processing_yolo_obj_width_m", "processing_yolo_obj_height_m", "confidence_threshold"):
                require(type(value) in (int, float), f"Invalid {section}.{key}")
    field = read_json(absolute(root, config["tag_map"]))
    require(isinstance(field, dict) and isinstance(field.get("tags"), list) and field["tags"],
            "Tag map must contain a nonempty tags array")
    seen = set()
    for tag in field["tags"]:
        require(isinstance(tag, dict) and integer(tag.get("ID"), 1, 2147483647), "Invalid tag ID")
        require(tag["ID"] not in seen, "Duplicate tag ID")
        seen.add(tag["ID"])
        pose = tag.get("pose", {})
        require(isinstance(pose, dict), "Invalid tag pose")
        translation, rotation = pose.get("translation", {}), pose.get("rotation", {})
        require(isinstance(rotation, dict), "Invalid tag rotation")
        quaternion = rotation.get("quaternion", {})
        for values, keys in ((translation, ("x", "y", "z")), (quaternion, ("W", "X", "Y", "Z"))):
            require(isinstance(values, dict) and all(type(values.get(k)) in (int, float) for k in keys),
                    "Invalid tag translation or quaternion")
        require(sum(v*v for v in (quaternion[k] for k in ("W", "X", "Y", "Z"))) > 0,
                "Tag quaternion cannot be zero")


def environment(root, config, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for name in MANAGED_ENV:
        env.pop(name, None)
    nt, udp = config["networktables"], config["udp"]
    env.update(VORTEX_RUNTIME_CONFIG=str(absolute(root, config["runtime_config"])),
               VORTEX_TAG_MAP=str(absolute(root, config["tag_map"])),
               VORTEX_NT_ENABLE=str(int(nt["enabled"])), VORTEX_NT_TABLE=nt["table"],
               VORTEX_UDP_ENABLE=str(int(udp["enabled"])),
               VORTEX_UDP_TARGET=udp["target"], VORTEX_UDP_PORT=str(udp["port"]))
    if nt["enabled"]:
        if nt["mode"] == "team":
            env["VORTEX_NT_TEAM"] = str(nt["team"])
        else:
            env["VORTEX_NT_SERVER"] = "127.0.0.1" if nt["mode"] == "local" else nt["server"]
    return env


def sha256(path):
    with Path(path).open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest(root):
    files = [root/name for name in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml")]
    files += sorted(p for p in (root/"src").rglob("*") if p.is_file() and p.suffix in (".rs", ".cpp", ".h", ".c"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0"+bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def binary_path(root, binary):
    return root/"target"/"release"/(binary+(".exe" if os.name == "nt" else ""))


def build(root, config, online=False, with_debug_bridge=False, cache_dir=None):
    rustup_cargo = Path.home()/".cargo"/"bin"/("cargo.exe" if os.name == "nt" else "cargo")
    cargo = str(rustup_cargo) if rustup_cargo.is_file() else shutil.which("cargo") or "cargo"
    pin = re.search(r'^\s*channel\s*=\s*"(\d+\.\d+\.\d+)"\s*$',
                    (root/"rust-toolchain.toml").read_text(), re.MULTILINE)
    require(pin is not None, "rust-toolchain.toml must pin an exact Rust version")
    command = [cargo, "+"+pin.group(1), "build", "--release", "--locked", "--no-default-features", "--bin", MAIN]
    if not online:
        command.append("--offline")
    bins = [MAIN]
    if with_debug_bridge:
        command += ["--bin", BRIDGE]
        bins.append(BRIDGE)
    features = PROFILES[config["build_profile"]]
    if features:
        command += ["--features", ",".join(features)]
    if cache_dir is not None:
        cache_dir = Path(cache_dir).resolve()
        command += ["--target-dir", str(cache_dir)]
    # An explicit build may replace old artifacts, so invalidate the old stamp first.
    stamp = root/"target"/"release"/"vortex-build.json"
    before = source_digest(root)
    stamp.unlink(missing_ok=True)
    build_env = dict(os.environ)
    # Cargo's default target location is part of the current launcher contract.
    for name in ("CARGO_TARGET_DIR", "CARGO_BUILD_TARGET", "RUSTUP_TOOLCHAIN"):
        build_env.pop(name, None)
    build_env["RUSTUP_AUTO_INSTALL"] = "1" if online else "0"
    subprocess.run(command, cwd=root, env=build_env, check=True)
    require(source_digest(root) == before, "Source changed during build; rebuild before launching")
    if cache_dir is not None and cache_dir != (root/"target").resolve():
        # Each staged release owns its executable. Never run a mutable cache artifact.
        stamp.parent.mkdir(parents=True, exist_ok=True)
        for name in bins:
            destination = binary_path(root, name)
            fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
            os.close(fd)
            try:
                shutil.copy2(cache_dir/"release"/destination.name, temporary)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    atomic_json(stamp, dict(schema_version=1, runtime="main", profile=config["build_profile"],
                           source_sha256=before, binaries={name: sha256(binary_path(root, name)) for name in bins}))


def checked_binary(root, config, binary=MAIN):
    executable = binary_path(root, binary)
    stamp_path = root/"target"/"release"/"vortex-build.json"
    require(executable.is_file() and stamp_path.is_file(),
            "Verified executable missing. Run python3 scripts/run_vortex.py --build first.")
    require(os.name == "nt" or os.access(executable, os.X_OK), "Runtime binary is not executable")
    stamp = read_json(stamp_path)
    require(isinstance(stamp, dict), "Invalid build fingerprint; rebuild before launching")
    require(stamp.get("schema_version") == 1 and stamp.get("runtime") == "main" and
            stamp.get("profile") == config["build_profile"] and
            stamp.get("source_sha256") == source_digest(root) and
            stamp.get("binaries", {}).get(binary) == sha256(executable),
            "Executable is stale or has a different build profile. Rebuild before launching.")
    return executable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-config", type=Path, default=Path("config/device.json"))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--build", action="store_true", help="Explicitly build the selected profile")
    actions.add_argument("--validate-config", action="store_true", help="Check settings and resource files without cameras")
    actions.add_argument("--check", action="store_true", help="Also verify the executable and build fingerprint")
    actions.add_argument("--check-debug-bridge", action="store_true", help="Verify the legacy diagnostic build without launching")
    actions.add_argument("--debug-bridge", action="store_true", help="Legacy single-camera diagnostic mode; not production")
    parser.add_argument("--online", action="store_true", help="Allow dependency downloads for an explicit build")
    parser.add_argument("--with-debug-bridge", action="store_true", help="Also build the legacy diagnostic executable")
    parser.add_argument("--build-cache", type=Path, help="Shared Cargo target directory; copy artifacts into this release")
    parser.add_argument("--camera", type=int, default=0, help="Camera for --debug-bridge only")
    args = parser.parse_args(argv)
    if (args.online or args.with_debug_bridge or args.build_cache is not None) and not args.build:
        parser.error("Build options require --build")
    try:
        root = ROOT
        config = validate_config(read_json(absolute(root, args.device_config)))
        validate_resources(root, config)
        if args.validate_config:
            print(f"Device settings valid: main / {config['build_profile']} / cameras {config['cameras']}")
            return 0
        if args.build:
            build(root, config, args.online, args.with_debug_bridge, args.build_cache)
            print(f"Verified build complete: {config['build_profile']}")
            return 0
        binary = BRIDGE if args.debug_bridge or args.check_debug_bridge else MAIN
        executable = checked_binary(root, config, binary)
        if args.check or args.check_debug_bridge:
            print(f"Verified executable: {executable}")
            return 0
        env = environment(root, config)
        if args.debug_bridge:
            require(integer(args.camera, 0, 5), "Invalid debug camera (0-5)")
            print("Legacy bridge diagnostic: production telemetry is not provided by this mode.", flush=True)
            env.update(VORTEX_BRIDGE_CAMERA=str(args.camera), VORTEX_BRIDGE_CONFIG=env["VORTEX_RUNTIME_CONFIG"],
                       VORTEX_BRIDGE_FRAME="/tmp/vortex_bridge_frame.jpg",
                       VORTEX_BRIDGE_STATE="/tmp/vortex_bridge_state.json")
            command = [str(executable)]
        else:
            command = [str(executable), ",".join(map(str, config["cameras"]))]
        os.chdir(root)
        os.execve(str(executable), command, env)
    except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Vortex launch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
