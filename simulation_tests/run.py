"""Generate and replay Vortex synthetic tests using a Python entry point."""
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

HERE = Path(__file__).resolve().parent


def command(args, env=None):
    return subprocess.run([str(arg) for arg in args], env=env, check=True)


def replay_command(args, dataset, env):
    if args.replay_executable:
        executable = args.replay_executable.resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"Replay executable not found: {executable}")
        if os.name == "nt" and (args.mingw_root / "bin").is_dir():
            env["PATH"] = str(args.mingw_root / "bin") + os.pathsep + env.get("PATH", "")
        return [executable, dataset, args.max_error_meters]

    cargo = shutil.which("cargo")
    rustup_cargo = Path.home() / ".cargo/bin/cargo.exe"
    if os.name == "nt" and rustup_cargo.is_file():
        cargo = str(rustup_cargo)
    if not cargo:
        raise FileNotFoundError("Rust Cargo is needed to build the production replay. "
                                "Use --generate-only for image generation, or "
                                "--replay-executable PATH for an already-built replay.")
    cmd = [cargo]
    pthread = args.mingw_root / "lib/libwinpthread.a"
    if os.name == "nt" and pthread.is_file():
        native = dataset / "native"
        native.mkdir(parents=True)
        shim = native / "winpthread.lib"
        shutil.copyfile(pthread, shim)
        env["PATH"] = str(args.mingw_root / "bin") + os.pathsep + env.get("PATH", "")
        env["APRILTAG_SYS_WINDOWS_PTHREAD_INCLUDE_DIR"] = str(args.mingw_root / "include")
        env["APRILTAG_SYS_WINDOWS_PTHREAD_STATIC_LIB"] = str(shim)
        # The bundled AprilTag sources predate C23's stricter function prototypes.
        env["CFLAGS"] = (env.get("CFLAGS", "") + " -std=gnu11").strip()
        cmd.append("+stable")
        target = ["--target", "x86_64-pc-windows-gnu"]
    else:
        target = []
    return cmd + ["run", "--release", "--locked", "--manifest-path", HERE / "replay/Cargo.toml"] + target + [
        "--", dataset, args.max_error_meters]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "extended", "stress"), default="extended")
    parser.add_argument("--case", help="Run one named scenario from the selected suite")
    parser.add_argument("--backend", choices=("fast", "blender"), default="fast")
    parser.add_argument("--blender", default="blender", help="Blender executable name or path")
    parser.add_argument("--mingw-root", type=Path, default=Path("C:/msys64/mingw64"))
    parser.add_argument("--max-error-meters", type=float, default=0.20)
    parser.add_argument("--output", type=Path, help="Fresh dataset directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true", help="Generate images without requiring Rust")
    mode.add_argument("--replay-executable", type=Path, help="Use a prebuilt replay instead of Cargo")
    args = parser.parse_args()
    if not math.isfinite(args.max_error_meters) or args.max_error_meters <= 0:
        parser.error("--max-error-meters must be positive and finite")
    if args.backend == "blender" and not shutil.which(args.blender):
        parser.error("Blender not found; supply --blender PATH or use --backend fast")
    name = f"{args.backend}_{args.suite}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    dataset = (args.output or HERE / "generated" / name).resolve()
    if dataset.exists():
        parser.error(f"Choose a fresh output directory: {dataset}")
    env = os.environ.copy()
    try:
        # Check the replay tool before spending time generating the dataset.
        replay = None if args.generate_only else replay_command(args, dataset, env)
        scenarios = "scenarios.json" if args.suite == "smoke" else f"scenarios_{args.suite}.json"
        generation = [sys.executable, HERE / "generate.py", "--backend", args.backend,
                      "--scenarios", HERE / scenarios, "--output", dataset]
        if args.case:
            generation += ["--case", args.case]
        command(generation)
        if args.backend == "blender":
            command([args.blender, "--background", "--python", HERE / "render_blender.py",
                     "--", "--dataset", dataset])
            command([sys.executable, HERE / "postprocess.py", "--dataset", dataset])
        print(f"Dataset: {dataset}", flush=True)
        if args.generate_only:
            print("Images ready. Replay was not run.")
            return 0
        result = subprocess.run([str(arg) for arg in replay], env=env)
        if (dataset / "report.json").is_file():
            command([sys.executable, HERE / "summarize.py", dataset])
            print(f"Report: {dataset / 'report.json'}", flush=True)
        return result.returncode if result.returncode >= 0 else 1
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Test run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
