"use strict";

const fs = require("fs/promises");
const path = require("path");
const os = require("os");
const { spawn } = require("child_process");
const { buildDeviceConfig } = require("./launch-config");
const { logFollower } = require("./job-progress");
const { nativeController, verifyController } = require("./native-controller");

function quote(value) { return `'${String(value).replace(/'/g, `'\\''`)}'`; }

function managedRoot(remoteTarget) {
  if (typeof remoteTarget !== "string" || !remoteTarget.startsWith("/") ||
      /[\x00-\x1f\x7f]/.test(remoteTarget) || remoteTarget.split("/").includes("..") ||
      path.posix.normalize(remoteTarget) !== remoteTarget || remoteTarget === "/") {
    throw new Error("Remote deployment directory must be a normalized absolute path");
  }
  return remoteTarget+".deployment";
}

async function packageLocally(settings, directory, appDir) {
  const source = path.resolve(appDir, settings.local_path);
  const devicePath = path.join(directory, "device.json");
  await fs.writeFile(devicePath, JSON.stringify(buildDeviceConfig(settings)), "utf8");
  const helper = await nativeController(appDir, process.platform, process.arch);
  const controller = await nativeController(appDir, "linux", settings.deployment_architecture === "x86_64" ? "x64" : "arm64");
  const args = ["pack", "--source", source, "--output", directory, "--device-config", devicePath, "--controller", controller];
  if (settings.runtime_config_path) args.push("--runtime-config", path.resolve(appDir, settings.runtime_config_path));
  if (settings.tag_map_path) args.push("--tag-map", path.resolve(appDir, settings.tag_map_path));
  await new Promise((resolve, reject) => {
    const child = spawn(helper, args, { windowsHide: true, cwd: source, shell: false });
    let output = "";
    const consume = (bytes) => { output = (output+bytes.toString("utf8")).slice(-16000); };
    child.stdout.on("data", consume);
    child.stderr.on("data", consume);
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? resolve() : reject(new Error(`Release packaging failed: ${output}`)));
  });
  return JSON.parse(await fs.readFile(path.join(directory, "package.json"), "utf8"));
}

async function stageProject(settings, remoteTarget, deps) {
  const device = deps.resume ? { build_profile: deps.resume.build_profile } : buildDeviceConfig(settings);
  const root = managedRoot(remoteTarget);
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "vortex-release-"));
  let conn = null;
  let metadata = null;
  let launched = false;
  const followLog = logFollower(deps.log);
  const emit = (percent, text, status = "active") => deps.progress({ percent, text, status });
  const checked = async (command) => {
    if (!conn) conn = await deps.connect(settings);
    return deps.execChecked(conn, command);
  };
  try {
    emit(2, deps.resume ? "Reconnecting to saved staging job..." : "Packaging release snapshot...");
    if (!deps.resume && !deps.package) {
      const architecture = (await checked("uname -m")).stdout.trim();
      if (!["aarch64", "x86_64"].includes(architecture)) throw new Error("Unsupported device architecture: "+architecture);
      settings = { ...settings, deployment_architecture: architecture };
    }
    metadata = deps.resume || await (deps.package || packageLocally)(settings, directory, deps.appDir);
    if (metadata.schema_version !== 1 || metadata.build_profile !== device.build_profile ||
        !/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(metadata.release_id) ||
        !/^[a-f0-9]{64}$/.test(metadata.archive_sha256) || !/^[a-f0-9]{64}$/.test(metadata.controller_sha256)) {
      throw new Error("Invalid release package metadata");
    }
    const incoming = `${root}/incoming/${metadata.release_id}`;
    const controller = `${incoming}/vortexctl`;
    const archive = `${incoming}/release.tar.gz`;
    deps.log(`Release ${metadata.release_id}: staging at ${root}; production will keep running.`);
    if (!deps.resume) {
      await checked(`mkdir -p -- ${quote(incoming)}`);
      const sftp = await deps.getSftp(conn);
      emit(10, "Uploading release archive...");
      await deps.upload(sftp, path.join(directory, "release.tar.gz"), archive, {
        step: (sent, chunk, total) => emit(10+Math.round(25*sent/Math.max(total, 1)), "Uploading release archive...")
      });
      await deps.upload(sftp, path.join(directory, "vortexctl"), controller);
      // Verify the small bootstrap controller before executing it. The controller
      // separately checks the entire archive and every extracted payload file.
      await checked(verifyController(controller, metadata.controller_sha256, quote));
      emit(38, "Starting isolated build...");
      const command = `${quote(controller)} start-job --root ${quote(root)} --kind stage ` +
        `--job-id ${quote(metadata.release_id)} --sha256 ${quote(metadata.archive_sha256)} --release-id ${quote(metadata.release_id)}`;
      // Save before submission so a lost SSH reply cannot cause a second launch.
      await deps.remember?.({ kind: "stage", ...metadata });
      launched = true;
      await checked(command);
    }
    launched = true;
    const poll = `${quote(controller)} status --root ${quote(root)} --release-id ${quote(metadata.release_id)}`;
    const start = Date.now();
    const timeout = deps.timeoutMs ?? 60*60*1000;
    const delay = deps.delay || ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    let lastState = null, starting = 0, disconnects = 0;
    while (Date.now()-start < timeout) {
      let response;
      try {
        response = await checked(poll);
        disconnects = 0;
      } catch (error) {
        if (conn) conn.end();
        conn = null;
        if (++disconnects >= 10) throw error;
        emit(40, "Connection interrupted; reconnecting to the existing build...");
        await delay(deps.pollMs ?? 1000);
        continue;
      }
      const job = JSON.parse(response.stdout);
      followLog(job);
      if (job.release_id !== metadata.release_id || job.activated !== false) throw new Error("Unexpected deployment job identity/state");
      if (job.state === "ready") {
        if (job.profile !== metadata.build_profile || job.release_path !== `${root}/releases/${metadata.release_id}`) {
          throw new Error("Unexpected staged release path/profile");
        }
        emit(100, "Release ready — not activated", "done");
        deps.log(`Release ready: ${job.release_path}. Running installation unchanged.`);
        await deps.finished?.();
        return job;
      }
      if (["failed", "interrupted"].includes(job.state) ||
          (job.state === "starting" && ++starting > 30 && !["active", "activating"].includes(job.unit_state))) {
        launched = false;
        const tail = await checked(`tail -n 60 -- ${quote(root+"/jobs/"+metadata.release_id+".log")} ` +
          `${quote(root+"/jobs/"+metadata.release_id+".controller.log")} 2>/dev/null || true`);
        if (tail.stdout.trim()) deps.log(tail.stdout.trim());
        await deps.finished?.();
        throw new Error(job.error || "Remote staging job did not start; inspect controller.log");
      }
      if (!["starting", "queued", "verifying", "extracting", "validating", "building", "checking"].includes(job.state)) {
        throw new Error(`Unknown deployment job state: ${job.state}`);
      }
      const percent = { starting: 40, queued: 40, verifying: 42, extracting: 48, validating: 55, building: 65, checking: 90 }[job.state];
      emit(percent, `${job.state}: ${metadata.release_id}`);
      if (job.state !== lastState) deps.log(`Release ${metadata.release_id}: ${job.state}`);
      lastState = job.state;
      await delay(deps.pollMs ?? 1000);
    }
    throw new Error("Stopped waiting for the remote build; inspect its saved job status before retrying");
  } catch (error) {
    emit(0, launched ? "Staging status unavailable — reconnect with Refresh Releases" :
      "Staging failed — running installation unchanged", "error");
    if (launched) deps.log(`Job may still be running. Status: ${root}/jobs/${metadata.release_id}.json`);
    throw error;
  } finally {
    if (conn) conn.end();
    await fs.rm(directory, { recursive: true, force: true });
  }
}

module.exports = { stageProject, packageLocally, managedRoot, quote };
