"use strict";

const fs = require("fs/promises");
const path = require("path");
const os = require("os");
const { spawn } = require("child_process");
const { buildDeviceConfig } = require("./launch-config");

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
  const helper = path.join(source, "scripts", "deploy_release.py");
  await fs.access(helper);
  const args = [helper, "pack", "--source", source, "--output", directory, "--device-config", devicePath];
  if (settings.runtime_config_path) args.push("--runtime-config", path.resolve(appDir, settings.runtime_config_path));
  if (settings.tag_map_path) args.push("--tag-map", path.resolve(appDir, settings.tag_map_path));
  const python = settings.python_executable || (process.platform === "win32" ? "python" : "python3");
  await new Promise((resolve, reject) => {
    const child = spawn(python, args, { windowsHide: true, cwd: source, shell: false });
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
  const device = buildDeviceConfig(settings);
  const root = managedRoot(remoteTarget);
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "vortex-release-"));
  let conn = null;
  let metadata = null;
  let launched = false;
  const emit = (percent, text, status = "active") => deps.progress({ percent, text, status });
  const checked = async (command) => {
    if (!conn) conn = await deps.connect(settings);
    return deps.execChecked(conn, command);
  };
  try {
    emit(2, "Packaging release snapshot...");
    metadata = await (deps.package || packageLocally)(settings, directory, deps.appDir);
    if (metadata.schema_version !== 1 || metadata.build_profile !== device.build_profile ||
        !/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(metadata.release_id) ||
        !/^[a-f0-9]{64}$/.test(metadata.archive_sha256) || !/^[a-f0-9]{64}$/.test(metadata.controller_sha256)) {
      throw new Error("Invalid release package metadata");
    }
    const incoming = `${root}/incoming/${metadata.release_id}`;
    const controller = `${incoming}/controller.py`;
    const archive = `${incoming}/release.tar.gz`;
    deps.log(`Release ${metadata.release_id}: staging at ${root}; production will keep running.`);
    await checked(`mkdir -p -- ${quote(incoming)}`);
    const sftp = await deps.getSftp(conn);
    emit(10, "Uploading release archive...");
    await deps.upload(sftp, path.join(directory, "release.tar.gz"), archive, {
      step: (sent, chunk, total) => emit(10+Math.round(25*sent/Math.max(total, 1)), "Uploading release archive...")
    });
    await deps.upload(sftp, path.join(directory, "controller.py"), controller);
    // Verify the small bootstrap controller before executing it. The controller
    // separately checks the entire archive and every extracted payload file.
    const verify = "import hashlib,sys; p=sys.argv[1]; expected=sys.argv[2]; " +
      "actual=hashlib.sha256(open(p,'rb').read()).hexdigest(); " +
      "sys.exit(0 if actual==expected else 1)";
    await checked(`python3 -c ${quote(verify)} ${quote(controller)} ${quote(metadata.controller_sha256)}`);
    emit(38, "Starting isolated build...");
    const command = `python3 ${quote(controller)} stage --root ${quote(root)} --archive ${quote(archive)} ` +
      `--sha256 ${quote(metadata.archive_sha256)} --release-id ${quote(metadata.release_id)} --with-debug-bridge`;
    // A detached job owns the lock and log independently of the GUI/SSH stream.
    await checked(`nohup ${command} > ${quote(incoming+"/controller.log")} 2>&1 < /dev/null &`);
    launched = true;
    const poll = `python3 ${quote(controller)} status --root ${quote(root)} --release-id ${quote(metadata.release_id)}`;
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
      if (job.release_id !== metadata.release_id || job.activated !== false) throw new Error("Unexpected deployment job identity/state");
      if (job.state === "ready") {
        if (job.profile !== metadata.build_profile || job.release_path !== `${root}/releases/${metadata.release_id}`) {
          throw new Error("Unexpected staged release path/profile");
        }
        emit(100, "Release ready — not activated", "done");
        deps.log(`Release ready: ${job.release_path}. Running installation unchanged.`);
        return job;
      }
      if (job.state === "failed" || (job.state === "starting" && ++starting > 30)) {
        if (job.state === "failed") launched = false;
        const tail = await checked(`tail -n 60 -- ${quote(root+"/jobs/"+metadata.release_id+".log")} ` +
          `${quote(incoming+"/controller.log")} 2>/dev/null || true`);
        if (tail.stdout.trim()) deps.log(tail.stdout.trim());
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
    emit(0, "Staging failed — running installation unchanged", "error");
    if (launched) deps.log(`Job may still be running. Status: ${root}/jobs/${metadata.release_id}.json`);
    throw error;
  } finally {
    if (conn) conn.end();
    await fs.rm(directory, { recursive: true, force: true });
  }
}

module.exports = { stageProject, packageLocally, managedRoot, quote };
