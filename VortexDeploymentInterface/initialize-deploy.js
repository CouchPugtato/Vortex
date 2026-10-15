"use strict";
const fs = require("fs/promises");
const path = require("path");
const os = require("os");
const crypto = require("crypto");
const { packageLocally, managedRoot, quote } = require("./staging-deploy");
const { logFollower } = require("./job-progress");

// Send the sudo password through SSH stdin, never argv, the shell or a file.
// No PTY means terminal echo cannot put it in stdout/logs.
function execAdministrator(conn, command, password = "") {
  if (typeof password !== "string" || /[\r\n\0]/.test(password)) return Promise.reject(new Error("Invalid administrator password"));
  return new Promise((resolve, reject) => {
    const prefix = password ? "sudo -S -k -p '' -- " : "sudo -n -- ";
    conn.exec(prefix+command, (error, channel) => {
      if (error) return reject(new Error("Could not open administrator setup channel"));
      let output = "", stderr = "";
      const timer = setTimeout(() => { channel.close(); reject(new Error("Administrator submission timed out; reconnect to its saved job")); }, 45000);
      channel.on("data", (bytes) => { output = (output+bytes).slice(-16000); });
      channel.stderr.on("data", (bytes) => { stderr = (stderr+bytes).slice(-4000); });
      channel.on("error", () => { clearTimeout(timer); reject(new Error("Administrator connection interrupted; reconnect to its saved job")); });
      channel.on("close", (code) => {
        clearTimeout(timer);
        // Never include raw privileged output in errors: sudo/plugin output can
        // include authentication details. Worker logs are read separately.
        if (code !== 0) {
          const authentication = /incorrect password|try again|password is required|sudoers|not allowed/i.test(stderr);
          const failure = new Error(authentication ?
            "Administrator authentication failed. Check the sudo password and account permissions, then retry Initialize Jetson." :
            "Administrator setup could not start. Refresh Releases to check its saved job, then retry Initialize Jetson.");
          failure.notSubmitted = authentication;
          return reject(failure);
        }
        resolve({ stdout: output });
      });
      channel.end(password ? password+"\n" : "");
    });
  });
}

async function initializeJetson(settings, remoteTarget, password, deps) {
  const root = managedRoot(remoteTarget);
  if (!/^\/[A-Za-z0-9_./-]+$/.test(root)) throw new Error("Use a deployment folder without spaces or special characters");
  const user = deps.user || settings.user || "vortex";
  if (!/^[a-z_][a-z0-9_-]*$/.test(user) || user === "root") throw new Error("Select a non-root Jetson account");
  const service = settings.startup_service_name || "vortex-vision.service";
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*\.service$/.test(service)) throw new Error("Invalid managed service name");
  let directory, conn, metadata = deps.resume;
  const operation = deps.resume?.operation_id || "initialize-"+crypto.randomUUID();
  const emit = (percent, text, status = "active") => deps.progress({ percent, text, status });
  const followLog = logFollower(deps.log);
  const checked = async (command) => {
    if (!conn) conn = await deps.connect(settings);
    return deps.execChecked(conn, command);
  };
  try {
    if (!deps.resume) {
      emit(2, "Checking Jetson connection...");
      // These tools exist on supported JetPack/Ubuntu images; no working project,
      // Rust installation, user manager or managed vision service is needed yet.
      await checked("{ test \"$(uname -s)\" = Linux && command -v python3 >/dev/null && command -v systemd-run >/dev/null && command -v sudo >/dev/null; } || " +
        "{ echo 'Initialization needs Ubuntu/JetPack with SSH, Python 3, systemd and sudo enabled.' >&2; exit 1; }");
      directory = await fs.mkdtemp(path.join(os.tmpdir(), "vortex-initialize-"));
      emit(5, "Preparing initialization snapshot...");
      metadata = await (deps.package || packageLocally)(settings, directory, deps.appDir);
      if (metadata.schema_version !== 1 || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(metadata.release_id) ||
          ![metadata.archive_sha256, metadata.controller_sha256, metadata.initializer_sha256].every((value) => /^[a-f0-9]{64}$/.test(value))) {
        throw new Error("Invalid initialization package metadata");
      }
      const incoming = `${root}/incoming/${metadata.release_id}`;
      await checked(`mkdir -p -- ${quote(incoming)} ${quote(root+"/jobs")}`);
      const sftp = await deps.getSftp(conn);
      emit(10, "Uploading setup snapshot...");
      for (const file of ["release.tar.gz", "controller.py", "initialize_jetson.py"]) {
        await deps.upload(sftp, path.join(directory, file), incoming+"/"+file);
      }
      const verify = "import hashlib,sys; " +
        "sys.exit(0 if all(hashlib.sha256(open(sys.argv[i],'rb').read()).hexdigest()==sys.argv[i+1] for i in (1,3)) else 1)";
      await checked(`python3 -c ${quote(verify)} ${quote(incoming+"/controller.py")} ${quote(metadata.controller_sha256)} `+
        `${quote(incoming+"/initialize_jetson.py")} ${quote(metadata.initializer_sha256)}`);
      // The initializer imports this verified controller under its normal module name.
      await checked(`cp -- ${quote(incoming+"/controller.py")} ${quote(incoming+"/deploy_release.py")}`);
      metadata = { ...metadata, kind: "initialize", operation_id: operation };
      await deps.remember?.(metadata);
      emit(20, "Starting administrator setup...");
      const command = `/usr/bin/python3 ${quote(incoming+"/initialize_jetson.py")} submit --root ${quote(root)} `+
        `--legacy-root ${quote(remoteTarget)} --job-id ${quote(operation)} --release-id ${quote(metadata.release_id)} `+
        `--sha256 ${quote(metadata.archive_sha256)} --user ${quote(user)} --service ${quote(service)}`;
      await (deps.execAdministrator || execAdministrator)(conn, command, password);
      password = "";
    }
    if (!metadata || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(metadata.release_id) ||
        !/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(operation)) throw new Error("Invalid saved initialization job");
    const helper = `${root}/incoming/${metadata.release_id}/initialize_jetson.py`;
    const deadline = Date.now()+(deps.timeoutMs ?? 3*60*60*1000);
    const delay = deps.delay || ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    let disconnects = 0, starting = 0;
    while (Date.now() < deadline) {
      let response;
      try {
        response = await checked(`python3 ${quote(helper)} status --root ${quote(root)} --job-id ${quote(operation)}`);
        disconnects = 0;
      } catch (error) {
        conn?.end(); conn = null;
        if (++disconnects >= 10) throw error;
        emit(20, "Setup continues on the Jetson; reconnecting...");
        await delay(deps.pollMs ?? 1000);
        continue;
      }
      const job = JSON.parse(response.stdout);
      if (job.operation_id !== operation) throw new Error("Unexpected initialization operation ID");
      followLog(job);
      if (job.state === "ready") {
        if (job.release_id !== metadata.release_id || job.profile !== metadata.build_profile || job.activated !== false ||
            job.release_path !== `${root}/releases/${metadata.release_id}` || job.verification !== "dependencies_build_smoke_service") {
          throw new Error("Unverified initialization result");
        }
        await deps.finished?.();
        emit(100, "Jetson initialized — release ready for Activate", "done");
        return job;
      }
      if (["failed", "interrupted"].includes(job.state)) {
        await deps.finished?.();
        throw new Error(job.error || "Initialization interrupted; retry Initialize Jetson");
      }
      if (job.state === "starting" && ++starting > 30 && !["active", "activating"].includes(job.unit_state)) {
        await deps.finished?.();
        throw new Error("Setup was not submitted. Check administrator credentials and retry Initialize Jetson.");
      }
      const percent = { starting: 20, queued: 20, verifying: 25, checking: 30, packages: 35,
        toolchain: 45, dependencies: 55, building: 65, validating: 85, service: 95 }[job.state];
      if (percent === undefined) throw new Error("Unknown initialization state: "+job.state);
      emit(percent, `Initialize Jetson: ${job.state}`);
      await delay(deps.pollMs ?? 1000);
    }
    throw new Error("Setup is still running. Refresh Releases to reconnect to its saved job.");
  } catch (error) {
    if (error.notSubmitted) await deps.finished?.();
    emit(0, error.message, "error");
    throw error;
  } finally {
    password = "";
    conn?.end();
    if (directory) await fs.rm(directory, { recursive: true, force: true });
  }
}

module.exports = { initializeJetson, execAdministrator };
