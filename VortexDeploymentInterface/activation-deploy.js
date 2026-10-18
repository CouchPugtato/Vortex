"use strict";
const crypto = require("crypto");
const { managedRoot, quote } = require("./staging-deploy");
const { logFollower } = require("./job-progress");

function identity(value) {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(value)) throw new Error("Stage/select a valid release first");
  return value;
}

async function releaseStatus(settings, remoteTarget, deps) {
  const root = managedRoot(remoteTarget);
  const release = settings.last_ready_release ? identity(settings.last_ready_release) : null;
  const conn = await deps.connect(settings);
  try {
    const helper = release ? `${root}/releases/${release}/tools/vortexctl` : `${root}/vortexctl`;
    const empty = JSON.stringify({ ready: [], active: { kind: "none" }, previous: { kind: "none" }, service_configured: false });
    const command = release ? `${quote(helper)} status --root ${quote(root)}` :
      `if [ -f ${quote(helper)} ]; then ${quote(helper)} status --root ${quote(root)}; ` +
      `else printf '%s\\n' ${quote(empty)}; fi`;
    const response = await deps.execChecked(conn, command);
    return JSON.parse(response.stdout);
  } finally { conn.end(); }
}

async function operateRelease(settings, remoteTarget, action, release, deps) {
  if (!["activate", "rollback"].includes(action)) throw new Error("Unknown release operation");
  const root = managedRoot(remoteTarget);
  const prepared = identity(release || settings.last_ready_release);
  const operation = deps.resume ? identity(deps.resume.operation_id) : "activation-"+crypto.randomUUID();
  const followLog = logFollower(deps.log);
  let conn = null, started = false;
  const checked = async (command) => {
    if (!conn) conn = await deps.connect(settings);
    return deps.execChecked(conn, command);
  };
  const emit = (percent, text, status = "active") => deps.progress({ percent, text, status });
  try {
    emit(2, "Preparing managed service...");
    if (!deps.resume) {
      const installer = `${root}/releases/${prepared}/tools/vortexctl`;
      const name = settings.startup_service_name || "vortex-vision.service";
      if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*\.service$/.test(name)) throw new Error("Invalid managed service name");
      // Read-only status locates an existing installation; only first-time preparation
      // needs sudo. Later activations use narrowly granted systemctl permissions.
      const overview = JSON.parse((await checked(`${quote(installer)} status --root ${quote(root)}`)).stdout);
      if (!overview.service_configured || overview.controller_current === false) {
        const install = `${quote(installer)} install-service --root ${quote(root)} ` +
          `--legacy-root ${quote(remoteTarget)} --service ${quote(name)} --user ${quote(deps.user || "vortex")}`;
        try { await checked(`sudo -n ${install}`); }
        catch (error) {
          throw new Error("Managed service setup needs administrator access. Open Jetson Setup and use Initialize Jetson, then retry Activate.");
        }
      }
      const command = `${quote(root+"/vortexctl")} start-job --kind ${quote(action)} --root ${quote(root)} --job-id ${quote(operation)}` +
        (action === "activate" ? ` --release-id ${quote(prepared)}` : "");
      await deps.remember?.({ kind: action, operation_id: operation, release_id: prepared });
      started = true;
      await checked(command);
    }
    const helper = `${root}/vortexctl`;
    started = true;
    const deadline = Date.now()+(deps.timeoutMs ?? 180000);
    const delay = deps.delay || ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    let disconnected = 0, starting = 0;
    while (Date.now() < deadline) {
      let response;
      try {
        response = await checked(`${quote(helper)} status --root ${quote(root)} --operation-id ${quote(operation)}`);
        disconnected = 0;
      } catch (error) {
        if (conn) conn.end();
        conn = null;
        if (++disconnected >= 10) throw error;
        await delay(deps.pollMs ?? 1000);
        continue;
      }
      const job = JSON.parse(response.stdout);
      followLog(job);
      if (job.operation_id !== operation) throw new Error("Unexpected activation operation ID");
      if (job.state === "active") {
        const kind = job.selection?.kind;
        if (job.action !== action || !["release", "legacy"].includes(kind) ||
            (action === "activate" && (kind !== "release" || job.selection.release_id !== prepared)) ||
            job.verification !== (kind === "release" ? "all_cameras_advancing" : "legacy_process_only")) {
          throw new Error("Unexpected activation result or unverified runtime health");
        }
        emit(100, action === "rollback" ? "Rollback verified" : "Release active — camera health verified", "done");
        await deps.finished?.();
        return job;
      }
      if (["rolled_back", "rollback_failed", "failed", "recovered", "interrupted"].includes(job.state)) {
        started = false;
        await deps.finished?.();
        const suffix = job.state === "rolled_back" ? "; previous installation restored" :
          job.state === "rollback_failed" ? `; rollback also failed: ${job.rollback_error}` : "";
        throw new Error((job.error || "Activation failed")+suffix);
      }
      if (job.state === "starting" && ++starting > 30 && !["active", "activating"].includes(job.unit_state)) {
        await deps.finished?.();
        started = false;
        throw new Error("Activation worker did not start; inspect its controller log");
      }
      const percent = { starting: 5, queued: 5, validating: 15, stopping: 35, switching: 50, verifying: 75 }[job.state];
      if (percent === undefined) throw new Error(`Unknown activation state: ${job.state}`);
      emit(percent, `${action}: ${job.state}`);
      await delay(deps.pollMs ?? 1000);
    }
    throw new Error("Activation still running; refresh its saved status before retrying");
  } catch (error) {
    emit(0, started ? "Release status unavailable — reconnect with Refresh Releases" :
      "Release operation failed — inspect logs/status", "error");
    if (started) deps.log(`Activation may continue. Saved status: ${root}/jobs/${operation}.json`);
    else deps.log(error.message);
    throw error;
  } finally { if (conn) conn.end(); }
}

module.exports = { operateRelease, releaseStatus };
