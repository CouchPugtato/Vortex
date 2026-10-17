"use strict";
const crypto = require("crypto");
const { managedRoot, quote } = require("./staging-deploy");

class ServiceDiagnostics {
  constructor(deps) { this.deps = deps; this.session = null; }
  get running() { return Boolean(this.session); }
  async start(settings, target) {
    if (this.running) throw new Error("Diagnostics already attached");
    const camera = Number(settings.preview_camera_index ?? 0);
    if (!Number.isInteger(camera) || camera < 0 || camera > 63) throw new Error("Invalid diagnostics camera");
    const root = managedRoot(target);
    const session = { settings, root, camera, client: crypto.randomUUID(), conn: null, lastFrame: null, error: null };
    this.session = session;
    try {
      await this.read(session);
      if (this.session !== session) return;
      this.deps.monitorState(true);
      this.loop(session);  // Observation only; no production process is launched here.
    } catch (error) {
      if (this.session === session) this.session = null;
      session.conn?.end();
      throw error;
    }
  }
  command(session, action = "diagnostics") {
    return `python3 ${quote(session.root+"/activate_release.py")} ${action} --root ${quote(session.root)} `+
      `--camera ${session.camera} --client-id ${quote(session.client)}`;
  }
  async read(session) {
    if (!session.conn) session.conn = await this.deps.connect(session.settings);
    if (this.session !== session) { session.conn.end(); return; }
    const requested = Date.now();
    const response = await this.deps.execChecked(session.conn, this.command(session));
    if (this.session !== session) return;
    const value = JSON.parse(response.stdout);
    if (value.schema_version !== 1 || typeof value.ready !== "boolean" || value.camera_index !== session.camera) {
      throw new Error("Unexpected service diagnostics response");
    }
    const requestMs = Date.now()-requested;
    if (value.ready && Number(value.state?.frame_age_ms || 0)+requestMs > 1500) {
      value.ready = false;
      value.reason = "Preview transfer is stale; waiting for a current frame";
    }
    if (value.ready) {
      const state = value.state;
      if (!state || state.camera_index !== session.camera || state.client_id !== session.client ||
          state.pid !== value.service?.pid || state.pid !== value.health?.pid ||
          state.release_id !== value.health?.release_id || state.launch_token !== value.health?.launch_token ||
          !/^frame-[0-9]+-[0-9]+\.jpg$/.test(state.frame_id) ||
          typeof value.jpeg_base64 !== "string" || value.jpeg_base64.length > 700000 || !value.jpeg_base64.startsWith("/9j/")) {
        throw new Error("Mismatched production preview frame");
      }
      // Repeated polls of a slow camera update status without reloading its image.
      if (session.lastFrame !== state.frame_id) {
        this.deps.frame({ state, dataUrl: "data:image/jpeg;base64,"+value.jpeg_base64 });
        session.lastFrame = state.frame_id;
      }
    } else {
      session.lastFrame = null;
      this.deps.frame({ state: null, dataUrl: null });
    }
    this.deps.status({ ...value, request_ms: requestMs, jpeg_base64: undefined, state: undefined });
    session.error = null;
  }
  async loop(session) {
    const delay = this.deps.delay || ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    while (this.session === session) {
      await delay(this.deps.pollMs ?? 250);
      if (this.session !== session) return;
      try { await this.read(session); }
      catch (error) {
        if (this.session !== session) return;
        session.conn?.end(); session.conn = null;
        session.lastFrame = null;
        this.deps.frame({ state: null, dataUrl: null });
        this.deps.status({ ready: false, reason: "Connection interrupted; reconnecting to production service" });
        if (session.error !== error.message) this.deps.log(`Service diagnostics: ${error.message}`);
        session.error = error.message;
      }
    }
  }
  async stop() {
    const session = this.session;
    if (!session) return;
    this.session = null;
    let timer;
    try {
      if (session.conn) await Promise.race([
        this.deps.execChecked(session.conn, this.command(session, "diagnostics-stop")),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("Detach timed out")), this.deps.stopTimeoutMs ?? 1000); })
      ]);
    } catch (_) { /* The lease expires in three seconds if the connection is gone. */ }
    finally {
      clearTimeout(timer);
      session.conn?.end();
      if (!this.session) {
        this.deps.frame({ state: null, dataUrl: null });
        this.deps.monitorState(false);
        this.deps.status({ ready: false, reason: "Diagnostics detached; production service continues" });
      }
    }
  }
}

module.exports = { ServiceDiagnostics };
