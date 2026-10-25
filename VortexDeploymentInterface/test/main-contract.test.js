"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { EventEmitter } = require("node:events");
const { Writable } = require("node:stream");

function harness(t, exitCode = () => 0, stage = async () => ({ state: "ready" }), operation = async () => ({ state: "active" }), initialize = async () => ({ state: "ready", release_id: "new" })) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "vortex-deployment-test-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const commands = [], writes = new Map(), handlers = new Map(), stageCalls = [], activationCalls = [], initializeCalls = [], diagnosticsCalls = [];
  const appEvents = new Map();
  let quits = 0, diagnosticsStops = 0;
  const overview = { ready: ["new"], active: { kind: "none" }, previous: { kind: "none" } };
  class FakeClient extends EventEmitter {
    connect() { queueMicrotask(() => this.emit("ready")); return this; }
    end() {}
    exec(command, callback) {
      commands.push(command);
      const channel = new EventEmitter();
      channel.stderr = new EventEmitter();
      callback(null, channel);
      setImmediate(() => {
        const code = exitCode(command);
        if (code !== 0) channel.stderr.emit("data", Buffer.from("simulated failure"));
        channel.emit("close", code);
      });
    }
    sftp(callback) {
      callback(null, { createWriteStream(destination) {
        const buffers = [];
        return new Writable({ write(bytes, encoding, done) {
          buffers.push(Buffer.from(bytes));
          writes.set(destination, Buffer.concat(buffers));
          done();
        } });
      } });
    }
  }
  const appDir = path.resolve(__dirname, "..");
  const electron = {
    app: { getPath: () => directory, whenReady: () => ({ then() {} }), on: (name, fn) => appEvents.set(name, fn), quit: () => { quits++; } },
    BrowserWindow: { getAllWindows: () => [] },
    ipcMain: { handle: (name, handler) => handlers.set(name, handler) }, dialog: {}, Menu: {}
  };
  const sandbox = { module: { exports: {} }, __dirname: appDir, process, Buffer, setTimeout,
    clearTimeout, require(name) {
      if (name === "electron") return electron;
      if (name === "ssh2") return { Client: FakeClient };
      if (name === "./service-diagnostics") return { ServiceDiagnostics: class {
        constructor() { this.running = false; }
        async start(...args) { diagnosticsCalls.push(args); this.running = true; }
        async stop() { this.running = false; diagnosticsStops++; }
      } };
      if (name === "./initialize-deploy") return { initializeJetson: async (...args) => { initializeCalls.push(args); return initialize(...args); } };
      if (name === "./launch-config") return require("../launch-config");
      if (name === "./maintenance") return require("../maintenance");
      if (name === "./robot-network") return require("../robot-network");
      if (name === "./network-setup") return require("../network-setup");
      if (name === "./radio-transfer") return require("../radio-transfer");
      if (name === "./activation-deploy") return {
        operateRelease: async (...args) => { activationCalls.push(args); return operation(...args); },
        releaseStatus: async () => overview
      };
      if (name === "./staging-deploy") return { stageProject: async (...args) => { stageCalls.push(args); return stage(...args); } };
      return require(name);
    } };
  const source = fs.readFileSync(path.join(appDir, "main.js"), "utf8");
  vm.runInNewContext(source+"\nmodule.exports = { APP_CONFIG_PATH, stopManagedStartupService, " +
    "beginManagedToolActivity, serviceGuard, syncRemoteStartupScript, buildMainProgram, startStaging, JOB_PATH, resumeDeployment };", sandbox);
  return { ...sandbox.module.exports, commands, writes, handlers, stageCalls, activationCalls, initializeCalls, diagnosticsCalls, appEvents,
    get quits() { return quits; }, get diagnosticsStops() { return diagnosticsStops; }, overview, directory, connection: new FakeClient() };
}

test("closing the app detaches production diagnostics without invoking removed legacy managers", async (t) => {
  const h = harness(t);
  let prevented = false;
  h.appEvents.get("before-quit")({ preventDefault() { prevented = true; } });
  await new Promise(setImmediate);
  assert(prevented);
  assert.equal(h.diagnosticsStops, 1);
  assert.equal(h.quits, 1);
  assert.equal(h.commands.length, 0);
});

const settings = { local_path: "C:/source/Vortex", remote_path: "/home/vortex/deployments",
  tag_map_path: "", host: "never-contacted", startup_camera_indices: "0,2", build_profile: "gpu",
  vortex_nt_enable: true, vortex_nt_mode: "local", vortex_udp_port: 5092, vortex_udp_target: "robot.local" };

test("packaged app stores preferences in writable userData", async (t) => {
  const h = harness(t);
  assert.equal(h.APP_CONFIG_PATH, path.join(h.directory, "deployment_tool_config.json"));
  await h.handlers.get("save-app-config")(null, settings);
  assert.equal(JSON.parse(fs.readFileSync(h.APP_CONFIG_PATH)).build_profile, "gpu");
});

test("failed service stop aborts and does not acquire an activity count", async (t) => {
  const h = harness(t, () => 1);
  await assert.rejects(h.beginManagedToolActivity({ ...settings, startup_service_name: "vortex-vision.service" }, "test"),
    /Service stop failed/);
  assert.equal(h.serviceGuard.activeCount, 0);
});

test("service names cannot become shell commands", async (t) => {
  const h = harness(t);
  await assert.rejects(h.stopManagedStartupService({ ...settings, startup_service_name: "vortex; echo bad" }), /valid .service/);
  assert.equal(h.commands.length, 0);
});

test("startup sync writes structured device settings and the common launcher", async (t) => {
  const h = harness(t);
  await h.syncRemoteStartupScript(h.connection, settings);
  const payload = [...h.writes.entries()].find(([file]) => file.includes("/config/device.json."))[1];
  const device = JSON.parse(payload.toString());
  assert.deepEqual(device.cameras, [0, 2]);
  assert.equal(device.build_profile, "gpu");
  assert.equal(device.udp.port, 5092);
  assert.equal(device.udp.target, "robot.local");
  assert.equal(device.networktables.mode, "local");
  const launcher = [...h.writes.entries()].find(([file]) => file.includes("/run_vortex.sh."))[1].toString();
  assert(launcher.includes("tools/vortexctl"));
  assert(h.commands.some((command) => command.startsWith("mv -f --")));
});

test("startup sync propagates a failed remote write/activation command", async (t) => {
  const h = harness(t, (command) => command.startsWith("mv ") ? 1 : 0);
  await assert.rejects(h.syncRemoteStartupScript(h.connection, settings), /Remote command failed/);
});

test("legacy build API stages a snapshot instead of building the running tree", async (t) => {
  const h = harness(t);
  await h.buildMainProgram(settings);
  assert.equal(h.stageCalls.length, 1);
  assert.equal(h.stageCalls[0][1], "/home/vortex/deployments/Vortex");
  assert.equal(h.commands.length, 0);
});

test("staging does not stop/resume the service and rejects concurrent app requests", async (t) => {
  let finish;
  const h = harness(t, () => 0, () => new Promise((resolve) => { finish = resolve; }));
  const cfg = { ...settings, startup_service_name: "vortex-vision.service" };
  assert.equal(h.startStaging(cfg).ok, true);
  assert.equal(h.startStaging(cfg).ok, false);
  assert.equal(h.serviceGuard.activeCount, 0);
  assert.equal(h.commands.length, 0);
  finish({ state: "ready" });
  await new Promise(setImmediate);
  assert.equal(h.startStaging({ ...cfg, build_profile: "invalid" }).ok, false);
});

test("diagnostics ignores legacy launch commands and only attaches to the service", async (t) => {
  const h = harness(t);
  const cfg = { ...settings, monitor_start_cmd: "unverified-program", preview_camera_index: 2 };
  assert.equal((await h.handlers.get("monitor-start")(null, cfg)).ok, true);
  assert.equal(h.diagnosticsCalls.length, 1);
  assert.equal(h.commands.length, 0);
});

test("activation IPC shares the staging guard while diagnostics attaches without service control", async (t) => {
  let finish;
  const h = harness(t, () => 0, undefined, () => new Promise((resolve) => { finish = resolve; }));
  const operate = h.handlers.get("release-operate");
  assert.equal((await operate(null, settings, "activate", "new")).ok, true);
  assert.equal((await operate(null, settings, "rollback", "new")).ok, false);
  assert.equal(h.activationCalls.length, 1);
  finish({ state: "active" });
  await new Promise(setImmediate);
  h.overview.active = { kind: "release", release_id: "new" };
  const result = await h.handlers.get("monitor-start")(null, { ...settings, last_ready_release: "new" });
  assert.equal(result.ok, true);
  assert.equal(h.diagnosticsCalls.length, 1);
  await h.handlers.get("monitor-stop")(null);
  assert.equal(h.commands.length, 0);
});


function savedJob(h, job, overrides = {}) {
  fs.writeFileSync(h.JOB_PATH, JSON.stringify({ schema_version: 1,
    endpoint: { host: settings.host, user: "vortex", port: "22", target: "/home/vortex/deployments/Vortex", ...overrides }, job }));
}

test("saved jobs reattach through IPC on the original device and block duplicate app operations", async (t) => {
  let finish;
  const h = harness(t, () => 0, () => new Promise((resolve) => { finish = resolve; }));
  savedJob(h, { kind: "stage", release_id: "old-job", build_profile: "cpu" });
  const [result] = await Promise.all([
    h.handlers.get("deployment-resume")(null, settings), h.resumeDeployment(settings)
  ]);
  assert.equal(result.attached, true);
  assert.equal(h.stageCalls.length, 1);
  assert.equal(h.stageCalls[0][0].build_profile, "cpu");
  assert.equal(h.stageCalls[0][2].resume.release_id, "old-job");
  assert.equal(h.startStaging(settings).ok, false);
  assert.equal((await h.handlers.get("release-operate")(null, settings, "activate", "new")).ok, false);
  assert.equal((await h.resumeDeployment(settings)).attached, true);
  assert.equal(h.stageCalls.length, 1);
  await h.stageCalls[0][2].finished();
  finish({ state: "ready", release_id: "old-job" });
  // Allow the persistent last-ready update to complete before temporary cleanup.
  for (let i = 0; i < 30 && !fs.existsSync(h.APP_CONFIG_PATH); i++) await new Promise((resolve) => setTimeout(resolve, 5));
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(JSON.parse(fs.readFileSync(h.APP_CONFIG_PATH)).last_ready_release, "old-job");
});

test("pending activation reconnects using its saved operation ID without preparing another release", async (t) => {
  const h = harness(t);
  savedJob(h, { kind: "rollback", operation_id: "activation-original", release_id: "new" });
  assert.equal((await h.resumeDeployment(settings)).attached, true);
  assert.equal(h.activationCalls[0][2], "rollback");
  assert.equal(h.activationCalls[0][4].resume.operation_id, "activation-original");
  await new Promise(setImmediate);
});

test("saved job from another device is never polled or overwritten", async (t) => {
  const h = harness(t);
  savedJob(h, { kind: "activate", operation_id: "original", release_id: "new" }, { host: "other-device" });
  assert.equal((await h.resumeDeployment(settings)).ok, false);
  assert.equal(h.stageCalls.length, 0);
  assert.equal(h.activationCalls.length, 0);
  assert.equal(h.startStaging(settings).ok, false);
  assert.equal(h.commands.length, 0);
});

test("saved job persistence contains only job identity and device location", async (t) => {
  const h = harness(t);
  await h.buildMainProgram(settings);
  const callbacks = h.stageCalls[0][2];
  await callbacks.remember({ kind: "stage", release_id: "original" });
  const saved = JSON.parse(fs.readFileSync(h.JOB_PATH));
  assert.deepEqual(Object.keys(saved).sort(), ["endpoint", "job", "schema_version"]);
  assert.equal(saved.endpoint.host, settings.host);
  assert(!fs.readFileSync(h.JOB_PATH, "utf8").includes("redstorm"));
  await callbacks.finished();
  assert.equal(fs.existsSync(h.JOB_PATH), false);
});


test("Initialize Jetson uses the shared operation guard and passes administrator credentials separately", async (t) => {
  let finish;
  const h = harness(t, () => 0, undefined, undefined, () => new Promise((resolve) => { finish = resolve; }));
  const handler = h.handlers.get("jetson-initialize");
  const cfg = { ...settings, user: "operator", port: "2222", pass: "ssh-secret" };
  assert.equal((await handler(null, cfg, "sudo-secret")).ok, true);
  assert.equal(h.initializeCalls[0][2], "sudo-secret");
  assert.equal(h.initializeCalls[0][3].user, "operator");
  assert.equal(h.initializeCalls[0][0].admin_password, undefined);
  assert.equal((await handler(null, cfg, "sudo-secret")).ok, false);
  assert.equal(h.startStaging(cfg).ok, false);
  assert.equal((await h.handlers.get("release-operate")(null, cfg, "activate", "new")).ok, false);
  finish({ state: "ready", release_id: "new" });
  for (let i = 0; i < 30 && !fs.existsSync(h.APP_CONFIG_PATH); i++) await new Promise((resolve) => setTimeout(resolve, 5));
  await new Promise((resolve) => setTimeout(resolve, 20));
});

test("SSH connection settings are editable and passwords are excluded from saved preferences", async (t) => {
  const h = harness(t);
  await h.handlers.get("save-app-config")(null, { ...settings, user: "operator", port: "2222", pass: "ssh-secret", admin_password: "sudo-secret" });
  const saved = JSON.parse(fs.readFileSync(h.APP_CONFIG_PATH));
  assert.equal(saved.user, "operator");
  assert.equal(saved.port, "2222");
  assert.equal(saved.pass, "");
  assert.equal(saved.admin_password, undefined);
  assert(!fs.readFileSync(h.APP_CONFIG_PATH, "utf8").includes("secret"));
});

test("saved initialization reattaches through the app without requiring a sudo password again", async (t) => {
  let finish;
  const h = harness(t, () => 0, undefined, undefined, () => new Promise((resolve) => { finish = resolve; }));
  savedJob(h, { kind: "initialize", operation_id: "initialize-original", release_id: "new", build_profile: "cpu" });
  assert.equal((await h.resumeDeployment(settings)).attached, true);
  assert.equal(h.initializeCalls[0][2], "");
  assert.equal(h.initializeCalls[0][3].resume.operation_id, "initialize-original");
  finish({ state: "ready", release_id: "new" });
  for (let i = 0; i < 30 && !fs.existsSync(h.APP_CONFIG_PATH); i++) await new Promise((resolve) => setTimeout(resolve, 5));
  await new Promise((resolve) => setTimeout(resolve, 20));
});

test("fresh app bootstrap defaults to robot Ethernet and pending USB jobs keep their endpoint", async t => {
  const h=harness(t);
  fs.writeFileSync(h.APP_CONFIG_PATH,"{}");
  let boot=await h.handlers.get("bootstrap")();
  assert.equal(boot.appConfig.connection_mode,"radio");
  assert.equal(boot.appConfig.host,"10.5.9.12");
  assert.equal(boot.appConfig.port,"5808");
  fs.writeFileSync(h.APP_CONFIG_PATH,JSON.stringify({host:"192.168.55.1",port:"22",user:"vortex"}));
  fs.writeFileSync(h.JOB_PATH,"{}");
  boot=await h.handlers.get("bootstrap")();
  assert.equal(boot.appConfig.connection_mode,"custom");
  assert.equal(boot.appConfig.host,"192.168.55.1");
  assert.equal(boot.appConfig.port,"22");
});
