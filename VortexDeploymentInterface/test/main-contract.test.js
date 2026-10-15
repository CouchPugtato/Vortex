"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { EventEmitter } = require("node:events");
const { Writable } = require("node:stream");

function harness(t, exitCode = () => 0, stage = async () => ({ state: "ready" })) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "vortex-deployment-test-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const commands = [], writes = new Map(), handlers = new Map(), stageCalls = [];
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
    app: { getPath: () => directory, whenReady: () => ({ then() {} }), on() {} },
    BrowserWindow: { getAllWindows: () => [] },
    ipcMain: { handle: (name, handler) => handlers.set(name, handler) }, dialog: {}, Menu: {}
  };
  const sandbox = { module: { exports: {} }, __dirname: appDir, process, Buffer, setTimeout,
    clearTimeout, require(name) {
      if (name === "electron") return electron;
      if (name === "ssh2") return { Client: FakeClient };
      if (name === "./launch-config") return require("../launch-config");
      if (name === "./staging-deploy") return { stageProject: async (...args) => { stageCalls.push(args); return stage(...args); } };
      return require(name);
    } };
  const source = fs.readFileSync(path.join(appDir, "main.js"), "utf8");
  vm.runInNewContext(source+"\nmodule.exports = { APP_CONFIG_PATH, stopManagedStartupService, " +
    "beginManagedToolActivity, serviceGuard, syncRemoteStartupScript, buildMainProgram, resolveMonitorCommand, startStaging };", sandbox);
  return { ...sandbox.module.exports, commands, writes, handlers, stageCalls, directory, connection: new FakeClient() };
}

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
  assert(launcher.includes("scripts/run_vortex.py"));
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

test("diagnostics cannot bypass the launcher via legacy command settings", async (t) => {
  const h = harness(t);
  const command = await h.resolveMonitorCommand(h.connection,
    { ...settings, monitor_start_cmd: "unverified-program", preview_camera_index: 2 });
  assert(command.endsWith("python3 scripts/run_vortex.py --debug-bridge --camera 2"));
  assert(!command.includes("unverified-program"));
});
