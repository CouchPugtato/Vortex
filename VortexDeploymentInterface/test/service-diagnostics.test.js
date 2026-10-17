"use strict";
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { ServiceDiagnostics } = require("../service-diagnostics");

const settings = { preview_camera_index: 2 };
const target = "/home/vortex/deployments/Vortex";
function harness(states = [{}]) {
  const commands = [], frames = [], statuses = [], monitor = [], delays = [], logs = [];
  let connections = 0;
  const deps = {
    connect: async () => { connections++; return { end() {} }; },
    execChecked: async (conn, command) => {
      commands.push(command);
      if (command.includes(" diagnostics-stop ")) return { stdout: "{}" };
      const next = states.shift() || {};
      if (next instanceof Error) throw next;
      const client = command.match(/--client-id '([^']+)'/)[1];
      const camera = Number(command.match(/--camera (\d+)/)[1]);
      const value = { schema_version: 1, ready: true, camera_index: camera,
        service: { pid: 44 }, health: { pid: 44, release_id: "active", launch_token: "nonce" },
        state: { pid: 44, release_id: "active", launch_token: "nonce", camera_index: camera,
          client_id: client, frame_id: "frame-44-7.jpg" }, jpeg_base64: Buffer.from([255, 216, 255, 224]).toString("base64") };
      return { stdout: JSON.stringify({ ...value, ...next, state: next.state ? { ...value.state, ...next.state } : value.state }) };
    },
    delay: async () => new Promise((resolve) => delays.push(resolve)),
    frame: (value) => frames.push(value), status: (value) => statuses.push(value),
    log: (value) => logs.push(value), monitorState: (value) => monitor.push(value)
  };
  const client = new ServiceDiagnostics(deps);
  return { client, deps, commands, frames, statuses, monitor, logs, get connections() { return connections; },
    tick: async () => { assert(delays.length); delays.shift()(); await new Promise(setImmediate); } };
}

test("attachment, camera switching and detach never control production or open cameras", async () => {
  const h = harness();
  await h.client.start(settings, target);
  assert.equal(h.frames[0].state.camera_index, 2);
  await h.client.stop();
  await h.client.start({ preview_camera_index: 0 }, target);
  assert.equal(h.frames.at(-1).state.camera_index, 0);
  await h.client.stop();
  await h.client.start({ preview_camera_index: 63 }, target);
  assert.equal(h.frames.at(-1).state.camera_index, 63);
  await h.client.stop();
  assert(h.commands.every((command) => command.includes(" diagnostics ") || command.includes(" diagnostics-stop ")));
  assert(!h.commands.some((command) => /pkill|systemctl|run_vortex|\/dev\/video| activate /.test(command)));
  assert.deepEqual(h.monitor, [true, false, true, false, true, false]);
});

test("repeated polls do not reload the same frame", async () => {
  const h = harness([{}, {}, { state: { frame_id: "frame-44-8.jpg" } }]);
  await h.client.start(settings, target);
  await h.tick();
  assert.equal(h.frames.length, 1);
  await h.tick();
  assert.equal(h.frames.length, 2);
  await h.client.stop();
});

test("outage clears stale images and reconnects without restarting a service", async () => {
  const h = harness([{}, new Error("disconnected"), {}]);
  await h.client.start(settings, target);
  await h.tick();
  assert.equal(h.frames.at(-1).dataUrl, null);
  assert(h.statuses.at(-1).reason.includes("reconnecting"));
  await h.tick();
  assert.equal(h.connections, 2);
  assert(h.frames.at(-1).dataUrl.startsWith("data:image/jpeg;"));
  await h.client.stop();
  const identities = h.commands.map((command) => command.match(/--client-id '([^']+)'/)[1]);
  assert.equal(new Set(identities).size, 1);
});

test("inactive service and stale preview are waiting states with no cached image", async () => {
  const h = harness([{}, { ready: false, reason: "Service is not running" }, {}]);
  await h.client.start(settings, target);
  await h.tick();
  assert.equal(h.frames.at(-1).dataUrl, null);
  assert.equal(h.client.running, true);
  await h.tick();
  assert(h.frames.at(-1).dataUrl);
  await h.client.stop();
});

test("mismatched camera, client, runtime, schema or image never reaches the renderer", async () => {
  for (const change of [{ state: { camera_index: 0 } }, { state: { client_id: "other" } },
    { state: { launch_token: "old" } }, { state: { pid: 1 } }, { schema_version: 2 },
    { jpeg_base64: "invalid" }, { state: { frame_id: "../../secret" } }]) {
    const h = harness([change]);
    await assert.rejects(h.client.start(settings, target), /Unexpected|Mismatched/);
    assert.equal(h.frames.length, 0);
    assert.equal(h.client.running, false);
  }
});

test("stopping a pending attachment does not publish a late frame or revive the monitor", async () => {
  const h = harness();
  let finish;
  h.deps.connect = async () => new Promise((resolve) => { finish = resolve; });
  const attaching = h.client.start(settings, target);
  await h.client.stop();
  finish({ end() {} });
  await attaching;
  assert.equal(h.client.running, false);
  assert(h.frames.every((frame) => frame.dataUrl === null));
  assert(!h.monitor.includes(true));
});

test("invalid camera indices are rejected before SSH", async () => {
  const h = harness();
  await assert.rejects(h.client.start({ preview_camera_index: -1 }, target), /Invalid/);
  assert.equal(h.connections, 0);
});

test("slow transfers discard an otherwise correctly paired frame", async () => {
  const h = harness([{ state: { frame_age_ms: 2000 } }]);
  await h.client.start(settings, target);
  assert.equal(h.frames.at(-1).dataUrl, null);
  assert(h.statuses.at(-1).reason.includes("stale"));
  await h.client.stop();
});

test("detaching a lost connection is bounded and relies on lease expiry", async () => {
  const h = harness();
  await h.client.start(settings, target);
  h.deps.stopTimeoutMs = 5;
  h.deps.execChecked = async () => new Promise(() => {});
  await h.client.stop();
  assert.equal(h.client.running, false);
  assert.equal(h.frames.at(-1).dataUrl, null);
  assert.equal(h.monitor.at(-1), false);
});
