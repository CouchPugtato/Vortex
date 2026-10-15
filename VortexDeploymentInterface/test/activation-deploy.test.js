"use strict";
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { operateRelease, releaseStatus } = require("../activation-deploy");

const settings = { last_ready_release: "new" };
const remote = "/home/vortex/deployments/Vortex";

function harness(states = ["active"], overview = { service_configured: true, controller_current: true }) {
  const commands = [], progress = [], logs = [];
  let operation, connects = 0;
  const deps = {
    pollMs: 0, delay: async () => {},
    connect: async () => { connects++; return { end() {} }; },
    execChecked: async (conn, command) => {
      commands.push(command);
      if (command.startsWith("nohup ")) operation = command.match(/--operation-id '([^']+)'/)[1];
      if (command.includes(" status ") && !command.includes("--operation-id")) return { stdout: JSON.stringify(overview) };
      if (command.includes(" status ")) {
        const next = states.shift() || "active";
        if (next instanceof Error) throw next;
        const value = typeof next === "string" ? { state: next } : next;
        return { stdout: JSON.stringify({ operation_id: operation, action: "activate", verification: "all_cameras_advancing",
          selection: { kind: "release", release_id: "new" }, ...value }) };
      }
      return { stdout: "{}" };
    },
    log: (value) => logs.push(value), progress: (value) => progress.push(value)
  };
  return { deps, commands, progress, logs, get connects() { return connects; } };
}

test("activation uses a detached saved operation and waits for verified camera health", async () => {
  const h = harness(["validating", "stopping", "switching", "verifying", "active"]);
  const result = await operateRelease(settings, remote, "activate", "new", h.deps);
  assert.equal(result.state, "active");
  assert(h.commands.some((command) => command.startsWith("nohup ") && command.includes(" activate ")));
  assert.equal(h.progress.at(-1).text, "Release active — camera health verified");
});

test("first activation prepares the managed service without hiding sudo errors", async () => {
  const h = harness(["active"], { service_configured: false });
  await operateRelease(settings, remote, "activate", "new", h.deps);
  assert(h.commands.some((command) => command.startsWith("sudo -n ") && command.includes(" install-service ")));
  const failed = harness([], { service_configured: false });
  const checked = failed.deps.execChecked;
  failed.deps.execChecked = async (conn, command) => {
    if (command.startsWith("sudo ")) throw new Error("a password is required");
    return checked(conn, command);
  };
  await assert.rejects(operateRelease(settings, remote, "activate", "new", failed.deps), /Run this once on the Jetson with sudo/);
  assert(!failed.commands.some((command) => command.startsWith("nohup ")));
});

test("automatic rollback is reported as a failed activation with restored installation", async () => {
  const h = harness([{ state: "rolled_back", error: "camera 2 missing" }]);
  await assert.rejects(operateRelease(settings, remote, "activate", "new", h.deps), /previous installation restored/);
  assert.equal(h.progress.at(-1).status, "error");
});

test("rollback failure never claims the previous release is healthy", async () => {
  const h = harness([{ state: "rollback_failed", error: "new failed", rollback_error: "old camera missing" }]);
  await assert.rejects(operateRelease(settings, remote, "activate", "new", h.deps), /rollback also failed/);
});

test("unverified, mismatched, or wrong-operation results cannot succeed", async () => {
  for (const change of [{ verification: "process_alive" }, { operation_id: "other" },
    { selection: { kind: "release", release_id: "other" } }, { action: "rollback" }]) {
    const h = harness([{ state: "active", ...change }]);
    await assert.rejects(operateRelease(settings, remote, "activate", "new", h.deps), /Unexpected/);
  }
});

test("temporary disconnection reconnects without starting another activation", async () => {
  const h = harness([new Error("disconnected"), "active"]);
  await operateRelease(settings, remote, "activate", "new", h.deps);
  assert.equal(h.connects, 2);
  assert.equal(h.commands.filter((command) => command.startsWith("nohup ")).length, 1);
});

test("manual rollback can restore a legacy process without claiming camera verification", async () => {
  const h = harness([{ state: "active", action: "rollback", selection: { kind: "legacy" }, verification: "legacy_process_only" }]);
  await operateRelease(settings, remote, "rollback", "new", h.deps);
  assert.equal(h.progress.at(-1).text, "Rollback verified");
  assert(h.commands.some((command) => command.startsWith("nohup ") && command.includes(" rollback ")));
});

test("invalid action/release is rejected before any remote command", async () => {
  const h = harness();
  await assert.rejects(operateRelease(settings, remote, "delete", "new", h.deps), /Unknown/);
  await assert.rejects(operateRelease({}, remote, "activate", "../bad", h.deps), /valid release/);
  assert.equal(h.commands.length, 0);
});

test("release overview is read without service mutation", async () => {
  const overview = { ready: ["new"], active: { kind: "none" } };
  const h = harness([], overview);
  assert.deepEqual(await releaseStatus(settings, remote, h.deps), overview);
  assert.equal(h.commands.length, 1);
  assert(h.commands[0].includes(" status "));
});

test("a fresh desktop can inspect the installed controller without a saved release ID", async () => {
  const h = harness([], { active: { kind: "release", release_id: "new" }, ready: ["new"] });
  const overview = await releaseStatus({}, remote, h.deps);
  assert.equal(overview.active.release_id, "new");
  assert(h.commands[0].startsWith("if [ -f "));
  assert(!h.commands[0].includes("sudo"));
});
