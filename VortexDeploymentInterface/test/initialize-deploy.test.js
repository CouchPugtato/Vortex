"use strict";
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { EventEmitter } = require("node:events");
const fs = require("node:fs/promises");
const path = require("node:path");
const { initializeJetson, execAdministrator } = require("../initialize-deploy");

const remote = "/home/vortex/deployments/Vortex";
const root = remote+".deployment";
const metadata = { schema_version: 1, release_id: "source-one", build_profile: "cpu", archive_sha256: "a".repeat(64),
  controller_sha256: "b".repeat(64), initializer_sha256: "c".repeat(64) };
function harness(states = ["ready"]) {
  let operation, saved, finished = 0, connections = 0;
  const commands = [], uploads = [], progress = [], logs = [], administrator = [];
  const deps = {
    appDir: __dirname, user: "vortex", pollMs: 0, delay: async () => {},
    connect: async () => { connections++; return { end() {} }; },
    execChecked: async (conn, command) => {
      commands.push(command);
      if (command.includes(" status ")) {
        const next = states.shift() || "ready";
        if (next instanceof Error) throw next;
        return { stdout: JSON.stringify({ operation_id: operation || "initialize-original", state: "ready", action: "initialize",
          release_id: metadata.release_id, profile: "cpu", release_path: `${root}/releases/${metadata.release_id}`,
          activated: false, verification: "dependencies_build_smoke_service", ...(typeof next === "string" ? { state: next } : next) }) };
      }
      return { stdout: "" };
    },
    getSftp: async () => ({}), upload: async (sftp, local, destination) => { await fs.access(local); uploads.push(destination); },
    package: async (settings, folder) => {
      for (const file of ["release.tar.gz", "vortexctl"]) await fs.writeFile(path.join(folder, file), "snapshot");
      return metadata;
    },
    execAdministrator: async (conn, command, password) => {
      assert(saved, "must save identity before privileged submission");
      assert(!command.includes(password));
      operation = command.match(/--job-id '([^']+)'/)[1];
      administrator.push({ command, password });
    },
    remember: async (job) => { saved = job; }, finished: async () => { finished++; },
    progress: (value) => progress.push(value), log: (value) => logs.push(value)
  };
  return { deps, commands, uploads, progress, logs, administrator,
    get saved() { return saved; }, get finished() { return finished; }, get connections() { return connections; } };
}

test("initialization uploads and verifies setup, then uses privileged stdin submission", async () => {
  const h = harness(["packages", "toolchain", "building", "service", "ready"]);
  const result = await initializeJetson({}, remote, "private-admin-value", h.deps);
  assert.equal(result.activated, false);
  assert.equal(h.uploads.length, 2);
  assert(h.commands.some((command) => command.includes("sha256sum")));
  assert.equal(h.administrator.length, 1);
  assert.equal(h.administrator[0].password, "private-admin-value");
  assert.equal(h.saved.kind, "initialize");
  assert(!JSON.stringify(h.saved).includes("private-admin-value"));
  assert.equal(h.finished, 1);
  assert.equal(h.progress.at(-1).status, "done");
  assert(!h.commands.some((command) => command.includes(" activate ")));
});
test("offline setup uploads a checked bundle and forwards its exact hash to the detached worker", async (t) => {
  const os=require("node:os"),crypto=require("node:crypto");
  const folder=await fs.mkdtemp(path.join(os.tmpdir(),"vortex-offline-init-"));t.after(()=>fs.rm(folder,{recursive:true,force:true}));
  const bytes=Buffer.from("offline archive"),sha=crypto.createHash("sha256").update(bytes).digest("hex");
  await fs.writeFile(path.join(folder,"offline.tar.gz"),bytes);
  await fs.writeFile(path.join(folder,"offline.json"),JSON.stringify({schema_version:1,kind:"vortex-offline",profile:"cpu",host:{architecture:"aarch64"},archive_bytes:bytes.length,archive_sha256:sha}));
  const h=harness();await initializeJetson({offline_bundle_path:path.join(folder,"offline.tar.gz")},remote,"admin",h.deps);
  assert(h.uploads.some(p=>p.endsWith("/offline.tar.gz")));
  assert(h.administrator[0].command.includes("--offline-bundle"));assert(h.administrator[0].command.includes(`--offline-sha256 '${sha}'`));assert.equal(h.saved.offline_sha256,sha);
});

test("setup reattaches without uploading again or asking for administrator credentials", async () => {
  const h = harness([{ state: "building", log_tail: "compiling release" }, "ready"]);
  h.deps.resume = { ...metadata, kind: "initialize", operation_id: "initialize-original" };
  h.deps.package = async () => { throw new Error("must not package"); };
  await initializeJetson({}, remote, "", h.deps);
  assert.equal(h.uploads.length, 0);
  assert.equal(h.administrator.length, 0);
  assert(h.commands.every((command) => command.includes(" status ")));
  assert(h.logs.includes("compiling release"));
});

test("a lost SSH connection follows the original setup job without another root command", async () => {
  const h = harness([new Error("SSH disconnected"), "ready"]);
  await initializeJetson({}, remote, "secret", h.deps);
  assert.equal(h.connections, 2);
  assert.equal(h.administrator.length, 1);
});

test("checksum/upload failures prevent privileged setup", async () => {
  for (const failure of ["upload", "sha256sum"]) {
    const h = harness();
    if (failure === "upload") h.deps.upload = async () => { throw new Error("upload failed"); };
    else {
      const checked = h.deps.execChecked;
      h.deps.execChecked = async (conn, command) => {
        if (command.includes("sha256sum")) throw new Error("checksum failed");
        return checked(conn, command);
      };
    }
    await assert.rejects(initializeJetson({}, remote, "secret", h.deps), /failed/);
    assert.equal(h.administrator.length, 0);
  }
});

test("failed or interrupted provisioning cannot report a prepared Jetson", async () => {
  for (const state of ["failed", "interrupted"]) {
    const h = harness([{ state, error: "dependency install failed" }]);
    await assert.rejects(initializeJetson({}, remote, "secret", h.deps), /dependency install failed/);
    assert.equal(h.finished, 1);
    assert.equal(h.progress.at(-1).status, "error");
  }
});

test("unverified or mismatched setup results remain failures", async () => {
  for (const change of [{ operation_id: "wrong" }, { activated: true }, { verification: "process_alive" },
    { release_path: "/wrong" }, { profile: "gpu" }, { release_id: "wrong" }]) {
    const h = harness([{ state: "ready", ...change }]);
    await assert.rejects(initializeJetson({}, remote, "secret", h.deps), /Unexpected|Unverified/);
    assert.equal(h.finished, 0);
  }
});

function administratorChannel(code = 0, stderr = "") {
  const channel = new EventEmitter();
  channel.stderr = new EventEmitter();
  let command, input;
  channel.end = (value) => { input = value; queueMicrotask(() => {
    if (stderr) channel.stderr.emit("data", Buffer.from(stderr));
    channel.emit("close", code);
  }); };
  const conn = { exec: (value, callback) => { command = value; callback(null, channel); } };
  return { conn, get command() { return command; }, get input() { return input; } };
}

test("administrator passwords travel only through stdin, including shell metacharacters", async () => {
  const h = administratorChannel();
  const secret = "admin' $(touch /tmp/test) `value`";
  await execAdministrator(h.conn, "/vortexctl initialize-submit", secret);
  assert(h.command.startsWith("sudo -S -k -p '' -- "));
  assert(!h.command.includes(secret));
  assert.equal(h.input, secret+"\n");
});

test("passwordless sudo is noninteractive and sends no password", async () => {
  const h = administratorChannel();
  await execAdministrator(h.conn, "/vortexctl initialize-submit", "");
  assert(h.command.startsWith("sudo -n -- "));
  assert.equal(h.input, "");
});

test("authentication errors are actionable without echoing privileged output", async () => {
  const h = administratorChannel(1, "Sorry, try again. private-value");
  await assert.rejects(execAdministrator(h.conn, "/vortexctl initialize-submit", "private-value"), (error) => {
    assert.equal(error.notSubmitted, true);
    assert(!error.message.includes("private-value"));
    assert(error.message.includes("authentication failed"));
    return true;
  });
  const setup = harness();
  setup.deps.execAdministrator = async () => { const error = new Error("authentication failed"); error.notSubmitted = true; throw error; };
  await assert.rejects(initializeJetson({}, remote, "secret", setup.deps), /authentication/);
  assert.equal(setup.finished, 1);
});

test("uncertain submission errors preserve the job for inspection rather than resubmitting", async () => {
  const h = harness();
  h.deps.execAdministrator = async () => { throw new Error("connection interrupted"); };
  await assert.rejects(initializeJetson({}, remote, "secret", h.deps), /connection interrupted/);
  assert.equal(h.saved.kind, "initialize");
  assert.equal(h.finished, 0);
});
