"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");
const { stageProject, packageLocally, managedRoot, quote } = require("../staging-deploy");

const settings = { local_path: "unused", build_profile: "cpu" };
const target = "/home/vortex/deployments/Vortex";
const root = target+".deployment";
const metadata = { schema_version: 1, release_id: "release_test_1", build_profile: "cpu",
  archive_sha256: "a".repeat(64), controller_sha256: "b".repeat(64) };
const ready = { release_id: metadata.release_id, state: "ready", activated: false,
  profile: "cpu", release_path: `${root}/releases/${metadata.release_id}` };

function fake(states = [ready]) {
  const commands = [], uploads = [], events = [], logs = [];
  let connects = 0, localDirectory;
  const connection = { end() {} };
  const deps = {
    appDir: __dirname, pollMs: 0, delay: async () => {},
    connect: async () => { connects++; return connection; },
    execChecked: async (conn, command) => {
      commands.push(command);
      if (command.includes(" status ")) {
        const value = states.shift() || ready;
        if (value instanceof Error) throw value;
        return { stdout: JSON.stringify(value), code: 0 };
      }
      return { stdout: command.startsWith("tail ") ? "compiler log" : "", code: 0 };
    },
    getSftp: async () => ({}),
    upload: async (sftp, local, remote, options) => {
      await fs.access(local);
      uploads.push(remote);
      options?.step(50, 50, 100);
    },
    package: async (config, directory) => {
      localDirectory = directory;
      await fs.writeFile(path.join(directory, "release.tar.gz"), "archive");
      await fs.writeFile(path.join(directory, "vortexctl"), "controller");
      return metadata;
    },
    progress: (event) => events.push(event), log: (line) => logs.push(line)
  };
  return { deps, commands, uploads, events, logs,
    get connects() { return connects; }, get directory() { return localDirectory; } };
}

test("archive upload and managed build never delete or restart the current installation", async () => {
  const h = fake([{ ...ready, state: "building" }, { ...ready, state: "checking" }, ready]);
  assert.equal((await stageProject(settings, target, h.deps)).state, "ready");
  assert.equal(h.uploads.length, 2);
  assert(h.uploads.every((file) => file.startsWith(`${root}/incoming/`)));
  assert(h.commands.some((command) => command.includes(" start-job ") && command.includes("--kind stage")));
  assert(!h.commands.some((command) => /systemctl|pkill|rm -rf|find |syncRemoteStartupScript/.test(command)));
  assert(h.commands.findIndex((command) => command.includes("sha256sum")) < h.commands.findIndex((command) => command.includes(" start-job ")));
  assert.equal(h.events.at(-1).text, "Release ready — not activated");
  await assert.rejects(fs.access(h.directory));
});

test("a disconnected SSH client reconnects to the same job without launching it twice", async () => {
  const h = fake([new Error("connection reset"), ready]);
  await stageProject(settings, target, h.deps);
  assert.equal(h.connects, 2);
  assert.equal(h.commands.filter((command) => command.includes(" start-job ")).length, 1);
});

test("failed builds report saved logs and never activate anything", async () => {
  const h = fake([{ ...ready, state: "failed", error: "compile failed" }]);
  await assert.rejects(stageProject(settings, target, h.deps), /compile failed/);
  assert(h.logs.includes("compiler log"));
  assert.equal(h.events.at(-1).status, "error");
  assert(!h.commands.some((command) => /systemctl|pkill|rm -rf/.test(command)));
  await assert.rejects(fs.access(h.directory));
});

test("device lock rejection is surfaced from the saved job", async () => {
  const h = fake([{ ...ready, state: "failed", error: "Another Vortex deployment is in progress" }]);
  await assert.rejects(stageProject(settings, target, h.deps), /Another Vortex deployment/);
});

test("checksum verification failure prevents launching the controller", async () => {
  const h = fake();
  const execute = h.deps.execChecked;
  h.deps.execChecked = async (conn, command) => {
    if (command.includes("sha256sum")) throw new Error("checksum mismatch");
    return execute(conn, command);
  };
  await assert.rejects(stageProject(settings, target, h.deps), /checksum mismatch/);
  assert(!h.commands.some((command) => command.includes(" start-job ")));
});

test("failed uploads do not start a remote build", async () => {
  const h = fake();
  h.deps.upload = async () => { throw new Error("transfer failed"); };
  await assert.rejects(stageProject(settings, target, h.deps), /transfer failed/);
  assert(!h.commands.some((command) => command.includes(" start-job ")));
});

test("local packaging failure contacts no Jetson", async () => {
  const h = fake();
  h.deps.package = async () => { throw new Error("invalid calibration"); };
  await assert.rejects(stageProject(settings, target, h.deps), /invalid calibration/);
  assert.equal(h.connects, 0);
});

test("timeout leaves managed job running and reports where to inspect it", async () => {
  const h = fake();
  h.deps.timeoutMs = 0;
  await assert.rejects(stageProject(settings, target, h.deps), /Stopped waiting/);
  assert(h.logs.some((line) => line.includes("Job may still be running")));
  assert(!h.commands.some((command) => /kill|rm -rf/.test(command)));
});

test("unexpected ready identity, profile, or destination cannot be reported as success", async () => {
  for (const change of [{ release_id: "other" }, { profile: "gpu" }, { release_path: "/running" }, { activated: true }]) {
    const h = fake([{ ...ready, ...change }]);
    await assert.rejects(stageProject(settings, target, h.deps), /Unexpected/);
  }
});

test("invalid remote directories fail before packaging or SSH", async () => {
  for (const destination of ["/", "relative", "/home/../etc", "/home//vortex", "/home/vortex\ncommand"]) {
    const h = fake();
    await assert.rejects(stageProject(settings, destination, h.deps), /normalized absolute/);
    assert.equal(h.connects, 0);
  }
  assert.equal(managedRoot(target), root);
  assert.equal(quote("a'b"), "'a'\\''b'");
});

test("real local packager snapshots the repository and selected configuration", async (t) => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "vortex-package-test-"));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const repository = path.resolve(__dirname, "../..");
  const result = await packageLocally({ ...settings, local_path: repository,
    runtime_config_path: path.join(repository, "config/config.json"),
    tag_map_path: path.join(repository, "config/2026-rebuilt-welded.json"),
    deployment_architecture: "aarch64" },
  directory, path.dirname(__dirname));
  const bytes = await fs.readFile(path.join(directory, "release.tar.gz"));
  assert.equal(crypto.createHash("sha256").update(bytes).digest("hex"), result.archive_sha256);
  assert.equal(bytes.length, result.archive_bytes);
  assert.equal(result.build_profile, "cpu");
});


test("reopening restores a stage job and compiler logs without packaging, uploading or submitting", async () => {
  const h = fake([{ ...ready, state: "building", log_tail: "Compiling vortex\n" }, ready]);
  h.deps.resume = { kind: "stage", ...metadata };
  h.deps.package = async () => { throw new Error("must not package on reconnect"); };
  let finished = 0;
  h.deps.finished = async () => finished++;
  await stageProject(settings, target, h.deps);
  assert.equal(h.uploads.length, 0);
  assert.equal(h.commands.filter((command) => command.includes(" start-job ")).length, 0);
  assert(h.commands.every((command) => command.includes(" status ")));
  assert(h.logs.some((line) => line.includes("Compiling vortex")));
  assert.equal(finished, 1);
});

test("save precedes submission and a lost submission reply keeps the job resumable", async () => {
  const h = fake();
  let saved, finished = false;
  h.deps.remember = async (job) => { saved = job; };
  h.deps.finished = async () => { finished = true; };
  const execute = h.deps.execChecked;
  h.deps.execChecked = async (conn, command) => {
    if (command.includes(" start-job ")) {
      assert.equal(saved.release_id, metadata.release_id);
      throw new Error("SSH lost reply");
    }
    return execute(conn, command);
  };
  await assert.rejects(stageProject(settings, target, h.deps), /SSH lost reply/);
  assert.equal(saved.kind, "stage");
  assert.equal(finished, false);
});

test("interrupted stages restore saved logs, clear the pending record and report failure", async () => {
  const h = fake([{ ...ready, state: "interrupted", error: "Worker killed", log_tail: "last compiler output" }]);
  let finished = false;
  h.deps.finished = async () => { finished = true; };
  await assert.rejects(stageProject(settings, target, h.deps), /Worker killed/);
  assert(finished);
  assert(h.logs.includes("last compiler output"));
});
