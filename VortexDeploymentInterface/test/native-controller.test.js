"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");
const { nativeController, verifyController } = require("../native-controller");
const { quote } = require("../staging-deploy");

test("bundled native controllers exist and match their checksum manifest", async () => {
  const app = path.resolve(__dirname, "..");
  for (const [platform, architecture] of [["win32", "x64"], ["linux", "arm64"], ["linux", "x64"]]) {
    assert((await fs.stat(await nativeController(app, platform, architecture))).size > 100000);
  }
});
test("unsupported native platforms fail explicitly", async () => {
  await assert.rejects(nativeController(__dirname, "darwin", "arm64"), /Unsupported controller platform/);
});
test("missing native resources explain how to prepare a source build", async (t) => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "vortex-native-test-"));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  await assert.rejects(nativeController(directory, "win32", "x64"), /npm run build:native/);
});
test("altered native artifacts are rejected before execution", async (t) => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "vortex-native-test-"));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const target = "x86_64-pc-windows-gnu/vortexctl.exe";
  await fs.mkdir(path.join(directory, "native", path.dirname(target)), { recursive: true });
  const filename = path.join(directory, "native", target);
  await fs.writeFile(filename, "original");
  await fs.writeFile(path.join(directory, "native/manifest.json"), JSON.stringify({
    [target]: crypto.createHash("sha256").update("original").digest("hex")
  }));
  assert.equal(await nativeController(directory, "win32", "x64"), filename);
  await fs.writeFile(filename, "changed");
  await assert.rejects(nativeController(directory, "win32", "x64"), /checksum mismatch/);
});
test("remote verification quotes filenames and grants execute only after matching SHA256", () => {
  const command = verifyController("/home/robot's tool/vortexctl", "a".repeat(64), quote);
  assert(command.includes("sha256sum -- '/home/robot'\\''s tool/vortexctl'"));
  assert(command.includes("&& chmod 755 --"));
  assert(!command.includes("python"));
  assert.throws(() => verifyController("/x", "invalid", quote), /Invalid controller checksum/);
});
test("Electron packaging includes native resources and verifies them before shipping", () => {
  const pkg = require("../package.json");
  assert(pkg.build.files.includes("native-controller.js"));
  assert(pkg.build.extraResources.some((r) => r.from === "native" && r.to === "native"));
  for (const script of ["prepack", "predist", "predist:win"]) assert.equal(pkg.scripts[script], "npm run check:native");
});
