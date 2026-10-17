"use strict";
const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function renderer() {
  const elements = new Map(), saves = [], applied = [], timers = [];
  const sandbox = { module: { exports: {} }, document: { querySelector: (selector) => {
    if (!elements.has(selector)) elements.set(selector, {
      value: "config.json", src: "", querySelectorAll: () => [], removeAttribute(name) { this[name] = ""; }
    });
    return elements.get(selector);
  } }, window: { vortexApi: { saveRuntimeConfig: async (...args) => saves.push(args) } },
  setTimeout: (fn) => { timers.push(fn); return 1; }, clearTimeout() {}, requestAnimationFrame: () => 1 };
  const source = fs.readFileSync(path.join(__dirname, "../renderer/app.js"), "utf8")
    .replace(/\ninit\(\)\.catch[^\n]*\n?$/, "");
  vm.runInNewContext(source+"\napplyBridgeState = (value) => applied.push(value);\n"+
    "renderDetectionTable = () => {}; renderRobotPose = () => {};\n"+
    "module.exports = { state, el, showServiceFrame, finishServiceFrame, previewSourceDims, scheduleRuntimeSave };",
  Object.assign(sandbox, { applied }));
  return { ...sandbox.module.exports, applied, saves, timers };
}

test("overlays change only when the corresponding queued image has loaded", () => {
  const h = renderer();
  const one = { dataUrl: "image-one", state: { camera_index: 2, frame_id: "one", source_width: 1920, source_height: 1080 } };
  const two = { dataUrl: "image-two", state: { camera_index: 2, frame_id: "two", source_width: 1280, source_height: 720 } };
  h.showServiceFrame(one);
  h.showServiceFrame(two);
  assert.equal(h.el.previewImage.src, "image-one");
  assert.equal(h.applied.length, 0);
  assert.equal(h.finishServiceFrame(), true);
  assert.equal(h.applied[0].frame_id, "one");
  assert.equal(h.el.previewImage.src, "image-two");
  assert.equal(h.previewSourceDims().w, 1920);
  h.finishServiceFrame();
  assert.equal(h.applied[1].frame_id, "two");
  assert.equal(h.previewSourceDims().w, 1280);
});

test("stale/disconnected preview clears queued images so late loads cannot revive them", () => {
  const h = renderer();
  h.showServiceFrame({ dataUrl: "image", state: { camera_index: 0 } });
  h.showServiceFrame(null);
  assert.equal(h.el.previewImage.src, "");
  assert.equal(h.finishServiceFrame(), false);
  assert.equal(h.applied.length, 0);
});

test("editing a runtime config during attachment saves locally without remote writes", async () => {
  const h = renderer();
  h.state.pipelineRunning = true;
  h.scheduleRuntimeSave();
  await h.timers.pop()();
  assert.equal(h.saves.length, 1);
  assert.equal(h.saves[0][0], "config.json");
});
