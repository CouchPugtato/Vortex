const state = {
  appConfig: null,
  runtimePath: "",
  tagMapPath: "",
  tagMap: null,
  robotPose: null,
  hoveredTagId: null,
  runtimeConfig: {},
  cameras: new Map(),
  remoteCameras: [],
  selectedCamera: null,
  currentLineCamera: null,
  viewMode: "apriltag",
  pipelineRunning: false
};
const CAMERA_ACTIVE_MS = 4000;
const TAG_OVERLAY_STICKY_MS = 450;

const el = {
  runtimePath: document.querySelector("#runtimePath"),
  tagMapPath: document.querySelector("#tagMapPath"),
  browseTagMap: document.querySelector("#browseTagMap"),
  loadTagMap: document.querySelector("#loadTagMap"),
  loadRuntime: document.querySelector("#loadRuntime"),
  saveRuntime: document.querySelector("#saveRuntime"),
  runtimeForm: document.querySelector("#runtimeForm"),
  deployBtn: document.querySelector("#deployBtn"),
  deployProgressWrap: document.querySelector("#deployProgressWrap"),
  deployProgress: document.querySelector("#deployProgress"),
  deployProgressText: document.querySelector("#deployProgressText"),
  monitorProgressWrap: document.querySelector("#monitorProgressWrap"),
  monitorProgress: document.querySelector("#monitorProgress"),
  monitorProgressText: document.querySelector("#monitorProgressText"),
  togglePipeline: document.querySelector("#togglePipeline"),
  nextCamera: document.querySelector("#nextCamera"),
  viewMode: document.querySelector("#viewMode"),
  cameraPill: document.querySelector("#cameraPill"),
  fps: document.querySelector("#fps"),
  monitorStatus: document.querySelector("#monitorStatus"),
  detectionTableHead: document.querySelector("#detectionTable thead"),
  detectionTableBody: document.querySelector("#detectionTable tbody"),
  previewStatus: document.querySelector("#previewStatus"),
  previewImage: document.querySelector("#previewImage"),
  previewOverlay: document.querySelector("#previewOverlay"),
  logs: document.querySelector("#logs"),
  robotPoseText: document.querySelector("#robotPoseText"),
  fieldMapCanvas: document.querySelector("#fieldMapCanvas")
};

function appendLog(msg) {
  const stamp = new Date().toLocaleTimeString();
  el.logs.textContent += `[${stamp}] ${msg}\n`;
  el.logs.scrollTop = el.logs.scrollHeight;
}

let deployFadeTimer = null;
let monitorFadeTimer = null;
let runtimeSyncTimer = null;

function setDeployIndicator(stateKind, text, percent = null) {
  if (deployFadeTimer) {
    clearTimeout(deployFadeTimer);
    deployFadeTimer = null;
  }
  const wrap = el.deployProgressWrap;
  wrap.classList.remove("is-hidden", "is-done", "is-error");

  if (stateKind === "hidden") {
    wrap.classList.add("is-hidden");
    return;
  }
  if (stateKind === "done") wrap.classList.add("is-done");
  if (stateKind === "error") wrap.classList.add("is-error");

  if (typeof percent === "number") {
    el.deployProgress.value = Math.max(0, Math.min(100, percent));
  }
  if (text) el.deployProgressText.textContent = text;

  if (stateKind === "done" || stateKind === "error") {
    deployFadeTimer = setTimeout(() => {
      setDeployIndicator("hidden", "");
    }, 1400);
  }
}

function setMonitorIndicator(stateKind, text, percent = null) {
  if (monitorFadeTimer) {
    clearTimeout(monitorFadeTimer);
    monitorFadeTimer = null;
  }
  const wrap = el.monitorProgressWrap;
  wrap.classList.remove("is-hidden", "is-done", "is-error");

  if (stateKind === "hidden") {
    wrap.classList.add("is-hidden");
    return;
  }
  if (stateKind === "done") wrap.classList.add("is-done");
  if (stateKind === "error") wrap.classList.add("is-error");

  if (typeof percent === "number") {
    el.monitorProgress.value = Math.max(0, Math.min(100, percent));
  }
  if (text) el.monitorProgressText.textContent = text;

  if (stateKind === "done" || stateKind === "error") {
    monitorFadeTimer = setTimeout(() => {
      setMonitorIndicator("hidden", "");
    }, 1400);
  }
}

function hydrateAppConfig(cfg) {
  state.appConfig = cfg;
  state.tagMapPath = String(cfg?.tag_map_path || state.tagMapPath || "");
  if (el.tagMapPath) el.tagMapPath.value = state.tagMapPath;
  if (state.selectedCamera == null) {
    state.selectedCamera = 0;
  }
}

function readAppConfigFromUI() {
  const cam = Number(state.selectedCamera || 0);
  const normalizedCam = Number.isFinite(cam) && cam >= 0 && cam <= 5 ? cam : 0;
  const tagMapPath = String(el.tagMapPath?.value || state.tagMapPath || "").trim();
  return {
    ...state.appConfig,
    preview_camera_index: normalizedCam,
    tag_map_path: tagMapPath
  };
}

function renderRuntimeForm(config) {
  el.runtimeForm.innerHTML = "";
  const orderedCamera = [
    "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
    "tag_size_m", "x_offset", "y_offset", "z_offset", "pitch_deg", "yaw_deg", "roll_deg"
  ];
  const orderedProcessing = [
    "smoothing_alpha",
    "black_level_offset",
    "sensor_gain",
    "red_balance",
    "blue_balance"
  ];
  const orderedObjectDetection = [
    "yolo_obj_width_m",
    "yolo_obj_height_m",
    "confidence_threshold"
  ];

  const sections = [
    { key: "camera", title: "Camera Constants", fields: orderedCamera },
    { key: "processing", title: "Processing Constants", fields: orderedProcessing },
    { key: "object_detection", title: "Object Detection Constants", fields: orderedObjectDetection }
  ];
  const controlSpec = {
    "processing.smoothing_alpha": { type: "range", min: 0, max: 1, step: 0.01 },
    "processing.black_level_offset": { type: "range", min: 0, max: 64, step: 1 },
    "processing.sensor_gain": { type: "range", min: 0.01, max: 2, step: 0.01 },
    "processing.red_balance": { type: "range", min: 0, max: 4096, step: 1 },
    "processing.blue_balance": { type: "range", min: 0, max: 4096, step: 1 },
    "object_detection.yolo_obj_width_m": { type: "text" },
    "object_detection.yolo_obj_height_m": { type: "text" },
    "object_detection.confidence_threshold": { type: "range", min: 0, max: 1, step: 0.01 }
  };
  const labelMap = {
    "object_detection.yolo_obj_width_m": "obj_width_m",
    "object_detection.yolo_obj_height_m": "obj_height_m",
    "object_detection.confidence_threshold": "confidence_threshold"
  };

  for (const section of sections) {
    const details = document.createElement("details");
    details.className = "runtime-section";
    details.open = true;

    const summary = document.createElement("summary");
    summary.textContent = section.title;
    details.appendChild(summary);

    const grid = document.createElement("div");
    grid.className = "runtime-grid";

    for (const key of section.fields) {
      const wrap = document.createElement("label");
      wrap.dataset.section = section.key;
      wrap.dataset.key = key;
      const fullKey = `${section.key}.${key}`;
      wrap.textContent = labelMap[fullKey] || key;
      const spec = controlSpec[`${section.key}.${key}`];
      const input = document.createElement("input");
      input.dataset.section = section.key;
      input.dataset.key = key;
      const value = config?.[section.key]?.[key];
      if (spec?.type === "checkbox") {
        input.type = "checkbox";
        input.className = "toggle-input";
        input.checked = Boolean(value);
        input.addEventListener("change", () => {
          scheduleLiveRuntimeSync();
        });
      } else if (spec?.type === "range") {
        input.type = "range";
        input.className = "slider-input";
        input.min = String(spec.min);
        input.max = String(spec.max);
        input.step = String(spec.step);
        input.value = String(value ?? spec.min);
        const valueEl = document.createElement("span");
        valueEl.className = "slider-value";
        valueEl.textContent = input.value;
        input.addEventListener("input", () => {
          valueEl.textContent = input.value;
          scheduleLiveRuntimeSync();
        });
        wrap.appendChild(input);
        wrap.appendChild(valueEl);
        grid.appendChild(wrap);
        continue;
      } else {
        input.type = "text";
        input.value = value ?? "";
        input.addEventListener("change", () => scheduleLiveRuntimeSync());
      }
      wrap.appendChild(input);
      grid.appendChild(wrap);
    }

    details.appendChild(grid);
    el.runtimeForm.appendChild(details);
  }
}

function readRuntimeConfigFromForm() {
  const result = { camera: {}, processing: {}, object_detection: {} };
  const inputs = el.runtimeForm.querySelectorAll("input");
  for (const input of inputs) {
    const section = input.dataset.section;
    const key = input.dataset.key;
    if (input.type === "checkbox") {
      result[section][key] = input.checked;
      continue;
    }
    const raw = input.value.trim();
    if (input.type !== "range" && (raw.toLowerCase() === "true" || raw.toLowerCase() === "false")) {
      result[section][key] = raw.toLowerCase() === "true";
      continue;
    }
    const num = Number(raw);
    result[section][key] = Number.isFinite(num) && raw !== "" ? num : raw;
  }
  return result;
}

function scheduleLiveRuntimeSync() {
  if (runtimeSyncTimer) clearTimeout(runtimeSyncTimer);
  runtimeSyncTimer = setTimeout(async () => {
    state.runtimeConfig = readRuntimeConfigFromForm();
    await window.vortexApi.saveRuntimeConfig(el.runtimePath.value.trim(), state.runtimeConfig).catch(() => {});
    if (state.pipelineRunning) {
      await window.vortexApi
        .syncRuntimeConfigRemote(readAppConfigFromUI(), state.runtimeConfig)
        .catch(() => {});
    }
  }, 180);
}

function ensureCameraOption(cam) {
  if (state.selectedCamera == null) state.selectedCamera = cam;
}

function getActiveCameras() {
  const now = Date.now();
  return [...state.cameras.entries()]
    .filter(([, data]) => now - Number(data.lastSeen || 0) <= CAMERA_ACTIVE_MS)
    .map(([cam]) => Number(cam))
    .sort((a, b) => a - b);
}

function getCameraCandidates() {
  const active = getActiveCameras();
  const remote = [...(state.remoteCameras || [])].sort((a, b) => a - b);
  if (remote.length >= 2) return remote;
  if (active.length >= 2) return active;
  if (remote.length === 1) return remote;
  if (active.length === 1) return active;
  return [];
}

function refreshCameraControls() {
  const cams = getCameraCandidates();
  el.nextCamera.disabled = !state.pipelineRunning || cams.length < 2;
}

async function refreshRemoteCameras() {
  try {
    const cfg = readAppConfigFromUI();
    const r = await window.vortexApi.listRemoteCameras(cfg);
    if (r?.ok && Array.isArray(r.cameras)) {
      state.remoteCameras = r.cameras
        .map((n) => Number(n))
        .filter((n) => Number.isFinite(n) && n >= 0 && n <= 5)
        .sort((a, b) => a - b);
      return state.remoteCameras;
    }
  } catch (_err) {
    // ignore
  }
  return state.remoteCameras || [];
}

function previewSourceDims() {
  if (el.previewImage?.naturalWidth > 0 && el.previewImage?.naturalHeight > 0) {
    return { w: el.previewImage.naturalWidth, h: el.previewImage.naturalHeight };
  }
  return { w: 1920, h: 1080 };
}

function normalizeTagMap(raw) {
  const tags = Array.isArray(raw?.tags) ? raw.tags : [];
  const outTags = [];
  for (const t of tags) {
    const id = Number(t?.ID ?? t?.id);
    const tx = Number(t?.pose?.translation?.x);
    const ty = Number(t?.pose?.translation?.y);
    const tz = Number(t?.pose?.translation?.z);
    if (!Number.isFinite(id) || !Number.isFinite(tx) || !Number.isFinite(ty)) continue;
    outTags.push({ id, x: tx, y: ty, z: Number.isFinite(tz) ? tz : 0 });
  }
  const length = Number(raw?.field?.length);
  const width = Number(raw?.field?.width);
  return {
    tags: outTags,
    field: {
      length: Number.isFinite(length) ? length : 16.541,
      width: Number.isFinite(width) ? width : 8.069
    }
  };
}

async function loadTagMapFromPath(mapPath, { log = true } = {}) {
  const p = String(mapPath || "").trim();
  if (!p) return;
  try {
    const raw = await window.vortexApi.loadTagMap(p);
    state.tagMap = normalizeTagMap(raw);
    if (log) appendLog(`Loaded tag map: ${p}`);
  } catch (err) {
    if (log) appendLog(`Tag map load failed: ${err.message}`);
  }
  drawFieldMap();
}

function drawFieldMap() {
  const canvas = el.fieldMapCanvas;
  if (!canvas) return;
  const mapForSize = state.tagMap;
  const fieldLength = Number(mapForSize?.field?.length) || 16.541;
  const fieldWidth = Number(mapForSize?.field?.width) || 8.069;
  const ratio = Math.max(0.25, Math.min(1.2, fieldWidth / Math.max(1e-6, fieldLength)));
  const targetHeight = Math.round(
    Math.max(220, Math.min(360, (canvas.clientWidth || 640) * ratio + 28))
  );
  canvas.style.height = `${targetHeight}px`;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const map = state.tagMap;
  if (!map || !map.field) {
    ctx.fillStyle = "#9a9a9a";
    ctx.font = "13px Segoe UI";
    ctx.fillText("No AprilTag map loaded.", 10, 20);
    return;
  }

  const tags = [...(map.tags || [])].sort((a, b) => a.id - b.id);
  const margin = 12;
  const usableW = Math.max(1, rect.width - margin * 2);
  const usableH = Math.max(1, rect.height - margin * 2);
  const sx = usableW / Math.max(1e-6, map.field.length);
  const sy = usableH / Math.max(1e-6, map.field.width);
  const s = Math.min(sx, sy);
  const fieldW = map.field.length * s;
  const fieldH = map.field.width * s;
  const ox = margin + (usableW - fieldW) * 0.5;
  const oy = (rect.height - fieldH) * 0.5;
  const toPx = (x, y) => [ox + x * s, oy + (map.field.width - y) * s];

  ctx.strokeStyle = "#4a4a4a";
  ctx.lineWidth = 1;
  ctx.strokeRect(ox, oy, fieldW, fieldH);

  // draw numbered circles exactly at each tag's true field position
  const baseR = 8;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const hoveredId = Number(state.hoveredTagId);
  const drawTag = (t, isHover) => {
    const [x, y] = toPx(t.x, t.y);
    const r = isHover ? baseR + 7 : baseR;
    ctx.fillStyle = "#151515";
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = isHover ? "#ffffff" : "#d8d8d8";
    ctx.lineWidth = isHover ? 2 : 1.4;
    ctx.stroke();
    ctx.fillStyle = "#f3f3f3";
    ctx.font = isHover ? "bold 12px Segoe UI" : "9px Segoe UI";
    ctx.fillText(String(t.id), x, y + 0.5);
  };
  for (const t of tags) {
    if (Number(t.id) === hoveredId) continue;
    drawTag(t, false);
  }
  const hovered = tags.find((t) => Number(t.id) === hoveredId);
  if (hovered) {
    drawTag(hovered, true);
  }

  if (state.robotPose && Number.isFinite(state.robotPose.x) && Number.isFinite(state.robotPose.y)) {
    const [rx, ry] = toPx(state.robotPose.x, state.robotPose.y);
    ctx.fillStyle = "#39d353";
    ctx.beginPath();
    ctx.arc(rx, ry, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#39d353";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rx - 7, ry - 7, 14, 14);
  }
}

function updateHoveredTagFromPointer(evt) {
  const canvas = el.fieldMapCanvas;
  const map = state.tagMap;
  if (!canvas || !map || !Array.isArray(map.tags)) {
    state.hoveredTagId = null;
    drawFieldMap();
    return;
  }
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) {
    state.hoveredTagId = null;
    return;
  }

  const mx = evt.clientX - rect.left;
  const my = evt.clientY - rect.top;
  const fieldLength = Number(map?.field?.length) || 16.541;
  const fieldWidth = Number(map?.field?.width) || 8.069;
  const margin = 12;
  const usableW = Math.max(1, rect.width - margin * 2);
  const usableH = Math.max(1, rect.height - margin * 2);
  const sx = usableW / Math.max(1e-6, fieldLength);
  const sy = usableH / Math.max(1e-6, fieldWidth);
  const s = Math.min(sx, sy);
  const fieldW = fieldLength * s;
  const fieldH = fieldWidth * s;
  const ox = margin + (usableW - fieldW) * 0.5;
  const oy = (rect.height - fieldH) * 0.5;
  const toPx = (x, y) => [ox + x * s, oy + (fieldWidth - y) * s];

  let nearestId = null;
  let nearestD2 = Infinity;
  for (const t of map.tags) {
    const tx = Number(t?.x);
    const ty = Number(t?.y);
    if (!Number.isFinite(tx) || !Number.isFinite(ty)) continue;
    const [px, py] = toPx(tx, ty);
    const dx = px - mx;
    const dy = py - my;
    const d2 = dx * dx + dy * dy;
    if (d2 < nearestD2) {
      nearestD2 = d2;
      nearestId = t.id;
    }
  }
  const hoverThresholdPx = 34;
  const nextHover = nearestD2 <= hoverThresholdPx * hoverThresholdPx ? nearestId : null;
  if (nextHover !== state.hoveredTagId) {
    state.hoveredTagId = nextHover;
    drawFieldMap();
  }
}

function parseBBoxPart(bboxText, idx) {
  const m = String(bboxText).match(/\[([^\]]+)\]/);
  if (!m) return NaN;
  const parts = m[1].split(",").map((s) => Number(s.trim()));
  return parts[idx];
}

function parseMonitorLine(line) {
  const header = line.match(/^Camera\s+(\d+):\s+([\d.]+)\s+FPS/);
  if (header) {
    const cam = Number(header[1]);
    const fps = Number(header[2]);
    if (!state.cameras.has(cam)) {
      state.cameras.set(cam, {
        fps: null,
        apriltags: [],
        objects: [],
        lastSeen: 0,
        tagFrameHistory: [],
        lastTagById: new Map()
      });
    }
    const obj = state.cameras.get(cam);
    obj.fps = fps;
    obj.lastSeen = Date.now();
    state.currentLineCamera = cam;
    state.selectedCamera = cam;
    ensureCameraOption(cam);
    if (state.selectedCamera == null || !state.cameras.has(Number(state.selectedCamera))) {
      state.selectedCamera = cam;
    }
    return;
  }

  const activeCamera = Number(
    state.currentLineCamera != null ? state.currentLineCamera : state.selectedCamera
  );
  if (!state.cameras.has(activeCamera)) return;
  const current = state.cameras.get(activeCamera);

  const tagLegacy = line.match(/Tag ID:\s*(\d+)\s*\|\s*Dist:\s*([\d.-]+)m\s*\|\s*X:\s*([\d.-]+)m\s*\|\s*Y:\s*([\d.-]+)m/);
  if (tagLegacy) {
    current.apriltags.push({
      id: Number(tagLegacy[1]),
      dist: Number(tagLegacy[2]),
      x: Number(tagLegacy[3]),
      y: Number(tagLegacy[4])
    });
    return;
  }

  const tagField = line.match(/Tag ID:\s*(\d+)\s*\|\s*Field X:\s*([\d.-]+)m\s*\|\s*Field Y:\s*([\d.-]+)m\s*\|\s*FloorErr:\s*([\d.-]+)m/);
  if (tagField) {
    current.apriltags.push({
      id: Number(tagField[1]),
      dist: Number(tagField[4]),
      x: Number(tagField[2]),
      y: Number(tagField[3])
    });
    return;
  }

  const obj = line.match(/Object:\s*(.+?)\s*\(([\d.-]+)\)\s*\|\s*Dist:\s*([\d.-]+)m\s*\|\s*X:\s*([\d.-]+)m\s*\|\s*Y:\s*([\d.-]+)m\s*\|\s*(BBox:.*)$/);
  if (obj) {
    current.objects.push({
      className: obj[1].trim(),
      conf: Number(obj[2]),
      dist: Number(obj[3]),
      x: Number(obj[4]),
      y: Number(obj[5]),
      bbox: obj[6],
      bboxWidth: parseBBoxPart(obj[6], 2),
      bboxHeight: parseBBoxPart(obj[6], 3)
    });
  }
}

function renderDetectionTable() {
  const selected = state.selectedCamera != null ? Number(state.selectedCamera) : 0;
  el.cameraPill.textContent = `Camera: ${selected}`;
  const cam = selected;
  const data = state.cameras.get(cam);
  if (!data) {
    el.fps.textContent = "FPS: -";
    el.detectionTableHead.innerHTML = "";
    el.detectionTableBody.innerHTML = "<tr><td>No data</td></tr>";
    return;
  }
  el.fps.textContent = `FPS: ${data.fps != null ? data.fps.toFixed(2) : "-"}`;
  const mode = el.viewMode.value;
  if (mode === "apriltag") {
    el.detectionTableHead.innerHTML = "<tr><th>ID</th><th>Field X (m)</th><th>Field Y (m)</th><th>Floor Err (m)</th><th>Seen / sec</th></tr>";
    const fpsValue = Number.isFinite(data.fps) && data.fps > 0 ? data.fps : 1;
    const windowSize = Math.max(1, Math.round(fpsValue));
    const recentFrames = (data.tagFrameHistory || []).slice(-windowSize);
    const counts = new Map();
    for (const frameIds of recentFrames) {
      const unique = new Set(frameIds || []);
      for (const id of unique) counts.set(id, (counts.get(id) || 0) + 1);
    }
    const ids = [...counts.keys()].sort((a, b) => a - b);
    if (ids.length === 0) {
      el.detectionTableBody.innerHTML = "<tr><td colspan='5'>No AprilTags</td></tr>";
    } else {
      let totalSeenPerSec = 0;
      const rows = ids.map((id) => {
        const seenCount = counts.get(id) || 0;
        const seenPerSec = seenCount * (fpsValue / Math.max(1, recentFrames.length));
        const seenPct = (seenCount / Math.max(1, recentFrames.length)) * 100;
        totalSeenPerSec += seenPerSec;
        const last = data.lastTagById?.get(id);
        const dist = Number.isFinite(last?.dist) ? last.dist.toFixed(3) : "-";
        const x = Number.isFinite(last?.x) ? last.x.toFixed(2) : "-";
        const y = Number.isFinite(last?.y) ? last.y.toFixed(2) : "-";
        return `<tr><td>${id}</td><td>${x}</td><td>${y}</td><td>${dist}</td><td>${seenPerSec.toFixed(2)} (${seenPct.toFixed(0)}%)</td></tr>`;
      });
      const avgSeenPerSec = totalSeenPerSec / ids.length;
      const avgSeenPct = (avgSeenPerSec / Math.max(1e-6, fpsValue)) * 100;
      rows.push(`<tr><td><b>Avg</b></td><td>-</td><td>-</td><td>-</td><td><b>${avgSeenPerSec.toFixed(2)} (${avgSeenPct.toFixed(0)}%)</b></td></tr>`);
      el.detectionTableBody.innerHTML = rows.join("");
    }
  } else {
    el.detectionTableHead.innerHTML = "<tr><th>Class</th><th>Conf</th><th>Distance (m)</th><th>X (m)</th><th>Y (m)</th><th>BBox</th></tr>";
    el.detectionTableBody.innerHTML = data.objects
      .map((r) => `<tr><td>${r.className}</td><td>${r.conf.toFixed(2)}</td><td>${r.dist.toFixed(2)}</td><td>${r.x.toFixed(2)}</td><td>${r.y.toFixed(2)}</td><td>${r.bbox}</td></tr>`)
      .join("") || "<tr><td colspan='6'>No objects</td></tr>";
  }
  drawOverlay();
  renderRobotPose();
  refreshCameraControls();
}

function renderRobotPose() {
  if (!el.robotPoseText) return;
  const p = state.robotPose;
  if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) {
    el.robotPoseText.textContent = "Robot Pose:";
    drawFieldMap();
    return;
  }
  const tags = Number(p.tags_used || 0);
  const zErr = Number.isFinite(p.floor_z_error_avg) ? p.floor_z_error_avg : 0;
  el.robotPoseText.textContent = `Robot Pose: x=${p.x.toFixed(2)} m, y=${p.y.toFixed(2)} m | tags=${tags} | floor z err=${zErr.toFixed(3)} m`;
  drawFieldMap();
}

function drawOverlay() {
  const canvas = el.previewOverlay;
  const img = el.previewImage;
  if (!canvas || !img) return;

  const rect = img.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const cam = Number(state.selectedCamera ?? 0);
  const data = state.cameras.get(cam);
  if (!data) return;

  const src = previewSourceDims();
  const sx = rect.width / Math.max(1, src.w);
  const sy = rect.height / Math.max(1, src.h);

  if (el.viewMode.value === "apriltag") {
    ctx.strokeStyle = "#00ff66";
    ctx.lineWidth = 2.5;
    const tags = data.apriltags || [];
    const now = Date.now();
    const drawList = [];
    for (const t of tags) drawList.push({ corners: t.corners });
    if (drawList.length === 0 && data.lastTagById instanceof Map) {
      for (const entry of data.lastTagById.values()) {
        if (!entry || !Array.isArray(entry.corners)) continue;
        if (now - Number(entry.lastSeenTs || 0) <= TAG_OVERLAY_STICKY_MS) {
          drawList.push({ corners: entry.corners });
        }
      }
    }
    for (const t of drawList) {
      if (!Array.isArray(t.corners) || t.corners.length !== 4) continue;
      const p = t.corners.map((c) => [Number(c[0]) * sx, Number(c[1]) * sy]);
      if (!p.every((xy) => Number.isFinite(xy[0]) && Number.isFinite(xy[1]))) continue;
      ctx.beginPath();
      ctx.moveTo(p[0][0], p[0][1]);
      ctx.lineTo(p[1][0], p[1][1]);
      ctx.lineTo(p[2][0], p[2][1]);
      ctx.lineTo(p[3][0], p[3][1]);
      ctx.closePath();
      ctx.stroke();
    }
    return;
  }

  if (el.viewMode.value === "object") {
    ctx.strokeStyle = "#00ff88";
    ctx.fillStyle = "#00ff88";
    ctx.lineWidth = 2;
    ctx.font = "12px Segoe UI";

    for (const o of data.objects || []) {
      const x = o.x * sx;
      const y = o.y * sy;
      const w = o.bboxWidth * sx;
      const h = o.bboxHeight * sy;
      if (![x, y, w, h].every(Number.isFinite)) continue;
      ctx.strokeRect(x, y, w, h);
      ctx.fillText(`${o.className} ${o.conf.toFixed(2)}`, x + 4, Math.max(12, y - 4));
    }
  }
}

async function init() {
  const boot = await window.vortexApi.bootstrap();
  hydrateAppConfig(boot.appConfig);
  state.runtimePath = boot.runtimeConfigPath;
  el.runtimePath.value = state.runtimePath;
  state.tagMapPath = String(state.appConfig?.tag_map_path || "");
  if (el.tagMapPath) el.tagMapPath.value = state.tagMapPath;
  state.runtimeConfig = boot.runtimeConfig;
  renderRuntimeForm(state.runtimeConfig);
  await loadTagMapFromPath(state.tagMapPath, { log: false });

  if (state.appConfig.monitor_start_cmd === "") {
    const applied = await window.vortexApi.applyMonitorPreset(readAppConfigFromUI(), "main_cam_0");
    hydrateAppConfig(applied);
    state.appConfig = applied;
    await window.vortexApi.saveAppConfig(applied);
  }

  el.loadRuntime.addEventListener("click", async () => {
    state.runtimePath = el.runtimePath.value.trim();
    try {
      state.runtimeConfig = await window.vortexApi.loadRuntimeConfig(state.runtimePath);
      renderRuntimeForm(state.runtimeConfig);
      appendLog(`Loaded runtime config: ${state.runtimePath}`);
    } catch (err) {
      appendLog(`Load failed: ${err.message}`);
    }
  });

  el.saveRuntime.addEventListener("click", async () => {
    state.runtimePath = el.runtimePath.value.trim();
    state.runtimeConfig = readRuntimeConfigFromForm();
    await window.vortexApi.saveRuntimeConfig(state.runtimePath, state.runtimeConfig);
    if (state.pipelineRunning) {
      const sync = await window.vortexApi.syncRuntimeConfigRemote(readAppConfigFromUI(), state.runtimeConfig);
      if (!sync?.ok) {
        appendLog(`Remote config sync failed: ${sync?.error || "unknown error"}`);
      }
    }
    appendLog(`Saved runtime config: ${state.runtimePath}`);
  });

  el.browseTagMap.addEventListener("click", async () => {
    const picked = await window.vortexApi.chooseFile({
      filters: [{ name: "JSON", extensions: ["json"] }]
    });
    if (!picked) return;
    el.tagMapPath.value = picked;
    state.tagMapPath = picked;
    state.appConfig = { ...state.appConfig, tag_map_path: picked };
    await window.vortexApi.saveAppConfig(readAppConfigFromUI());
    await loadTagMapFromPath(picked);
  });

  el.loadTagMap.addEventListener("click", async () => {
    const p = String(el.tagMapPath.value || "").trim();
    state.tagMapPath = p;
    state.appConfig = { ...state.appConfig, tag_map_path: p };
    await window.vortexApi.saveAppConfig(readAppConfigFromUI());
    await loadTagMapFromPath(p);
  });

  el.deployBtn.addEventListener("click", async () => {
    const cfg = readAppConfigFromUI();
    state.runtimeConfig = readRuntimeConfigFromForm();
    await window.vortexApi.saveRuntimeConfig(el.runtimePath.value.trim(), state.runtimeConfig);
    await window.vortexApi.saveAppConfig(cfg);
    setDeployIndicator("active", "0%", 0);
    await window.vortexApi.deployStart(cfg);
    appendLog("Deploy started.");
  });

  el.togglePipeline.addEventListener("click", async () => {
    let cfg = readAppConfigFromUI();
    if (!state.pipelineRunning) {
      try {
        setMonitorIndicator("active", "Starting...", 5);
        cfg = await window.vortexApi.applyMonitorPreset(cfg, "main_cam_0");
        state.appConfig = cfg;
        await window.vortexApi.saveAppConfig(cfg);
        await refreshRemoteCameras();
        const m = await window.vortexApi.monitorStart(cfg);
        if (!m?.ok) throw new Error(m?.error || "monitor start failed");
        const p = await window.vortexApi.previewStart(cfg);
        if (!p?.ok) throw new Error(p?.error || "preview start failed");
        state.pipelineRunning = true;
        el.togglePipeline.textContent = "Stop Monitor";
        refreshCameraControls();
      } catch (err) {
        const msg = String(err?.message || err);
        setMonitorIndicator("error", "✕ Start failed", 0);
        if (msg.includes("authentication methods failed")) {
          appendLog(`Start failed: ${msg}. Check vortex_config.json credentials.`);
        } else {
          appendLog(`Start failed: ${msg}`);
        }
      }
    } else {
      try {
        setMonitorIndicator("hidden", "");
        await window.vortexApi.previewStop();
        await window.vortexApi.monitorStop();
      } catch (err) {
        appendLog(`Stop error: ${err.message}`);
      }
      state.pipelineRunning = false;
      el.togglePipeline.textContent = "Start Monitor";
      refreshCameraControls();
    }
  });

  el.nextCamera.addEventListener("click", async () => {
    await refreshRemoteCameras();
    const cams = getCameraCandidates();
    if (cams.length === 0) {
      refreshCameraControls();
      return;
    }
    const current = Number(state.selectedCamera);
    const idx = cams.indexOf(current);
    state.selectedCamera = cams[(idx + 1 + cams.length) % cams.length];
    await window.vortexApi.saveAppConfig(state.appConfig).catch(() => {});
    if (state.pipelineRunning) {
      try {
        let cfg = readAppConfigFromUI();
        cfg = await window.vortexApi.applyMonitorPreset(cfg, "main_cam_0");
        state.appConfig = cfg;
        await window.vortexApi.saveAppConfig(cfg).catch(() => {});
        await window.vortexApi.previewStop();
        await window.vortexApi.monitorStop();
        const m = await window.vortexApi.monitorStart(cfg);
        if (!m?.ok) {
          appendLog(`Monitor restart failed: ${m?.error || "unknown error"}`);
        } else {
          const p = await window.vortexApi.previewStart(cfg);
          if (!p?.ok) appendLog(`Preview restart failed: ${p?.error || "unknown error"}`);
          state.pipelineRunning = true;
          el.togglePipeline.textContent = "Stop Monitor";
        }
      } catch (err) {
        appendLog(`Camera switch error: ${err.message}`);
      }
    }
    renderDetectionTable();
    refreshCameraControls();
  });

  el.viewMode.addEventListener("change", () => {
    state.viewMode = el.viewMode.value;
    renderDetectionTable();
  });

  window.vortexApi.onLog((msg) => appendLog(msg));
  window.vortexApi.onMonitorLine((line) => {
    parseMonitorLine(line);
    renderDetectionTable();
    refreshCameraControls();
  });
  window.vortexApi.onMonitorState((running) => {
    el.monitorStatus.textContent = `Monitor: ${running ? "Running" : "Stopped"}`;
    if (running) {
      state.pipelineRunning = true;
      el.togglePipeline.textContent = "Stop Monitor";
    }
    if (!running) {
      state.pipelineRunning = false;
      el.togglePipeline.textContent = "Start Monitor";
      window.vortexApi.previewStop().catch(() => {});
    }
    refreshCameraControls();
  });
  window.vortexApi.onPreviewState((running) => {
    el.previewStatus.textContent = `Feed: ${running ? "Running" : "Stopped"}`;
  });
  window.vortexApi.onPreviewFrame((dataUrl) => {
    el.previewImage.src = dataUrl;
    drawOverlay();
  });
  window.vortexApi.onBridgeState((bridge) => {
    const cam = Number(bridge?.camera_index);
    if (!Number.isFinite(cam)) return;
    if (!state.cameras.has(cam)) {
      state.cameras.set(cam, {
        fps: null,
        apriltags: [],
        objects: [],
        lastSeen: 0,
        tagFrameHistory: [],
        lastTagById: new Map()
      });
    }
    const row = state.cameras.get(cam);
    row.lastSeen = Date.now();
    if (Number.isFinite(Number(bridge?.fps))) row.fps = Number(bridge.fps);
    if (Array.isArray(bridge?.apriltags)) {
      row.apriltags = bridge.apriltags.map((t) => ({
        id: Number(t?.id),
        dist: Number(t?.floor_z_error ?? t?.z ?? 0),
        x: Number(t?.x ?? 0),
        y: Number(t?.y ?? 0),
        corners: Array.isArray(t?.corners) ? t.corners : []
      }));
      const idsThisFrame = row.apriltags.map((t) => t.id).filter((id) => Number.isFinite(id));
      row.tagFrameHistory = row.tagFrameHistory || [];
      row.tagFrameHistory.push(idsThisFrame);
      if (row.tagFrameHistory.length > 300) row.tagFrameHistory.splice(0, row.tagFrameHistory.length - 300);
      row.lastTagById = row.lastTagById || new Map();
      for (const t of row.apriltags) {
        row.lastTagById.set(t.id, {
          dist: t.dist,
          x: t.x,
          y: t.y,
          corners: t.corners,
          lastSeenTs: Date.now()
        });
      }
    }
    if (Array.isArray(bridge?.objects)) {
      row.objects = bridge.objects.map((o) => ({
        className: String(o?.class_name ?? ""),
        conf: Number(o?.confidence ?? 0),
        dist: Number(o?.z ?? 0),
        x: Number(Array.isArray(o?.bbox) ? o.bbox[0] : 0),
        y: Number(Array.isArray(o?.bbox) ? o.bbox[1] : 0),
        bbox: `BBox: [${Array.isArray(o?.bbox) ? o.bbox.join(",") : ""}]`,
        bboxWidth: Number(Array.isArray(o?.bbox) ? o.bbox[2] : 0),
        bboxHeight: Number(Array.isArray(o?.bbox) ? o.bbox[3] : 0)
      }));
    }
    if (bridge?.robot_pose && Number.isFinite(Number(bridge.robot_pose.x)) && Number.isFinite(Number(bridge.robot_pose.y))) {
      state.robotPose = {
        x: Number(bridge.robot_pose.x),
        y: Number(bridge.robot_pose.y),
        tags_used: Number(bridge.robot_pose.tags_used || 0),
        floor_z_error_avg: Number(bridge.robot_pose.floor_z_error_avg || 0)
      };
    } else {
      state.robotPose = null;
    }
    if (!state.tagMap && Number.isFinite(Number(bridge?.field?.length)) && Number.isFinite(Number(bridge?.field?.width))) {
      state.tagMap = { tags: [], field: { length: Number(bridge.field.length), width: Number(bridge.field.width) } };
    }
    if (state.selectedCamera == null) state.selectedCamera = cam;
    renderDetectionTable();
    drawOverlay();
  });
  window.vortexApi.onDeployProgress((p) => {
    const status = String(p?.status || "");
    const percent = Number(p?.percent || 0);
    const cur = Number(p?.current || 0);
    const total = Number(p?.total || 0);
    const msg = total > 0 ? `${percent}% (${cur}/${total})` : `${percent}%`;
    if (status === "done") {
      setDeployIndicator("done", "✓ Deployed", 100);
    } else if (status === "error") {
      setDeployIndicator("error", "✕ Deploy failed", percent);
    } else {
      setDeployIndicator("active", msg, percent);
    }
  });
  window.vortexApi.onMonitorStartProgress((p) => {
    const status = String(p?.status || "");
    const percent = Number(p?.percent || 0);
    const text = String(p?.text || `${percent}%`);
    if (status === "done") {
      setMonitorIndicator("done", text, 100);
    } else if (status === "error") {
      setMonitorIndicator("error", text, percent);
    } else if (status === "hidden") {
      setMonitorIndicator("hidden", "");
    } else {
      setMonitorIndicator("active", text, percent);
    }
  });

  setDeployIndicator("hidden", "");
  setMonitorIndicator("hidden", "");
  await refreshRemoteCameras();
  refreshCameraControls();
  renderRobotPose();
  if (el.fieldMapCanvas) {
    el.fieldMapCanvas.addEventListener("mousemove", updateHoveredTagFromPointer);
    el.fieldMapCanvas.addEventListener("mouseleave", () => {
      if (state.hoveredTagId != null) {
        state.hoveredTagId = null;
        drawFieldMap();
      }
    });
  }

  el.previewImage.onload = () => drawOverlay();
  window.addEventListener("resize", () => {
    drawOverlay();
    drawFieldMap();
  });
}

init().catch((err) => appendLog(`Bootstrap failed: ${err.message}`));
