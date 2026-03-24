const state = {
  appConfig: null,
  runtimePath: "",
  tagMapPath: "",
  onnxPath: "",
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
  jetsonWatts: document.querySelector("#jetsonWatts"),
  applyJetsonWatts: document.querySelector("#applyJetsonWatts"),
  startupCameras: document.querySelector("#startupCameras"),
  ntEnable: document.querySelector("#ntEnable"),
  ntMode: document.querySelector("#ntMode"),
  ntTeam: document.querySelector("#ntTeam"),
  ntServer: document.querySelector("#ntServer"),
  ntTable: document.querySelector("#ntTable"),
  udpEnable: document.querySelector("#udpEnable"),
  udpTarget: document.querySelector("#udpTarget"),
  udpPort: document.querySelector("#udpPort"),
  tagMapPath: document.querySelector("#tagMapPath"),
  onnxPath: document.querySelector("#onnxPath"),
  browseTagMap: document.querySelector("#browseTagMap"),
  loadTagMap: document.querySelector("#loadTagMap"),
  browseOnnx: document.querySelector("#browseOnnx"),
  uploadOnnxBuild: document.querySelector("#uploadOnnxBuild"),
  onnxProgressWrap: document.querySelector("#onnxProgressWrap"),
  onnxProgress: document.querySelector("#onnxProgress"),
  onnxProgressText: document.querySelector("#onnxProgressText"),
  objectDetectionForm: document.querySelector("#objectDetectionForm"),
  loadRuntime: document.querySelector("#loadRuntime"),
  saveRuntime: document.querySelector("#saveRuntime"),
  runtimeForm: document.querySelector("#runtimeForm"),
  deployBtn: document.querySelector("#deployBtn"),
  buildMainBtn: document.querySelector("#buildMainBtn"),
  deployProgressWrap: document.querySelector("#deployProgressWrap"),
  deployProgress: document.querySelector("#deployProgress"),
  deployProgressText: document.querySelector("#deployProgressText"),
  buildProgressWrap: document.querySelector("#buildProgressWrap"),
  buildProgress: document.querySelector("#buildProgress"),
  buildProgressText: document.querySelector("#buildProgressText"),
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

const CONTROL_SPEC = {
  "processing.smoothing_alpha": { type: "range", min: 0, max: 1, step: 0.01 },
  "processing.black_level_offset": { type: "range", min: 0, max: 64, step: 1 },
  "processing.sensor_gain": { type: "range", min: 0.01, max: 2, step: 0.01 },
  "processing.red_balance": { type: "range", min: 0, max: 4096, step: 1 },
  "processing.blue_balance": { type: "range", min: 0, max: 4096, step: 1 },
  "object_detection.use_nn": { type: "checkbox", defaultValue: true },
  "object_detection.yolo_obj_width_m": { type: "text" },
  "object_detection.yolo_obj_height_m": { type: "text" },
  "object_detection.confidence_threshold": { type: "range", min: 0, max: 1, step: 0.01 }
};

const LABEL_MAP = {
  "object_detection.use_nn": "use_nn",
  "object_detection.yolo_obj_width_m": "obj_width_m",
  "object_detection.yolo_obj_height_m": "obj_height_m",
  "object_detection.confidence_threshold": "confidence_threshold"
};

function appendLog(msg) {
  const stamp = new Date().toLocaleTimeString();
  el.logs.textContent += `[${stamp}] ${msg}\n`;
  el.logs.scrollTop = el.logs.scrollHeight;
}

let deployFadeTimer = null;
let monitorFadeTimer = null;
let onnxFadeTimer = null;
let buildFadeTimer = null;
let runtimeSyncTimer = null;
let overlayRaf = 0;
let fieldRaf = 0;
let pendingPreviewDataUrl = null;
let previewLoadPending = false;

function scheduleOverlayDraw() {
  if (overlayRaf) return;
  overlayRaf = requestAnimationFrame(() => {
    overlayRaf = 0;
    drawOverlay();
  });
}

function scheduleFieldDraw() {
  if (fieldRaf) return;
  fieldRaf = requestAnimationFrame(() => {
    fieldRaf = 0;
    drawFieldMap();
  });
}

function pushPreviewFrame(dataUrl) {
  pendingPreviewDataUrl = dataUrl;
  if (previewLoadPending) return;
  const next = pendingPreviewDataUrl;
  pendingPreviewDataUrl = null;
  if (!next) return;
  previewLoadPending = true;
  el.previewImage.src = next;
}

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

function setOnnxIndicator(stateKind, text, percent = null) {
  if (onnxFadeTimer) {
    clearTimeout(onnxFadeTimer);
    onnxFadeTimer = null;
  }
  const wrap = el.onnxProgressWrap;
  wrap.classList.remove("is-hidden", "is-done", "is-error");

  if (stateKind === "hidden") {
    wrap.classList.add("is-hidden");
    return;
  }
  if (stateKind === "done") wrap.classList.add("is-done");
  if (stateKind === "error") wrap.classList.add("is-error");

  if (typeof percent === "number") {
    el.onnxProgress.value = Math.max(0, Math.min(100, percent));
  }
  if (text) el.onnxProgressText.textContent = text;

  if (stateKind === "done" || stateKind === "error") {
    onnxFadeTimer = setTimeout(() => {
      setOnnxIndicator("hidden", "");
    }, 1600);
  }
}

function setBuildIndicator(stateKind, text, percent = null) {
  if (buildFadeTimer) {
    clearTimeout(buildFadeTimer);
    buildFadeTimer = null;
  }
  const wrap = el.buildProgressWrap;
  wrap.classList.remove("is-hidden", "is-done", "is-error");

  if (stateKind === "hidden") {
    wrap.classList.add("is-hidden");
    return;
  }
  if (stateKind === "done") wrap.classList.add("is-done");
  if (stateKind === "error") wrap.classList.add("is-error");

  if (typeof percent === "number") {
    el.buildProgress.value = Math.max(0, Math.min(100, percent));
  }
  if (text) el.buildProgressText.textContent = text;

  if (stateKind === "done" || stateKind === "error") {
    buildFadeTimer = setTimeout(() => {
      setBuildIndicator("hidden", "");
    }, 1800);
  }
}

function hydrateAppConfig(cfg) {
  state.appConfig = cfg;
  state.tagMapPath = String(cfg?.tag_map_path || state.tagMapPath || "");
  state.onnxPath = String(cfg?.onnx_model_path || state.onnxPath || "");
  if (el.tagMapPath) el.tagMapPath.value = state.tagMapPath;
  if (el.onnxPath) el.onnxPath.value = state.onnxPath;
  if (state.selectedCamera == null) {
    state.selectedCamera = 0;
  }
  if (el.jetsonWatts) {
    const watts = Number(cfg?.jetson_max_watts);
    const allowed = new Set([7, 15, 25]);
    const selected = Number.isFinite(watts) && allowed.has(Math.round(watts)) ? Math.round(watts) : 15;
    el.jetsonWatts.value = String(selected);
  }
  if (el.startupCameras) {
    el.startupCameras.value = normalizeCameraIndices(cfg?.startup_camera_indices || "0,1");
  }
  if (el.ntEnable) el.ntEnable.checked = Boolean(cfg?.vortex_nt_enable);
  if (el.ntMode) el.ntMode.value = normalizeNtMode(cfg?.vortex_nt_mode || "team");
  if (el.ntTeam) el.ntTeam.value = String(cfg?.vortex_nt_team || "509");
  if (el.ntServer) el.ntServer.value = String(cfg?.vortex_nt_server || "");
  if (el.ntTable) el.ntTable.value = String(cfg?.vortex_nt_table || "/Vortex/Vision");
  if (el.udpEnable) el.udpEnable.checked = cfg?.vortex_udp_enable == null ? true : Boolean(cfg?.vortex_udp_enable);
  if (el.udpTarget) el.udpTarget.value = String(cfg?.vortex_udp_target || "192.168.1.24");
  if (el.udpPort) el.udpPort.value = String(normalizePort(cfg?.vortex_udp_port, 5809));
  syncNetworkFieldState();
}

function normalizeCameraIndices(raw) {
  const parts = String(raw || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => Number(s))
    .filter((n) => Number.isFinite(n) && n >= 0 && n <= 63)
    .map((n) => String(Math.trunc(n)));
  if (parts.length === 0) return "0,1";
  return [...new Set(parts)].join(",");
}

function normalizeNtMode(raw) {
  const mode = String(raw || "").trim().toLowerCase();
  if (mode === "team" || mode === "custom" || mode === "local") return mode;
  return "team";
}

function normalizePort(raw, fallback = 5809) {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  const i = Math.trunc(n);
  if (i < 1 || i > 65535) return fallback;
  return i;
}

function syncNetworkFieldState() {
  const ntEnabled = Boolean(el.ntEnable?.checked);
  const ntMode = normalizeNtMode(el.ntMode?.value || "team");
  if (el.ntMode) el.ntMode.disabled = !ntEnabled;
  if (el.ntTeam) el.ntTeam.disabled = !ntEnabled || ntMode !== "team";
  if (el.ntServer) el.ntServer.disabled = !ntEnabled || ntMode !== "custom";
  if (el.ntTable) el.ntTable.disabled = !ntEnabled;

  const udpEnabled = Boolean(el.udpEnable?.checked);
  if (el.udpTarget) el.udpTarget.disabled = !udpEnabled;
  if (el.udpPort) el.udpPort.disabled = !udpEnabled;
}

function readAppConfigFromUI() {
  const cam = Number(state.selectedCamera || 0);
  const normalizedCam = Number.isFinite(cam) && cam >= 0 && cam <= 5 ? cam : 0;
  const tagMapPath = String(el.tagMapPath?.value || state.tagMapPath || "").trim();
  const onnxPath = String(el.onnxPath?.value || state.onnxPath || "").trim();
  const wattsRaw = Number(el.jetsonWatts?.value ?? state.appConfig?.jetson_max_watts ?? 15);
  const allowed = new Set([7, 15, 25]);
  const jetsonWatts = Number.isFinite(wattsRaw) && allowed.has(Math.round(wattsRaw))
    ? Math.round(wattsRaw)
    : 15;
  const startupCameraIndices = normalizeCameraIndices(
    el.startupCameras?.value || state.appConfig?.startup_camera_indices || "0,1"
  );
  const ntEnable = Boolean(el.ntEnable?.checked ?? state.appConfig?.vortex_nt_enable);
  const ntMode = normalizeNtMode(el.ntMode?.value || state.appConfig?.vortex_nt_mode || "team");
  const ntTeam = String(el.ntTeam?.value || state.appConfig?.vortex_nt_team || "509").trim() || "509";
  const ntServer = String(el.ntServer?.value || state.appConfig?.vortex_nt_server || "").trim();
  const ntTable = String(el.ntTable?.value || state.appConfig?.vortex_nt_table || "/Vortex/Vision").trim() || "/Vortex/Vision";
  const udpEnable = Boolean(el.udpEnable?.checked ?? state.appConfig?.vortex_udp_enable ?? true);
  const udpTarget = String(el.udpTarget?.value || state.appConfig?.vortex_udp_target || "192.168.1.24").trim();
  const udpPort = normalizePort(el.udpPort?.value || state.appConfig?.vortex_udp_port || 5809, 5809);
  return {
    ...state.appConfig,
    preview_camera_index: normalizedCam,
    tag_map_path: tagMapPath,
    onnx_model_path: onnxPath,
    jetson_max_watts: jetsonWatts,
    startup_camera_indices: startupCameraIndices,
    vortex_nt_enable: ntEnable,
    vortex_nt_mode: ntMode,
    vortex_nt_team: ntTeam,
    vortex_nt_server: ntServer,
    vortex_nt_table: ntTable,
    vortex_udp_enable: udpEnable,
    vortex_udp_target: udpTarget,
    vortex_udp_port: udpPort
  };
}

function renderRuntimeForm(config) {
  el.runtimeForm.innerHTML = "";
  const orderedCameraIntrinsics = [
    "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "tag_size_m"
  ];
  const orderedCameraTranslation = [
    "x_offset", "y_offset", "z_offset", "pitch_deg", "yaw_deg", "roll_deg"
  ];
  const orderedProcessing = [
    "smoothing_alpha",
    "black_level_offset",
    "sensor_gain",
    "red_balance",
    "blue_balance"
  ];
  const sections = [
    { key: "camera", title: "Camera Intrinsics Profile", fields: orderedCameraIntrinsics },
    { key: "camera", title: "Camera Translation", fields: orderedCameraTranslation },
    { key: "processing", title: "Processing Constants", fields: orderedProcessing }
  ];

  for (const section of sections) {
    const details = document.createElement("details");
    details.className = "runtime-section";

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
      wrap.textContent = LABEL_MAP[fullKey] || key;
      const spec = CONTROL_SPEC[`${section.key}.${key}`];
      const input = document.createElement("input");
      input.dataset.section = section.key;
      input.dataset.key = key;
      const value = config?.[section.key]?.[key];
      if (spec?.type === "checkbox") {
        input.type = "checkbox";
        input.className = "toggle-input";
        input.checked = value == null ? Boolean(spec.defaultValue) : Boolean(value);
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

function renderObjectDetectionForm(config) {
  if (!el.objectDetectionForm) return;
  el.objectDetectionForm.innerHTML = "";
  const fields = ["use_nn", "yolo_obj_width_m", "yolo_obj_height_m", "confidence_threshold"];
  const grid = document.createElement("div");
  grid.className = "runtime-grid";

  for (const key of fields) {
    const section = "object_detection";
    const wrap = document.createElement("label");
    wrap.dataset.section = section;
    wrap.dataset.key = key;
    const fullKey = `${section}.${key}`;
    wrap.textContent = LABEL_MAP[fullKey] || key;
    const spec = CONTROL_SPEC[fullKey];
    const input = document.createElement("input");
    input.dataset.section = section;
    input.dataset.key = key;
    const value = config?.[section]?.[key];

    if (spec?.type === "checkbox") {
      input.type = "checkbox";
      input.className = "toggle-input";
      input.checked = value == null ? Boolean(spec.defaultValue) : Boolean(value);
      input.addEventListener("change", () => scheduleLiveRuntimeSync());
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

  el.objectDetectionForm.appendChild(grid);
}

function collectRuntimeInputs(container, formResult) {
  const inputs = container?.querySelectorAll("input") || [];
  for (const input of inputs) {
    const section = input.dataset.section;
    const key = input.dataset.key;
    if (!section || !key || !formResult[section]) continue;
    if (input.type === "checkbox") {
      formResult[section][key] = input.checked;
      continue;
    }
    const raw = input.value.trim();
    if (input.type !== "range" && (raw.toLowerCase() === "true" || raw.toLowerCase() === "false")) {
      formResult[section][key] = raw.toLowerCase() === "true";
      continue;
    }
    const num = Number(raw);
    formResult[section][key] = Number.isFinite(num) && raw !== "" ? num : raw;
  }
}

function readRuntimeConfigFromForm() {
  const formResult = { camera: {}, processing: {}, object_detection: {} };
  collectRuntimeInputs(el.runtimeForm, formResult);
  collectRuntimeInputs(el.objectDetectionForm, formResult);
  const merged = JSON.parse(JSON.stringify(state.runtimeConfig || {}));
  if (!merged.camera_profiles || typeof merged.camera_profiles !== "object") {
    merged.camera_profiles = {};
  }
  const cam = Number(state.selectedCamera);
  const profileId = Number.isFinite(cam) && cam >= 0 && cam <= 5 ? cam : 0;
  merged.camera_profiles[String(profileId)] = formResult.camera;
  if (!merged.camera || typeof merged.camera !== "object" || profileId === 0) {
    merged.camera = formResult.camera;
  }
  merged.processing = formResult.processing;
  merged.object_detection = formResult.object_detection;
  return merged;
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

function cloneCameraConfig(camera) {
  return JSON.parse(JSON.stringify(camera || {}));
}

function ensureCameraProfilesForIds(ids) {
  if (!state.runtimeConfig || typeof state.runtimeConfig !== "object") return;
  if (!state.runtimeConfig.camera || typeof state.runtimeConfig.camera !== "object") {
    state.runtimeConfig.camera = {};
  }
  if (!state.runtimeConfig.camera_profiles || typeof state.runtimeConfig.camera_profiles !== "object") {
    state.runtimeConfig.camera_profiles = {};
  }
  for (const id of ids) {
    const key = String(id);
    if (!state.runtimeConfig.camera_profiles[key] || typeof state.runtimeConfig.camera_profiles[key] !== "object") {
      state.runtimeConfig.camera_profiles[key] = cloneCameraConfig(state.runtimeConfig.camera);
    }
  }
}

function renderRuntimeFormForCurrentCamera() {
  const ids = [0, 1, 2, 3, 4, 5];
  ensureCameraProfilesForIds(ids);
  const cam = Number(state.selectedCamera);
  const profileId = Number.isFinite(cam) && cam >= 0 && cam <= 5 ? cam : 0;
  const cfg = JSON.parse(JSON.stringify(state.runtimeConfig || {}));
  const profile = cfg?.camera_profiles?.[String(profileId)];
  if (profile && typeof profile === "object") {
    cfg.camera = cloneCameraConfig(profile);
  }
  renderRuntimeForm(cfg);
  renderObjectDetectionForm(cfg);
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
  scheduleFieldDraw();
  return;
}
  const tags = Number(p.tags_used || 0);
  const zErr = Number.isFinite(p.floor_z_error_avg) ? p.floor_z_error_avg : 0;
  el.robotPoseText.textContent = `Robot Pose: x=${p.x.toFixed(2)} m, y=${p.y.toFixed(2)} m | tags=${tags} | floor z err=${zErr.toFixed(3)} m`;
  scheduleFieldDraw();
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
  state.onnxPath = String(state.appConfig?.onnx_model_path || "");
  if (el.tagMapPath) el.tagMapPath.value = state.tagMapPath;
  if (el.onnxPath) el.onnxPath.value = state.onnxPath;
  state.runtimeConfig = boot.runtimeConfig;
  renderRuntimeFormForCurrentCamera();
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
      renderRuntimeFormForCurrentCamera();
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

  el.jetsonWatts.addEventListener("change", async () => {
    const vRaw = Math.round(Number(el.jetsonWatts.value) || 15);
    const v = [7, 15, 25].includes(vRaw) ? vRaw : 15;
    el.jetsonWatts.value = String(v);
    state.appConfig = { ...state.appConfig, jetson_max_watts: v };
    await window.vortexApi.saveAppConfig(readAppConfigFromUI()).catch(() => {});
  });

  const saveNetworkConfig = async () => {
    if (el.startupCameras) {
      el.startupCameras.value = normalizeCameraIndices(el.startupCameras.value);
    }
    if (el.udpPort) {
      el.udpPort.value = String(normalizePort(el.udpPort.value, 5809));
    }
    syncNetworkFieldState();
    state.appConfig = readAppConfigFromUI();
    await window.vortexApi.saveAppConfig(state.appConfig).catch(() => {});
  };

  [
    el.startupCameras,
    el.ntEnable,
    el.ntMode,
    el.ntTeam,
    el.ntServer,
    el.ntTable,
    el.udpEnable,
    el.udpTarget,
    el.udpPort
  ].forEach((input) => {
    if (!input) return;
    const evt = input.type === "checkbox" || input.tagName === "SELECT" ? "change" : "blur";
    input.addEventListener(evt, saveNetworkConfig);
  });

  el.applyJetsonWatts.addEventListener("click", async () => {
    const vRaw = Math.round(Number(el.jetsonWatts.value) || 15);
    const v = [7, 15, 25].includes(vRaw) ? vRaw : 15;
    el.jetsonWatts.value = String(v);
    state.appConfig = { ...state.appConfig, jetson_max_watts: v };
    const cfg = readAppConfigFromUI();
    await window.vortexApi.saveAppConfig(cfg).catch(() => {});
    const res = await window.vortexApi.setJetsonPowerLimit(cfg, v);
    if (!res?.ok) {
      if (res?.rebootRequired || res?.rebooting) {
        appendLog("Power mode change is in progress (reboot/verification).");
      } else {
        appendLog(`Power apply failed: ${res?.error || "unknown error"}`);
      }
      return;
    }
    appendLog(`Power mode applied: request ${v}W -> mode ${res.modeId} (${res.modeWatts}W)`);
    if (res.rebooted) appendLog("Jetson rebooted and power mode verified.");
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

  el.browseOnnx.addEventListener("click", async () => {
    const picked = await window.vortexApi.chooseFile({
      filters: [{ name: "ONNX", extensions: ["onnx"] }]
    });
    if (!picked) return;
    el.onnxPath.value = picked;
    state.onnxPath = picked;
    state.appConfig = { ...state.appConfig, onnx_model_path: picked };
    await window.vortexApi.saveAppConfig(readAppConfigFromUI());
  });

  el.uploadOnnxBuild.addEventListener("click", async () => {
    const onnx = String(el.onnxPath.value || "").trim();
    if (!onnx) {
      appendLog("Select an ONNX file first.");
      return;
    }
    setOnnxIndicator("active", "Starting...", 5);
    state.onnxPath = onnx;
    state.appConfig = { ...state.appConfig, onnx_model_path: onnx };
    await window.vortexApi.saveAppConfig(readAppConfigFromUI());
    const res = await window.vortexApi.uploadOnnxBuild(readAppConfigFromUI(), onnx);
    if (!res?.ok) {
      appendLog(`ONNX upload/build failed: ${res?.error || "unknown error"}`);
    } else {
      appendLog(`ONNX upload/build complete: ${res.remoteEngine}`);
    }
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

  el.buildMainBtn.addEventListener("click", async () => {
    const cfg = readAppConfigFromUI();
    setBuildIndicator("active", "[0/0] Queued...", 0);
    const res = await window.vortexApi.buildMainStart(cfg);
    if (!res?.ok) {
      setBuildIndicator("error", "✕ Build start failed", 0);
      appendLog(`Build start failed: ${res?.error || "unknown error"}`);
    } else {
      appendLog("Main build started.");
    }
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
          appendLog(`Start failed: ${msg}. SSH credentials are fixed to vortex/redstorm on port 22.`);
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
    state.runtimeConfig = readRuntimeConfigFromForm();
    const current = Number(state.selectedCamera);
    const idx = cams.indexOf(current);
    state.selectedCamera = cams[(idx + 1 + cams.length) % cams.length];
    renderRuntimeFormForCurrentCamera();
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
    pushPreviewFrame(dataUrl);
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
    if (state.selectedCamera == null) {
      state.selectedCamera = cam;
      renderRuntimeFormForCurrentCamera();
    }
    renderDetectionTable();
    scheduleOverlayDraw();
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
  window.vortexApi.onOnnxUploadProgress((p) => {
    const status = String(p?.status || "");
    const percent = Number(p?.percent || 0);
    const text = String(p?.text || `${percent}%`);
    if (status === "done") {
      setOnnxIndicator("done", text, 100);
    } else if (status === "error") {
      setOnnxIndicator("error", text, percent);
    } else if (status === "hidden") {
      setOnnxIndicator("hidden", "");
    } else {
      setOnnxIndicator("active", text, percent);
    }
  });
  window.vortexApi.onMainBuildProgress((p) => {
    const status = String(p?.status || "");
    const percent = Number(p?.percent || 0);
    const text = String(p?.text || `${percent}%`);
    if (status === "done") {
      setBuildIndicator("done", text, 100);
    } else if (status === "error") {
      setBuildIndicator("error", text, percent);
    } else if (status === "hidden") {
      setBuildIndicator("hidden", "");
    } else {
      setBuildIndicator("active", text, percent);
    }
  });

  setDeployIndicator("hidden", "");
  setMonitorIndicator("hidden", "");
  setOnnxIndicator("hidden", "");
  setBuildIndicator("hidden", "");
  await refreshRemoteCameras();
  refreshCameraControls();
  renderRobotPose();
  if (el.fieldMapCanvas) {
    el.fieldMapCanvas.addEventListener("mousemove", updateHoveredTagFromPointer);
    el.fieldMapCanvas.addEventListener("mouseleave", () => {
      if (state.hoveredTagId != null) {
        state.hoveredTagId = null;
        scheduleFieldDraw();
      }
    });
  }

  el.previewImage.onload = () => {
    previewLoadPending = false;
    scheduleOverlayDraw();
    if (pendingPreviewDataUrl) pushPreviewFrame(pendingPreviewDataUrl);
  };
  window.addEventListener("resize", () => {
    scheduleOverlayDraw();
    scheduleFieldDraw();
  });
}

init().catch((err) => appendLog(`Bootstrap failed: ${err.message}`));
