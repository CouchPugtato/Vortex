const state = {
  appConfig: null,
  runtimePath: "",
  runtimeConfig: {},
  cameras: new Map(),
  selectedCamera: null,
  currentLineCamera: null,
  viewMode: "apriltag",
  pipelineRunning: false
};
const CAMERA_ACTIVE_MS = 4000;

const el = {
  runtimePath: document.querySelector("#runtimePath"),
  loadRuntime: document.querySelector("#loadRuntime"),
  saveRuntime: document.querySelector("#saveRuntime"),
  runtimeForm: document.querySelector("#runtimeForm"),
  deployBtn: document.querySelector("#deployBtn"),
  deployProgressWrap: document.querySelector("#deployProgressWrap"),
  deployProgress: document.querySelector("#deployProgress"),
  deployProgressText: document.querySelector("#deployProgressText"),
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
  logs: document.querySelector("#logs")
};

function appendLog(msg) {
  const stamp = new Date().toLocaleTimeString();
  el.logs.textContent += `[${stamp}] ${msg}\n`;
  el.logs.scrollTop = el.logs.scrollHeight;
}

let deployFadeTimer = null;

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

function hydrateAppConfig(cfg) {
  state.appConfig = cfg;
  if (state.selectedCamera == null) {
    const raw = Number(cfg.preview_camera_index);
    state.selectedCamera = Number.isFinite(raw) && raw >= 0 && raw <= 7 ? raw : 0;
  }
}

function readAppConfigFromUI() {
  return {
    ...state.appConfig,
    preview_camera_index: Number(state.selectedCamera || 0)
  };
}

function renderRuntimeForm(config) {
  el.runtimeForm.innerHTML = "";
  const orderedCamera = [
    "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
    "tag_size_m", "x_offset", "y_offset", "z_offset", "pitch_deg", "yaw_deg", "roll_deg"
  ];
  const orderedProcessing = [
    "smoothing_alpha", "resolution_scale_factor", "yolo_obj_width_m", "yolo_obj_height_m"
  ];

  const sections = [
    { key: "camera", title: "Camera Constants", fields: orderedCamera },
    { key: "processing", title: "Processing Constants", fields: orderedProcessing }
  ];

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
      wrap.textContent = key;
      const input = document.createElement("input");
      input.dataset.section = section.key;
      input.dataset.key = key;
      input.value = config?.[section.key]?.[key] ?? "";
      wrap.appendChild(input);
      grid.appendChild(wrap);
    }

    details.appendChild(grid);
    el.runtimeForm.appendChild(details);
  }
}

function readRuntimeConfigFromForm() {
  const result = { camera: {}, processing: {} };
  const inputs = el.runtimeForm.querySelectorAll("input");
  for (const input of inputs) {
    const section = input.dataset.section;
    const key = input.dataset.key;
    const raw = input.value.trim();
    const num = Number(raw);
    result[section][key] = Number.isFinite(num) && raw !== "" ? num : raw;
  }
  return result;
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

function refreshCameraControls() {
  const active = getActiveCameras();
  el.nextCamera.disabled = !state.pipelineRunning || active.length < 2;
}

function previewSourceDims() {
  if (el.previewImage?.naturalWidth > 0 && el.previewImage?.naturalHeight > 0) {
    return { w: el.previewImage.naturalWidth, h: el.previewImage.naturalHeight };
  }
  return { w: 1920, h: 1080 };
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
      state.cameras.set(cam, { fps: null, apriltags: [], objects: [], lastSeen: 0 });
    }
    const obj = state.cameras.get(cam);
    obj.fps = fps;
    obj.apriltags = [];
    obj.objects = [];
    obj.lastSeen = Date.now();
    state.currentLineCamera = cam;
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

  const tag = line.match(/Tag ID:\s*(\d+)\s*\|\s*Dist:\s*([\d.-]+)m\s*\|\s*X:\s*([\d.-]+)m\s*\|\s*Y:\s*([\d.-]+)m/);
  if (tag) {
    current.apriltags.push({
      id: Number(tag[1]),
      dist: Number(tag[2]),
      x: Number(tag[3]),
      y: Number(tag[4])
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
    el.detectionTableHead.innerHTML = "<tr><th>ID</th><th>Distance (m)</th><th>X (m)</th><th>Y (m)</th></tr>";
    el.detectionTableBody.innerHTML = data.apriltags
      .map((r) => `<tr><td>${r.id}</td><td>${r.dist.toFixed(2)}</td><td>${r.x.toFixed(2)}</td><td>${r.y.toFixed(2)}</td></tr>`)
      .join("") || "<tr><td colspan='4'>No AprilTags</td></tr>";
  } else {
    el.detectionTableHead.innerHTML = "<tr><th>Class</th><th>Conf</th><th>Distance (m)</th><th>X (m)</th><th>Y (m)</th><th>BBox</th></tr>";
    el.detectionTableBody.innerHTML = data.objects
      .map((r) => `<tr><td>${r.className}</td><td>${r.conf.toFixed(2)}</td><td>${r.dist.toFixed(2)}</td><td>${r.x.toFixed(2)}</td><td>${r.y.toFixed(2)}</td><td>${r.bbox}</td></tr>`)
      .join("") || "<tr><td colspan='6'>No objects</td></tr>";
  }
  drawOverlay();
  refreshCameraControls();
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

  if (el.viewMode.value !== "object") return;
  const cam = Number(state.selectedCamera ?? 0);
  const data = state.cameras.get(cam);
  if (!data || !data.objects) return;

  const src = previewSourceDims();
  const sx = rect.width / Math.max(1, src.w);
  const sy = rect.height / Math.max(1, src.h);

  ctx.strokeStyle = "#00ff88";
  ctx.fillStyle = "#00ff88";
  ctx.lineWidth = 2;
  ctx.font = "12px Segoe UI";

  for (const o of data.objects) {
    const x = o.x * sx;
    const y = o.y * sy;
    const w = o.bboxWidth * sx;
    const h = o.bboxHeight * sy;
    if (![x, y, w, h].every(Number.isFinite)) continue;
    ctx.strokeRect(x, y, w, h);
    ctx.fillText(`${o.className} ${o.conf.toFixed(2)}`, x + 4, Math.max(12, y - 4));
  }
}

async function init() {
  const boot = await window.vortexApi.bootstrap();
  hydrateAppConfig(boot.appConfig);
  state.runtimePath = boot.runtimeConfigPath;
  el.runtimePath.value = state.runtimePath;
  state.runtimeConfig = boot.runtimeConfig;
  renderRuntimeForm(state.runtimeConfig);

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
    appendLog(`Saved runtime config: ${state.runtimePath}`);
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
        cfg = await window.vortexApi.applyMonitorPreset(cfg, "main_cam_0");
        state.appConfig = cfg;
        await window.vortexApi.saveAppConfig(cfg);
        const m = await window.vortexApi.monitorStart(cfg);
        if (!m?.ok) throw new Error(m?.error || "monitor start failed");
        const p = await window.vortexApi.previewStart(cfg);
        if (!p?.ok) throw new Error(p?.error || "preview start failed");
        state.pipelineRunning = true;
        el.togglePipeline.textContent = "Stop Monitor";
        refreshCameraControls();
      } catch (err) {
        const msg = String(err?.message || err);
        if (msg.includes("authentication methods failed")) {
          appendLog(`Start failed: ${msg}. Check vortex_config.json credentials.`);
        } else {
          appendLog(`Start failed: ${msg}`);
        }
      }
    } else {
      try {
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
    const cams = getActiveCameras();
    if (cams.length === 0) {
      refreshCameraControls();
      return;
    }
    const current = Number(state.selectedCamera);
    const idx = cams.indexOf(current);
    state.selectedCamera = cams[(idx + 1 + cams.length) % cams.length];
    state.appConfig.preview_camera_index = Number(state.selectedCamera || 0);
    await window.vortexApi.saveAppConfig(state.appConfig).catch(() => {});
    if (state.pipelineRunning) {
      try {
        await window.vortexApi.previewStop();
        const r = await window.vortexApi.previewStart(readAppConfigFromUI());
        if (!r?.ok) appendLog(`Preview restart failed: ${r?.error || "unknown error"}`);
      } catch (err) {
        appendLog(`Preview restart error: ${err.message}`);
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

  setDeployIndicator("hidden", "");
  refreshCameraControls();

  el.previewImage.onload = () => drawOverlay();
  window.addEventListener("resize", drawOverlay);
}

init().catch((err) => appendLog(`Bootstrap failed: ${err.message}`));
