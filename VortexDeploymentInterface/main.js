const { app, BrowserWindow, ipcMain, dialog, Menu } = require("electron");
const path = require("path");
const fs = require("fs/promises");
const fsSync = require("fs");
const { Client } = require("ssh2");
const crypto = require("crypto");
const { LAUNCHER, buildDeviceConfig } = require("./launch-config");
const { stageProject } = require("./staging-deploy");
const { operateRelease, releaseStatus } = require("./activation-deploy");
const { initializeJetson } = require("./initialize-deploy");
const { ServiceDiagnostics } = require("./service-diagnostics");

const APP_CONFIG_PATH = path.join(app.getPath("userData"), "deployment_tool_config.json");
const JOB_PATH = path.join(app.getPath("userData"), "pending_deployment_job.json");
const LEGACY_REPO_CONFIG_PATH = path.resolve(__dirname, "..", "config", "deployment_tool_config.json");
const LEGACY_APP_CONFIG_PATH = path.join(__dirname, "vortex_config.json");
const LEGACY_USERDATA_CONFIG_PATH = path.join(app.getPath("userData"), "vortex_config.json");
const DEFAULT_SSH = Object.freeze({
  port: "22",
  user: "vortex"
});
const serviceGuard = {
  activeCount: 0,
  lastSettings: null,
  resumeInFlight: null
};
let quitCleanupInProgress = false;
let stagingInFlight = false;
const serviceDiagnostics = new ServiceDiagnostics({
  connect: connectSsh, execChecked, log: emitLog, monitorState: emitMonitorState,
  frame: (packet) => { const win = getWindow(); if (win) win.webContents.send("service-frame", packet); },
  status: (status) => {
    emitPreviewState(Boolean(status.ready));
    const win = getWindow(); if (win) win.webContents.send("service-status", status);
  }
});

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1200,
    minHeight: 760,
    backgroundColor: "#0f1115",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  serviceDiagnostics.stop();
  monitorManager.stop();
  previewManager.stop();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", (event) => {
  if (quitCleanupInProgress) return;
  quitCleanupInProgress = true;
  event.preventDefault();
  (async () => {
    try {
      await serviceDiagnostics.stop();
      await monitorManager.stop();
      previewManager.stop();
      await restartManagedStartupServiceOnExit();
    } catch (err) {
      emitLog(`Shutdown cleanup error: ${err?.message || String(err)}`);
    } finally {
      app.quit();
    }
  })();
});

function getWindow() {
  return BrowserWindow.getAllWindows()[0] || null;
}

function emitLog(msg) {
  const win = getWindow();
  if (win) win.webContents.send("log", String(msg));
}

function emitMonitorState(isRunning) {
  const win = getWindow();
  if (win) win.webContents.send("monitor-state", isRunning);
}

function emitMonitorLine(line) {
  const win = getWindow();
  if (win) win.webContents.send("monitor-line", line);
}

function emitPreviewState(isRunning) {
  const win = getWindow();
  if (win) win.webContents.send("preview-state", isRunning);
}

function emitPreviewFrame(dataUrl) {
  const win = getWindow();
  if (win) win.webContents.send("preview-frame", dataUrl);
}

function emitBridgeState(state) {
  const win = getWindow();
  if (win) win.webContents.send("bridge-state", state);
}

function emitDeployProgress(progress) {
  const win = getWindow();
  if (win) win.webContents.send("deploy-progress", progress);
}

function emitReleaseState(state) {
  const win = getWindow();
  if (win) win.webContents.send("release-state", state);
}

function emitMonitorStartProgress(progress) {
  const win = getWindow();
  if (win) win.webContents.send("monitor-start-progress", progress);
}
function emitOnnxUploadProgress(progress) {
  const win = getWindow();
  if (win) win.webContents.send("onnx-upload-progress", progress);
}
function emitMainBuildProgress(progress) {
  const win = getWindow();
  if (win) win.webContents.send("main-build-progress", progress);
}

function isDetectionLine(line) {
  const s = String(line || "").trim();
  if (!s) return false;
  if (/^Camera\s+\d+:\s+[\d.]+\s+FPS/.test(s)) return true;
  if (/Tag ID:\s*\d+\s*\|/.test(s)) return true;
  if (/Object:\s*.+\|\s*Dist:/.test(s)) return true;
  return false;
}

function defaultAppConfig() {
  return {
    host: "192.168.55.1",
    port: DEFAULT_SSH.port,
    user: DEFAULT_SSH.user,
    pass: "",
    remote_path: "/home/vortex/deployments",
    local_path: path.resolve(__dirname, ".."),
    tag_map_path: path.join("..", "config", "2026-rebuilt-welded.json"),
    onnx_model_path: "",
    monitor_start_cmd: "",
    monitor_stop_cmd: "pkill -f orin_bridge || true",
    startup_service_name: "",
    preview_remote_path: "/tmp/vortex_preview.jpg",
    preview_state_path: "/tmp/vortex_bridge_state.json",
    preview_capture_cmd: "",
    jetson_max_watts: 15,
    startup_camera_indices: "0,1",
    build_profile: "cpu",
    vortex_nt_enable: false,
    vortex_nt_mode: "team",
    vortex_nt_team: "509",
    vortex_nt_server: "",
    vortex_nt_table: "/Vortex/Vision",
    vortex_udp_enable: true,
    vortex_udp_target: "192.168.1.24",
    vortex_udp_port: 5809
  };
}

const sessionPasswords = new Map();
function withSshConfig(cfg = {}) {
  const user = cfg.user || DEFAULT_SSH.user;
  const port = String(cfg.port || DEFAULT_SSH.port);
  const host = String(cfg.host || "192.168.55.1").trim();
  if (!/^[a-z_][a-z0-9_-]*$/.test(user) || user === "root") throw new Error("Use a non-root Jetson account");
  if (!/^\d+$/.test(port) || Number(port) < 1 || Number(port) > 65535) throw new Error("Invalid SSH port");
  if (!host || /[\s\x00-\x1f\x7f]/.test(host)) throw new Error("Invalid Jetson address");
  const key = `${host}:${port}:${user}`;
  if (typeof cfg.pass === "string" && cfg.pass) sessionPasswords.set(key, cfg.pass);
  return { ...cfg, host, user, port, pass: sessionPasswords.get(key) || "" };
}

function normalizeServiceName(name) {
  const value = String(name || "").trim();
  if (value && !/^[a-zA-Z0-9][a-zA-Z0-9_.@:-]*\.service$/.test(value)) {
    throw new Error("Startup service must be a valid .service unit name");
  }
  return value;
}

function hasManagedService(settings) {
  return !!normalizeServiceName(settings?.startup_service_name);
}

async function stopManagedStartupService(settings, reason = "operation") {
  if (!hasManagedService(settings)) return;
  const cfg = withSshConfig(settings);
  const serviceName = normalizeServiceName(cfg.startup_service_name);
  const conn = await connectSsh(cfg);
  try {
    const stopSvc = await execCommand(conn, `sudo -n systemctl stop ${shSingle(serviceName)}`);
    if (stopSvc.code !== 0) {
      throw new Error(`Service stop failed (${reason}): ${stopSvc.stderr || stopSvc.stdout}`);
    } else {
      emitLog(`Service stopped (${reason}): ${serviceName}`);
    }
  } finally {
    conn.end();
  }
}

async function startManagedStartupService(settings, reason = "exit") {
  if (!hasManagedService(settings)) return;
  const cfg = withSshConfig(settings);
  const serviceName = normalizeServiceName(cfg.startup_service_name);
  const conn = await connectSsh(cfg);
  try {
    const startSvc = await execCommand(conn, `sudo -n systemctl start ${shSingle(serviceName)}`);
    if (startSvc.code !== 0) {
      throw new Error(`Service start failed (${reason}): ${startSvc.stderr || startSvc.stdout}`);
    } else {
      emitLog(`Service started (${reason}): ${serviceName}`);
    }
  } finally {
    conn.end();
  }
}

async function beginManagedToolActivity(settings, label) {
  if (!hasManagedService(settings)) return;
  serviceGuard.lastSettings = withSshConfig(settings);
  const wasIdle = serviceGuard.activeCount === 0;
  if (wasIdle) {
    await stopManagedStartupService(serviceGuard.lastSettings, label);
  }
  serviceGuard.activeCount += 1;
}

function endManagedToolActivity() {
  if (serviceGuard.activeCount > 0) {
    serviceGuard.activeCount -= 1;
  }
  if (
    serviceGuard.activeCount === 0 &&
    serviceGuard.lastSettings &&
    hasManagedService(serviceGuard.lastSettings) &&
    !serviceGuard.resumeInFlight
  ) {
    const resumeSettings = serviceGuard.lastSettings;
    serviceGuard.resumeInFlight = startManagedStartupService(resumeSettings, "tool-idle")
      .catch((err) => {
        emitLog(`Service auto-resume failed: ${err?.message || String(err)}`);
      })
      .finally(() => {
        serviceGuard.resumeInFlight = null;
      });
  }
}

async function restartManagedStartupServiceOnExit() {
  if (!serviceGuard.lastSettings || !hasManagedService(serviceGuard.lastSettings)) return;
  await startManagedStartupService(serviceGuard.lastSettings, "tool-closed");
}

function resolvePathFromAppDir(candidatePath, fallback = "") {
  const raw = String(candidatePath || "").trim();
  if (!raw) return fallback;
  const resolved = path.resolve(__dirname, raw);
  if (!fsSync.existsSync(resolved)) return fallback;
  return resolved;
}

function portablePathFromAppDir(candidatePath) {
  const raw = String(candidatePath || "").trim();
  if (!raw) return "";
  const abs = path.resolve(__dirname, raw);
  const rel = path.relative(__dirname, abs);
  if (rel && !rel.startsWith("..") && !path.isAbsolute(rel)) return rel;
  return abs;
}

async function readPersistedAppConfig() {
  const userCfg = await readJson(APP_CONFIG_PATH, null);
  if (userCfg) return userCfg;
  const repoCfg = await readJson(LEGACY_REPO_CONFIG_PATH, null);
  if (repoCfg) {
    await writeJson(APP_CONFIG_PATH, repoCfg);
    return repoCfg;
  }
  const legacyUserCfg = await readJson(LEGACY_USERDATA_CONFIG_PATH, null);
  if (legacyUserCfg) {
    await writeJson(APP_CONFIG_PATH, legacyUserCfg);
    return legacyUserCfg;
  }
  const legacyCfg = await readJson(LEGACY_APP_CONFIG_PATH, null);
  if (legacyCfg) {
    await writeJson(APP_CONFIG_PATH, legacyCfg);
    return legacyCfg;
  }
  return {};
}

async function readJson(filePath, fallback) {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw);
  } catch (_err) {
    return fallback;
  }
}

async function writeJson(filePath, obj) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${crypto.randomUUID()}.tmp`;
  try {
    await fs.writeFile(temporary, JSON.stringify(obj, null, 2), "utf8");
    await fs.rename(temporary, filePath);
  } finally {
    await fs.unlink(temporary).catch(() => {});
  }
}

function toPosix(p) {
  return p.replace(/\\/g, "/");
}

function basenameOrDefault(p, fallback) {
  const name = path.basename(p || "");
  if (!name || name === "." || name === "..") return fallback;
  return name;
}

function inferRemoteDeployFolder(remoteBase, localPath) {
  return `${(remoteBase || "").replace(/\/+$/, "")}/${basenameOrDefault(localPath, "Vortex")}`;
}

function shSingle(s) {
  return `'${String(s).replace(/'/g, `'\\''`)}'`;
}

function buildRunVortexScript(settings) {
  buildDeviceConfig(settings);
  return LAUNCHER;
}

async function syncRemoteStartupScript(conn, settings) {
  const remoteTarget = inferRemoteDeployFolder(settings.remote_path, settings.local_path);
  const runScriptPath = `${remoteTarget}/run_vortex.sh`;
  const content = Buffer.from(buildRunVortexScript(settings), "utf8");
  const sftp = await getSftp(conn);
  await syncTagMapRemote(conn, settings);
  const devicePath = `${remoteTarget}/config/device.json`;
  await execChecked(conn, `mkdir -p -- ${shSingle(path.posix.dirname(devicePath))}`);
  await sftpAtomicWrite(conn, sftp, devicePath,
    Buffer.from(JSON.stringify(buildDeviceConfig(settings), null, 2)+"\n"));
  await sftpAtomicWrite(conn, sftp, runScriptPath, content);
  await execChecked(conn, `chmod +x -- ${shSingle(runScriptPath)}`);
  emitLog("Production launcher and device settings synced (camera, build profile, NT, UDP).");
}

function tailForLog(text, limit = 6000) {
  const t = String(text || "").trim();
  if (!t) return "";
  if (t.length <= limit) return t;
  return `...[output truncated, showing last ${limit} chars]\n${t.slice(-limit)}`;
}

function stripAnsi(text) {
  return String(text || "").replace(/\x1B\[[0-9;]*[A-Za-z]/g, "");
}

function parseNvpmodelModes(configText) {
  const text = String(configText || "");
  const byId = new Map();
  const blockRe = /<\s*POWER_MODEL\b([^>]*)>([\s\S]*?)<\/\s*POWER_MODEL\s*>/gi;
  let m = null;
  while ((m = blockRe.exec(text)) != null) {
    const attrs = String(m[1] || "");
    const body = String(m[2] || "");
    const id = Number((attrs.match(/\bID\s*=\s*([0-9]+)/i) || [])[1]);
    if (!Number.isFinite(id)) continue;
    const name = String((attrs.match(/\bNAME\s*=\s*("[^"]+"|'[^']+'|[^\s>]+)/i) || [])[1] || `MODE_${id}`)
      .replace(/['"]/g, "");
    let watts = Number((body.match(/\bPOWER_BUDGET\s+([0-9]+(?:\.[0-9]+)?)/i) || [])[1]);
    if (Number.isFinite(watts) && watts > 200) watts /= 1000.0;
    if (!Number.isFinite(watts)) {
      const w = name.match(/([0-9]+(?:\.[0-9]+)?)\s*W/i);
      if (w) watts = Number(w[1]);
    }
    if (!Number.isFinite(watts) && /MAXN/i.test(name)) watts = 999;
    byId.set(id, { id, watts, name });
  }

  // fallback parser for nontagged style
  const lineRe = /POWER_MODEL[^\n\r]*\bID\s*=\s*([0-9]+)[^\n\r]*/gi;
  const lineMatches = [];
  let lm = null;
  while ((lm = lineRe.exec(text)) != null) {
    lineMatches.push({
      id: Number(lm[1]),
      line: String(lm[0] || ""),
      index: Number(lm.index),
      end: Number(lineRe.lastIndex)
    });
  }
  for (let i = 0; i < lineMatches.length; i += 1) {
    const cur = lineMatches[i];
    const id = cur.id;
    if (!Number.isFinite(id)) continue;
    const nameMatch = cur.line.match(/\bNAME\s*=\s*("[^"]+"|'[^']+'|[^\s>]+)/i);
    const name = nameMatch ? String(nameMatch[1]).replace(/['"]/g, "") : `MODE_${id}`;
    const start = cur.end;
    const end = i + 1 < lineMatches.length ? lineMatches[i + 1].index : text.length;
    const section = text.slice(start, end);
    let watts = Number((section.match(/\bPOWER_BUDGET\s+([0-9]+(?:\.[0-9]+)?)/i) || [])[1]);
    if (Number.isFinite(watts) && watts > 200) watts /= 1000.0;
    if (!Number.isFinite(watts)) {
      const w = name.match(/([0-9]+(?:\.[0-9]+)?)\s*W/i);
      if (w) watts = Number(w[1]);
    }
    if (!Number.isFinite(watts) && /MAXN/i.test(name)) watts = 999;
    if (!byId.has(id)) byId.set(id, { id, watts, name });
  }

  const modes = [...byId.values()];
  modes.sort((a, b) => {
    const aw = Number.isFinite(a.watts) ? a.watts : -1;
    const bw = Number.isFinite(b.watts) ? b.watts : -1;
    return aw - bw || a.id - b.id;
  });
  return modes;
}

function parseNvpmodelModesFromQuery(queryText) {
  const text = String(queryText || "");
  const byId = new Map();
  const re = /\bID\s*[:=]\s*([0-9]+)[^\n\r]*\bNAME\s*[:=]\s*([^\n\r]+)/gi;
  let m = null;
  while ((m = re.exec(text)) != null) {
    const id = Number(m[1]);
    if (!Number.isFinite(id)) continue;
    const name = String(m[2] || "").replace(/[<>"']/g, "").trim() || `MODE_${id}`;
    let watts = NaN;
    const w = name.match(/([0-9]+(?:\.[0-9]+)?)\s*W/i);
    if (w) watts = Number(w[1]);
    if (!Number.isFinite(watts) && /MAXN/i.test(name)) watts = 999;
    byId.set(id, { id, watts, name });
  }
  return [...byId.values()].sort((a, b) => a.id - b.id);
}

function parseCurrentNvpmodelState(text) {
  const raw = String(text || "");
  const name = ((raw.match(/Current mode:\s*NV Power Mode:\s*([^\r\n]+)/i) || [])[1] || "").trim();
  const id = Number((raw.match(/Current mode:[\s\S]*?\n\s*([0-9]+)\s*(?:\r?\n|$)/i) || [])[1]);
  return {
    name: name || "",
    id: Number.isFinite(id) ? id : null
  };
}

function pickTargetPowerMode(modes, requestedWatts) {
  const req = Number(requestedWatts);
  const exact = modes.find((m) =>
    Number.isFinite(m?.watts) && Math.round(Number(m.watts)) === Math.round(req)
  ) || modes.find((m) => new RegExp(`\\b${Math.round(req)}\\s*W\\b`, "i").test(String(m?.name || "")));
  if (exact) return exact;

  const numeric = modes.filter((x) => Number.isFinite(x.watts));
  if (numeric.length === 0) return modes[0] || null;
  let chosen = numeric[0];
  for (const mode of numeric) {
    if (mode.watts <= req) chosen = mode;
  }
  return chosen;
}

async function waitForJetsonReconnect(settings, timeoutMs = 180000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const conn = await connectSsh(settings);
      conn.end();
      return true;
    } catch (_err) {
      await new Promise((r) => setTimeout(r, 3000));
    }
  }
  return false;
}

async function setJetsonPowerLimit(settings, requestedWatts) {
  const cfg = withSshConfig(settings);
  const conn = await connectSsh(settings);
  try {
    const req = Number(requestedWatts);
    if (!Number.isFinite(req) || req <= 0) throw new Error(`Invalid wattage: ${requestedWatts}`);

    const cfgPathRes = await execCommand(
      conn,
      "bash -lc \"for f in /etc/nvpmodel.conf /etc/nvpmodel/*.conf /etc/nvpmodel_*.conf; do [ -f \\\"$f\\\" ] && echo \\\"$f\\\"; done | head -n 1\""
    );
    const cfgPath = String(cfgPathRes.stdout || "").trim() || "/etc/nvpmodel.conf";
    const cfgText = await execCommand(conn, `bash -lc "cat '${cfgPath}' 2>/dev/null || true"`);
    let modes = parseNvpmodelModes(cfgText.stdout || "");
    if (modes.length === 0) {
      const q = await execCommand(conn, "bash -lc \"nvpmodel -q --verbose 2>/dev/null || nvpmodel -q 2>/dev/null || true\"");
      modes = parseNvpmodelModesFromQuery(`${q.stdout || ""}\n${q.stderr || ""}`);
    }
    if (modes.length === 0) throw new Error(`Could not parse nvpmodel modes (config path: ${cfgPath})`);

    const chosen = pickTargetPowerMode(modes, req);
    if (!chosen) throw new Error("No power mode candidates found");
    emitLog(`Power mode target selected: requested ${req}W -> mode ${chosen.id} (${chosen.name})`);

    const beforeQ = await execCommand(conn, "bash -lc \"nvpmodel -q --verbose 2>/dev/null || nvpmodel -q 2>/dev/null || true\"");
    const before = parseCurrentNvpmodelState(`${beforeQ.stdout || ""}\n${beforeQ.stderr || ""}`);

    const pw = shSingle(cfg.pass || "");
    const applyLog = `/tmp/vortex_nvp_apply_${Date.now()}.log`;
    const apply = await execCommand(
      conn,
      `bash -lc "echo ${pw} | sudo -S -p '' bash -lc \\\"printf 'YES\\\\n' | /usr/sbin/nvpmodel -m ${chosen.id}\\\" > '${applyLog}' 2>&1; code=\\$?; cat '${applyLog}'; exit \\$code"`
    );
    const out = String((apply.stdout || "") + "\n" + (apply.stderr || "")).trim();
    const applyCode = Number.isFinite(Number(apply.code)) ? Number(apply.code) : null;
    const likelyRebootDisconnect = applyCode == null;
    const rebootRequired =
      /reboot required/i.test(out) ||
      /DO YOU WANT TO REBOOT NOW/i.test(out) ||
      likelyRebootDisconnect;
    const rebootAlreadyStarted =
      /rebooting/i.test(out) ||
      /Connection to .* closed/i.test(out) ||
      likelyRebootDisconnect;
    if (applyCode !== null && applyCode !== 0 && !rebootRequired) {
      throw new Error(out || `nvpmodel apply failed (exit=${applyCode})`);
    }

    if (rebootRequired) {
      if (rebootAlreadyStarted) {
        emitLog(`Jetson accepted mode ${chosen.name} and already started reboot.`);
      } else {
        emitLog(`Jetson requires reboot to apply mode ${chosen.name}. Rebooting now...`);
        try {
          await execCommand(
            conn,
            `bash -lc "echo ${pw} | sudo -S -p '' reboot >/tmp/vortex_reboot_${Date.now()}.log 2>&1 || true"`
          );
        } catch (_err) {
          // expected to disconnect quickly once reboot starts
        }
      }
      conn.end();

      emitLog("Waiting for Jetson to come back online...");
      const online = await waitForJetsonReconnect(settings, 180000);
      if (!online) {
        return {
          ok: false,
          rebootRequired: true,
          rebooting: true,
          error: "Jetson reboot started but SSH did not come back within 180s."
        };
      }
      emitLog("Jetson is back online. Verifying applied power mode...");
      const verifyConn = await connectSsh(settings);
      try {
        const afterQ = await execCommand(
          verifyConn,
          "bash -lc \"nvpmodel -q --verbose 2>/dev/null || nvpmodel -q 2>/dev/null || true\""
        );
        const after = parseCurrentNvpmodelState(`${afterQ.stdout || ""}\n${afterQ.stderr || ""}`);
        if (after.id === chosen.id || (after.name && after.name === chosen.name)) {
          emitLog(`Jetson power mode applied after reboot: ${after.name || chosen.name} (${after.id ?? chosen.id})`);
          return {
            ok: true,
            requestedWatts: req,
            modeId: chosen.id,
            modeWatts: chosen.watts,
            modeName: chosen.name,
            rebootRequired: true,
            rebooting: false,
            rebooted: true,
            previousModeId: before.id,
            previousModeName: before.name
          };
        }
        return {
          ok: false,
          error: `Reboot completed, but mode verify failed. Expected ${chosen.name} (${chosen.id}), got ${after.name || "unknown"} (${after.id ?? "?"}).`,
          rebootRequired: true,
          rebooting: false
        };
      } finally {
        verifyConn.end();
      }
    }

    emitLog(`Jetson power mode set: requested ${req}W, applied mode ${chosen.id} (${chosen.watts}W, ${chosen.name})`);
    return {
      ok: true,
      requestedWatts: req,
      modeId: chosen.id,
      modeWatts: chosen.watts,
      modeName: chosen.name,
      rebootRequired: false,
      previousModeId: before.id,
      previousModeName: before.name
    };
  } catch (err) {
    const msg = err?.message || String(err);
    emitLog(`Jetson power mode set failed: ${msg}`);
    return { ok: false, error: msg };
  } finally {
    conn.end();
  }
}

function connectSsh(settings) {
  return new Promise((resolve, reject) => {
    const conn = new Client();
    const cfg = withSshConfig(settings);
    conn
      .on("ready", () => resolve(conn))
      .on("error", (err) => reject(err))
      .connect({
        host: cfg.host,
        port: Number(cfg.port || 22),
        username: cfg.user,
        password: cfg.pass,
        readyTimeout: 8000
      });
  });
}

function execCommand(conn, command, { stream = false } = {}) {
  return new Promise((resolve, reject) => {
    conn.exec(command, (err, channel) => {
      if (err) return reject(err);
      if (stream) return resolve(channel);

      let stdout = "";
      let stderr = "";
      channel.on("data", (data) => {
        stdout += data.toString("utf8");
      });
      channel.stderr.on("data", (data) => {
        stderr += data.toString("utf8");
      });
      channel.on("close", (code) => {
        resolve({ code, stdout, stderr });
      });
    });
  });
}

async function execChecked(conn, command) {
  const result = await execCommand(conn, command);
  if (result.code !== 0) {
    throw new Error(`Remote command failed (${result.code}): ${tailForLog(result.stderr || result.stdout)}`);
  }
  return result;
}

async function sftpAtomicWrite(conn, sftp, destination, bytes) {
  const temporary = `${destination}.${crypto.randomUUID()}.tmp`;
  try {
    await sftpWriteFile(sftp, temporary, bytes);
    await execChecked(conn, `mv -f -- ${shSingle(temporary)} ${shSingle(destination)}`);
  } finally {
    await execCommand(conn, `rm -f -- ${shSingle(temporary)}`).catch(() => {});
  }
}

function getSftp(conn) {
  return new Promise((resolve, reject) => {
    conn.sftp((err, sftp) => {
      if (err) reject(err);
      else resolve(sftp);
    });
  });
}

function sftpFastPut(sftp, localPath, remotePath, options = {}) {
  return new Promise((resolve, reject) => {
    sftp.fastPut(localPath, remotePath, options, (err) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

function sftpReadFile(sftp, remotePath) {
  return new Promise((resolve, reject) => {
    const stream = sftp.createReadStream(remotePath);
    const chunks = [];
    stream.on("data", (chunk) => chunks.push(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(Buffer.concat(chunks)));
  });
}

function sftpWriteFile(sftp, remotePath, bytes) {
  return new Promise((resolve, reject) => {
    const ws = sftp.createWriteStream(remotePath, { flags: "w", mode: 0o644 });
    ws.on("error", reject);
    ws.on("close", resolve);
    ws.end(bytes);
  });
}

async function deployProject(settings) {
  const result = await stageProject(settings, inferRemoteDeployFolder(settings.remote_path, settings.local_path), {
    appDir: __dirname,
    ...jobPersistence(settings),
    connect: connectSsh,
    execChecked,
    getSftp,
    upload: sftpFastPut,
    log: emitLog,
    progress: (event) => emitDeployProgress({ ...event, current: event.percent, total: 100 })
  });
  if (result.release_id) {
    const persisted = await readPersistedAppConfig();
    await writeJson(APP_CONFIG_PATH, { ...persisted, last_ready_release: result.release_id });
    emitReleaseState({ ready: [result.release_id] });
  }
  return result;
}

// Legacy IPC callers also stage a new snapshot; they cannot rebuild the active tree.
async function buildMainProgram(settings) {
  return deployProject(settings);
}

async function resolveMonitorCommand(conn, settings) {
  const remoteTarget = inferRemoteDeployFolder(settings.remote_path, settings.local_path);
  const camera = Number(settings.preview_camera_index ?? 0);
  if (!Number.isInteger(camera) || camera < 0 || camera > 5) throw new Error("Invalid debug camera (0-5)");
  // Legacy diagnostics remain explicit until service-attached preview is implemented.
  // No hidden builds, feature fallback, dependency installation, or lockfile changes.
  await syncRemoteStartupScript(conn, settings);
  await execChecked(conn, `cd ${shSingle(remoteTarget)} && python3 scripts/run_vortex.py --check-debug-bridge`);
  emitLog("Starting legacy bridge diagnostics; this replaces production camera ownership while monitoring.");
  return `cd ${shSingle(remoteTarget)} && python3 scripts/run_vortex.py --debug-bridge --camera ${camera}`;
}

const monitorManager = {
  conn: null,
  channel: null,
  restartTimer: null,
  stopping: false,
  settings: null,
  activityHeld: false,
  async start(settings) {
    if (this.conn) throw new Error("Monitor already running");
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    this.settings = settings;
    this.stopping = false;
    try {
      emitMonitorStartProgress({ status: "starting", percent: 10, text: "Connecting..." });
      this.conn = await connectSsh(settings);

      emitMonitorStartProgress({ status: "starting", percent: 55, text: "Resolving monitor..." });
      const resolvedStart = await resolveMonitorCommand(this.conn, settings);
      emitMonitorStartProgress({ status: "starting", percent: 70, text: "Freeing camera devices..." });
      await execCommand(
        this.conn,
        "bash -lc \"pkill -f dumapril-taglocalization >/dev/null 2>&1 || true; pkill -f orin_bridge >/dev/null 2>&1 || true; sleep 0.25\""
      );
      emitMonitorStartProgress({ status: "starting", percent: 85, text: "Launching monitor..." });
      const cmd = `bash -lc "${String(resolvedStart).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
      this.channel = await execCommand(this.conn, cmd, { stream: true });
      emitMonitorState(true);
      emitMonitorStartProgress({ status: "done", percent: 100, text: "✓ Monitor started" });
      emitLog(`Monitor started: ${resolvedStart}`);

      let pending = "";
      this.channel.on("data", (data) => {
        pending += data.toString("utf8");
        const lines = pending.split(/\r?\n/);
        pending = lines.pop() || "";
        for (const line of lines) {
          emitMonitorLine(line);
          if (!isDetectionLine(line)) emitLog(line);
        }
      });
      this.channel.stderr.on("data", (data) => {
        const msg = data.toString("utf8").trim();
        if (msg) emitLog(msg);
      });
      this.channel.on("close", async (code, signal) => {
        const restartSettings = this.settings;
        if (pending.trim()) {
          emitMonitorLine(pending.trim());
          if (!isDetectionLine(pending.trim())) emitLog(pending.trim());
        }
        if (this.activityHeld && this.stopping) {
          endManagedToolActivity();
          this.activityHeld = false;
        }
        this.cleanup();
        emitMonitorState(false);
        emitMonitorStartProgress({ status: "hidden", percent: 0, text: "" });
        if (!this.stopping) {
          const exitCode = typeof code === "number" ? code : "unknown";
          emitLog(`Monitor stream ended (exit=${exitCode}${signal ? `, signal=${signal}` : ""}).`);
          emitLog("Monitor exited unexpectedly. Restarting in 2s...");
          this.restartTimer = setTimeout(async () => {
            this.restartTimer = null;
            try {
              await this.start(restartSettings);
            } catch (err) {
              emitLog(`Monitor auto-restart failed: ${err?.message || String(err)}`);
            }
          }, 2000);
        }
      });
    } catch (err) {
      this.cleanup();
      emitMonitorState(false);
      emitMonitorStartProgress({ status: "error", percent: 0, text: "✕ Start failed" });
      throw err;
    }
  },
  async stop() {
    this.stopping = true;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    try {
      if (this.settings && this.settings.monitor_stop_cmd) {
        const stopConn = await connectSsh(this.settings);
        await execCommand(stopConn, this.settings.monitor_stop_cmd);
        stopConn.end();
      }
      if (this.channel) this.channel.close();
    } catch (err) {
      emitLog(`Monitor stop failed: ${err.message}`);
    } finally {
      if (this.activityHeld) {
        endManagedToolActivity();
        this.activityHeld = false;
      }
      this.cleanup();
      emitMonitorState(false);
      emitMonitorStartProgress({ status: "hidden", percent: 0, text: "" });
    }
  },
  cleanup() {
    if (this.channel) this.channel = null;
    if (this.conn) {
      this.conn.end();
      this.conn = null;
    }
    this.settings = null;
  }
};

async function syncTagMapRemote(conn, settings) {
  if (!String(settings.tag_map_path || "").trim()) return;
  const localPath = path.resolve(__dirname, settings.tag_map_path);
  const map = await readJson(localPath, null);
  if (!Array.isArray(map?.tags) || map.tags.length === 0) throw new Error("Selected tag map is missing or invalid");
  const remoteTarget = inferRemoteDeployFolder(settings.remote_path, settings.local_path);
  const destination = `${remoteTarget}/config/apriltag_map.json`;
  await execChecked(conn, `mkdir -p -- ${shSingle(path.posix.dirname(destination))}`);
  await sftpAtomicWrite(conn, await getSftp(conn), destination,
    Buffer.from(JSON.stringify(map)+"\n"));
  emitLog(`Tag map synced: ${destination}`);
}

const previewManager = {
  running: false,
  latestFrameDataUrl: null,
  latestBridgeState: null,
  emitTimer: null,
  emitPending: false,
  scheduleEmit() {
    if (this.emitPending) return;
    this.emitPending = true;
    this.emitTimer = setTimeout(() => {
      this.emitPending = false;
      if (!this.running) return;
      if (this.latestFrameDataUrl) emitPreviewFrame(this.latestFrameDataUrl);
      if (this.latestBridgeState) emitBridgeState(this.latestBridgeState);
    }, 120);
  },
  async start(settings) {
    if (this.running) return;
    this.running = true;
    this.latestFrameDataUrl = null;
    this.latestBridgeState = null;
    this.emitPending = false;
    emitPreviewState(true);
    emitLog("Preview started.");
    let conn = null;
    let sftp = null;
    while (this.running) {
      try {
        if (!conn) {
          conn = await connectSsh(settings);
          sftp = await getSftp(conn);
        }
        const bytes = await sftpReadFile(sftp, settings.preview_remote_path);
        const mime = bytes[0] === 0x89 ? "image/png" : "image/jpeg";
        this.latestFrameDataUrl = `data:${mime};base64,${bytes.toString("base64")}`;
        try {
          const stateBytes = await sftpReadFile(
            sftp,
            settings.preview_state_path || "/tmp/vortex_bridge_state.json"
          );
          const parsed = JSON.parse(stateBytes.toString("utf8"));
          this.latestBridgeState = parsed;
        } catch (_stateErr) {
          // state file may not exist yet, ignore and keep preview alive
        }
        this.scheduleEmit();
      } catch (err) {
        const msg = err?.message || String(err);
        emitLog(`Preview error: ${msg}`);
        if (conn) {
          try { conn.end(); } catch (_e) {}
          conn = null;
          sftp = null;
        }
        if (msg.includes("All configured authentication methods failed")) {
          emitLog("Authentication failed. Check the Jetson address, account and SSH password in Jetson Setup.");
          this.running = false;
        }
        await new Promise((r) => setTimeout(r, 450));
        continue;
      }
      await new Promise((r) => setTimeout(r, 90));
    }
    if (conn) {
      try { conn.end(); } catch (_e) {}
    }
    emitPreviewState(false);
    emitLog("Preview stopped.");
  },
  stop() {
    this.running = false;
    if (this.emitTimer) {
      clearTimeout(this.emitTimer);
      this.emitTimer = null;
    }
    this.emitPending = false;
    this.latestFrameDataUrl = null;
    this.latestBridgeState = null;
  }
};

ipcMain.handle("bootstrap", async () => {
  const appCfg = withSshConfig({ ...defaultAppConfig(), ...(await readPersistedAppConfig()) });
  appCfg.pass = "";
  delete appCfg.admin_password;
  delete appCfg.preview_camera_index;
  appCfg.local_path = resolvePathFromAppDir(appCfg.local_path, path.resolve(__dirname, ".."));
  appCfg.tag_map_path = resolvePathFromAppDir(
    appCfg.tag_map_path,
    resolvePathFromAppDir(path.join("..", "config", "2026-rebuilt-welded.json"), "")
  );
  if (appCfg.onnx_model_path) {
    appCfg.onnx_model_path = resolvePathFromAppDir(appCfg.onnx_model_path, "");
  }
  const runtimeConfigPath = path.resolve(__dirname, appCfg.runtime_config_path ||
    path.join(appCfg.local_path, "config", "config.json"));
  const runtimeConfig = await readJson(runtimeConfigPath, {});
  return { appConfig: appCfg, runtimeConfigPath, runtimeConfig };
});

ipcMain.handle("choose-folder", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"]
  });
  if (result.canceled || !result.filePaths[0]) return null;
  return result.filePaths[0];
});

ipcMain.handle("choose-file", async (_evt, opts) => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: Array.isArray(opts?.filters) ? opts.filters : undefined
  });
  if (result.canceled || !result.filePaths[0]) return null;
  return result.filePaths[0];
});

ipcMain.handle("load-tag-map", async (_evt, filePath) => {
  const mapPath = path.resolve(String(filePath || ""));
  const data = await readJson(mapPath, null);
  if (!data) throw new Error(`Failed to load tag map: ${mapPath}`);
  return data;
});

ipcMain.handle("upload-onnx-build-engine", async (_evt, settings, localOnnxPath) => {
  let conn = null;
  await beginManagedToolActivity(settings, "onnx-upload-build");
  try {
    conn = await connectSsh(settings);
    emitOnnxUploadProgress({ status: "starting", percent: 5, text: "Connecting..." });
    const localPath = path.resolve(String(localOnnxPath || ""));
    if (!localPath || !fsSync.existsSync(localPath)) {
      throw new Error(`ONNX file not found: ${localPath}`);
    }
    if (path.extname(localPath).toLowerCase() !== ".onnx") {
      throw new Error("Selected file is not .onnx");
    }

    const remoteTarget = inferRemoteDeployFolder(settings.remote_path, settings.local_path);
    const fileName = path.basename(localPath);
    const stem = fileName.replace(/\.onnx$/i, "");
    const remoteModelsDir = `${remoteTarget}/models`;
    const remoteOnnx = `${remoteModelsDir}/${fileName}`;
    const remoteEngine = `${remoteModelsDir}/${stem}.engine`;
    const canonicalOnnx = `${remoteModelsDir}/rockpaperscizzors.onnx`;
    const canonicalEngine = `${remoteModelsDir}/rockpaperscizzors.engine`;

    const hasProject = await execCommand(conn, `test -d '${remoteTarget}/model_builder' && echo OK || echo NO`);
    if (!String(hasProject.stdout || "").includes("OK")) {
      throw new Error(
        `Remote project not found at ${remoteTarget}. Deploy first so model_builder exists on the target.`
      );
    }

    emitLog(`Uploading ONNX: ${fileName}`);
    await execCommand(conn, `mkdir -p '${remoteModelsDir}'`);
    const sftp = await getSftp(conn);
    const localSize = Number(fsSync.statSync(localPath)?.size || 0);
    let lastPercent = -1;
    emitOnnxUploadProgress({ status: "uploading", percent: 10, text: "Uploading 0%" });
    await sftpFastPut(sftp, localPath, remoteOnnx, {
      step: (transferred, _chunk, total) => {
        const denom = Number(total) || localSize || 1;
        const frac = Math.max(0, Math.min(1, Number(transferred || 0) / denom));
        const stage = Math.round(frac * 100);
        const percent = 10 + Math.round(frac * 60);
        if (percent !== lastPercent) {
          lastPercent = percent;
          emitOnnxUploadProgress({
            status: "uploading",
            percent,
            text: `Uploading ${stage}%`
          });
        }
      }
    });
    if (remoteOnnx !== canonicalOnnx) {
      await execCommand(conn, `cp '${remoteOnnx}' '${canonicalOnnx}'`);
      emitLog(`Canonical ONNX updated: ${canonicalOnnx}`);
    }
    emitLog(`Remote ONNX: ${remoteOnnx}`);

    const builderBin = `${remoteTarget}/model_builder/build/build_engine`;
    const hasBuilder = await execCommand(conn, `test -x '${builderBin}' && echo OK || echo NO`);
    if (!String(hasBuilder.stdout || "").includes("OK")) {
      emitLog("Building model_builder/build_engine on remote...");
      emitOnnxUploadProgress({ status: "building", percent: 75, text: "Building model builder..." });
      const build = await execCommand(
        conn,
        `bash -lc "cd '${remoteTarget}/model_builder' && mkdir -p build && cd build && cmake .. && make -j4"`
      );
      if (build.code !== 0) {
        const out = String((build.stdout || "") + "\n" + (build.stderr || "")).trim();
        if (out) emitLog(tailForLog(out));
        throw new Error("Failed to build model_builder/build_engine on remote");
      }
    }

    emitLog(`Generating engine: ${stem}.engine`);
    emitOnnxUploadProgress({ status: "building", percent: 85, text: "Generating engine..." });
    const gen = await execCommand(
      conn,
      `bash -lc "cd '${remoteTarget}' && ./model_builder/build/build_engine '${remoteOnnx}' '${remoteEngine}'"`
    );
    const genOut = String((gen.stdout || "") + "\n" + (gen.stderr || "")).trim();
    if (genOut) emitLog(tailForLog(genOut));
    if (gen.code !== 0) {
      throw new Error("ONNX -> engine conversion failed");
    }

    if (remoteEngine !== canonicalEngine) {
      await execCommand(conn, `cp '${remoteEngine}' '${canonicalEngine}'`);
      emitLog(`Canonical engine updated: ${canonicalEngine}`);
    }

    emitLog(`Engine generated: ${remoteEngine}`);
    emitOnnxUploadProgress({ status: "done", percent: 100, text: "✓ Upload & build complete" });
    return { ok: true, remoteOnnx, remoteEngine, canonicalOnnx, canonicalEngine };
  } catch (err) {
    const msg = err?.message || String(err);
    emitLog(`ONNX upload/build failed: ${msg}`);
    emitOnnxUploadProgress({ status: "error", percent: 0, text: "✕ Upload/build failed" });
    return { ok: false, error: msg };
  } finally {
    if (conn) conn.end();
    endManagedToolActivity();
  }
});

ipcMain.handle("save-app-config", async (_evt, appConfig) => {
  buildDeviceConfig(appConfig || {});
  const sanitized = withSshConfig({ ...(appConfig || {}) });
  sanitized.pass = "";
  delete sanitized.admin_password;
  delete sanitized.preview_camera_index;
  sanitized.local_path = portablePathFromAppDir(sanitized.local_path);
  sanitized.tag_map_path = portablePathFromAppDir(sanitized.tag_map_path);
  sanitized.onnx_model_path = portablePathFromAppDir(sanitized.onnx_model_path);
  if (sanitized.runtime_config_path) {
    sanitized.runtime_config_path = portablePathFromAppDir(sanitized.runtime_config_path);
  }
  await writeJson(APP_CONFIG_PATH, sanitized);
  return { ok: true };
});

ipcMain.handle("load-runtime-config", async (_evt, runtimePath) => {
  const data = await readJson(runtimePath, null);
  if (!data) throw new Error(`Failed to load ${runtimePath}`);
  return data;
});

ipcMain.handle("save-runtime-config", async (_evt, runtimePath, config) => {
  await writeJson(runtimePath, config);
  return { ok: true };
});

ipcMain.handle("sync-runtime-config-remote", async (_evt, settings, config) => {
  const conn = await connectSsh(settings);
  try {
    const remoteTarget = inferRemoteDeployFolder(settings.remote_path, settings.local_path);
    const remoteConfigPath = `${remoteTarget}/config/config.json`;
    const remoteParent = path.posix.dirname(remoteConfigPath);
    buildDeviceConfig(settings);
    await execChecked(conn, `mkdir -p -- ${shSingle(remoteParent)}`);
    const sftp = await getSftp(conn);
    const payload = Buffer.from(`${JSON.stringify(config, null, 2)}\n`, "utf8");
    await sftpAtomicWrite(conn, sftp, remoteConfigPath, payload);
    emitLog(`Runtime config synced to remote: ${remoteConfigPath}`);
    return { ok: true, remoteConfigPath };
  } catch (err) {
    const msg = err?.message || String(err);
    emitLog(`Runtime config remote sync failed: ${msg}`);
    return { ok: false, error: msg };
  } finally {
    conn.end();
  }
});

ipcMain.handle("set-jetson-power-limit", async (_evt, settings, watts) => {
  return setJetsonPowerLimit(settings, watts);
});

function startStaging(settings) {
  if (stagingInFlight) return { ok: false, error: "A staging operation is already running in this app" };
  if (fsSync.existsSync(JOB_PATH)) return { ok: false, error: "A saved deployment job needs reconnection. Use Refresh Releases first." };
  try {
    buildDeviceConfig(settings);
  } catch (error) {
    return { ok: false, error: error.message };
  }
  stagingInFlight = true;
  deployProject(settings)
    .catch((error) => emitLog(`Release staging failed: ${error.message}`))
    .finally(() => { stagingInFlight = false; });
  return { ok: true };
}

ipcMain.handle("deploy-start", async (_evt, settings) => startStaging(settings));
ipcMain.handle("build-main-start", async (_evt, settings) => startStaging(settings));

function jobEndpoint(settings) {
  const connection = withSshConfig(settings);
  return { host: connection.host, user: connection.user, port: connection.port,
    target: inferRemoteDeployFolder(settings.remote_path, settings.local_path) };
}

function jobPersistence(settings) {
  return {
    remember: async (job) => writeJson(JOB_PATH, { schema_version: 1, endpoint: jobEndpoint(settings), job }),
    finished: async () => fs.rm(JOB_PATH, { force: true })
  };
}

function activationDependencies(settings) {
  return { connect: connectSsh, execChecked, log: emitLog, progress: emitDeployProgress,
    user: settings ? withSshConfig(settings).user : DEFAULT_SSH.user, ...(settings ? jobPersistence(settings) : {}) };
}

async function resumeDeployment(settings) {
  // Refresh can supply a newly entered SSH password to an attachment already reconnecting.
  withSshConfig(settings);
  if (stagingInFlight) return { ok: true, attached: true };
  if (!fsSync.existsSync(JOB_PATH)) return { ok: true, attached: false };
  // Claim attachment before the first await, including concurrent Refresh/startup requests.
  stagingInFlight = true;
  let saved;
  try {
    saved = JSON.parse(await fs.readFile(JOB_PATH, "utf8"));
    if (saved.schema_version !== 1 || JSON.stringify(saved.endpoint) !== JSON.stringify(jobEndpoint(settings))) {
      stagingInFlight = false;
      return { ok: false, error: "Saved job belongs to another device or deployment folder. Restore those connection settings to reconnect." };
    }
    if (!["stage", "activate", "rollback", "initialize"].includes(saved.job?.kind)) throw new Error("Invalid saved deployment job");
  } catch (error) {
    stagingInFlight = false;
    throw error;
  }
  emitLog(`Reconnecting to saved ${saved.job.kind} job. No new job will be launched.`);
  emitDeployProgress({ status: "active", percent: 2, text: "Reconnecting to saved job..." });
  const target = saved.endpoint.target;
  const deps = { ...activationDependencies(settings), resume: saved.job };
  const work = saved.job.kind === "initialize" ?
    initializeJetson(settings, target, "", { ...deps, appDir: __dirname }) : saved.job.kind === "stage" ?
    stageProject({ ...settings, build_profile: saved.job.build_profile }, target, { ...deps, appDir: __dirname }) :
    operateRelease(settings, target, saved.job.kind, saved.job.release_id, deps);
  work.then(async (result) => {
    if (["stage", "initialize"].includes(saved.job.kind)) {
      const config = await readPersistedAppConfig();
      await writeJson(APP_CONFIG_PATH, { ...config, last_ready_release: result.release_id });
      emitReleaseState({ ready: [result.release_id] });
    } else {
      emitReleaseState(await releaseStatus(settings, target, activationDependencies()));
    }
  }).catch((error) => emitLog(`Saved deployment job: ${error.message}`))
    .finally(() => { stagingInFlight = false; });
  return { ok: true, attached: true };
}

ipcMain.handle("deployment-resume", async (_evt, settings) => resumeDeployment(settings));

ipcMain.handle("jetson-initialize", async (_evt, settings, administratorPassword) => {
  if (stagingInFlight || fsSync.existsSync(JOB_PATH) || monitorManager.conn || serviceGuard.activeCount) {
    return { ok: false, error: "Finish/reconnect the current operation and stop diagnostics before initializing" };
  }
  try {
    buildDeviceConfig(settings);
    withSshConfig(settings);
    if (typeof administratorPassword !== "string" || /[\r\n\0]/.test(administratorPassword)) throw new Error("Invalid administrator password");
  } catch (error) { return { ok: false, error: error.message }; }
  stagingInFlight = true;
  initializeJetson(settings, inferRemoteDeployFolder(settings.remote_path, settings.local_path), administratorPassword, {
    ...activationDependencies(settings), appDir: __dirname, getSftp, upload: sftpFastPut
  }).then(async (result) => {
    const config = await readPersistedAppConfig();
    await writeJson(APP_CONFIG_PATH, { ...config, last_ready_release: result.release_id });
    emitReleaseState(await releaseStatus({ ...settings, last_ready_release: result.release_id },
      inferRemoteDeployFolder(settings.remote_path, settings.local_path), activationDependencies()));
  }).catch((error) => emitLog(`Jetson initialization: ${error.message}`))
    .finally(() => { stagingInFlight = false; });
  administratorPassword = "";
  return { ok: true };
});

ipcMain.handle("release-status", async (_evt, settings) => {
  const resumed = await resumeDeployment(settings);
  if (!resumed.ok) throw new Error(resumed.error);
  if (resumed.attached) return { job_attached: true };
  const state = await releaseStatus(settings, inferRemoteDeployFolder(settings.remote_path, settings.local_path), activationDependencies());
  emitReleaseState(state);
  return state;
});

ipcMain.handle("release-operate", async (_evt, settings, action, release) => {
  if (stagingInFlight || fsSync.existsSync(JOB_PATH) || monitorManager.conn || serviceGuard.activeCount) {
    return { ok: false, error: "Wait for the current operation and stop diagnostics before switching releases" };
  }
  stagingInFlight = true;
  operateRelease(settings, inferRemoteDeployFolder(settings.remote_path, settings.local_path), action, release, activationDependencies(settings))
    .then(async () => {
      const state = await releaseStatus(settings, inferRemoteDeployFolder(settings.remote_path, settings.local_path), activationDependencies());
      emitReleaseState(state);
    })
    .catch((error) => emitLog(`Release operation failed: ${error.message}`))
    .finally(() => { stagingInFlight = false; });
  return { ok: true };
});

ipcMain.handle("monitor-start", async (_evt, settings) => {
  try {
    if (stagingInFlight || fsSync.existsSync(JOB_PATH)) throw new Error("Reconnect to the saved deployment job before attaching diagnostics");
    emitMonitorStartProgress({ status: "starting", percent: 10, text: "Attaching to production service..." });
    await serviceDiagnostics.start(settings, inferRemoteDeployFolder(settings.remote_path, settings.local_path));
    emitMonitorStartProgress({ status: "done", percent: 100, text: "Production diagnostics attached" });
    return { ok: true };
  } catch (error) {
    emitMonitorStartProgress({ status: "error", percent: 0, text: error.message });
    return { ok: false, error: error.message };
  }
});

ipcMain.handle("monitor-stop", async () => {
  await serviceDiagnostics.stop();
  return { ok: true };
});

ipcMain.handle("preview-start", async () => serviceDiagnostics.running ? { ok: true } :
  { ok: false, error: "Attach production diagnostics first. Activate an updated release if necessary." });
ipcMain.handle("preview-stop", async () => ({ ok: true }));

ipcMain.handle("list-remote-cameras", async (_evt, settings) => {
  const overview = await releaseStatus(settings, inferRemoteDeployFolder(settings.remote_path, settings.local_path), activationDependencies());
  if (overview.active?.kind === "release") return { ok: true, cameras: overview.configured_cameras || [] };
  const conn = await connectSsh(settings);
  try {
    const grouped = await execCommand(
      conn,
      "bash -lc \"if command -v v4l2-ctl >/dev/null 2>&1; then v4l2-ctl --list-devices 2>/dev/null | awk 'BEGIN{best=\"\"} /^[^[:space:]].*:$/{ if(best!=\"\"){ print best; best=\"\" }; next } /^[[:space:]]*\\/dev\\/video[0-9]+/{ gsub(/^[[:space:]]*/, \"\", $0); sub(/^.*video/, \"\", $0); n=$0+0; if(best==\"\" || n<best) best=n } END{ if(best!=\"\") print best }' | sort -n | uniq; fi\""
    );
    const groupedCameras = String(grouped.stdout || "")
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => Number(s))
      .filter((n) => Number.isFinite(n));
    if (groupedCameras.length > 0) {
      return { ok: true, cameras: groupedCameras };
    }
    const out = await execCommand(
      conn,
      "bash -lc \"ls -1 /dev/video* 2>/dev/null | sed -E 's#.*/video##' | grep -E '^[0-9]+$' | sort -n | uniq\""
    );
    const cameras = String(out.stdout || "")
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => Number(s))
      .filter((n) => Number.isFinite(n));
    return { ok: true, cameras };
  } catch (err) {
    return { ok: false, error: err?.message || String(err), cameras: [] };
  } finally {
    conn.end();
  }
});

ipcMain.handle("apply-monitor-preset", async (_evt, appConfig, preset) => {
  if (!["main_cam_0", "main_cam_0_2", "main_defaults"].includes(preset)) throw new Error("Unknown monitor preset");
  return {
    ...appConfig,
    monitor_start_cmd: "python3 scripts/run_vortex.py --debug-bridge",
    monitor_stop_cmd: "pkill -f orin_bridge || true",
    preview_remote_path: "/tmp/vortex_bridge_frame.jpg",
    preview_state_path: "/tmp/vortex_bridge_state.json"
  };
});
