const { app, BrowserWindow, ipcMain, dialog, Menu } = require("electron");
const path = require("path");
const fs = require("fs/promises");
const fsSync = require("fs");
const { Client } = require("ssh2");

const APP_CONFIG_PATH = path.resolve(__dirname, "..", "config", "deployment_tool_config.json");
const LEGACY_APP_CONFIG_PATH = path.join(__dirname, "vortex_config.json");
const LEGACY_USERDATA_CONFIG_PATH = path.join(app.getPath("userData"), "vortex_config.json");
const DEFAULT_RUNTIME_CONFIG_PATH = path.resolve(__dirname, "..", "config", "config.json");
const FIXED_SSH = Object.freeze({
  port: "22",
  user: "vortex",
  pass: "redstorm"
});
const serviceGuard = {
  activeCount: 0,
  lastSettings: null
};
let quitCleanupInProgress = false;

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
    port: FIXED_SSH.port,
    user: FIXED_SSH.user,
    pass: FIXED_SSH.pass,
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
    jetson_max_watts: 15
  };
}

function withFixedSshConfig(cfg) {
  return {
    ...(cfg || {}),
    port: FIXED_SSH.port,
    user: FIXED_SSH.user,
    pass: FIXED_SSH.pass
  };
}

function normalizeServiceName(name) {
  return String(name || "").trim();
}

function hasManagedService(settings) {
  return !!normalizeServiceName(settings?.startup_service_name);
}

async function stopManagedStartupService(settings, reason = "operation") {
  if (!hasManagedService(settings)) return;
  const cfg = withFixedSshConfig(settings);
  const serviceName = normalizeServiceName(cfg.startup_service_name);
  const conn = await connectSsh(cfg);
  try {
    const stopSvc = await execCommand(conn, `sudo systemctl stop ${serviceName}`);
    if (stopSvc.code !== 0) {
      emitLog(`Service stop failed (${reason}): ${stopSvc.stderr || stopSvc.stdout}`);
    } else {
      emitLog(`Service stopped (${reason}): ${serviceName}`);
    }
  } finally {
    conn.end();
  }
}

async function startManagedStartupService(settings, reason = "exit") {
  if (!hasManagedService(settings)) return;
  const cfg = withFixedSshConfig(settings);
  const serviceName = normalizeServiceName(cfg.startup_service_name);
  const conn = await connectSsh(cfg);
  try {
    const startSvc = await execCommand(conn, `sudo systemctl start ${serviceName}`);
    if (startSvc.code !== 0) {
      emitLog(`Service start failed (${reason}): ${startSvc.stderr || startSvc.stdout}`);
    } else {
      emitLog(`Service started (${reason}): ${serviceName}`);
    }
  } finally {
    conn.end();
  }
}

async function beginManagedToolActivity(settings, label) {
  if (!hasManagedService(settings)) return;
  serviceGuard.lastSettings = withFixedSshConfig(settings);
  const wasIdle = serviceGuard.activeCount === 0;
  serviceGuard.activeCount += 1;
  if (wasIdle) {
    await stopManagedStartupService(serviceGuard.lastSettings, label);
  }
}

function endManagedToolActivity() {
  if (serviceGuard.activeCount > 0) {
    serviceGuard.activeCount -= 1;
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
  await fs.writeFile(filePath, JSON.stringify(obj, null, 2), "utf8");
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

function stripOuterQuotes(s) {
  const t = String(s || "").trim();
  if ((t.startsWith("'") && t.endsWith("'")) || (t.startsWith('"') && t.endsWith('"'))) {
    return t.slice(1, -1);
  }
  return t;
}

function shSingle(s) {
  return `'${String(s).replace(/'/g, `'\\''`)}'`;
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

async function setJetsonPowerLimit(settings, requestedWatts) {
  const cfg = withFixedSshConfig(settings);
  const conn = await connectSsh(settings);
  try {
    const req = Number(requestedWatts);
    if (!Number.isFinite(req) || req <= 0) throw new Error(`Invalid wattage: ${requestedWatts}`);

    const cfgPathRes = await execCommand(
      conn,
      "bash -lc \"for f in /etc/nvpmodel.conf /etc/nvpmodel/*.conf /etc/nvpmodel_*.conf; do [ -f \\\"$f\\\" ] && echo \\\"$f\\\"; done | head -n 1\""
    );
    const cfgPath = String(cfgPathRes.stdout || "").trim() || "/etc/nvpmodel.conf";
    const cfg = await execCommand(conn, `bash -lc "cat '${cfgPath}' 2>/dev/null || true"`);
    let modes = parseNvpmodelModes(cfg.stdout || "");
    if (modes.length === 0) {
      const q = await execCommand(conn, "bash -lc \"nvpmodel -q --verbose 2>/dev/null || nvpmodel -q 2>/dev/null || true\"");
      modes = parseNvpmodelModesFromQuery(`${q.stdout || ""}\n${q.stderr || ""}`);
    }
    if (modes.length === 0) throw new Error(`Could not parse nvpmodel modes (config path: ${cfgPath})`);

    const numeric = modes.filter((x) => Number.isFinite(x.watts));
    let chosen = numeric[0] || modes[0];
    for (const mode of numeric) {
      if (mode.watts <= req) chosen = mode;
    }
    const pw = shSingle(cfg.pass || "");
    const applyLog = `/tmp/vortex_nvp_apply_${Date.now()}.log`;
    const apply = await execCommand(
      conn,
      `bash -lc "echo ${pw} | sudo -S -p '' nvpmodel -m ${chosen.id} > '${applyLog}' 2>&1; code=\\$?; cat '${applyLog}'; exit \\$code"`
    );
    const out = String((apply.stdout || "") + "\n" + (apply.stderr || "")).trim();
    const rebootRequired = /reboot required/i.test(out) || /DO YOU WANT TO REBOOT NOW/i.test(out);
    if (apply.code !== 0) {
      if (rebootRequired) {
        return { ok: false, rebootRequired: true, error: "Reboot required for this power mode. Reboot Jetson, then apply again." };
      }
      throw new Error(out || `nvpmodel apply failed (exit=${apply.code})`);
    }
    emitLog(`Jetson power mode set: requested ${req}W, applied mode ${chosen.id} (${chosen.watts}W, ${chosen.name})`);
    if (rebootRequired) emitLog("Jetson reported reboot required for this power mode change.");
    return {
      ok: true,
      requestedWatts: req,
      modeId: chosen.id,
      modeWatts: chosen.watts,
      modeName: chosen.name,
      rebootRequired
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
    const cfg = withFixedSshConfig(settings);
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

function getSftp(conn) {
  return new Promise((resolve, reject) => {
    conn.sftp((err, sftp) => {
      if (err) reject(err);
      else resolve(sftp);
    });
  });
}

function sftpMkdir(sftp, remotePath) {
  return new Promise((resolve) => {
    sftp.mkdir(remotePath, { mode: 0o755 }, () => resolve());
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

async function walkFiles(rootDir) {
  const out = [];
  async function walk(current) {
    const entries = await fs.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(current, entry.name);
      if (entry.name === "target" || entry.name === "node_modules" || entry.name === ".git") continue;
      if (entry.isDirectory()) await walk(full);
      else out.push(full);
    }
  }
  await walk(rootDir);
  return out;
}

async function deployProject(settings) {
  const cfg = withFixedSshConfig(settings);
  const conn = await connectSsh(settings);
  try {
    emitLog(`Connected to ${cfg.host}:${cfg.port}`);
    const remoteTarget = inferRemoteDeployFolder(settings.remote_path, settings.local_path);
    const normalizedTarget = String(remoteTarget).replace(/\/+$/, "");
    if (!normalizedTarget || normalizedTarget === "/" || normalizedTarget.split("/").length < 3) {
      throw new Error(`Refusing to delete unsafe remote path: ${remoteTarget}`);
    }

    emitLog(`Clearing remote folder: ${remoteTarget}`);
    await execCommand(conn, `rm -rf '${remoteTarget}'`);
    await execCommand(conn, `mkdir -p '${remoteTarget}'`);
    emitLog(`Remote folder: ${remoteTarget}`);

    const sftp = await getSftp(conn);
    const files = await walkFiles(settings.local_path);
    const total = files.length || 1;
    emitDeployProgress({ percent: 0, current: 0, total, status: "starting" });

    const createdDirs = new Set([toPosix(remoteTarget)]);
    let i = 0;
    for (const file of files) {
      const rel = path.relative(settings.local_path, file);
      const relPosix = toPosix(rel);
      const remoteFile = toPosix(path.posix.join(remoteTarget, relPosix));
      const parent = path.posix.dirname(remoteFile);
      if (!createdDirs.has(parent)) {
        await execCommand(conn, `mkdir -p '${parent}'`);
        createdDirs.add(parent);
      }
      emitLog(`Upload: ${relPosix}`);
      await sftpFastPut(sftp, file, remoteFile);
      i += 1;
      emitDeployProgress({
        percent: Math.round((i / total) * 100),
        current: i,
        total,
        status: "uploading"
      });
    }
    emitLog("Deployment finished successfully.");
    emitDeployProgress({ percent: 100, current: total, total, status: "done" });
  } finally {
    conn.end();
  }
}

async function buildMainProgram(settings) {
  const conn = await connectSsh(settings);
  try {
    emitMainBuildProgress({ status: "starting", percent: 0, text: "Connecting..." });
    const remoteTarget = inferRemoteDeployFolder(settings.remote_path, settings.local_path);
    const hasCargo = await execCommand(conn, `test -f '${remoteTarget}/Cargo.toml' && echo OK || echo NO`);
    if (!String(hasCargo.stdout || "").includes("OK")) {
      throw new Error(`No Cargo.toml found in ${remoteTarget}. Deploy first.`);
    }

    let total = 1;
    try {
      const meta = await execCommand(
        conn,
        `bash -lc "cd '${remoteTarget}' && cargo metadata --format-version 1 --locked 2>/dev/null"`
      );
      if (meta.code === 0) {
        const parsed = JSON.parse(String(meta.stdout || "{}"));
        const count = Array.isArray(parsed?.resolve?.nodes) ? parsed.resolve.nodes.length : 1;
        total = Math.max(1, Number(count) || 1);
      }
    } catch (_err) {}
    emitLog(`[0/${total}] Starting remote build`);
    emitMainBuildProgress({ status: "active", percent: 0, text: `[0/${total}] Starting` });

    const cmd = `bash -lc "cd '${remoteTarget}' && cargo build -vv --release --bin dumapril-taglocalization 2>&1"`;
    const channel = await execCommand(conn, cmd, { stream: true });
    await new Promise((resolve, reject) => {
      let pending = "";
      const seen = new Set();
      const onLine = (raw) => {
        const line = stripAnsi(raw).trim();
        if (!line) return;
        const bracket = line.match(/\[([0-9]+)\/([0-9]+)\]/);
        if (bracket) {
          const cur = Number(bracket[1]);
          const ttl = Number(bracket[2]);
          if (Number.isFinite(cur) && Number.isFinite(ttl) && ttl > 0) {
            const pct = Math.max(0, Math.min(99, Math.round((cur * 100) / ttl)));
            emitLog(line);
            emitMainBuildProgress({ status: "active", percent: pct, text: bracket[0] });
            return;
          }
        }
        const step = line.match(/^(Compiling|Fresh)\s+([^\s]+)\s+/);
        if (step) {
          const action = String(step[1]);
          const crate = String(step[2]);
          if (crate) seen.add(crate);
          const cur = seen.size;
          const synthetic = `[${cur}/${total}] ${action} ${crate}`;
          emitLog(synthetic);
          const pct = Math.max(0, Math.min(99, Math.round((cur * 100) / Math.max(1, total))));
          emitMainBuildProgress({ status: "active", percent: pct, text: synthetic });
          return;
        }
        if (/^Finished\b/i.test(line)) {
          emitLog(`[${total}/${total}] ${line}`);
          emitMainBuildProgress({ status: "active", percent: 99, text: `[${total}/${total}] Finished` });
        }
      };

      const onData = (buf) => {
        pending += buf.toString("utf8");
        const lines = pending.split(/\r?\n/);
        pending = lines.pop() || "";
        for (const ln of lines) onLine(ln);
      };
      channel.on("data", onData);
      channel.stderr.on("data", onData);
      channel.on("error", reject);
      channel.on("close", (code) => {
        if (pending.trim()) onLine(pending.trim());
        if (code === 0) {
          emitMainBuildProgress({ status: "done", percent: 100, text: "✓ Main build complete" });
          resolve();
        } else {
          emitMainBuildProgress({ status: "error", percent: 0, text: "✕ Main build failed" });
          reject(new Error(`Remote build failed with exit code ${code}`));
        }
      });
    });
  } finally {
    conn.end();
  }
}

async function commandExistsInDir(conn, dir, binaryPath) {
  const probe = await execCommand(
    conn,
    `test -x '${dir}/${binaryPath}' && echo OK || echo NO`
  );
  return String(probe.stdout || "").includes("OK");
}

function splitMonitorCommand(monitorStartCmd) {
  const cmd = String(monitorStartCmd || "").trim();
  const m = cmd.match(/^cd\s+(['"])(.*?)\1\s*&&\s*(.+)$/);
  if (!m) return { cwd: null, tail: cmd };
  return { cwd: m[2], tail: m[3] };
}

function withBridgeEnv(cmdTail, binary) {
  const tail = String(cmdTail || "").trim();
  if (binary !== "orin_bridge") return tail;
  const envParts = [];
  if (!/\bYOLO_ENGINE=/.test(tail)) envParts.push(`YOLO_ENGINE='models/rockpaperscizzors.engine'`);
  if (envParts.length === 0) return tail;
  return `${envParts.join(" ")} ${tail}`;
}

async function resolveMonitorCommand(conn, settings) {
  const cfg = withFixedSshConfig(settings);
  const raw = String(settings.monitor_start_cmd || "").trim();
  const { cwd, tail } = splitMonitorCommand(raw);
  const invoked = String(tail || raw);
  const binaryMatch = invoked.match(/\.\/([^\s]+)/);
  const binaryRelPath = binaryMatch ? binaryMatch[1] : "dumapril-taglocalization";
  const binary = path.posix.basename(binaryRelPath);
  const likelyTail = withBridgeEnv(tail || raw, binary);
  const hasBinaryRef = likelyTail.includes(binary);

  if (!hasBinaryRef) return raw;

  async function ensureYoloEngineInDir(dir) {
    if (binary !== "orin_bridge") return;
    try {
      const hasEngine = await execCommand(
        conn,
        `test -f '${dir}/models/rockpaperscizzors.engine' && echo OK || echo NO`
      );
      if (String(hasEngine.stdout || "").includes("OK")) return;

      const hasOnnx = await execCommand(
        conn,
        `test -f '${dir}/models/rockpaperscizzors.onnx' && echo OK || echo NO`
      );
      if (!String(hasOnnx.stdout || "").includes("OK")) {
        emitLog("YOLO engine missing and ONNX source not found; object detection disabled.");
        return;
      }

      const hasTrtExec = await execCommand(conn, "bash -lc \"command -v trtexec >/dev/null 2>&1 && echo OK || echo NO\"");
      if (!String(hasTrtExec.stdout || "").includes("OK")) {
        emitLog("YOLO engine missing and trtexec not found on remote; object detection disabled.");
        return;
      }

      const trtLog = `/tmp/vortex_trtexec_${Date.now()}.log`;
      emitLog("Building TensorRT engine from models/rockpaperscizzors.onnx...");
      const build = await execCommand(
        conn,
        `bash -lc "cd '${dir}' && trtexec --onnx=models/rockpaperscizzors.onnx --saveEngine=models/rockpaperscizzors.engine --fp16 --workspace=2048 > '${trtLog}' 2>&1; code=\\$?; tail -n 120 '${trtLog}'; exit \\$code"`
      );
      const out = String((build.stdout || "") + "\n" + (build.stderr || "")).trim();
      if (out) emitLog(tailForLog(out));
      if (build.code === 0) emitLog("TensorRT engine created: models/rockpaperscizzors.engine");
      else emitLog("TensorRT engine build failed; object detection may remain disabled.");
    } catch (_err) {
      emitLog("Failed to prepare TensorRT engine automatically.");
    }
  }

  const derived = inferRemoteDeployFolder(settings.remote_path, settings.local_path);
  const candidates = [
    cwd,
    derived,
    settings.remote_path,
    `/home/${cfg.user}`,
    `/home/${cfg.user}/deployments`,
    `/home/${cfg.user}/deployments/Vortex`
  ]
    .map((v) => stripOuterQuotes(v))
    .filter(Boolean)
    .filter((v, i, arr) => arr.indexOf(v) === i);

  for (const dir of candidates) {
    try {
      if (await commandExistsInDir(conn, dir, binaryRelPath)) {
        await ensureYoloEngineInDir(dir);
        const resolved = `cd '${dir}' && ${likelyTail}`;
        emitLog(`Resolved monitor dir: ${dir}`);
        return resolved;
      }
    } catch (_err) {
      // try next candidate
    }
  }

  async function buildBridgeInDir(dir, cargoCmd = "cargo") {
    const logPath = `/tmp/vortex_orin_bridge_build_${Date.now()}.log`;
    const variants = [
      { label: "GPU+TensorRT", features: "gpu,tensorrt" },
      { label: "TensorRT", features: "tensorrt" },
      { label: "GPU", features: "gpu" },
      { label: "CPU", features: "" }
    ];

    const runBuild = async (features, pre = "") => {
      const featureArgs = features ? ` --features ${features}` : "";
      return execCommand(
        conn,
        `bash -lc "cd '${dir}' && ${pre}${cargoCmd} build --release --bin orin_bridge${featureArgs} > '${logPath}' 2>&1; code=\\$?; tail -n 180 '${logPath}'; exit \\$code"`
      );
    };

    const isLockErr = (text) =>
      text.includes("lock file version 4") ||
      text.includes("failed to parse lock file") ||
      text.includes("feature `edition2024` is required");

    let sawLockErr = false;
    let last = null;

    for (const v of variants) {
      emitLog(`Trying orin_bridge build variant: ${v.label}${v.features ? ` (${v.features})` : ""}`);
      const out = await runBuild(v.features);
      last = out;
      if (out.code === 0) {
        emitLog(`Built orin_bridge variant: ${v.label}`);
        return { ok: true, output: out };
      }
      const text = String((out.stdout || "") + "\n" + (out.stderr || "")).trim();
      sawLockErr = sawLockErr || isLockErr(text);
      if (text) emitLog(tailForLog(text));
    }

    if (!sawLockErr) return { ok: false, output: last };

    emitLog(`Lockfile incompatible on remote in ${dir}; retrying build variants without Cargo.lock...`);
    for (const v of variants) {
      emitLog(`Retrying orin_bridge build variant: ${v.label}${v.features ? ` (${v.features})` : ""}`);
      const out = await runBuild(v.features, "rm -f Cargo.lock && ");
      last = out;
      if (out.code === 0) {
        emitLog(`Built orin_bridge variant after lock reset: ${v.label}`);
        return { ok: true, output: out };
      }
      const text = String((out.stdout || "") + "\n" + (out.stderr || "")).trim();
      if (text) emitLog(tailForLog(text));
    }
    return { ok: false, output: last };
  }

  async function ensureBridgeBuildDeps() {
    const pw = shSingle(cfg.pass || "");
    emitLog("Ensuring remote build dependencies for orin_bridge...");
    const depInstall = await execCommand(
      conn,
      `bash -lc "echo ${pw} | sudo -S -p '' apt-get update && echo ${pw} | sudo -S -p '' apt-get install -y build-essential pkg-config clang cmake libssl-dev libclang-dev libturbojpeg0-dev libjpeg-dev zlib1g-dev v4l-utils libv4l-dev"`
    );
    if (depInstall.code !== 0) {
      const out = String((depInstall.stdout || "") + "\n" + (depInstall.stderr || "")).trim();
      if (out) emitLog(tailForLog(out));
      throw new Error("failed to install remote build dependencies");
    }
  }

  let sawMissingCargo = false;
  let sawOldCargo = false;
  for (const dir of candidates) {
    try {
      const hasCargo = await execCommand(conn, `test -f '${dir}/Cargo.toml' && echo OK || echo NO`);
      if (!String(hasCargo.stdout || "").includes("OK")) continue;
      emitLog(`Building orin bridge on remote: ${dir}`);
      const build = await buildBridgeInDir(dir);
      if (!build.ok) {
        emitLog(`Remote build failed in ${dir}`);
        const out = String((build.output.stdout || "") + "\n" + (build.output.stderr || "")).trim();
        if (out) {
          const clipped = tailForLog(out);
          emitLog(clipped);
          emitLog(`Remote build exit code: ${build.output.code}`);
          if (clipped.includes("cargo: command not found")) {
            sawMissingCargo = true;
          }
          if (
            clipped.includes("feature `edition2024` is required") ||
            clipped.includes("lock file version 4") ||
            clipped.includes("-Znext-lockfile-bump")
          ) {
            sawOldCargo = true;
          }
        }
        continue;
      }
      if (await commandExistsInDir(conn, dir, `target/release/${binary}`)) {
        await ensureYoloEngineInDir(dir);
        const suffix = likelyTail.includes(binary)
          ? likelyTail.split(binary).slice(1).join(binary)
          : "";
        const cmdTail = withBridgeEnv(`./target/release/${binary}${suffix}`, binary);
        const resolved = `cd '${dir}' && ${cmdTail}`;
        emitLog(`Resolved monitor dir after build: ${dir}`);
        return resolved;
      }
    } catch (_err) {
      // continue
    }
  }

  try {
    const scan = await execCommand(
      conn,
      `find '/home/${cfg.user}' -maxdepth 6 -type f -name '${binary}' 2>/dev/null | head -n 1`
    );
    const found = String(scan.stdout || "").trim();
    if (found) {
      const dir = path.posix.dirname(found);
      const suffix = likelyTail.includes(binary)
        ? likelyTail.split(binary).slice(1).join(binary)
        : "";
      const normalized = dir.replace(/\/+$/, "");
      let runDir = normalized;
      let runCmd = `./${binary}${suffix}`;

      if (normalized.endsWith("/target/release")) {
        runDir = normalized.replace(/\/target\/release$/, "");
        runCmd = `./target/release/${binary}${suffix}`;
      }

      await ensureYoloEngineInDir(runDir);
      const resolved = `cd '${runDir}' && ${withBridgeEnv(runCmd, binary)}`;
      emitLog(`Resolved monitor dir by scan: ${runDir}`);
      return resolved;
    }
  } catch (_err) {
    // fall through
  }

  if (sawMissingCargo) {
    emitLog("Cargo not found. Attempting remote install via apt...");
    try {
      const pw = shSingle(cfg.pass || "");
      const install = await execCommand(
        conn,
        `bash -lc "echo ${pw} | sudo -S -p '' apt-get update && echo ${pw} | sudo -S -p '' apt-get install -y cargo"`
      );
      if (install.code !== 0) {
        const out = String((install.stdout || "") + "\n" + (install.stderr || "")).trim();
        if (out) emitLog(tailForLog(out));
        throw new Error("automatic cargo install failed");
      }
      emitLog("Cargo install succeeded. Retrying bridge build...");
      await ensureBridgeBuildDeps();

      for (const dir of candidates) {
        try {
          const hasCargo = await execCommand(conn, `test -f '${dir}/Cargo.toml' && echo OK || echo NO`);
          if (!String(hasCargo.stdout || "").includes("OK")) continue;
          const build = await buildBridgeInDir(dir);
          if (!build.ok) {
            const out = String((build.output.stdout || "") + "\n" + (build.output.stderr || "")).trim();
            if (out) emitLog(tailForLog(out));
            emitLog(`Remote build exit code: ${build.output.code}`);
            continue;
          }
          if (await commandExistsInDir(conn, dir, `target/release/${binary}`)) {
            await ensureYoloEngineInDir(dir);
            const suffix = likelyTail.includes(binary)
              ? likelyTail.split(binary).slice(1).join(binary)
              : "";
            const cmdTail = withBridgeEnv(`./target/release/${binary}${suffix}`, binary);
            const resolved = `cd '${dir}' && ${cmdTail}`;
            emitLog(`Resolved monitor dir after auto-install: ${dir}`);
            return resolved;
          }
        } catch (_err) {
          // continue
        }
      }
    } catch (_err) {
      throw new Error(
        "Cargo is not installed on remote host and auto-install failed. Install Rust/Cargo on Orin or deploy a prebuilt ./target/release/orin_bridge."
      );
    }

    throw new Error(
      "Cargo was installed, but orin_bridge build still failed. Check remote build logs."
    );
  }

  if (sawOldCargo) {
    emitLog("Remote Cargo is too old. Attempting rustup stable install...");
    try {
      const pw = shSingle(cfg.pass || "");
      // Ensure curl exists for rustup bootstrap
      await execCommand(
        conn,
        `bash -lc "command -v curl >/dev/null 2>&1 || (echo ${pw} | sudo -S -p '' apt-get update && echo ${pw} | sudo -S -p '' apt-get install -y curl)"`
      );
      const rustup = await execCommand(
        conn,
        `bash -lc "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable"`
      );
      if (rustup.code !== 0) {
        const out = String((rustup.stdout || "") + "\n" + (rustup.stderr || "")).trim();
        if (out) emitLog(tailForLog(out));
        throw new Error("rustup install failed");
      }

      const cargoCmd = "$HOME/.cargo/bin/cargo";
      await ensureBridgeBuildDeps();
      for (const dir of candidates) {
        try {
          const hasCargo = await execCommand(conn, `test -f '${dir}/Cargo.toml' && echo OK || echo NO`);
          if (!String(hasCargo.stdout || "").includes("OK")) continue;
          const build = await buildBridgeInDir(dir, cargoCmd);
          if (!build.ok) {
            const out = String((build.output.stdout || "") + "\n" + (build.output.stderr || "")).trim();
            if (out) emitLog(tailForLog(out));
            emitLog(`Remote build exit code: ${build.output.code}`);
            continue;
          }
          if (await commandExistsInDir(conn, dir, `target/release/${binary}`)) {
            await ensureYoloEngineInDir(dir);
            const suffix = likelyTail.includes(binary)
              ? likelyTail.split(binary).slice(1).join(binary)
              : "";
            const cmdTail = withBridgeEnv(`./target/release/${binary}${suffix}`, binary);
            const resolved = `cd '${dir}' && ${cmdTail}`;
            emitLog(`Resolved monitor dir after rustup upgrade: ${dir}`);
            return resolved;
          }
        } catch (_err) {
          // continue
        }
      }
    } catch (_err) {
      throw new Error(
        "Remote Cargo is too old and rustup auto-upgrade failed. Install a newer Rust toolchain on Orin or deploy a prebuilt ./target/release/orin_bridge."
      );
    }

    throw new Error("Rust toolchain upgraded, but orin_bridge build still failed.");
  }
  throw new Error("No monitor binary found on remote host. Deploy/build first.");
}

const monitorManager = {
  conn: null,
  channel: null,
  stopping: false,
  settings: null,
  activityHeld: false,
  async start(settings) {
    if (this.conn) throw new Error("Monitor already running");
    this.settings = settings;
    this.stopping = false;
    try {
      emitMonitorStartProgress({ status: "starting", percent: 10, text: "Connecting..." });
      this.conn = await connectSsh(settings);
      await syncTagMapRemote(this.conn, settings);

      emitMonitorStartProgress({ status: "starting", percent: 55, text: "Resolving monitor..." });
      const resolvedStart = await resolveMonitorCommand(this.conn, settings);
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
        if (pending.trim()) {
          emitMonitorLine(pending.trim());
          if (!isDetectionLine(pending.trim())) emitLog(pending.trim());
        }
        if (this.activityHeld) {
          endManagedToolActivity();
          this.activityHeld = false;
        }
        this.cleanup();
        emitMonitorState(false);
        emitMonitorStartProgress({ status: "hidden", percent: 0, text: "" });
        if (!this.stopping) {
          const exitCode = typeof code === "number" ? code : "unknown";
          emitLog(`Monitor stream ended (exit=${exitCode}${signal ? `, signal=${signal}` : ""}).`);
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
  try {
    const localPath = path.resolve(String(settings.tag_map_path || ""));
    if (!localPath || !fsSync.existsSync(localPath)) {
      emitLog("Tag map not found locally; using remote/default map path if present.");
      return { ok: false, reason: "missing-local-map" };
    }
    const sftp = await getSftp(conn);
    const targets = [
      inferRemoteDeployFolder(settings.remote_path, settings.local_path),
      settings.remote_path,
      `/home/${withFixedSshConfig(settings).user}/deployments/Vortex`
    ]
      .map((p) => String(p || "").replace(/\/+$/, ""))
      .filter(Boolean)
      .filter((v, i, arr) => arr.indexOf(v) === i);
    let wrote = 0;
    for (const t of targets) {
      const remotePath = `${t}/config/apriltag_map.json`;
      const remoteParent = path.posix.dirname(remotePath);
      await execCommand(conn, `mkdir -p '${remoteParent}'`);
      await sftpFastPut(sftp, localPath, remotePath);
      emitLog(`Tag map synced to remote: ${remotePath}`);
      wrote += 1;
    }
    return { ok: wrote > 0, remotePath: `${targets[0]}/config/apriltag_map.json` };
  } catch (err) {
    emitLog(`Tag map sync failed: ${err?.message || String(err)}`);
    return { ok: false, error: err?.message || String(err) };
  }
}

const previewManager = {
  running: false,
  async start(settings) {
    if (this.running) return;
    this.running = true;
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
        emitPreviewFrame(`data:${mime};base64,${bytes.toString("base64")}`);
        try {
          const stateBytes = await sftpReadFile(
            sftp,
            settings.preview_state_path || "/tmp/vortex_bridge_state.json"
          );
          const parsed = JSON.parse(stateBytes.toString("utf8"));
          emitBridgeState(parsed);
        } catch (_stateErr) {
          // state file may not exist yet, ignore and keep preview alive
        }
      } catch (err) {
        const msg = err?.message || String(err);
        emitLog(`Preview error: ${msg}`);
        if (conn) {
          try { conn.end(); } catch (_e) {}
          conn = null;
          sftp = null;
        }
        if (msg.includes("All configured authentication methods failed")) {
          emitLog("Authentication failed. Fixed credentials are vortex/redstorm on port 22.");
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
  }
};

ipcMain.handle("bootstrap", async () => {
  const appCfg = withFixedSshConfig({ ...defaultAppConfig(), ...(await readPersistedAppConfig()) });
  delete appCfg.preview_camera_index;
  appCfg.local_path = resolvePathFromAppDir(appCfg.local_path, path.resolve(__dirname, ".."));
  appCfg.tag_map_path = resolvePathFromAppDir(
    appCfg.tag_map_path,
    resolvePathFromAppDir(path.join("..", "config", "2026-rebuilt-welded.json"), "")
  );
  if (appCfg.onnx_model_path) {
    appCfg.onnx_model_path = resolvePathFromAppDir(appCfg.onnx_model_path, "");
  }
  const runtimeConfigPath = path.resolve(__dirname, appCfg.runtime_config_path || DEFAULT_RUNTIME_CONFIG_PATH);
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
  const sanitized = withFixedSshConfig({ ...(appConfig || {}) });
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
    await execCommand(conn, `mkdir -p '${remoteParent}'`);
    const sftp = await getSftp(conn);
    const payload = Buffer.from(`${JSON.stringify(config, null, 2)}\n`, "utf8");
    await sftpWriteFile(sftp, remoteConfigPath, payload);
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

ipcMain.handle("deploy-start", async (_evt, settings) => {
  try {
    await beginManagedToolActivity(settings, "deploy");
  } catch (err) {
    const msg = err?.message || String(err);
    emitLog(`Service pre-stop failed: ${msg}`);
    return { ok: false, error: msg };
  }
  deployProject(settings)
    .then(() => emitLog("Deploy complete."))
    .catch((err) => {
      emitLog(`Deploy failed: ${err.message}`);
      emitDeployProgress({ percent: 0, current: 0, total: 0, status: "error" });
    })
    .finally(() => {
      endManagedToolActivity();
    });
  return { ok: true };
});

ipcMain.handle("build-main-start", async (_evt, settings) => {
  try {
    await beginManagedToolActivity(settings, "main-build");
  } catch (err) {
    const msg = err?.message || String(err);
    emitLog(`Service pre-stop failed: ${msg}`);
    return { ok: false, error: msg };
  }
  buildMainProgram(settings)
    .then(() => emitLog("Main build complete."))
    .catch((err) => {
      emitLog(`Main build failed: ${err.message}`);
      emitMainBuildProgress({ status: "error", percent: 0, text: "✕ Main build failed" });
    })
    .finally(() => {
      endManagedToolActivity();
    });
  return { ok: true };
});

ipcMain.handle("monitor-start", async (_evt, settings) => {
  emitMonitorStartProgress({ status: "starting", percent: 5, text: "Starting..." });
  let activityStarted = false;
  try {
    await beginManagedToolActivity(settings, "monitor");
    activityStarted = true;
    await monitorManager.start(settings);
    monitorManager.activityHeld = true;
    return { ok: true };
  } catch (err) {
    if (activityStarted && monitorManager.activityHeld) {
      endManagedToolActivity();
      monitorManager.activityHeld = false;
    } else if (activityStarted) {
      endManagedToolActivity();
    }
    const msg = err?.message || String(err);
    emitLog(`Monitor start failed: ${msg}`);
    if (msg.includes("All configured authentication methods failed")) {
      emitLog("Authentication failed. Fixed credentials are vortex/redstorm on port 22.");
    }
    emitMonitorStartProgress({ status: "error", percent: 0, text: "✕ Start failed" });
    return { ok: false, error: msg };
  }
});

ipcMain.handle("monitor-stop", async () => {
  await monitorManager.stop();
  return { ok: true };
});

ipcMain.handle("preview-start", async (_evt, settings) => {
  try {
    previewManager.start(settings);
    return { ok: true };
  } catch (err) {
    const msg = err?.message || String(err);
    emitLog(`Preview start failed: ${msg}`);
    return { ok: false, error: msg };
  }
});

ipcMain.handle("preview-stop", async () => {
  previewManager.stop();
  return { ok: true };
});

ipcMain.handle("list-remote-cameras", async (_evt, settings) => {
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
  const remoteFolder = inferRemoteDeployFolder(appConfig.remote_path, appConfig.local_path);
  const out = { ...appConfig };
  const cam = Number(appConfig.preview_camera_index || 0);
  if (preset === "main_cam_0") {
    out.monitor_start_cmd = `cd '${remoteFolder}' && VORTEX_BRIDGE_CAMERA=${cam} VORTEX_BRIDGE_FRAME='/tmp/vortex_bridge_frame.jpg' VORTEX_BRIDGE_STATE='/tmp/vortex_bridge_state.json' ./target/release/orin_bridge 2>&1`;
  } else if (preset === "main_cam_0_2") {
    out.monitor_start_cmd = `cd '${remoteFolder}' && VORTEX_BRIDGE_CAMERA=${cam} VORTEX_BRIDGE_FRAME='/tmp/vortex_bridge_frame.jpg' VORTEX_BRIDGE_STATE='/tmp/vortex_bridge_state.json' ./target/release/orin_bridge 2>&1`;
  } else if (preset === "main_defaults") {
    out.monitor_start_cmd = `cd '${remoteFolder}' && VORTEX_BRIDGE_CAMERA=${cam} VORTEX_BRIDGE_FRAME='/tmp/vortex_bridge_frame.jpg' VORTEX_BRIDGE_STATE='/tmp/vortex_bridge_state.json' ./target/release/orin_bridge 2>&1`;
  }
  out.monitor_stop_cmd = "pkill -f orin_bridge || true";
  out.preview_remote_path = "/tmp/vortex_bridge_frame.jpg";
  out.preview_state_path = "/tmp/vortex_bridge_state.json";
  return out;
});
