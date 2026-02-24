const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs/promises");
const fsSync = require("fs");
const { Client } = require("ssh2");

const APP_CONFIG_PATH = path.join(__dirname, "vortex_config.json");
const DEFAULT_RUNTIME_CONFIG_PATH = path.resolve(__dirname, "..", "config", "config.json");

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
    port: "22",
    user: "jetson",
    pass: "",
    remote_path: "/home/jetson/deployments",
    local_path: path.resolve(__dirname, ".."),
    monitor_start_cmd: "",
    monitor_stop_cmd: "pkill -f orin_bridge || true",
    startup_service_name: "",
    preview_remote_path: "/tmp/vortex_preview.jpg",
    preview_state_path: "/tmp/vortex_bridge_state.json",
    preview_capture_cmd: ""
  };
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

function connectSsh(settings) {
  return new Promise((resolve, reject) => {
    const conn = new Client();
    conn
      .on("ready", () => resolve(conn))
      .on("error", (err) => reject(err))
      .connect({
        host: settings.host,
        port: Number(settings.port || 22),
        username: settings.user,
        password: settings.pass,
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

function sftpFastPut(sftp, localPath, remotePath) {
  return new Promise((resolve, reject) => {
    sftp.fastPut(localPath, remotePath, (err) => {
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
  const conn = await connectSsh(settings);
  try {
    emitLog(`Connected to ${settings.host}:${settings.port}`);
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

async function resolveMonitorCommand(conn, settings) {
  const raw = String(settings.monitor_start_cmd || "").trim();
  const { cwd, tail } = splitMonitorCommand(raw);
  const invoked = String(tail || raw);
  const binaryMatch = invoked.match(/\.\/([^\s]+)/);
  const binaryRelPath = binaryMatch ? binaryMatch[1] : "dumapril-taglocalization";
  const binary = path.posix.basename(binaryRelPath);
  const likelyTail = tail || raw;
  const hasBinaryRef = likelyTail.includes(binary);

  if (!hasBinaryRef) return raw;

  const derived = inferRemoteDeployFolder(settings.remote_path, settings.local_path);
  const candidates = [
    cwd,
    derived,
    settings.remote_path,
    `/home/${settings.user}`,
    `/home/${settings.user}/deployments`,
    `/home/${settings.user}/deployments/Vortex`
  ]
    .map((v) => stripOuterQuotes(v))
    .filter(Boolean)
    .filter((v, i, arr) => arr.indexOf(v) === i);

  for (const dir of candidates) {
    try {
      if (await commandExistsInDir(conn, dir, binaryRelPath)) {
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
    const first = await execCommand(
      conn,
      `bash -lc "cd '${dir}' && ${cargoCmd} build --release --bin orin_bridge > '${logPath}' 2>&1; code=\\$?; tail -n 180 '${logPath}'; exit \\$code"`
    );
    if (first.code === 0) return { ok: true, output: first };

    const out1 = String((first.stdout || "") + "\n" + (first.stderr || "")).trim();
    const lockErr =
      out1.includes("lock file version 4") ||
      out1.includes("failed to parse lock file") ||
      out1.includes("feature `edition2024` is required");
    if (!lockErr) return { ok: false, output: first };

    emitLog(`Lockfile incompatible on remote in ${dir}; retrying without Cargo.lock...`);
    const second = await execCommand(
      conn,
      `bash -lc "cd '${dir}' && rm -f Cargo.lock && ${cargoCmd} build --release --bin orin_bridge > '${logPath}' 2>&1; code=\\$?; tail -n 180 '${logPath}'; exit \\$code"`
    );
    if (second.code === 0) return { ok: true, output: second };
    return { ok: false, output: second };
  }

  async function ensureBridgeBuildDeps() {
    const pw = shSingle(settings.pass || "");
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
        const suffix = likelyTail.includes(binary)
          ? likelyTail.split(binary).slice(1).join(binary)
          : "";
        const resolved = `cd '${dir}' && ./target/release/${binary}${suffix}`;
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
      `find '/home/${settings.user}' -maxdepth 6 -type f -name '${binary}' 2>/dev/null | head -n 1`
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

      const resolved = `cd '${runDir}' && ${runCmd}`;
      emitLog(`Resolved monitor dir by scan: ${runDir}`);
      return resolved;
    }
  } catch (_err) {
    // fall through
  }

  if (sawMissingCargo) {
    emitLog("Cargo not found. Attempting remote install via apt...");
    try {
      const pw = shSingle(settings.pass || "");
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
            const suffix = likelyTail.includes(binary)
              ? likelyTail.split(binary).slice(1).join(binary)
              : "";
            const resolved = `cd '${dir}' && ./target/release/${binary}${suffix}`;
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
      const pw = shSingle(settings.pass || "");
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
            const suffix = likelyTail.includes(binary)
              ? likelyTail.split(binary).slice(1).join(binary)
              : "";
            const resolved = `cd '${dir}' && ./target/release/${binary}${suffix}`;
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
  async start(settings) {
    if (this.conn) throw new Error("Monitor already running");
    this.settings = settings;
    this.stopping = false;
    try {
      emitMonitorStartProgress({ status: "starting", percent: 10, text: "Connecting..." });
      this.conn = await connectSsh(settings);

      if (settings.startup_service_name) {
        emitMonitorStartProgress({ status: "starting", percent: 20, text: "Stopping service..." });
        const stopSvc = await execCommand(
          this.conn,
          `sudo systemctl stop ${settings.startup_service_name}`
        );
        if (stopSvc.code !== 0) {
          emitLog(`Service stop failed: ${stopSvc.stderr || stopSvc.stdout}`);
        } else {
          emitLog(`Service stopped: ${settings.startup_service_name}`);
        }
      }

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
        if (this.settings && this.settings.startup_service_name) {
          try {
            const restart = await execCommand(
              this.conn,
              `sudo systemctl start ${this.settings.startup_service_name}`
            );
            if (restart.code === 0) {
              emitLog(`Service restarted: ${this.settings.startup_service_name}`);
            }
          } catch (_err) {}
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
          emitLog("Authentication failed. Update credentials in vortex_config.json.");
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
  const appCfg = { ...defaultAppConfig(), ...(await readJson(APP_CONFIG_PATH, {})) };
  delete appCfg.preview_camera_index;
  appCfg.local_path = path.resolve(__dirname, appCfg.local_path || "..");
  const runtimeConfigPath = appCfg.runtime_config_path || DEFAULT_RUNTIME_CONFIG_PATH;
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

ipcMain.handle("save-app-config", async (_evt, appConfig) => {
  const sanitized = { ...(appConfig || {}) };
  delete sanitized.preview_camera_index;
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

ipcMain.handle("deploy-start", async (_evt, settings) => {
  deployProject(settings)
    .then(() => emitLog("Deploy complete."))
    .catch((err) => {
      emitLog(`Deploy failed: ${err.message}`);
      emitDeployProgress({ percent: 0, current: 0, total: 0, status: "error" });
    });
  return { ok: true };
});

ipcMain.handle("monitor-start", async (_evt, settings) => {
  emitMonitorStartProgress({ status: "starting", percent: 5, text: "Starting..." });
  try {
    await monitorManager.start(settings);
    return { ok: true };
  } catch (err) {
    const msg = err?.message || String(err);
    emitLog(`Monitor start failed: ${msg}`);
    if (msg.includes("All configured authentication methods failed")) {
      emitLog("Authentication failed. Update credentials in vortex_config.json.");
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
