"use strict";
const net = require("node:net");

const USB_HOST = "192.168.55.1";
function robotAddress(team = 509, device = 12) {
  team = Number(team); device = Number(device);
  if (!Number.isInteger(team) || team < 1 || team > 25599) throw new Error("Team number must produce a valid 10.TE.AM address (1–25599)");
  if (!Number.isInteger(device) || device < 6 || device > 19) throw new Error("Use an unused robot device address from 6 through 19");
  return `10.${Math.floor(team/100)}.${team%100}.${device}`;
}
function connectionSettings(input = {}) {
  const mode = input.connection_mode || (input.host ? "custom" : "radio");
  if (!["radio", "usb", "custom"].includes(mode)) throw new Error("Unknown connection mode");
  const host = mode === "radio" ? robotAddress(input.radio_team ?? 509, input.radio_device ?? 12) : mode === "usb" ? USB_HOST : String(input.host || "").trim();
  const port = mode === "usb" ? 22 : Number(input.port || (mode === "radio" ? 5808 : 22));
  if (!host || /[\s\x00-\x1f\x7f]/.test(host) || !Number.isInteger(port) || port < 1 || port > 65535) throw new Error("Invalid device address or SSH port");
  if (mode === "radio" && (port < 5800 || port > 5810)) throw new Error("Robot radio SSH port must be an unused team TCP port (5800–5810)");
  if (input.host_key && !/^[a-f0-9]{64}$/.test(input.host_key)) throw new Error("Invalid saved device identity");
  return { ...input, connection_mode: mode, host, port: String(port) };
}
function migrateConnection(saved = {}, pending = false) {
  if (saved.connection_mode) return connectionSettings(saved);
  if (saved.host && (saved.host !== USB_HOST || pending)) return connectionSettings({ ...saved, connection_mode: "custom" });
  return connectionSettings({ ...saved, connection_mode: "radio", port: "5808", radio_team: saved.radio_team ?? saved.vortex_nt_team ?? 509, radio_device: 12 });
}
function endpoints(settings) {
  const cfg = connectionSettings(settings);
  const hosts = [cfg.host];
  if (cfg.connection_mode === "radio" && cfg.radio_hostname) {
    if (!/^[a-zA-Z0-9][a-zA-Z0-9.-]{0,250}\.local$/.test(cfg.radio_hostname)) throw new Error("Use a .local device hostname");
    hosts.push(cfg.radio_hostname);
  }
  const ports = cfg.connection_mode === "radio" ? [...new Set([Number(cfg.port), 22])] : [Number(cfg.port)];
  return hosts.flatMap(host => ports.map(port => ({ host, port })));
}
function probe(endpoint, timeoutMs = 1200) {
  return new Promise(resolve => {
    let banner = "", done = false;
    const socket = net.connect(endpoint);
    const timer = setTimeout(() => finish(false, "No SSH response; check the robot network, address and port"), timeoutMs);
    const finish = (reachable, reason) => {
      if (done) return;
      done = true; clearTimeout(timer); socket.destroy(); resolve({ ...endpoint, reachable, reason });
    };
    socket.setTimeout(timeoutMs, () => finish(false, "No SSH response; check the robot network, address and port"));
    socket.on("error", error => finish(false, error.code === "ECONNREFUSED" ? "Device reachable but SSH port is closed" : error.code === "ENOTFOUND" ? "Hostname could not be resolved" : "Device is unreachable"));
    socket.on("data", bytes => {
      banner += bytes.toString("ascii");
      if (/^SSH-2\.0-/m.test(banner)) finish(true, "SSH available");
      else if (banner.length > 1024) finish(false, "Port is not an SSH service");
    });
    socket.on("end", () => finish(false, "Port closed before an SSH greeting"));
  });
}
async function discover(settings, probeEndpoint = probe) {
  return Promise.all(endpoints(settings).map(endpoint => probeEndpoint(endpoint)));
}
async function connect(settings, Client, password, probeEndpoint = probe) {
  const cfg = connectionSettings(settings);
  const choices = endpoints(cfg);
  let last;
  for (const endpoint of choices) {
    // Custom/USB connections retain their previous direct connection behavior.
    if (cfg.connection_mode === "radio" && !(await probeEndpoint(endpoint)).reachable) continue;
    try {
      return await new Promise((resolve, reject) => {
        const conn = new Client();
        conn.on("handshake", negotiated => { conn.vortexHostAlgorithm = negotiated.srvHostKey; });
        conn.on("ready", () => { conn.vortexEndpoint = endpoint; resolve(conn); });
        conn.on("error", error => { conn.end(); reject(error); });
        conn.connect({ ...endpoint, username: cfg.user || "vortex", password, readyTimeout: cfg.connect_timeout_ms || 8000,
          keepaliveInterval: 5000, keepaliveCountMax: 3, hostHash: "sha256",
          algorithms: cfg.host_key_algorithm ? { serverHostKey: [cfg.host_key_algorithm] } : undefined,
          hostVerifier: key => { conn.vortexHostKey = key; return !cfg.host_key || key === cfg.host_key; } });
      });
    } catch (error) {
      last = error;
      if (error.level === "client-authentication" || /host.*verif/i.test(error.message)) throw error;
    }
  }
  throw last || new Error("Jetson not found on the robot network. Check team/address and Ethernet; use USB recovery for first-time setup.");
}
module.exports = { USB_HOST, robotAddress, connectionSettings, migrateConnection, endpoints, probe, discover, connect };
