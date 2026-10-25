"use strict";
const crypto = require("node:crypto");
const { nativeController, verifyController } = require("./native-controller");
const fs = require("node:fs/promises");
const { quote } = require("./staging-deploy");
const { execAdministrator } = require("./initialize-deploy");
const { connectionSettings } = require("./robot-network");

async function helper(settings, deps) {
  const conn = await deps.connect(settings);
  try {
    const arch = (await deps.execChecked(conn, "uname -m")).stdout.trim();
    if (!["aarch64", "x86_64"].includes(arch)) throw new Error("Robot networking requires an Ubuntu/JetPack Jetson or x64 Linux device");
    const home = (await deps.execChecked(conn, "printf '%s' \"$HOME\"")).stdout.trim();
    if (!/^\/[A-Za-z0-9_./-]+$/.test(home)) throw new Error("Unsupported remote home folder");
    const remote = `${home}/.cache/vortex-network/${crypto.randomUUID()}/vortexctl`;
    const local = await (deps.controller || nativeController)(deps.appDir, "linux", arch === "aarch64" ? "arm64" : "x64");
    const checksum = crypto.createHash("sha256").update(await fs.readFile(local)).digest("hex");
    await deps.execChecked(conn, `mkdir -p -- ${quote(remote.slice(0, remote.lastIndexOf("/")))}`);
    await deps.upload(await deps.getSftp(conn), local, remote);
    await deps.execChecked(conn, verifyController(remote, checksum, quote));
    return { conn, remote };
  } catch (error) { conn.end(); throw error; }
}
async function inspect(settings, deps) {
  const { conn, remote } = await helper(settings, deps);
  try { return JSON.parse((await deps.execChecked(conn, `${quote(remote)} network-info`)).stdout); }
  finally { conn.end(); }
}
async function configure(settings, iface, administratorPassword, deps) {
  const administrator = deps.admin || ((conn, command, password) => execAdministrator(conn, command, password, "network"));
  const desired = connectionSettings({ ...settings, connection_mode: "radio", port: settings.radio_port || "5808" });
  if (!/^[a-zA-Z0-9_-]{1,15}$/.test(iface)) throw new Error("Select an Ethernet interface");
  const { conn, remote } = await helper(settings, deps);
  const identity = conn.vortexHostKey;
  const algorithm = conn.vortexHostAlgorithm;
  if (!/^[a-f0-9]{64}$/.test(identity || "")) { conn.end(); throw new Error("Could not verify the original Jetson SSH identity"); }
  let operation;
  try {
    const command = `${quote(remote)} network-submit --interface ${quote(iface)} --team ${Number(desired.radio_team ?? 509)} --device ${Number(desired.radio_device ?? 12)} --ssh-port ${Number(desired.port)} --user ${quote(settings.user || "vortex")}`;
    operation = JSON.parse((await administrator(conn, command, administratorPassword)).stdout);
    if (operation.schema_version !== 1 || operation.address !== desired.host || !/^[a-f0-9]{32}$/.test(operation.operation_id || "") || !/^[a-f0-9]{32}$/.test(operation.token || "")) throw new Error("Unexpected network setup response");
  } finally { conn.end(); }
  deps.log("Ethernet setup started. Reconnecting to the robot address; an unconfirmed change will roll back automatically.");
  const delay = deps.delay || (ms => new Promise(resolve => setTimeout(resolve, ms)));
  // Connect only to the requested address and port; never confirm over a USB or
  // mDNS fallback that could hide a broken robot-network route.
  const next = { ...desired, connection_mode: "custom", host_key: identity, host_key_algorithm: algorithm, connect_timeout_ms: 1500 };
  let last;
  for (let attempt = 0; attempt < (deps.attempts ?? 18); attempt++) {
    await delay(deps.retryMs ?? 2000);
    let active;
    try {
      active = await deps.connect(next);
      if (active.vortexHostKey !== identity) throw new Error("Jetson identity changed; network setup will roll back");
      const worker = `/var/lib/vortex-network/${operation.operation_id}/vortexctl`;
      const status = JSON.parse((await deps.execChecked(active, `${quote(worker)} network-status --operation-id ${quote(operation.operation_id)}`)).stdout);
      if (["rolled_back","rollback_failed","failed"].includes(status.state)) throw new Error(status.error || "Ethernet setup rolled back");
      if (status.state !== "awaiting_confirmation") continue;
      const command = `${quote(worker)} network-confirm --operation-id ${quote(operation.operation_id)} --token ${quote(operation.token)}`;
      const result = JSON.parse((await administrator(active, command, administratorPassword)).stdout);
      if (result.state !== "confirmed" || result.address !== desired.host) throw new Error("Ethernet setup was not confirmed");
      deps.log(`Robot Ethernet ready: ${desired.host}:${desired.port}. USB recovery remains available.`);
      return { ...desired, host_key: identity, host_key_algorithm: algorithm, radio_port: desired.port,
        radio_hostname: settings.radio_hostname || (operation.hostname ? operation.hostname.split(".")[0]+".local" : ""), pass: "" };
    } catch (error) {
      last = error;
      if (/identity|authentication|host.*verif|rolled back/i.test(error.message)) break;
    } finally { active?.end(); }
  }
  throw new Error(`Could not confirm the robot Ethernet connection. It will revert automatically; USB recovery remains available. ${last?.message || "Check the laptop is connected to the robot network."}`);
}
module.exports = { inspect, configure };
