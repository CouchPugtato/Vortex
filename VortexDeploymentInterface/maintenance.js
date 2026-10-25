"use strict";
const { managedRoot, quote } = require("./staging-deploy");
async function maintain(settings, target, action, deps) {
  if (!["cleanup-preview", "cleanup", "validate"].includes(action)) throw new Error("Unknown maintenance action");
  const root = managedRoot(target);
  const command = `${quote(root+"/vortexctl")} `+(action === "validate" ? "validate-device --duration-seconds 150" :
    "cleanup --keep-releases 3 --max-age-days 14"+(action === "cleanup" ? " --apply" : ""))+` --root ${quote(root)}`;
  const conn = await deps.connect(settings);
  try { return JSON.parse((await deps.execChecked(conn, command)).stdout); }
  finally { conn.end(); }
}
module.exports = { maintain };
