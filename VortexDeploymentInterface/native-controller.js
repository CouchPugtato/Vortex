"use strict";
const fs = require("fs/promises");
const path = require("path");
const crypto = require("crypto");

async function nativeController(appDir, platform, architecture) {
  const target = platform === "win32" && architecture === "x64" ? "x86_64-pc-windows-gnu" :
    platform === "linux" && architecture === "arm64" ? "aarch64-unknown-linux-musl" :
    platform === "linux" && architecture === "x64" ? "x86_64-unknown-linux-musl" : null;
  if (!target) throw new Error(`Unsupported controller platform: ${platform}/${architecture}`);
  const directory = process.resourcesPath && require("fs").existsSync(path.join(process.resourcesPath, "native")) ?
    path.join(process.resourcesPath, "native") : path.join(appDir, "native");
  const manifest = JSON.parse(await fs.readFile(path.join(directory, "manifest.json"), "utf8").catch(() => {
    throw new Error("Native deployment tools are missing. Run npm run build:native before starting or packaging the app.");
  }));
  const name = target+"/vortexctl"+(platform === "win32" ? ".exe" : "");
  const filename = path.join(directory, name);
  const bytes = await fs.readFile(filename);
  if (!/^[a-f0-9]{64}$/.test(manifest[name]) || crypto.createHash("sha256").update(bytes).digest("hex") !== manifest[name]) {
    throw new Error(`Native controller checksum mismatch: ${target}`);
  }
  return filename;
}

function verifyController(filename, checksum, quote) {
  if (!/^[a-f0-9]{64}$/.test(checksum)) throw new Error("Invalid controller checksum");
  return `test "$(sha256sum -- ${quote(filename)} | cut -d ' ' -f 1)" = ${quote(checksum)} && chmod 755 -- ${quote(filename)}`;
}
module.exports = { nativeController, verifyController };
