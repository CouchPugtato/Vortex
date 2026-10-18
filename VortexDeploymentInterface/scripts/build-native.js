"use strict";
// Build machines need Rust and Zig (for portable static Linux executables).
// The packaged application does not run compilers or need Python.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { spawnSync } = require("child_process");
const app = path.resolve(__dirname, "..");
const project = path.resolve(app, "..", "deployment");
const targets = ["x86_64-pc-windows-gnu", "aarch64-unknown-linux-musl", "x86_64-unknown-linux-musl"];
const native = path.join(app, "native");
const manifestPath = path.join(native, "manifest.json");
const check = process.argv.includes("--check");
const install = process.argv.includes("--import");
const pin = fs.readFileSync(path.join(project, "rust-toolchain.toml"), "utf8").match(/channel\s*=\s*"(\d+\.\d+\.\d+)"/)[1];
const hash = (filename) => crypto.createHash("sha256").update(fs.readFileSync(filename)).digest("hex");
function run(exe, args) {
  const result = spawnSync(exe, args, { cwd: project, stdio: "inherit", shell: false });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${exe} failed (${result.status})`);
}
const manifest = check ? JSON.parse(fs.readFileSync(manifestPath, "utf8")) : {};
if (!check && !install) run("rustup", ["toolchain", "install", pin, "--profile", "minimal"]);
function validateBinary(filename, target) {
  const bytes = fs.readFileSync(filename);
  if (target.includes("windows")) {
    if (bytes.length < 64 || bytes.toString("ascii", 0, 2) !== "MZ") throw new Error("Expected Windows executable: "+filename);
    const pe = bytes.readUInt32LE(60);
    if (pe+6 > bytes.length || bytes.toString("ascii", pe, pe+4) !== "PE\0\0" || bytes.readUInt16LE(pe+4) !== 0x8664) {
      throw new Error("Expected Windows x64 executable: "+filename);
    }
  } else {
    if (bytes.length < 64 || !bytes.subarray(0, 6).equals(Buffer.from([127, 69, 76, 70, 2, 1])) ||
        bytes.readUInt16LE(18) !== (target.startsWith("aarch64") ? 183 : 62)) {
      throw new Error("Wrong Linux controller architecture: "+filename);
    }
    const offset = Number(bytes.readBigUInt64LE(32)), size = bytes.readUInt16LE(54), count = bytes.readUInt16LE(56);
    if (size < 56 || count > 128 || offset+size*count > bytes.length) throw new Error("Invalid ELF program headers");
    for (let i = 0; i < count; i++) if (bytes.readUInt32LE(offset+i*size) === 3) {
      throw new Error("Linux controller must be statically linked: "+filename);
    }
  }
}
for (const target of targets) {
  const binary = "vortexctl"+(target.includes("windows") ? ".exe" : "");
  const name = target+"/"+binary;
  const destination = path.join(native, name);
  if (!check) {
    if (!install) {
      run("rustup", ["target", "add", "--toolchain", pin, target]);
      run("rustup", ["run", pin, "cargo", target.includes("linux") ? "zigbuild" : "build", "--release", "--locked", "--target", target]);
    }
    const input = path.join(process.env.VORTEXCTL_ARTIFACTS || path.join(project, "target"), target, "release", binary);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(input, destination);
    if (!target.includes("windows")) fs.chmodSync(destination, 0o755);
    manifest[name] = hash(destination);
  } else if (hash(destination) !== manifest[name]) {
    throw new Error(`Missing or changed native artifact: ${name}. Run npm run build:native.`);
  }
  validateBinary(destination, target);
}
if (!check) fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2)+"\n");
console.log(check ? "Native controller artifacts verified." : "Native controller artifacts prepared.");
