"use strict";
// Run on a disposable Windows CI runner: this installs and uninstalls the app.
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { spawn, spawnSync } = require("node:child_process");
const assert = require("node:assert/strict");
const net = require("node:net");
const { setTimeout: delay } = require("node:timers/promises");

function run(file, args, options = {}) {
  const result = spawnSync(file, args, { encoding: "utf8", windowsHide: true, timeout: 180000, ...options });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `${file}: ${result.stderr || result.stdout}`);
  return result.stdout;
}
async function debugPort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}
async function evaluate(url, expression) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    const timer = setTimeout(() => { ws.close(); reject(new Error("Renderer inspection timed out")); }, 5000);
    ws.addEventListener("open", () => ws.send(JSON.stringify({ id: 1, method: "Runtime.evaluate", params: { expression, returnByValue: true } })));
    ws.addEventListener("error", () => { clearTimeout(timer); reject(new Error("Renderer debugger disconnected")); });
    ws.addEventListener("message", message => {
      const response = JSON.parse(message.data);
      if (response.id !== 1) return;
      clearTimeout(timer);
      ws.close();
      if (response.error || response.result?.exceptionDetails) reject(new Error(JSON.stringify(response)));
      else resolve(response.result.result.value);
    });
  });
}

async function main() {
  assert.equal(process.platform, "win32", "Installer smoke test requires Windows");
  const probeOnly = process.argv.includes("--probe-only");
  const launch = !probeOnly || process.argv.includes("--launch");
  const work = fs.mkdtempSync(path.join(os.tmpdir(), "vortex-installer-"));
  const installed = probeOnly ? path.resolve(process.argv[process.argv.indexOf("--probe-only") + 1]) : path.join(work, "app");
  const dist = path.resolve(__dirname, "../dist");
  const installer = probeOnly ? null : fs.readdirSync(dist).filter(n => /^VortexSetup-.*-x64\.exe$/.test(n));
  if (!probeOnly) assert.equal(installer.length, 1, "Expected exactly one NSIS installer");
  const executable = path.join(installed, "Vortex Deployment Interface.exe");
  let gui;
  let installedByTest = false;
  try {
    if (!probeOnly) {
      assert.equal(process.env.CI, "true", "Installation is restricted to disposable CI runners; use --probe-only for an existing unpacked app");
      run(path.join(dist, installer[0]), ["/S", `/D=${installed}`]);
      installedByTest = true;
    }
    assert.ok(fs.existsSync(executable), "Installed application executable is missing");
    const probe = `
      const path = require('path');
      const cp = require('child_process');
      const root = path.join(process.resourcesPath, 'app.asar');
      const {nativeController} = require(path.join(root, 'native-controller.js'));
      (async () => {
        for (const [platform, arch] of [['win32','x64'],['linux','x64'],['linux','arm64']]) {
          const controller = await nativeController(root, platform, arch);
          if (platform === 'win32') {
            const result = cp.spawnSync(controller, ['--help'], {encoding:'utf8', windowsHide:true});
            if (result.status !== 0 || !['offline-create','validate-device','cleanup'].every(x => result.stdout.includes(x))) throw new Error('Installed controller failed');
          }
        }
        for (const name of ['maintenance','initialize-deploy','staging-deploy','service-diagnostics','robot-network','network-setup','radio-transfer']) require(path.join(root, name+'.js'));
        const fs = require('fs');
        for (const name of ['renderer/index.html','renderer/app.js','renderer/styles.css','preload.js']) if (!fs.existsSync(path.join(root,name))) throw new Error('Missing '+name);
        console.log('INSTALLED_PROBE_PASS');
      })().catch(error => {console.error(error);process.exitCode=1});
    `;
    const output = run(executable, ["-e", probe], { env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" } });
    assert.ok(output.includes("INSTALLED_PROBE_PASS"), "Installed application probe did not finish");
    if (launch) {
      const env = { ...process.env };
      delete env.ELECTRON_RUN_AS_NODE;
      const port = await debugPort();
      gui = spawn(executable, [`--user-data-dir=${path.join(work, "profile")}`, "--deployment-smoke-test", "--disable-gpu", `--remote-debugging-port=${port}`, "--remote-debugging-address=127.0.0.1"], { windowsHide: true, env, stdio: ["ignore", "pipe", "pipe"] });
      let errors = "";
      gui.stdout.resume();
      gui.stderr.on("data", chunk => { errors += chunk; });
      let startupError;
      gui.on("error", error => { startupError = error; });
      await delay(10000);
      if (startupError) throw startupError;
      assert.equal(gui.exitCode, null, `Application exited during startup: ${errors}`);
      assert.ok(!/Unable to find Electron app|Cannot find module|UnhandledPromiseRejection/i.test(errors), errors);
      const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = targets.find(target => target.type === "page" && /renderer\/index\.html$/.test(target.url));
      assert.ok(page, "Installed renderer page did not load");
      const ui = await evaluate(page.webSocketDebuggerUrl, "({title:document.title,setup:!!document.querySelector('#jetsonSetup'),api:typeof window.vortexApi?.networkSetup,ready:document.readyState,styled:document.styleSheets.length>0,connection:document.querySelector('#connectionMode')?.value})");
      assert.deepEqual(ui, { title: "Vortex Deployment Interface", setup: true, api: "function", ready: "complete", styled: true, connection: "radio" });
      await evaluate(page.webSocketDebuggerUrl, "document.querySelector('#connectionMode').value='usb';document.querySelector('#connectionMode').dispatchEvent(new Event('change'));true");
      await delay(300);
      assert.deepEqual(await evaluate(page.webSocketDebuggerUrl, "({host:document.querySelector('#jetsonHost').value,port:document.querySelector('#jetsonPort').value})"), {host:"192.168.55.1",port:"22"});
      await evaluate(page.webSocketDebuggerUrl, "document.querySelector('#connectionMode').value='radio';document.querySelector('#connectionMode').dispatchEvent(new Event('change'));true");
      await delay(300);
      assert.deepEqual(await evaluate(page.webSocketDebuggerUrl, "({host:document.querySelector('#jetsonHost').value,port:document.querySelector('#jetsonPort').value})"), {host:"10.5.9.12",port:"5808"});
      run("taskkill", ["/PID", String(gui.pid), "/T", "/F"]);
      gui = null;
    }
    const result = { passed: true, installed: !probeOnly, controllersVerified: 3, guiStartup: launch };
    if (!probeOnly) {
      const uninstaller = fs.readdirSync(installed).find(n => /^Uninstall.*\.exe$/i.test(n));
      assert.ok(uninstaller, "Uninstaller is missing");
      run(path.join(installed, uninstaller), ["/S"]);
      for (let n = 0; n < 60 && fs.existsSync(executable); n++) await delay(500);
      assert.ok(!fs.existsSync(executable), "Uninstaller left the application executable behind");
      result.uninstalled = true;
      fs.writeFileSync(path.join(dist, "installer-smoke.json"), JSON.stringify(result, null, 2)+"\n");
    }
    console.log(JSON.stringify(result));
  } finally {
    if (gui && gui.pid) spawnSync("taskkill", ["/PID", String(gui.pid), "/T", "/F"], { windowsHide: true });
    // Preserve failed installations and logs for CI inspection instead of deleting evidence.
    if (installedByTest && fs.existsSync(executable)) console.error(`Installer evidence retained at ${work}`);
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
