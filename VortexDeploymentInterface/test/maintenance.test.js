"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");
const { maintain } = require("../maintenance");
const { offlineMetadata } = require("../initialize-deploy");
test("maintenance uses the installed controller without service control or sudo", async () => {
  for (const action of ["cleanup-preview", "cleanup", "validate"]) {
    let ended=false, command;
    const response=await maintain({},"/home/robot/Vortex",action,{
      connect:async()=>({end(){ended=true;}}),execChecked:async(_,cmd)=>{command=cmd;return {stdout:'{"schema_version":1}'};}
    });
    assert.equal(response.schema_version,1);assert(ended);
    assert(command.startsWith("'/home/robot/Vortex.deployment/vortexctl' "));
    assert.equal(command.includes(" --apply"),action==="cleanup");
    assert(!/sudo|systemctl|python|rm /.test(command));
  }
});
test("maintenance rejects unsupported actions before connecting and closes failures",async()=>{
  await assert.rejects(maintain({},"/home/robot/Vortex","destroy",{connect(){throw new Error("must not connect");}}),/Unknown maintenance/);
  let ended=false;
  await assert.rejects(maintain({},"/home/robot/Vortex","cleanup",{connect:async()=>({end(){ended=true;}}),execChecked:async()=>{throw new Error("busy");}}),/busy/);assert(ended);
});
test("offline bundle checks require matching content, size and sidecar metadata",async(t)=>{
  const directory=await fs.mkdtemp(path.join(os.tmpdir(),"vortex-offline-test-"));t.after(()=>fs.rm(directory,{recursive:true,force:true}));
  const filename=path.join(directory,"offline.tar.gz"), sidecar=path.join(directory,"offline.json");
  const bytes=Buffer.from("offline content");await fs.writeFile(filename,bytes);
  const metadata={schema_version:1,kind:"vortex-offline",profile:"cpu",archive_bytes:bytes.length,archive_sha256:crypto.createHash("sha256").update(bytes).digest("hex")};
  await fs.writeFile(sidecar,JSON.stringify(metadata));assert.deepEqual(await offlineMetadata(filename),metadata);
  await fs.writeFile(filename,"different bytes");await assert.rejects(offlineMetadata(filename),/Invalid offline|checksum mismatch/);
  await fs.writeFile(filename,bytes);await fs.writeFile(sidecar,JSON.stringify({...metadata,archive_sha256:"0".repeat(64)}));await assert.rejects(offlineMetadata(filename),/checksum mismatch/);
});
