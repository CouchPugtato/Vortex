"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs/promises");
const os=require("node:os");
const path=require("node:path");
const {inspect,configure}=require("../network-setup");
async function harness(t) {
  const dir=await fs.mkdtemp(path.join(os.tmpdir(),"vortex-network-test-"));
  t.after(()=>fs.rm(dir,{recursive:true,force:true}));
  const binary=path.join(dir,"controller");await fs.writeFile(binary,"fixture");
  const calls=[],logs=[],connections=[];
  const op={schema_version:1,operation_id:"1".repeat(32),token:"2".repeat(32),address:"10.5.9.12",hostname:"jetson"};
  const deps={appDir:dir,attempts:2,delay:async()=>{},log:s=>logs.push(s),
    controller:async()=>binary,getSftp:async()=>({}),upload:async()=>{},
    connect:async cfg=>{connections.push(cfg);return {vortexHostKey:"a".repeat(64),end(){}};},
    execChecked:async(conn,cmd)=>({stdout:cmd==="uname -m" ? "aarch64" : cmd.includes("printf") ? "/home/vortex" : cmd.includes("network-info") ? JSON.stringify({interfaces:[]}) : cmd.includes("network-status") ? JSON.stringify({state:"awaiting_confirmation"}) : ""}),
    admin:async(conn,cmd,password)=>{calls.push({cmd,password});return {stdout:JSON.stringify(cmd.includes("network-submit")?op:{state:"confirmed",address:op.address})};}};
  return {deps,calls,logs,connections,op};
}
const settings={connection_mode:"usb",radio_team:509,radio_device:12,user:"vortex",pass:"ssh-secret"};
test("setup confirms only over the new address and exact SSH host identity",async t=>{
  const h=await harness(t);const result=await configure(settings,"eth0","admin-secret",h.deps);
  assert.equal(result.connection_mode,"radio");assert.equal(result.host,"10.5.9.12");assert.equal(result.port,"5808");assert.equal(result.pass,"");assert.equal(result.radio_hostname,"jetson.local");
  assert.equal(h.connections[1].connection_mode,"custom");assert.equal(h.connections[1].host_key,"a".repeat(64));
  assert.ok(h.calls.every(c=>c.password==="admin-secret"&&!c.cmd.includes("admin-secret")));
  assert.ok(!h.logs.join("").includes(h.op.token));
});
test("inspection needs no privileged command or existing vision installation",async t=>{
  const h=await harness(t);assert.deepEqual(await inspect(settings,h.deps),{interfaces:[]});assert.equal(h.calls.length,0);
});
test("a different device at the new address cannot confirm a change",async t=>{
  const h=await harness(t);const connect=h.deps.connect;let n=0;
  h.deps.connect=async cfg=>{const c=await connect(cfg);if(n++)c.vortexHostKey="b".repeat(64);return c;};
  await assert.rejects(configure(settings,"eth0","",h.deps),/identity changed/);
  assert.equal(h.calls.length,1);
});
test("lost route leaves the server timer responsible for rollback",async t=>{
  const h=await harness(t);const connect=h.deps.connect;let n=0;
  h.deps.connect=async cfg=>{if(n++)throw new Error("unreachable");return connect(cfg);};
  await assert.rejects(configure(settings,"eth0","",h.deps),/revert automatically/);
  assert.equal(h.calls.length,1);
});
test("invalid addressing or interface fails before privileged mutation",async t=>{
  const h=await harness(t);
  await assert.rejects(configure({...settings,radio_device:200},"eth0","",h.deps),/device address/);
  await assert.rejects(configure(settings,"eth0;reboot","",h.deps),/interface/);
  assert.equal(h.connections.length,0);
});
