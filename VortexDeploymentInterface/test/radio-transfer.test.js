"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs/promises");
const os=require("node:os");
const path=require("node:path");
const {Writable}=require("node:stream");
const {upload,radioLimited}=require("../radio-transfer");
test("radio transfers preserve bytes, honor the cap and report final progress",async t=>{
  const dir=await fs.mkdtemp(path.join(os.tmpdir(),"radio-upload-"));t.after(()=>fs.rm(dir,{recursive:true,force:true}));
  const file=path.join(dir,"input");const bytes=Buffer.alloc(512,42);await fs.writeFile(file,bytes);
  const received=[],progress=[];const start=performance.now();
  await upload({createWriteStream:()=>new Writable({write(chunk,encoding,cb){received.push(chunk);cb();}})},file,"remote",{step:(sent,chunk,total)=>progress.push([sent,total])},2048);
  assert.deepEqual(Buffer.concat(received),bytes);assert.deepEqual(progress.at(-1),[512,512]);assert.ok(performance.now()-start>=220);
  assert.equal(radioLimited({connection_mode:"radio"}),true);assert.equal(radioLimited({connection_mode:"usb"}),false);assert.equal(radioLimited({connection_mode:"radio",radio_bandwidth_mode:false}),false);
});
