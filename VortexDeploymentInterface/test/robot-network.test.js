"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const net = require("node:net");
const { EventEmitter } = require("node:events");
const network = require("../robot-network");
test("robot Ethernet is the fresh default and addresses handle leading zeros", () => {
  assert.deepEqual([network.connectionSettings().host,network.connectionSettings().port], ["10.5.9.12","5808"]);
  for (const [team, host] of [[1,"10.0.1.12"],[916,"10.9.16.12"],[9106,"10.91.6.12"],[25599,"10.255.99.12"]]) assert.equal(network.robotAddress(team), host);
  for (const [team, device] of [[0,12],[25600,12],[509,5],[509,20],[1.5,12]]) assert.throws(() => network.robotAddress(team,device));
});
test("USB and custom modes remain explicit; legacy addresses and pending jobs survive migration", () => {
  assert.equal(network.connectionSettings({connection_mode:"usb",host:"wrong",port:5808}).host, network.USB_HOST);
  assert.equal(network.connectionSettings({connection_mode:"usb"}).port,"22");
  assert.equal(network.migrateConnection({host:"192.168.55.1",port:"22"}).connection_mode,"radio");
  assert.equal(network.migrateConnection({host:"192.168.55.1",port:"22"},true).connection_mode,"custom");
  assert.equal(network.migrateConnection({host:"jetson.local",port:"2222"}).port,"2222");
  assert.throws(() => network.connectionSettings({connection_mode:"radio",port:22}));
  assert.throws(() => network.connectionSettings({host:"bad host"}));
  assert.throws(() => network.connectionSettings({host:"jetson",host_key:"bad"}));
});
test("discovery checks only configured addresses and team ports, without subnet scanning", async () => {
  const checked = [];
  const results = await network.discover({connection_mode:"radio",radio_hostname:"jetson.local"}, async endpoint => { checked.push(endpoint); return {...endpoint,reachable:endpoint.port===22}; });
  assert.deepEqual(checked,[{host:"10.5.9.12",port:5808},{host:"10.5.9.12",port:22},{host:"jetson.local",port:5808},{host:"jetson.local",port:22}]);
  assert.equal(results.filter(r=>r.reachable).length,2);
  assert.throws(() => network.endpoints({connection_mode:"radio",radio_hostname:"other.example.com"}));
});
test("real TCP probing distinguishes SSH, closed ports and a non-SSH server", async t => {
  const server = net.createServer(socket => socket.end("SSH-2.0-rehearsal\r\n"));
  await new Promise(resolve => server.listen(0,"127.0.0.1",resolve));
  t.after(()=>server.close());
  assert.equal((await network.probe({host:"127.0.0.1",port:server.address().port})).reachable,true);
  const bad = net.createServer(socket => socket.end("HTTP/1.1 200 OK\r\n"));
  await new Promise(resolve => bad.listen(0,"127.0.0.1",resolve));
  const port = bad.address().port;
  assert.equal((await network.probe({host:"127.0.0.1",port})).reachable,false);
  await new Promise(resolve=>bad.close(resolve));
  assert.match((await network.probe({host:"127.0.0.1",port})).reason,/closed/);
});
test("radio falls back to initial SSH port 22 and pins the same device identity", async () => {
  const calls=[];
  class Client extends EventEmitter {
    connect(cfg) { calls.push(cfg); queueMicrotask(()=>cfg.hostVerifier("a".repeat(64)) ? this.emit("ready") : this.emit("error",new Error("Host verification failed"))); }
    end() {}
  }
  const cfg={connection_mode:"radio",host_key:"a".repeat(64)};
  const conn=await network.connect(cfg,Client,"secret",async e=>({...e,reachable:e.port===22}));
  assert.equal(conn.vortexEndpoint.port,22);
  assert.equal(calls[0].password,"secret");
  assert.equal(conn.vortexHostKey,cfg.host_key);
  await assert.rejects(network.connect({...cfg,host_key:"b".repeat(64)},Client,"secret",async()=>({reachable:true})),/verification/);
});
test("authentication rejection does not spray credentials across fallback hosts", async () => {
  let attempts=0;
  class Client extends EventEmitter { connect() { attempts++; const e=new Error("Authentication failed");e.level="client-authentication";queueMicrotask(()=>this.emit("error",e)); } end() {} }
  await assert.rejects(network.connect({connection_mode:"radio",radio_hostname:"other.local"},Client,"secret",async()=>({reachable:true})),/Authentication/);
  assert.equal(attempts,1);
});
