"use strict";
const fs = require("node:fs");
const { pipeline } = require("node:stream/promises");
const { setTimeout: delay } = require("node:timers/promises");
function radioLimited(settings) { return settings.connection_mode === "radio" && settings.radio_bandwidth_mode !== false; }
async function upload(sftp, local, remote, options = {}, bytesPerSecond = 128*1024) {
  const total = fs.statSync(local).size;
  const start = performance.now();
  let sent = 0;
  await pipeline(fs.createReadStream(local, { highWaterMark: 16*1024 }), async function* (source) {
    for await (const chunk of source) {
      sent += chunk.length;
      const wait = sent/bytesPerSecond*1000-(performance.now()-start);
      if (wait > 0) await delay(wait);
      yield chunk;
      options.step?.(sent, chunk.length, total);
    }
  }, sftp.createWriteStream(remote, { flags: "w", mode: 0o600 }));
}
module.exports = { radioLimited, upload };
