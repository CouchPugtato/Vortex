"use strict";

// Status returns a bounded log tail. Emit only new output, including when the
// tail rolls over, rather than printing the same compiler messages every poll.
function logFollower(log) {
  let previous = "";
  return (job) => {
    const next = job.log_tail;
    if (typeof next !== "string" || next === previous) return;
    let overlap = Math.min(previous.length, next.length);
    while (overlap && !previous.endsWith(next.slice(0, overlap))) overlap--;
    const added = next.slice(overlap);
    if (added.trim()) log(added.trimEnd());
    previous = next;
  };
}

module.exports = { logFollower };
