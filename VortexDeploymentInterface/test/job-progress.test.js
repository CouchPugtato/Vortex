"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { logFollower } = require("../job-progress");

test("restored and rolling log tails print each new segment once", () => {
  const output = [];
  const follow = logFollower((line) => output.push(line));
  follow({ log_tail: "one\ntwo\n" });
  follow({ log_tail: "one\ntwo\n" });
  follow({ log_tail: "two\nthree\n" });
  follow({ log_tail: "two\nthree\nfour\n" });
  assert.deepEqual(output, ["one\ntwo", "three", "four"]);
});
