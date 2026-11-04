const {test} = require('node:test');
const assert = require('node:assert/strict');
const {LatestPoll} = require('../recorder/static/latest-poll.js');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const deferred = () => {let resolve; const promise = new Promise(r => resolve = r); return {promise, resolve};};
const response = value => ({ok: true, status: 200, json: async () => value});
const options = {url: () => '/preview.jpg', read: r => r.json(), onValue: () => {}, onError: () => {}, interval: 5};

test('slow transfer has only one request in flight, then jumps to newest data', async () => {
  const original = global.fetch, gate = deferred(), seen = [];
  let active = 0, peak = 0, calls = 0, newest = 1;
  global.fetch = async () => {
    calls++; active++; peak = Math.max(peak, active);
    const frame = newest;
    if (calls === 1) await gate.promise;
    active--;
    return response(frame);
  };
  const poll = new LatestPoll({...options, onValue: v => seen.push(v)});
  try {
    poll.start();
    await sleep(40);
    newest = 80;
    assert.equal(calls, 1);
    gate.resolve();
    await sleep(30);
    assert.equal(peak, 1);
    assert.deepEqual(seen.slice(0, 2), [1, 80]);
  } finally {poll.stop(); global.fetch = original;}
});

test('pause and immediate resume cannot display or overlap an old response', async () => {
  const original = global.fetch, gate = deferred(), seen = [], disposed = [];
  let calls = 0;
  // Model a response completing despite abort, including a pending image decode.
  global.fetch = async () => response(++calls);
  const poll = new LatestPoll({...options,
    read: async r => {const v = await r.json(); if (v === 1) await gate.promise; return v;},
    onValue: v => seen.push(v), dispose: v => disposed.push(v),
  });
  try {
    poll.start(); await sleep(10); poll.stop(); poll.start();
    assert.equal(calls, 1);
    gate.resolve(); await sleep(30);
    assert.equal(seen.includes(1), false);
    assert.deepEqual(disposed, [1]);
    assert.equal(seen[0], 2);
    poll.stop();
    const stoppedAt = calls;
    await sleep(20);
    assert.equal(calls, stoppedAt);
  } finally {poll.stop(); global.fetch = original;}
});

test('deadline covers the response body and rejects data completed after abort', async () => {
  const original = global.fetch, gate = deferred(), seen = [], disposed = [], errors = [];
  let signal;
  global.fetch = async (_url, args) => {signal = args.signal; return response(1);};
  const poll = new LatestPoll({...options, timeout: 20,
    read: async () => {await gate.promise; return 'late';},
    onValue: v => seen.push(v), onError: e => errors.push(e), dispose: v => disposed.push(v),
  });
  try {
    poll.start(); await sleep(35);
    assert.equal(signal.aborted, true);
    gate.resolve(); await sleep(5);
    assert.deepEqual(seen, []);
    assert.deepEqual(disposed, ['late']);
    assert.equal(errors.length, 1);
  } finally {poll.stop(); global.fetch = original;}
});

test('freshness rejection releases the image and retries instead of freezing', async () => {
  const original = global.fetch, disposed = [];
  let calls = 0;
  global.fetch = async () => response(++calls);
  const poll = new LatestPoll({...options,
    onValue: () => {throw new Error('Image too old');}, dispose: v => disposed.push(v),
  });
  try {
    poll.start(); await sleep(230);
    assert.equal(calls, 2);
    assert.deepEqual(disposed, [1, 2]);
  } finally {poll.stop(); global.fetch = original;}
});

test('no new frame is a normal response and resumes on the next request', async () => {
  const original = global.fetch, seen = [], errors = [];
  let calls = 0;
  global.fetch = async () => ++calls === 1 ? {ok: true, status: 204} : response(2);
  const poll = new LatestPoll({...options, onValue: v => seen.push(v), onError: e => errors.push(e)});
  try {
    poll.start(); await sleep(20);
    assert.equal(seen[0], 2);
    assert.deepEqual(errors, []);
  } finally {poll.stop(); global.fetch = original;}
});
