// Inference: π0.5 policy rollout on the excavator (hub app inference: routes in hub/apps/inference/__init__.py, state in hub/apps/inference/core.py): vanilla JS, no dependencies.
// API calls go through ctx (/api/inference/...), DOM lookups stay inside the view root, and hidden views stop
// polling the camera / twin images.
export async function mount(root, ctx) {
  ctx.css('style.css');
  root.innerHTML = `
    <header class="ui-header top">
      <span class="ui-mark" aria-hidden="true"></span>
      <h1 class="ui-title">Excavator inference</h1>
      <div class="pills">
        <span class="pill" id="pill-sse">events</span>
        <span class="pill" id="pill-pi">Pi</span>
        <span class="pill" id="pill-bridge">bridge</span>
        <span class="pill" id="pill-esp32">ESP32 bridge</span>
        <span class="pill" id="pill-job">idle</span>
      </div>
    </header>

    <div class="inference-main">
      <section class="panes">
        <figure class="pane">
          <figcaption>Pi camera <span class="muted" id="cam-info"></span> <span class="tag cloak" id="cam-cloak" hidden title="CLOAK checkpoint: the twin-rendered robot mask (tinted) is sent with every frame">CLOAK</span></figcaption>
          <div class="imgbox"><img id="img-cam" alt="Pi camera"><div class="overlay" id="cam-err"></div></div>
        </figure>
        <figure class="pane">
          <figcaption>Observation sent to the model <span class="muted" id="obs-format">320×256, real camera</span> <span class="tag" id="obs-src">-</span></figcaption>
          <div class="imgbox"><img id="img-obs" alt="observation sent to the model"><div class="overlay" id="obs-err"></div></div>
          <div class="caption mono" id="obs-state"></div>
        </figure>
        <figure class="pane small" id="twin-pane" hidden>
          <figcaption>Estimated pose <span class="tag" id="est-label">-</span>
            <select id="twin-camera" aria-label="view"></select>
          </figcaption>
          <div class="imgbox"><img id="img-twin" alt="estimated pose"><div class="overlay" id="twin-err"></div></div>
        </figure>
      </section>

      <section class="strip" id="sensors">
        <div class="cell"><label>Accel (g)</label><b id="s-accel">-</b></div>
        <div class="cell"><label>Gyro (dps)</label><b id="s-gyro">-</b></div>
        <div class="cell"><label>Bucket pitch</label><b id="s-pitch">-</b><small id="s-mount"></small></div>
        <div class="cell"><label>Estimate pitch</label><b id="s-est">-</b><small id="s-estq"></small></div>
        <div class="cell"><label>Camera</label><b id="s-cam">-</b><small id="s-cam2"></small></div>
        <div class="cell"><label>IMU rate</label><b id="s-imuhz">-</b></div>
        <div class="cell"><label>Bridge</label><b id="s-bridge">-</b><small id="s-bridge2"></small></div>
      </section>

      <section class="card" id="twin-state-card">
        <h2>State estimate <span class="tag" id="ts-live">-</span> <span class="muted small" id="ts-age"></span>
          <select id="ts-camera" aria-label="twin view"></select></h2>
        <div class="tsgrid">
          <div class="imgbox ts"><img id="img-twin-state" class="stale" alt="the twin at the live joint estimate"><div class="stalemark" id="ts-stale">waiting for the estimate</div><div class="overlay" id="ts-err"></div></div>
          <div>
            <table class="tsjoints"><thead><tr><th>joint</th><th>angle</th><th>source</th><th>age</th></tr></thead><tbody id="ts-joints"></tbody></table>
            <p class="muted small" id="ts-note"></p>
            <p class="muted small" id="ts-imus"></p>
          </div>
        </div>
      </section>

      <section class="cols single">
        <div class="card" id="policy-card">
          <h2>Policy</h2>
          <p class="pointer" id="policy-pointer">Calibration: loading…</p>
          <form id="policy-form" autocomplete="off">
            <div class="grid">
              <label>Profile <select id="profile"></select></label>
              <label>Checkpoint <select id="checkpoint"><option value="">(loading)</option></select></label>
              <label class="wide">Lambda endpoint <input id="gateway" readonly placeholder="pick a checkpoint" spellcheck="false" title="SSH-forwarded CLOAK policy server (configs/hub/local_checkpoints.json)"></label>
              <label>Prompt group <select id="prompt-group"></select></label>
              <label class="wide">Prompt <input id="prompt" spellcheck="false"></label>
              <label data-motion>Chunk <input id="chunk" type="number" min="1" max="50" step="1"></label>
              <label data-motion>Hz <input id="hz" type="number" min="0.5" max="30" step="0.5"></label>
              <label data-motion>Max seconds <input id="max-seconds" type="number" min="0.1" max="900" step="1"></label>
              <label data-motion>Max steps <input id="max-steps" type="number" min="1" step="1" placeholder="none"></label>
              <label data-motion>Start pose <input id="start-pose" value="none" spellcheck="false" title="none, or slew,boom,stick,bucket in degrees"></label>

            </div>
            <div class="muted small" id="ckpt-info"></div>
            <div class="buttons">
              <button type="submit" id="btn-start" class="go btn-primary">Start</button>
              <button type="button" id="btn-stop">Stop</button>
              <button type="button" id="btn-reset" title="Stop any job, release every pin, clear the last run and put the policy settings back to their defaults">Reset</button>
              <button type="button" id="btn-refresh-ckpt">Refresh checkpoints</button>
            </div>
            <p class="muted small">Start drives the excavator (there is no dry run); Stop and Reset release every pin.</p>
          </form>
          <details class="advanced" id="endpoint-panel">
            <summary>Lambda endpoints</summary>
            <form id="endpoint-form" autocomplete="off">
              <div class="grid">
                <label>Label <input id="ep-label" placeholder="π0.5 · my run · step 900" maxlength="80" required></label>
                <label>Endpoint <input id="ep-url" placeholder="ws://127.0.0.1:8006" spellcheck="false" required></label>
                <label class="wide">Prompt <input id="ep-prompt" placeholder="scoop the dirt" spellcheck="false" required></label>
              </div>
              <p class="muted small">Start the SSH port forward to the Lambda box first. Add connects once, checks that the
                server is a 4-joint CLOAK π0.5 joint-target policy (10 actions at 10 Hz, calibrated joint space), then saves it to
                configs/hub/local_checkpoints.json.</p>
              <div class="buttons">
                <button type="submit" id="btn-ep-add">Check &amp; add</button>
                <button type="button" id="btn-ep-remove" disabled>Remove selected checkpoint</button>
              </div>
              <p class="small" id="ep-status" role="status"></p>
            </form>
          </details>
        </div>
      </section>

      <details class="card calpanel" id="cal-card">
        <summary><h2>Hardware calibration <span class="muted" id="cal-session"></span></h2>
          <span class="muted small">(boom, stick and bucket IMUs: joint state, ranges, speeds, base turn, pin signs)</span></summary>
        <div class="calsummary" id="cal-summary">loading...</div>
        <div class="autocard" id="calib-card">
          <div class="autogrid">
            <div class="prereq">
              <b>Before it moves (by hand)</b>
              <ul id="cal-prereqs"></ul>
              <label class="chk"><input type="checkbox" id="cal-prereqs-ok"> done: the machine is ready to move</label>
            </div>
            <div class="autobody">
              <p class="small" id="cal-motion"></p>
              <div class="buttons">
                <label class="chk" title="turn the base left and right (300 / 600 ms, each undone)"><input type="checkbox" id="cal-slew" checked> base turn</label>
                <label title="round trips per joint">Round trips <input id="cal-visits" type="number" min="1" max="5" step="1" value="2"></label>
                <button id="btn-cal-live" class="bigdanger" title="drives the machine and writes the hardware map">Calibrate</button>
                <button id="btn-cal-stop" disabled>Stop</button>
              </div>
              <div class="muted small" id="cal-why"></div>
              <div class="autosum" id="cal-last"></div>
            </div>
          </div>
        </div>
        <div class="calfoot">
          <div class="row"><b>Log</b> <button id="btn-report-show">Show the last run's log</button>
            <span class="muted small" id="cal-report-info"></span></div>
          <pre id="cal-report" class="mono" hidden></pre>
        </div>
      </details>

      <section class="card" id="run-card">
        <h2>Run <span class="muted" id="run-info">no job</span></h2>
        <div class="progress"><span id="p-phase">idle</span><div class="bar"><div id="p-bar"></div></div><span class="mono" id="p-time"></span></div>
        <div class="stats">
          <div><label>Step</label><b id="r-step">-</b></div>
          <div><label>Latency</label><b id="r-lat">-</b></div>
          <div><label>Dropped stale</label><b id="r-skip">-</b></div>
          <div><label>State</label><b id="r-state">-</b></div>
          <div><label>IMU - estimate</label><b id="r-imu">-</b></div>
        </div>
        <canvas id="pitch-chart" height="140" aria-label="IMU pitch vs estimate"></canvas>
        <div class="legend"><span class="sw est"></span>estimate pitch <span class="sw imu"></span>IMU pitch <span class="sw lat"></span>latency (ms, right)</div>
        <div class="two">
          <div>
            <h3>Last action chunk / targets</h3>
            <pre id="r-actions" class="mono">-</pre>
            <h3>Compiled pulses</h3>
            <table class="pulses" id="r-pulses"><thead><tr><th>phase</th><th>t (s)</th><th>joint</th><th>dir</th><th>GPIO</th><th>ms</th></tr></thead><tbody></tbody></table>
          </div>
          <div>
            <h3>Log</h3>
            <pre id="log" class="log mono"></pre>
          </div>
        </div>
      </section>

    </div>
  `;
  const $ = (id) => root.querySelector('#' + id);
  const timers = [];
  let disposed = false;
  const S = { config: null, state: null, ckpts: null, lastStep: null,
              chart: [], pulses: [], job: null, twinDirty: true, obsDirty: true };
  const deg = (r) => (r === null || r === undefined) ? '-' : (r * 180 / Math.PI).toFixed(1);
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d);
  const vec = (a, d = 2) => Array.isArray(a) ? a.map((x) => fmt(x, d)).join(' ') : '-';

  const api = (path, body) => ctx.api(path, body);   // GET json, or POST json; throws Error(error)

  function pill(id, cls, text) { const el = $(id); el.className = 'pill ' + (cls || ''); if (text !== undefined) el.textContent = text; }

  // ------------------------------------------------------------------ log
  function logLine(line, job) {
    const el = $('log');
    const div = document.createElement('div');
    const t = new Date().toLocaleTimeString();
    div.textContent = `${t} ${job ? '[' + job + '] ' : ''}${line}`;
    if (/ERROR|FAILED|refused|EMERGENCY|WATCHDOG/i.test(line)) div.className = 'err';
    else if (/WARNING/i.test(line)) div.className = 'warn';
    else if (/\[LIVE\]/.test(line)) div.className = 'live';
    const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    el.appendChild(div);
    while (el.childNodes.length > 1500) el.removeChild(el.firstChild);
    if (stick) el.scrollTop = el.scrollHeight;
  }

  // ------------------------------------------------------------------ images
  // Polls one image endpoint; paused while the view is hidden (ctx.active()) and stopped on unmount.
  function imageLoop(img, errEl, urlFn, minMs, shouldLoad) {
    let busy = false;
    const tick = async () => {
      if (busy || disposed) return;
      if (!ctx.active() || (shouldLoad && !shouldLoad())) { setTimeout(tick, minMs); return; }
      busy = true;
      const t0 = performance.now();
      try {
        const r = await fetch(ctx.url(urlFn()), { cache: 'no-store' });
        if (r.status === 204) { /* no new frame */ }
        else if (!r.ok) {
          let msg = `${r.status}`;
          try { msg = (await r.json()).error || msg; } catch { /* not json */ }
          errEl.textContent = msg; errEl.classList.add('show');
        } else {
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          const old = img.src;
          img.src = url;
          if (old && old.startsWith('blob:')) setTimeout(() => URL.revokeObjectURL(old), 1000);
          errEl.classList.remove('show');
          if (img.onframe) img.onframe(r);
        }
      } catch (e) { errEl.textContent = String(e); errEl.classList.add('show'); }
      busy = false;
      setTimeout(tick, Math.max(0, minMs - (performance.now() - t0)));
    };
    tick();
    return tick;
  }

  // ------------------------------------------------------------------ sensors
  function renderSensors(s) {
    if (!s) return;
    const pi = s.pi || {}, br = s.bridge || {};
    if (pi.ok === null || pi.ok === undefined) pill('pill-pi', '', 'Pi ?');
    else {
      const lk = pi.link || {};
      const label = pi.ok ? `Pi: ${lk.kind || ''} ${(lk.host || '').split(':')[0]}${lk.status_ms != null ? ' · ' + Math.round(lk.status_ms) + ' ms' : ''}` : 'Pi offline (all links)';
      pill('pill-pi', pi.ok ? ((pi.camera || {}).live && (pi.imu || {}).live ? 'ok' : 'warn') : 'bad', label);
      $('pill-pi').title = Object.entries(lk.probes || {}).map(([h, r]) => `${h}: ${r.ok ? r.ms + ' ms' : r.error}`).join('\n');
    }
    pill('pill-bridge', br.ok ? 'ok' : (br.ok === false ? 'bad' : ''), br.ok ? 'bridge' : 'bridge offline');
    pill('pill-esp32', br.ok ? (br.esp32 ? 'ok' : 'bad') : '', br.ok ? (br.esp32 ? 'ESP32 bridge' : 'ESP32 bridge offline') : 'ESP32 bridge ?');
    const imu = pi.imu || {}, cam = pi.camera || {};
    $('s-accel').textContent = vec(imu.accel_g, 3);
    $('s-gyro').textContent = vec(imu.gyro_dps, 1);
    $('s-pitch').textContent = imu.pitch_deg != null ? `${fmt(imu.pitch_deg, 1)}°` : (pi.ok ? (imu.error || (imu.live ? '-' : 'IMU not live')) : '-');
    $('s-mount').textContent = imu.pitch_source || '';
    $('s-imuhz').textContent = imu.live ? `${fmt(imu.hz, 0)} Hz, age ${fmt((imu.age_s || 0) * 1000, 0)} ms` : '-';
    $('s-cam').textContent = cam.live ? `${fmt(cam.fps, 1)} fps, age ${fmt((cam.age_s || 0) * 1000, 0)} ms` : (pi.ok ? (cam.error || 'camera not live') : (pi.error || '-'));
    $('s-cam2').textContent = pi.ok ? `proxy ${fmt(cam.proxy_fps, 1)} fps · Pi: ${(pi.link || {}).kind || ''} ${((pi.link || {}).host || '').split(':')[0]} · ${fmt(pi.rtt_ms, 0)} ms` : '';
    $('cam-info').textContent = cam.live ? `${fmt(cam.fps, 0)} fps` : '';
    const est = s.estimate;
    $('s-est').textContent = est ? `${fmt(est.pitch_deg, 1)}°` : '-';
    $('s-estq').textContent = est ? `boom ${deg(est.q.boom)}° bucket ${deg(est.q.bucket)}°` : 'no estimate yet';
    renderTwinState(s.twin_state);
    $('s-bridge').textContent = br.ok ? (br.esp32 ? 'ESP32 bridge online' : 'ESP32 bridge offline') : (br.error || '-');
    $('s-bridge2').textContent = br.host || '';
  }

  // ------------------------------------------------------------------ twin state estimate
  // The twin posed at the live joint estimate (/api/twin_state, carried on every sensors event).
  const SOURCES = { imu: 'IMU', dead_reckoned: 'dead-reckoned', welded: 'welded', locked: 'locked', assumed: 'assumed (standard pose)' };
  const SILENT_S = 1.5;                                   // no sensors event for this long: stale whatever it said
  const ageText = (s) => (s === null || s === undefined) ? '-' : (s < 1 ? `${Math.round(s * 1000)} ms` : `${fmt(s, 1)} s`);
  function renderTwinState(ts) {
    if (!ts) return;
    S.twinState = ts; S.twinStateAt = Date.now();
    const key = Object.values(ts.joints).map((j) => j.rad.toFixed(4)).join(',');
    if (key !== S.twinStateKey) { S.twinStateKey = key; S.twinStateDirty = true; }
    $('ts-joints').innerHTML = Object.entries(ts.joints).map(([n, j]) => {
      const fixed = j.source === 'welded' || j.source === 'locked';
      const old = !fixed && (j.age_s === null || j.age_s > ts.stale_s);
      return `<tr><td>${esc(n)}</td><td class="num">${fmt(j.deg, 1)}°</td><td><span class="src ${esc(j.source)}">${esc(SOURCES[j.source] || j.source)}</span></td>`
        + `<td class="num${old ? ' old' : ''}">${fixed ? 'fixed' : ageText(j.age_s)}</td></tr>`;
    }).join('');
    const a = ts.arm_imus || {};
    $('ts-note').textContent = (a.calibrated ? `Arm IMUs calibrated${a.session ? ' (' + a.session + ')' : ''}.` : `Arm IMUs not used: ${a.error || 'no calibration'}.`)
      + (ts.job_source ? ` Job estimate: ${ts.job_source}.` : '')
      + (ts.convention && ts.convention !== 'model' ? ' Angles: boom = underside above horizontal, stick = arm link above horizontal (0 = flat), as a phone level reads them.' : '');
    const imus = Object.entries(a.imus || {});
    $('ts-imus').textContent = imus.length ? 'Link IMUs: ' + imus.map(([n, i]) => `${n} ${i.live ? fmt(i.hz, 0) + ' Hz' : 'not live'} · ${ageText(i.age_s)}`).join(', ') : 'Link IMUs: none reported by the Pi';
    renderTwinLive();
  }
  function renderTwinLive() {                             // also on a timer, so a silent server reads as stale
    const ts = S.twinState;
    if (!ts) return;
    const silent = (Date.now() - S.twinStateAt) / 1000;
    const live = ts.live && silent < SILENT_S;
    const el = $('ts-live'); el.textContent = live ? 'live' : 'stale'; el.className = 'tag ' + (live ? 'sent' : 'assumed');
    const age = ts.age_s === null ? null : ts.age_s + silent;
    $('ts-age').textContent = (age === null ? 'not every moving joint has an estimate' : `age ${ageText(age)}`)
      + (silent >= SILENT_S ? ` · no update for ${fmt(silent, 0)} s` : '');
    $('ts-stale').hidden = live;
    $('ts-stale').textContent = age === null ? 'not live: no estimate (homed pose shown)' : `not live: ${ageText(age)} old`;
    $('img-twin-state').classList.toggle('stale', !live);
  }

  // ------------------------------------------------------------------ state
  function renderState(st) {
    if (!st) return;
    S.state = st;
    const job = st.job;
    if (job && (job.status === 'running' || job.status === 'starting')) {
      pill('pill-job', 'live', `${job.kind} running`);
    } else if (job) {
      pill('pill-job', job.status === 'done' ? '' : (job.status === 'stopped' ? 'warn' : 'bad'), `${job.kind} ${job.status}`);
    } else pill('pill-job', '', 'idle');
    if (job) $('run-info').textContent = `${job.id} · ${job.kind} · ${job.status}${job.error ? ' · ' + job.error : ''}`;
    const busy = !!st.busy;
    $('btn-start').disabled = busy;
    renderCalButtons();
    $('btn-stop').disabled = !busy;
    if (st.q_est) S.twinDirty = true;
    if (st.q_source) setEstLabel(st.q_source);
  }

  // ------------------------------------------------------------------ calibration
  const esc = (x) => String(x ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const ticks = (t) => esc(t).replace(/✓/g, '<span class="y">✓</span>').replace(/✗/g, '<span class="n">✗</span>');
  function runText(r, label) {
    if (!r) return `<div><b>${label}</b> <span class="muted">none yet</span></div>`;
    const kv = (o, u) => Object.entries(o || {}).map(([k, v]) => `${k} ${v == null ? '—' : fmt(v, u === 's' ? 2 : 0) + u}`).join(', ');
    const joints = Object.keys(r.ranges_deg || {}).length ? Object.entries(r.ranges_deg).map(([j, v]) => `${j} ${fmt(v, 1)}° (${kv((r.speeds_deg_s || {})[j], '°/s')}; onset ${kv((r.dead_time_s || {})[j], 's')})`).join(' · ') : '';
    const sl = r.slew ? (r.slew.error ? `base turn skipped: ${r.slew.error}` : `base left ${fmt(r.slew.left, 1)}°/s, right ${fmt(r.slew.right, 1)}°/s, sign ${r.slew.sign}`) : '';
    const rev = Object.entries(r.signs || {}).filter(([, v]) => v === -1).map(([j]) => j);
    return `<div><b>${label}</b> ${esc(r.at_text)} · <b>${esc(r.status)}</b>${r.in_map ? ' · in the hardware map' : ''}${r.error ? ' — ' + esc(r.error) : ''}`
      + (joints ? `<br>${esc(joints)}` : '') + (sl ? `<br>${esc(sl)}` : '') + (rev.length ? `<br><span class="n">reversed pins: ${esc(rev.join(', '))}</span>` : '')
      + (r.failures.length ? `<br><span class="n">${r.failures.map(esc).join('<br>')}</span>` : '')
      + (r.warnings.length ? `<br><span class="muted">${r.warnings.map(esc).join('<br>')}</span>` : '') + '</div>';
  }
  function renderCalibration(g) {
    if (!g || g.error || !g.summary) { $('cal-summary').textContent = g ? (g.error || 'unavailable') : 'unavailable'; return; }
    S.cal = g;
    $('cal-summary').innerHTML = ticks(g.summary.text);
    $('cal-session').textContent = g.map.arm_imus.session ? `session ${g.map.arm_imus.session}` : '(not calibrated)';
    $('cal-prereqs').innerHTML = g.manual_prerequisites.map((x) => `<li>${esc(x)}</li>`).join('');
    $('cal-motion').textContent = g.motion;
    $('cal-last').innerHTML = runText(g.last, 'Last run') + runText(g.last_sim, 'Last simulated run (CLI)');
    // Runs need calibrated, live arm IMUs (the runner refuses otherwise); the rest of the calibration is informational.
    const pp = $('policy-pointer'), imus = !!(g.map.arm_imus && g.map.arm_imus.calibrated);
    pp.className = 'pointer ' + (imus ? 'info' : 'warn');
    pp.textContent = imus
      ? `Arm IMUs calibrated${g.map.arm_imus.session ? ` (session ${g.map.arm_imus.session})` : ''}: runs read the joint state from them.`
      : 'Arm IMUs not calibrated: Start is refused until you run Hardware calibration (below).';
    renderCalButtons();
  }
  function renderCalButtons() {
    const busy = !!(S.state && S.state.busy); const tick = $('cal-prereqs-ok').checked;
    const running = busy && S.state.job && S.state.job.kind === 'calibrate';
    $('btn-cal-live').disabled = busy || !tick;
    $('btn-cal-stop').disabled = !running;
    $('cal-why').textContent = running ? 'calibration running: every line is in the log below' : busy ? 'another job is running'
      : !tick ? 'Calibrate needs the prerequisites tick.' : 'The button drives the machine and writes the hardware map.';
  }
  async function calJob() {
    const body = { slew: $('cal-slew').checked, visits: Number($('cal-visits').value) || 2, prereqs_ok: $('cal-prereqs-ok').checked };
    try {
      const r = await api('calibrate', body);
      logLine(`calibration started (${r.id})`, r.id);
    } catch (e) { logLine('calibration refused: ' + e.message); }
    renderCalButtons();
  }
  async function showReport() {
    try {
      const r = await api('calibration/report');
      const pre = $('cal-report'); pre.hidden = false; pre.textContent = r.text || r.error || 'no run yet';
      $('cal-report-info').textContent = r.run ? `${r.run.name} · ${r.run.status}` : '';
    } catch (e) { logLine('log: ' + e.message); }
  }
  async function loadCalibration() { try { renderCalibration(await api('calibration')); } catch (e) { renderCalibration({ error: e.message }); } }

  async function startJob(path, body) {
    try {
      const r = await api(path, body);
      logLine(`started ${r.kind} job ${r.id}`);
      S.chart = []; S.pulses = []; renderPulses(); drawChart();
    } catch (e) { logLine('refused: ' + e.message); }
  }

  // ------------------------------------------------------------------ policy
  function ckptLabel(r, c) { return c.label || r.label || `${r.name} · step ${c.step}`; }
  // Every Lambda-served checkpoint in configs/hub/local_checkpoints.json, grouped by run group, task or profile.
  function renderCheckpoints() {
    const sel = $('checkpoint'); const prev = sel.value; sel.innerHTML = '';
    const all = (S.ckpts && S.ckpts.runs) || []; const profiles = Object.keys((S.config && S.config.profiles) || {});
    S.ckptDetails = {};
    const groups = new Map();
    for (const r of all) for (const c of (r.checkpoints || [])) {
      const g = !profiles.includes(r.profile) ? 'other (no runner for this profile)'
        : r.group ? r.group : r.task ? `${r.task === 'bottlecap' ? 'bottle cap' : r.task} ablation` : r.profile;
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push({ r, c });
    }
    const n = [...groups.values()].reduce((a, x) => a + x.length, 0);
    const o0 = document.createElement('option'); o0.value = ''; o0.textContent = n ? '(choose a checkpoint)' : '(no checkpoints yet)'; sel.appendChild(o0);
    const order = [$('profile').value, ...profiles.filter((p) => p !== $('profile').value), ...[...groups.keys()].filter((g) => !profiles.includes(g))];
    for (const g of order) {
      const opts = groups.get(g); if (!opts) continue;
      opts.sort((a, b) => ((a.r.order ?? 0) - (b.r.order ?? 0)) || ((b.r.version || 0) - (a.r.version || 0)) || String(b.r.created_at || '').localeCompare(String(a.r.created_at || '')) || ((b.c.step || 0) - (a.c.step || 0)));
      const og = document.createElement('optgroup'); og.label = g.replace(/_/g, ' ');
      for (const { r, c } of opts) {
        const o = document.createElement('option');
        const runnable = profiles.includes(r.profile);
        o.value = c.gateway_url || ''; o.disabled = !c.gateway_url || !runnable;
        if (c.gateway_url && runnable) S.ckptDetails[c.gateway_url] = r;
        o.textContent = ckptLabel(r, c);
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    $('ckpt-info').textContent = S.ckpts ? `${S.ckpts.source}${S.ckpts.error ? ' · ' + S.ckpts.error : ''} · ${all.length} checkpoints` : '';
    const def = S.ckpts && S.ckpts.default;
    if (prev && S.ckptDetails[prev]?.profile === $('profile').value && [...sel.options].some((o) => o.value === prev && !o.disabled)) sel.value = prev;
    else if (def && def.profile === $('profile').value && !$('gateway').value) { sel.value = def.gateway_url; $('gateway').value = def.gateway_url; }
    updateCheckpointControls();
  }
  async function loadCheckpoints(refresh) {
    $('ckpt-info').textContent = 'loading checkpoints...';
    try { S.ckpts = await api('checkpoints' + (refresh ? '?refresh=1' : '')); } catch (e) { S.ckpts = { runs: [], source: 'error', error: e.message }; }
    renderCheckpoints();
  }
  function startPolicy(ev) {
    ev.preventDefault();
    const body = {
      profile: $('profile').value, gateway_url: $('gateway').value.trim(), prompt: $('prompt').value,
      chunk: Number($('chunk').value), hz: Number($('hz').value), max_seconds: Number($('max-seconds').value),
      max_steps: $('max-steps').value ? Number($('max-steps').value) : null,
      start_pose: $('start-pose').value.trim() || 'none',
    };
    startJob('policy/start', body);
  }


  // A CLOAK checkpoint (pi05_tabletop_excavator, mode other than baseline / pretrained) always gets the twin-rendered robot mask: the Pi
  // camera shows it tinted at the live estimate, tagged CLOAK.
  function cloakMasked() {
    const r = (S.ckptDetails || {})[$('gateway').value.trim()];
    return !!r && r.profile === 'pi05_tabletop_excavator' && !!r.mode && !['baseline', 'pretrained'].includes(r.mode);
  }

  function updateCheckpointControls() {
    $('cam-cloak').hidden = !cloakMasked();
    const entry = (S.ckptDetails || {})[$('gateway').value.trim()];
    $('btn-ep-remove').disabled = !entry;
    $('obs-format').textContent = '224×224, real camera';
  }
  async function addEndpoint(ev) {
    ev.preventDefault();
    const body = { label: $('ep-label').value, gateway_url: $('ep-url').value.trim(), prompt: $('ep-prompt').value || $('prompt').value };
    $('btn-ep-add').disabled = true; $('ep-status').textContent = `Connecting to ${body.gateway_url}…`;
    try {
      const e = await api('checkpoints/add', body);
      $('ep-status').textContent = `Added ${e.label} (${e.gateway_url}).`;
      $('ep-label').value = ''; $('ep-url').value = '';
      await loadCheckpoints(true);
      $('checkpoint').value = e.gateway_url; $('gateway').value = e.gateway_url; updateCheckpointControls();
    } catch (err) { $('ep-status').textContent = err.message; }
    finally { $('btn-ep-add').disabled = false; }
  }
  async function removeEndpoint() {
    const entry = (S.ckptDetails || {})[$('gateway').value.trim()];
    if (!entry || !confirm(`Remove ${entry.label} (${$('gateway').value.trim()}) from the checkpoint list?`)) return;
    try {
      await api('checkpoints/remove', { checkpoint_id: entry.key });
      $('ep-status').textContent = `Removed ${entry.label}.`;
      $('gateway').value = ''; await loadCheckpoints(true);
    } catch (err) { $('ep-status').textContent = err.message; }
  }

  // ------------------------------------------------------------------ run stream
  function renderStep(d) {
    S.lastStep = d;
    $('r-step').textContent = d.step;
    $('r-lat').textContent = `${fmt(d.latency_ms, 0)} ms`;
    $('r-skip').textContent = `${d.skipped_total} (last ${d.n_skipped})`;
    $('r-state').textContent = vec(d.state, 3);
    $('r-imu').textContent = d.imu_minus_est_deg === null || d.imu_minus_est_deg === undefined ? '-' : `${fmt(d.imu_minus_est_deg, 1)}°`;
    const acts = (d.actions || []).map((a, i) => `a${i}: [${vec(a, 3)}]`).join('\n');
    const tg = (d.targets || []).map((q) => `[${q.map((x) => deg(x)).join(', ')}]°`).join(' ');
    $('r-actions').textContent = `${acts}\ntargets: ${tg}\npulses ${d.n_pulses}, schedule ${fmt(d.schedule_s, 2)} s, server ${d.server_ms ?? '-'} ms`;
    S.chart.push({ est: (d.est_pitch_rad || 0) * 180 / Math.PI, imu: d.imu_pitch_rad === null ? null : d.imu_pitch_rad * 180 / Math.PI, lat: d.latency_ms });
    if (S.chart.length > 300) S.chart.shift();
    drawChart();
    S.obsDirty = true;
  }
  function renderPulses() {
    const tb = $('r-pulses').querySelector('tbody');
    tb.innerHTML = S.pulses.slice(-120).map((p) => `<tr><td>${p.phase}${p.step !== null && p.step !== undefined ? ' ' + p.step : ''}</td><td>${fmt(p.t_start_s, 3)}</td><td>${p.joint}</td><td>${p.direction > 0 ? '+' : '-'}</td><td>${p.gpio}</td><td>${p.duration_ms}</td></tr>`).join('');
    tb.scrollTop = tb.scrollHeight;
  }
  function drawChart() {
    const cv = $('pitch-chart'); const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight; cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
    const css = getComputedStyle(root), tok = (n, d) => css.getPropertyValue(n).trim() || d;
    const cMuted = tok('--muted', '#66706a'), cLine = tok('--line', '#d3d7d0');
    const pts = S.chart; if (!pts.length) { g.fillStyle = cMuted; g.font = `12px ${tok('--font-body', 'sans-serif')}`; g.fillText('pitch trace appears when a policy run streams steps', 10, 20); return; }
    const vals = pts.flatMap((p) => [p.est, p.imu]).filter((v) => v !== null && v !== undefined);
    let lo = Math.min(...vals), hi = Math.max(...vals); if (hi - lo < 5) { const m = (hi + lo) / 2; lo = m - 2.5; hi = m + 2.5; }
    const latMax = Math.max(100, ...pts.map((p) => p.lat || 0));
    const X = (i) => 36 + (w - 76) * (pts.length === 1 ? 1 : i / (pts.length - 1));
    const Y = (v) => 8 + (h - 22) * (1 - (v - lo) / (hi - lo));
    const YL = (v) => 8 + (h - 22) * (1 - v / latMax);
    g.strokeStyle = cLine; g.lineWidth = 1; g.fillStyle = cMuted; g.font = '10px ui-monospace, monospace';
    for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4, y = Y(v); g.beginPath(); g.moveTo(36, y); g.lineTo(w - 40, y); g.stroke(); g.fillText(v.toFixed(0) + '°', 2, y + 3); }
    g.fillText(latMax.toFixed(0) + 'ms', w - 38, 14);
    const line = (key, color, Yf) => { g.strokeStyle = color; g.lineWidth = 1.6; g.beginPath(); let on = false;
      pts.forEach((p, i) => { const v = p[key]; if (v === null || v === undefined) { on = false; return; } const x = X(i), y = Yf(v); if (!on) { g.moveTo(x, y); on = true; } else g.lineTo(x, y); }); g.stroke(); };
    line('lat', tok('--line-strong', '#b5bbb3'), YL); line('est', tok('--info', '#2f5d8a'), Y); line('imu', tok('--warn', '#a86f00'), Y);
  }

  function setEstLabel(src) {
    const el = $('est-label'); el.textContent = src || '-';
    el.className = 'tag' + (src && src.startsWith('assumed') ? ' assumed' : '');
  }
  function renderProgress(d) {
    const bar = $('p-bar');
    if (!d) { $('p-phase').textContent = 'idle'; $('p-time').textContent = ''; bar.style.width = '0'; bar.classList.remove('busy'); return; }
    S.twinDirty = true;
    if (d.estimate) setEstLabel(d.estimate);
    const names = { setup: 'connecting', start_pose: 'moving to start pose',
      infer: 'waiting for inference', step: 'step', policy: 'executing pulses' };
    $('p-phase').textContent = (names[d.phase] || d.phase) + (d.step !== null && d.step !== undefined ? ` ${d.step}` : '');
    if (d.planned_s) {
      bar.classList.remove('busy'); bar.style.width = Math.min(100, 100 * d.elapsed_s / d.planned_s) + '%';
      $('p-time').textContent = `${fmt(d.elapsed_s, 1)} / ${fmt(d.planned_s, 1)} s`;
    } else { bar.classList.add('busy'); $('p-time').textContent = `${fmt(d.elapsed_s, 1)} s`; }
  }

  // ------------------------------------------------------------------ events
  // The SSE stream stays open while the view is hidden: a job e-stops when no page is connected.
  // EventSource retries a dropped stream by itself, but not a non-200 answer (the app restarting: 503).
  function connect() {
    if (S.es) S.es.close();
    const es = S.es = ctx.events('events');
    es.onopen = () => pill('pill-sse', 'ok', 'events');
    es.onerror = () => {
      pill('pill-sse', 'bad', 'events lost');
      if (es.readyState === EventSource.CLOSED && !disposed) setTimeout(() => { if (S.es === es && !disposed) connect(); }, 2000);
    };
    es.onmessage = (m) => {
      let ev; try { ev = JSON.parse(m.data); } catch { return; }
      const d = ev.data;
      switch (ev.type) {
        case 'hello': renderState(d.state); renderSensors(d.sensors); break;
        case 'state': renderState(d); break;
        case 'sensors': renderSensors(d); break;
        case 'log': logLine(d.line, d.job); break;
        case 'step': renderStep(d); break;
        case 'pulses': for (const p of d.pulses) S.pulses.push({ ...p, phase: d.phase, step: d.step }); if (S.pulses.length > 400) S.pulses = S.pulses.slice(-400); renderPulses(); break;
        case 'estimate': S.twinDirty = true; setEstLabel(d.source); break;
        case 'progress': renderProgress(d); break;
        case 'start_pose': logLine(`start pose: ${d.n_pulses} pulses compiled (${d.estimate})`, d.job); break;
        case 'job_end': renderProgress(null); logLine(`job ${d.id} ${d.status}${d.error ? ': ' + d.error : ''}`, d.id); loadCalibration(); if (d.kind === 'calibrate' && !$('cal-report').hidden) showReport(); S.obsDirty = true; break;
        case 'calibration': renderCalibration(d); break;
        case 'estop': logLine(d.phase === 'begin' ? `EMERGENCY STOP: ${d.reason}` : `EMERGENCY STOP done: bridge release ${d.bridge_release && d.bridge_release.ok ? 'acknowledged' : JSON.stringify(d.bridge_release)}`); break;
        case 'release': logLine('bridge release-all: ' + JSON.stringify(d)); break;
        default: break;
      }
    };
  }

  // ------------------------------------------------------------------ init
  async function init() {
    try { S.config = await api('config'); } catch (e) { logLine('config: ' + e.message); return; }
    const c = S.config;
    $('profile').innerHTML = Object.keys(c.profiles).map((p) => `<option value="${p}">${c.profiles[p].label || p}</option>`).join('');
    $('prompt-group').innerHTML = Object.keys(promptGroups()).map((g) => `<option value="${g}">${g}</option>`).join('');
    $('prompt-group').onchange = () => { $('prompt').value = promptGroups()[$('prompt-group').value]; };
    applyDefaults();
    $('twin-camera').innerHTML = c.twin_cameras.map((x) => `<option>${x}</option>`).join('');
    $('twin-camera').onchange = () => { S.twinDirty = true; };
    $('ts-camera').innerHTML = c.twin_cameras.map((x) => `<option>${x}</option>`).join('');
    $('ts-camera').onchange = () => { S.twinStateDirty = true; };

    $('btn-cal-live').onclick = calJob;
    $('btn-cal-stop').onclick = async () => { try { await api('stop', {}); logLine('calibration: stop sent (the routine stops at the next press and releases, then release-all)'); } catch (e) { logLine('stop: ' + e.message); } };
    $('cal-prereqs-ok').onchange = renderCalButtons;
    $('btn-report-show').onclick = showReport;
    $('policy-form').onsubmit = startPolicy;
    $('btn-stop').onclick = async () => { try { await api('stop', {}); } catch (e) { logLine('stop: ' + e.message); } };
    $('btn-refresh-ckpt').onclick = () => loadCheckpoints(true);
    $('profile').onchange = () => {
      $('gateway').value = '';
      renderCheckpoints();
    };
    $('checkpoint').onchange = () => {
      const url = $('checkpoint').value;
      if (url) {
        $('gateway').value = url;
        const r = (S.ckptDetails || {})[url];
        if (r && r.prompt) $('prompt').value = r.prompt;
        if (r && r.profile && r.profile !== $('profile').value) {
          $('profile').value = r.profile;
          renderCheckpoints(); $('checkpoint').value = url; $('gateway').value = url;
        }
      }
      updateCheckpointControls();
    };
    $('gateway').addEventListener('input', updateCheckpointControls);    // a typed URL can be a CLOAK checkpoint too
    $('endpoint-form').onsubmit = addEndpoint;
    $('btn-ep-remove').onclick = removeEndpoint;
    timers.push(setInterval(renderTwinLive, 500));
    window.addEventListener('resize', drawChart);

    $('img-obs').onframe = (r) => {
      const src = r.headers.get('X-Observation') || ''; const sent = src.startsWith('sent');
      const cloak = sent && /CLOAK (on|off)/.exec(src);
      $('obs-src').textContent = !sent ? 'preview' : cloak ? (cloak[1] === 'on' ? 'sent · CLOAK on' : 'sent · CLOAK off') : 'sent';
      $('obs-src').className = 'tag' + (!sent ? '' : cloak && cloak[1] === 'on' ? ' cloak' : ' sent');
      $('obs-state').textContent = sent ? src.replace(/^sent; /, '') : 'no run yet: live frame with the training preprocessing (nothing sent)';
    };
    $('img-cam').onframe = (r) => { const c = r.headers.get('X-Cloak'); $('cam-cloak').textContent = c ? `CLOAK · ${c}` : 'CLOAK'; };
    imageLoop($('img-cam'), $('cam-err'), () => 'camera.jpg' + (cloakMasked() ? '?mask=1' : ''), 100);
    let lastTwin = 0;
    imageLoop($('img-twin'), $('twin-err'), () => { S.twinDirty = false; lastTwin = Date.now(); return 'twin_frame?camera=' + encodeURIComponent($('twin-camera').value); }, 150,
      () => !$('twin-pane').hidden && (S.twinDirty || Date.now() - lastTwin > 3000));   // pane hidden: no renders
    let lastTwinState = 0;
    imageLoop($('img-twin-state'), $('ts-err'), () => { S.twinStateDirty = false; lastTwinState = Date.now(); return 'twin_state.jpg?camera=' + encodeURIComponent($('ts-camera').value); }, 150,
      () => S.twinStateDirty || Date.now() - lastTwinState > 5000);   // re-render only when the pose changed
    let lastObs = 0;
    imageLoop($('img-obs'), $('obs-err'), () => { S.obsDirty = false; lastObs = Date.now(); return 'observation.jpg'; }, 200,
      () => S.obsDirty || Date.now() - lastObs > 1000);

    connect();
    drawChart();
    loadCalibration(); loadCheckpoints(false);
  }

  // The policy settings as a fresh page shows them (from the config route).
  function promptGroups() { return { default: S.config.default_prompt, ...S.config.group_prompts }; }
  function applyDefaults() {
    const c = S.config;
    $('profile').value = c.default_profile;
    $('prompt-group').value = c.group_prompts.dig ? 'dig' : 'default';
    $('prompt').value = promptGroups()[$('prompt-group').value];
    $('chunk').value = c.defaults.chunk; $('hz').value = c.defaults.hz; $('max-seconds').value = c.defaults.max_seconds;
    $('max-steps').value = ''; $('start-pose').value = c.defaults.start_pose;
  }

  // Reset: the server e-stops (every pin released) and forgets the last job and observation; the page
  // clears the run views and puts the policy settings back to their defaults (no checkpoint chosen).
  async function reset() {
    const b = $('btn-reset'); b.disabled = true;
    try {
      const r = await api('reset', {});
      logLine(r.cleared ? 'reset: policy back to its starting state' : 'reset: pins released; the job is still stopping');
    } catch (e) { logLine('reset: ' + e.message); }
    b.disabled = false;
    S.chart = []; S.pulses = []; renderPulses(); drawChart(); renderProgress(null);
    S.twinDirty = S.obsDirty = true;
    if (S.config) {
      applyDefaults();
      $('gateway').value = ''; renderCheckpoints(); $('checkpoint').value = ''; updateCheckpointControls();
    }
  }

  ctx.onShow(() => { S.twinDirty = S.twinStateDirty = S.obsDirty = true; drawChart(); });
  $('btn-reset').onclick = reset;

  await init();
  return () => {
    disposed = true;
    timers.forEach((t) => clearInterval(t));
    window.removeEventListener('resize', drawChart);
    if (S.es) S.es.close();
  };
}
