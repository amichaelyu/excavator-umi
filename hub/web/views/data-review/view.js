// Excavator data review (hub app "data-review": hub/apps/data_review/__init__.py,
// hub/apps/data_review/core.py). One master clock in episode seconds drives the two panes.
//   camera video.mp4   : t -> t + video_offset_s                    (VFR pts = raw camera time - first frame)
//   replay*.mp4 / fk   : source_t_s -> replay t_s; legacy exports use t * time_scale
// These mirror the functions in hub/apps/data_review/core.py (unit-tested there).
//
// All state lives inside mount(). The source (policy config) picker comes from the index API. Deep link: ?episode=<id>.

const RESYNC_MS = 500, DRIFT_S = 0.06, STEP_S = 0.05, INDEX_REFRESH_MS = 15000;
const JOINTS = ["slew", "boom", "stick", "bucket"];   // the twin's joints; the server sends the list too
const SIM_CAMERA = "overview";                    // the replay.mp4 camera
const MARK = {good: "✓", bad: "✗", skip: "–"};

const PAGE = `
<header class="ui-header">
  <span class="ui-mark" aria-hidden="true"></span>
  <h1 class="ui-title">Data review</h1>
  <div class="ui-header-end">
    <label class="field" id="source-field" hidden>Config
      <select id="source"></select>
    </label>
    <label class="field">Episode
      <select id="episode"></select>
    </label>
  </div>
</header>

<div class="main">
  <section class="panes">
    <figure class="pane" id="camera-pane">
      <figcaption>
        <span class="title">Camera</span>
        <select id="camera-source" aria-label="Camera source">
          <option value="original">Original video</option>
          <option value="slam">ORB-SLAM3</option>
        </select>
      </figcaption>
      <div class="stage">
        <video id="cam-video" muted playsinline preload="auto"></video>
        <canvas id="cam-slam" hidden aria-label="ORB-SLAM3 camera path, top and side views"></canvas>
        <div class="stage-msg" id="cam-msg" hidden></div>
      </div>
    </figure>

    <figure class="pane" id="sim-pane">
      <figcaption>
        <span class="title">MuJoCo twin</span>
      </figcaption>
      <div class="stage">
        <video id="sim-video" muted playsinline preload="auto"></video>
        <img id="sim-img" alt="Server-rendered MuJoCo frame" hidden>
        <div class="stage-msg" id="sim-msg" hidden></div>
      </div>
    </figure>
  </section>

  <section class="transport" aria-label="Playback">
    <button id="play" type="button" class="primary btn-primary" aria-label="Play">Play</button>
    <input id="scrub" type="range" min="0" max="1" step="0.001" value="0" aria-label="Episode time">
    <span id="time" class="mono">0.00 / 0.00 s</span>
    <label class="field">Speed
      <select id="speed">
        <option>0.25</option><option>0.5</option><option>0.75</option>
        <option selected>1</option><option>1.5</option><option>2</option>
      </select>
    </label>
    <label class="check"><input type="checkbox" id="loop"> Loop</label>
  </section>

  <section class="timeline-wrap" aria-label="ORB-SLAM3 tracking timeline">
    <canvas id="timeline" height="28"></canvas>
    <div class="legend">
      <span><i class="sw tracked"></i>tracked</span>
      <span><i class="sw imu_only"></i>imu_only</span>
      <span><i class="sw interpolated"></i>interpolated</span>
      <span class="muted" id="sync-info"></span>
    </div>
  </section>

  <div class="info-row">
  <section class="annotate" id="annotate" aria-label="Annotate episode">
    <div class="ann-head">
      <h2>Annotate</h2>
      <span id="ann-state" class="ann-state">unlabeled</span>
    </div>
    <label class="stack">Task prompt (default)
      <input id="ann-default-prompt" type="text" maxlength="512" spellcheck="true" placeholder="scoop the dirt">
    </label>
    <details class="groups" id="ann-groups">
      <summary>Group prompts (override the default)</summary>
      <label class="stack">dig
        <input id="ann-group-dig" type="text" maxlength="512" data-group="dig" placeholder="(uses the default)">
      </label>
      <label class="stack">ball
        <input id="ann-group-ball" type="text" maxlength="512" data-group="ball" placeholder="(uses the default)">
      </label>
    </details>
    <div class="progress" aria-hidden="true"><div id="ann-bar-good" class="bar good"></div><div id="ann-bar-bad" class="bar bad"></div><div id="ann-bar-skip" class="bar skip"></div></div>
    <p id="ann-summary" class="ann-summary mono"><span id="ann-summary-main">-</span><span id="ann-summary-ready"></span><span id="ann-summary-prompts"></span></p>
    <div id="ann-overall" class="accuracy" hidden>
      <p class="ann-summary" title="IK path / actuated twin vs SLAM path. Ranges: thirds of the per-episode RMSE (best third good, worst third bad)">IK accuracy to SLAM data (RMSE)</p>
      <table class="mono">
        <thead><tr><th></th><th colspan="2">IK</th><th colspan="2">twin</th></tr>
          <tr><th></th><th>tip mm</th><th>angle °</th><th>tip mm</th><th>angle °</th></tr></thead>
        <tbody id="acc-body"></tbody>
      </table>
    </div>
    <div class="verdicts" role="group" aria-label="Verdict">
      <button type="button" class="verdict good" data-verdict="good" aria-keyshortcuts="G">Good <kbd>G</kbd></button>
      <button type="button" class="verdict bad" data-verdict="bad" aria-keyshortcuts="B">Bad <kbd>B</kbd></button>
    </div>
    <button type="button" id="ann-prev" class="next" aria-keyshortcuts="P">Previous <kbd>P</kbd></button>
    <button type="button" id="ann-next" class="next" aria-keyshortcuts="N">Next <kbd>N</kbd></button>
    <label class="stack">Episode prompt <span id="ann-prompt-src" class="muted"></span>
      <textarea id="ann-prompt" rows="2" maxlength="512" placeholder="(group / default prompt)"></textarea>
      <span id="ann-resolved" class="resolved mono" aria-live="polite"></span>
    </label>
  </section>

  <section class="readout" aria-label="State at t">
    <dl>
      <div><dt>t (episode)</dt><dd id="r-t" class="mono">-</dd></div>
      <div><dt>ORB-SLAM3 tracking</dt><dd id="r-status">-</dd></div>
      <div><dt id="r-q-label">boom / bucket</dt><dd id="r-q" class="mono">-</dd></div>
      <div><dt>tip xyz (model, mm)</dt><dd id="r-tip" class="mono">-</dd></div>
      <div><dt>tip error vs target</dt><dd id="r-err" class="mono">-</dd></div>
      <div><dt>IK path vs SLAM path (episode MSE / RMSE)</dt><dd id="r-mse-ik" class="mono">-</dd></div>
      <div><dt>actuated twin vs SLAM path (episode MSE / RMSE)</dt><dd id="r-mse-twin" class="mono">-</dd></div>
      <div><dt>IK path vs SLAM path (overall, all episodes)</dt><dd id="r-all-ik" class="mono">-</dd></div>
      <div><dt>actuated twin vs SLAM path (overall, all episodes)</dt><dd id="r-all-twin" class="mono">-</dd></div>
      <div><dt>retarget scale / time_scale</dt><dd id="r-scale" class="mono">-</dd></div>
      <div><dt>retarget yaw / heading source</dt><dd id="r-heading" class="mono">-</dd></div>
      <div><dt>label (task)</dt><dd id="r-label">-</dd></div>
      <div><dt>mount: bucket point in camera (x right, y down, z optical)</dt><dd id="r-mount" class="mono">-</dd></div>
    </dl>
  </section>
  </div>

  <section class="plots">
    <figure><img id="joints-png" alt="Joint angle plot for this episode"><figcaption>joints.png: IK vs command vs dynamic joint angles</figcaption></figure>
  </section>
</div>
<div id="toast" class="toast" role="status" aria-live="polite" hidden></div>
`;

// ------------------------------------------------------------------ pure helpers
function interp(x, xs, ys) {
  const n = xs.length;
  if (!n) return 0;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const f = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
  return ys[lo] + f * (ys[hi] - ys[lo]);
}
function lastIndexLE(xs, x) {
  let lo = 0, hi = xs.length - 1, ans = 0;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (xs[m] <= x) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}
function fmt(x, d = 2) { return Number.isFinite(x) ? x.toFixed(d) : "-"; }
// "20260927T095829Z-3d0c0ed4" -> "0958·3d0c0ed4"
function shortId(id) {
  const m = /^\d{8}T(\d{4})\d{2}Z-([0-9a-f]+)$/i.exec(id);
  return m ? `${m[1]}·${m[2]}` : id;
}
export async function mount(root, ctx) {
  ctx.css("style.css");
  root.innerHTML = PAGE;
  const $ = (id) => root.querySelector("#" + id);
  const $$ = (sel) => root.querySelectorAll(sel);
  let disposed = false;

  const S = {
    index: null, ep: null, t: 0, playing: false, speed: 1, wall0: 0, t0: 0,
    camSource: "original", mode: "twin", serverRender: false,
    source: null,                                 // source filter (index.sources: one per policy config)
    lastResync: 0, drift: {cam: null, sim: null}, overall: null,
  };

  // ------------------------------------------------------------------ mapping
  function originalTime(t) { return t + (S.ep.video_offset_s || 0); }
  function replayTime(t) {
    const c = S.ep.replay_clock;
    return c ? interp(t, c.source_t_s, c.replay_t_s) : t * S.ep.time_scale;
  }
  function replayRate(t) {
    const c = S.ep.replay_clock;
    if (!c) return S.ep.time_scale;
    if (t < c.source_t_s[0] || t >= c.source_t_s.at(-1)) return 0;
    const i = Math.min(lastIndexLE(c.source_t_s, t), c.source_t_s.length - 2);
    return (c.replay_t_s[i + 1] - c.replay_t_s[i]) / (c.source_t_s[i + 1] - c.source_t_s[i]);
  }

  // Followers: a <video> plus how episode time maps onto it.
  function camFollower() {
    return S.camSource === "slam" ? null : {el: $("cam-video"), map: originalTime, rate: 1, key: "cam"};
  }
  function simFollower() {
    return S.serverRender ? null : {el: $("sim-video"), map: replayTime, rate: replayRate(S.t), key: "sim"};
  }
  const followers = () => [camFollower(), simFollower()].filter(Boolean);

  function clampToVideo(el, v) {
    const d = Number.isFinite(el.duration) ? el.duration : Infinity;
    return Math.max(0, Math.min(v, d - 0.001));
  }
  function seekFollower(f) {
    const target = clampToVideo(f.el, f.map(S.t) + 0.001);
    if (Math.abs(f.el.currentTime - target) > 1e-3) f.el.currentTime = target;
  }
  function playFollower(f) {
    if (f.rate <= 0) { f.el.pause(); seekFollower(f); return; }
    f.el.playbackRate = Math.max(0.0625, Math.min(16, S.speed * f.rate));
    seekFollower(f);
    const past = Number.isFinite(f.el.duration) && f.map(S.t) >= f.el.duration - 0.01;
    if (!past) f.el.play().catch(() => {});
  }

  // ------------------------------------------------------------------ clock
  function now() { return S.playing ? S.t0 + (performance.now() - S.wall0) / 1000 * S.speed : S.t; }
  function play() {
    if (!S.ep) return;
    if (S.t >= S.ep.duration_s - 1e-3) S.t = 0;
    S.playing = true; S.t0 = S.t; S.wall0 = performance.now(); S.lastResync = performance.now();
    followers().forEach(playFollower);
    $("play").textContent = "Pause"; $("play").setAttribute("aria-label", "Pause");
    if (S.serverRender) requestSimFrame();
  }
  function pause() {
    S.t = S.ep ? Math.min(now(), S.ep.duration_s) : now(); S.playing = false;
    [$("cam-video"), $("sim-video")].forEach((v) => v.pause());
    if (S.ep) followers().forEach(seekFollower);
    $("play").textContent = "Play"; $("play").setAttribute("aria-label", "Play");
    if (S.serverRender) requestSimFrame();
    render();
  }
  function seek(t) {
    if (!S.ep) return;
    S.t = Math.max(0, Math.min(t, S.ep.duration_s));
    if (S.playing) { S.t0 = S.t; S.wall0 = performance.now(); followers().forEach(playFollower); }
    else followers().forEach(seekFollower);
    if (S.serverRender) requestSimFrame();
    render();
  }
  function setSpeed(v) {
    S.t = now(); S.t0 = S.t; S.wall0 = performance.now(); S.speed = v;
    if (S.playing) followers().forEach(playFollower);
  }

  function resync() {
    for (const f of followers()) {
      const want = clampToVideo(f.el, f.map(S.t));
      const drift = f.el.currentTime - want;
      S.drift[f.key] = drift;
      if (f.el.seeking) continue;
      const past = Number.isFinite(f.el.duration) && f.map(S.t) >= f.el.duration - 0.01;
      if (past) { if (!f.el.paused) f.el.pause(); continue; }
      if (Math.abs(drift) > DRIFT_S) f.el.currentTime = want;
      if (f.el.paused) playFollower(f);
    }
  }

  function tick() {
    if (disposed) return;
    if (S.playing && S.ep) {
      S.t = now();
      for (const f of followers()) {
        if (f.rate <= 0) { f.el.pause(); seekFollower(f); continue; }
        const rate = Math.max(0.0625, Math.min(16, S.speed * f.rate));
        if (Math.abs(f.el.playbackRate - rate) > 0.001) f.el.playbackRate = rate;
        if (f.el.paused) playFollower(f);
      }
      if (S.t >= S.ep.duration_s) {
        if ($("loop").checked) { seek(0); }
        else pause();
      }
      if (performance.now() - S.lastResync > RESYNC_MS) { S.lastResync = performance.now(); resync(); }
      render();
    }
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------------ readout + timeline
  function render() {
    const e = S.ep;
    if (!e) return;
    const t = S.t;
    $("scrub").value = String(t);
    $("time").textContent = `${fmt(t)} / ${fmt(e.duration_s)} s`;
    $("r-t").textContent = `${fmt(t, 3)} s  (replay ${fmt(replayTime(t), 3)} s)`;
    const st = e.status[lastIndexLE(e.frame_t, t)] || "-";
    const rs = $("r-status"); rs.textContent = st; rs.className = `status-${st}`;
    const fk = e.fk.twin;
    if (fk) {
      const r = replayTime(t);
      const q = fk._q.map((qi) => interp(r, fk.t, qi) * 180 / Math.PI);
      $("r-q-label").textContent = (fk.joints || e.joints || JOINTS).join(" / ");
      $("r-q").textContent = q.map((x) => `${x >= 0 ? "+" : ""}${x.toFixed(1)}°`).join("  ");
      const tip = [0, 1, 2].map((i) => interp(r, fk.t, fk._tip[i]) * 1000);
      $("r-tip").textContent = tip.map((x) => x.toFixed(0)).join(", ");
      $("r-err").textContent = `${fmt(interp(r, fk.t, fk.tip_err) * 1000, 1)} mm`;
    }
    const pe = e.path_error, fmtErr = (x) => !x ? "-"
      : `tip ${fmt(x.tip_mse_mm2, 1)} mm² / ${fmt(x.tip_rmse_mm, 1)} mm`
        + (x.tip_rmse_umi_mm != null ? ` (${fmt(x.tip_rmse_umi_mm, 1)} mm UMI)` : "")
        + (x.angle_rmse_deg != null ? ` · bucket angle ${fmt(x.angle_mse_deg2, 2)} deg² / ${fmt(x.angle_rmse_deg, 2)}°` : "");
    $("r-mse-ik").textContent = fmtErr(pe?.ik);
    renderAccuracy();
    $("r-mse-twin").textContent = pe?.twin ? fmtErr(pe.twin) : (pe ? "not recorded (re-solved IK only)" : "-");
    $("r-scale").textContent = `${fmt(e.retarget.scale, 3)} × / ${fmt(e.time_scale, 2)}`;
    const yaw = e.yaw_deg ?? e.retarget.yaw_deg;
    $("r-heading").textContent = `${fmt(yaw, 1)}°  ${e.heading_source || e.retarget.heading_source || "(not recorded)"}`;
    const mt = e.mount;
    $("r-mount").textContent = mt
      ? `[${(mt.bucket_point_in_camera_m || []).map((x) => (x * 1000).toFixed(0)).join(", ")}] mm  q[${(mt.rotation_quat_wxyz || []).map((x) => Number(x).toFixed(3)).join(", ")}]  ${mt.measured === false ? "UNMEASURED " : ""}${mt.mount_source || ""}`
      : "-";
    $("r-label").textContent = `${e.group || "dig"}${e.task ? "  (" + e.task + ")" : ""}`;
    const d = S.drift;
    const ms = (x) => (x == null ? "-" : `${(x * 1000).toFixed(0)} ms`);
    $("sync-info").textContent = `drift camera ${ms(d.cam)}, twin ${ms(d.sim)}`
      + (S.camSource === "original" && !e.original_video_synced ? " (original video unsynced)" : "");
    drawTimeline();
    if (S.camSource === "slam") drawSlam();
  }

  const COLORS = {};
  function drawTimeline() {
    const c = $("timeline"), e = S.ep;
    if (!e || root.hidden) return;
    const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const css = getComputedStyle(root);
    for (const k of ["tracked", "imu_only", "interpolated"]) COLORS[k] = css.getPropertyValue(`--${k}`).trim();
    const x = (t) => (t / e.duration_s) * w;
    const n = e.frame_t.length;
    for (let k = 0; k < n; k++) {
      const a = e.frame_t[k], b = k + 1 < n ? e.frame_t[k + 1] : e.duration_s;
      g.fillStyle = COLORS[e.status[k]] || "#888";
      g.fillRect(x(a), 4, Math.max(1, x(b) - x(a) + 0.5), h - 8);
    }
    g.fillStyle = css.getPropertyValue("--ink").trim() || "#000";
    g.fillRect(Math.round(x(S.t)) - 1, 0, 2, h);
  }
  function timelineSeek(ev) {
    if (!S.ep) return;
    const r = $("timeline").getBoundingClientRect();
    seek(((ev.clientX - r.left) / r.width) * S.ep.duration_s);
  }

  // ------------------------------------------------------------------ server-rendered sim frames
  const simReq = {busy: false, pending: false, url: null};
  function requestSimFrame() {
    if (disposed || !S.serverRender || !S.ep) return;
    if (simReq.busy) { simReq.pending = true; return; }
    simReq.busy = true; simReq.pending = false;
    const t = Math.min(now(), S.ep.duration_s);
    const q = new URLSearchParams({episode: S.ep.id, t: t.toFixed(3), camera: SIM_CAMERA, mode: S.mode});
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 10000);
    fetch(ctx.url(`sim_frame?${q}`), {cache: "no-store", signal: ctl.signal})
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
        return r.blob();
      })
      .then((b) => {
        if (!S.serverRender || disposed) return;
        const u = URL.createObjectURL(b), old = simReq.url;
        simReq.url = u; $("sim-img").src = u;
        if (old) URL.revokeObjectURL(old);
        msg("sim-msg", "");
      })
      .catch((err) => msg("sim-msg", `Server render failed: ${err.message}`))
      .finally(() => {
        clearTimeout(timer); simReq.busy = false;
        if (S.serverRender && (simReq.pending || S.playing)) requestSimFrame();
      });
  }

  // ------------------------------------------------------------------ ORB-SLAM3 view
  // The episode's ORB-SLAM3 camera path (trajectory.csv, world +z up, metres): top (x-y) and side (x-z) views side
  // by side, coloured by tracking status, with the camera at t and its optical axis. The path is drawn once per
  // episode and canvas size; each frame only adds the marker.
  const slam = {key: null, img: null, map: null};
  function drawSlam() {
    const c = $("cam-slam"), e = S.ep;
    if (!e || c.hidden || root.hidden) return;
    const tr = e.trajectory, P = tr ? tr.p : [];
    if (!P.some(Boolean)) { msg("cam-msg", "This reconstruction has no camera poses."); return; }
    const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
    const key = `${e.id}|${c.width}x${c.height}`;
    if (slam.key !== key) {
      const css = getComputedStyle(root);
      for (const k of ["tracked", "imu_only", "interpolated"]) COLORS[k] = css.getPropertyValue(`--${k}`).trim();
      const pad = 18, half = w / 2, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const p of P) if (p) for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); }
      const span = Math.max(1e-3, ...[0, 1, 2].map((i) => hi[i] - lo[i]));
      const s = Math.min((half - 2 * pad) / span, (h - 2 * pad - 14) / span);   // one scale for both views
      const views = [[0, 1], [0, 2]].map(([a, b], v) => ({
        a, b, x: (p) => v * half + half / 2 + (p[a] - (lo[a] + hi[a]) / 2) * s,
        y: (p) => (h + 14) / 2 - (p[b] - (lo[b] + hi[b]) / 2) * s}));
      const off = document.createElement("canvas"); off.width = c.width; off.height = c.height;
      const g = off.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.fillStyle = css.getPropertyValue("--muted").trim() || "#888"; g.font = "12px system-ui, sans-serif";
      g.fillText("top (x-y)", pad, 16); g.fillText("side (x-z)", half + pad, 16);
      g.fillRect(half, 24, 1, h - 32);
      g.lineWidth = 2; g.lineCap = "round";
      for (const v of views) {
        for (let k = 1; k < P.length; k++) {
          if (!P[k - 1] || !P[k]) continue;                    // no pose: a gap in the path
          g.strokeStyle = COLORS[e.status[k]] || "#888";
          g.beginPath(); g.moveTo(v.x(P[k - 1]), v.y(P[k - 1])); g.lineTo(v.x(P[k]), v.y(P[k])); g.stroke();
        }
      }
      slam.key = key; slam.img = off; slam.map = {views, s};
    }
    const g = c.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.width, c.height); g.drawImage(slam.img, 0, 0);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    let k = Math.max(0, lastIndexLE(e.frame_t, S.t));
    while (k > 0 && !P[k]) k--;                              // no pose at t: the last one before it
    const p = P[k], f = tr.fwd[k];
    if (!p) return;
    const ink = getComputedStyle(root).getPropertyValue("--accent").trim() || "#f2b705";   // visible on the dark stage
    for (const v of slam.map.views) {
      const x = v.x(p), y = v.y(p);                          // the axis shortens as it leaves the view plane
      g.strokeStyle = ink; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + 22 * f[v.a], y - 22 * f[v.b]); g.stroke();
      g.fillStyle = ink; g.beginPath(); g.arc(x, y, 4.5, 0, 2 * Math.PI); g.fill();
    }
  }

  // ------------------------------------------------------------------ good/bad annotation
  // Labels live on the server (POST labels/...), in the app's labels file:
  // a success flag and a task prompt per episode. Verdict clicks save immediately; prompt fields
  // save on blur or Enter. A field the user is editing
  // (focused, or dirty with unsaved edits) is never overwritten by a refresh, and refreshes update
  // values in place instead of rebuilding the panel.
  const A = {labels: null, summary: null, verdicts: {}, busy: false};
  const KEEP = Symbol("keep-dirty");             // save() result: not saved, keep the edit

  let toastTimer = null;
  function toast(text, isError = false) {
    const t = $("toast");
    t.textContent = text; t.classList.toggle("error", isError); t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 5000 : 1800);
  }

  const api = (path, body) => ctx.api(path, body);

  // ---- edit-safe text fields
  const F = {};                                   // id -> {dirty, ep, busy, again}
  // Labels belong to the recording, so every source listing it shares one verdict and prompt.
  const recId = () => S.ep && (S.ep.recording || S.ep.id);
  function field(id) { return F[id] || (F[id] = {dirty: false, ep: null, busy: false, again: false}); }
  function isEditing(id) { return field(id).dirty || document.activeElement === $(id); }
  // Apply a server value unless the user is editing the field.
  function setField(id, v) {
    if (isEditing(id)) return false;
    const el = $(id);
    if (el.value !== (v ?? "")) el.value = v ?? "";
    return true;
  }
  // Episode switch: the per-episode fields belong to the new episode.
  function forceField(id, v) {
    const f = field(id); f.dirty = false; f.ep = null; f.again = false;
    $(id).value = v ?? "";
  }
  function markDirty(id) {
    const f = field(id);
    if (!f.dirty) f.ep = recId() || null;
    f.dirty = true;
  }
  function bindField(id, save, {enter = true} = {}) {
    const el = $(id);
    el.addEventListener("input", () => { markDirty(id); if (id === "ann-prompt") showResolved(); });
    el.addEventListener("blur", () => commitField(id, save));
    el.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") el.blur();               // leave the box so G/B/S/N/P work again
      else if (enter && ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); commitField(id, save); }
    });
    field(id).save = save;
  }
  // One save in flight per field; edits made meanwhile are saved right after (coalesced).
  async function commitField(id, save = field(id).save) {
    const f = field(id), el = $(id);
    if (!f.dirty) return;
    if (f.busy) { f.again = true; return; }
    f.busy = true;
    const value = el.value, ep = f.ep;
    try {
      const server = await save(value, ep);
      if (server === KEEP) return;
      if (el.value === value && f.ep === ep) {           // no newer typing: take the server's value
        f.dirty = false; f.ep = null;
        if (typeof server === "string" && (ep === null || ep === recId())) el.value = server;
      }
    } catch (err) {
      toast(`Save failed: ${err.message}`, true);       // stays dirty; refreshes will not clobber it
    } finally {
      f.busy = false;
      if (f.again) { f.again = false; if (f.dirty) commitField(id, save); }
    }
  }

  function entryFor(id) {
    const L = A.labels || {};
    return (L.labels || {})[id] || (L.pending || {})[id] || null;
  }
  function storeEntry(id, entry) {
    A.labels = A.labels || {};
    for (const k of ["labels", "pending"]) { A.labels[k] = A.labels[k] || {}; delete A.labels[k][id]; }
    if (entry) A.labels[typeof entry.success === "boolean" ? "labels" : "pending"][id] = entry;
  }
  // The index's group for an episode ("dig" unless episodes.csv labels it, e.g. "ball").
  function epGroup(id) {
    const e = S.index?.episodes.find((x) => x.recording === id);
    return (e && e.group) || (recId() === id && S.ep.group) || "dig";
  }

  // ---- task prompt: episode prompt > group prompt > default_prompt (same order as the server)
  function promptChain(id) {
    const L = A.labels || {}, g = epGroup(id);
    const ep = (entryFor(id)?.prompt || "").trim();
    const grp = ((L.group_prompts || {})[g] || "").trim(), def = (L.default_prompt || "").trim();
    const fallback = grp ? {prompt: grp, source: g} : def ? {prompt: def, source: "default"} : {prompt: "", source: null};
    return {group: g, fallback, resolved: ep ? {prompt: ep, source: "episode override"} : fallback};
  }
  function showPrompt(force = false) {
    if (!S.ep) return;
    const c = promptChain(recId());
    if (force) forceField("ann-prompt", entryFor(recId())?.prompt || "");
    else setField("ann-prompt", entryFor(recId())?.prompt || "");
    $("ann-prompt").placeholder = c.fallback.prompt || "(no prompt: set a default)";
    $("ann-prompt-src").textContent = `(${c.group} episode; blank = ${c.fallback.source ? "from " + c.fallback.source : "missing"})`;
    showResolved();
  }
  // Exactly the string the server resolves for this episode (the task prompt).
  function showResolved() {
    if (!S.ep) return;
    const id = recId(), c = promptChain(id);
    const server = A.summary?.resolved_prompts?.[id];
    const p = server ?? c.resolved.prompt;
    const el = $("ann-resolved");
    el.textContent = p ? `Prompt: "${p}"  (${c.resolved.source})` : "Prompt: (none)";
    el.classList.toggle("missing", !p);
    if (field("ann-prompt").dirty) el.textContent += "  [unsaved edit]";
  }
  function fillMeta() {
    const L = A.labels || {};
    setField("ann-default-prompt", L.default_prompt || "");
    setField("ann-group-dig", (L.group_prompts || {}).dig || "");
    setField("ann-group-ball", (L.group_prompts || {}).ball || "");
  }

  function applySummary(sum) {
    A.summary = sum; A.verdicts = sum.verdicts || {};
    const n = sum.total || 0, pct = (x) => `${n ? (100 * x / n) : 0}%`;
    $("ann-bar-good").style.width = pct(sum.good); $("ann-bar-bad").style.width = pct(sum.bad);
    $("ann-bar-skip").style.width = pct(sum.skip);
    const parts = [`${sum.labeled}/${n} labeled`, `${sum.good} good`, `${sum.bad} bad`];
    if (sum.skip) parts.push(`${sum.skip} skipped`);
    let ready;
    if (sum.train) {
      const left = sum.train.unlabeled.length;
      ready = sum.train_ready ? "train ready: yes" : `train ready: no (${left} train left)`;
      $("ann-summary").title = `train ${sum.train.labeled}/${sum.train.total}, holdout ${sum.holdout.labeled}/${sum.holdout.total}; file ${sum.labels_file}`;
    } else ready = "train ready: no split.json";
    $("ann-summary-main").textContent = parts.join(" · ") + " · ";
    $("ann-summary-ready").textContent = ready;
    $("ann-summary-ready").className = sum.train_ready ? "ready" : "";
    $("ann-summary-prompts").textContent = sum.prompts
      ? ` · prompts ${sum.prompts.resolved}/${n}${sum.prompts_ready ? "" : " (train missing)"}` : "";
    for (const o of $("episode").options) {
      const e = S.index?.episodes.find((x) => x.id === o.value);
      if (e) { const t = episodeLabel(e); if (o.textContent !== t) o.textContent = t; }
    }
    showResolved();
  }

  async function refreshLabels(quiet = false) {
    try {
      const [labels, sum] = await Promise.all([api("labels"), api("labels/summary")]);
      A.labels = labels;
      fillMeta();
      applySummary(sum);
      showAnnotation(false);
    } catch (err) { if (!quiet) toast(`Labels unavailable: ${err.message}`, true); }
  }

  // force: an episode was (re)loaded, so the per-episode fields are replaced; otherwise (refresh)
  // only fields the user is not editing are updated.
  function showAnnotation(force = false) {
    if (!S.ep) return;
    const e = entryFor(recId()), v = A.verdicts[recId()] || (e && e.verdict) || null;
    showPrompt(force);
    const st = $("ann-state");
    st.textContent = v ? v : "unlabeled"; st.className = `ann-state ${v || ""}`;
    st.title = e?.updated_at ? `saved ${e.updated_at}` : "";
    for (const b of $$(".verdict")) b.classList.toggle("active", b.dataset.verdict === v);
  }

  async function saveVerdict(verdict) {
    if (!S.ep || A.busy) return;
    const id = recId();
    A.busy = true; $$(".verdict").forEach((b) => { b.disabled = true; });
    try {
      const out = await api(`labels/${encodeURIComponent(id)}`, {verdict});
      storeEntry(id, out.entry);
      applySummary(out.summary);
      if (recId() === id) showAnnotation(false);
      toast(verdict ? `${shortId(id)}: ${verdict} saved` : `${shortId(id)}: saved`);
    } catch (err) {
      toast(`Save failed: ${err.message}`, true);
    } finally {
      A.busy = false; $$(".verdict").forEach((b) => { b.disabled = false; });
    }
  }

  async function saveEpisodePrompt(value, ep) {
    if (!ep) return KEEP;
    let v = value.replace(/\s+/g, " ").trim();
    if (v === promptChain(ep).fallback.prompt) v = "";      // same as the group/default: no override
    const old = entryFor(ep)?.prompt || "";
    if (v === old) return old;                               // unchanged: nothing to save
    const out = await api(`labels/${encodeURIComponent(ep)}`, {prompt: v});
    storeEntry(ep, out.entry); applySummary(out.summary);
    if (ep === recId()) setTimeout(() => showPrompt(false));
    toast(`${shortId(ep)}: prompt ${v ? "saved" : "reset to group/default"}`);
    return out.entry?.prompt || "";
  }

  function metaSaver(key, what, {group = null} = {}) {
    return async (value) => {
      const v = value.replace(/\s+/g, " ").trim();
      const body = group ? {group_prompts: {[group]: v}} : {[key]: v};
      const out = await api("labels/meta", body);
      A.labels = {...(A.labels || {}), default_prompt: out.default_prompt, group_prompts: out.group_prompts};
      applySummary(out.summary);
      setTimeout(() => showPrompt(false));
      toast(`${what} saved`);
      return group ? (out.group_prompts[group] || "") : (out[key] ?? "");
    };
  }

  // Episodes visited before the current one, most recent last; P walks back through them.
  const HISTORY = [];

  function previousEpisode() {
    const ids = new Set([...$("episode").options].map((o) => o.value));
    while (HISTORY.length) {
      const id = HISTORY.pop();
      if (ids.has(id) && id !== S.ep?.id) { $("episode").value = id; loadEpisode(id, true); return; }
    }
    toast("No previous episode");
  }

  function nextEpisode() {
    const ids = [...$("episode").options].map((o) => o.value);
    if (!ids.length) return;
    const id = ids[(ids.indexOf(S.ep?.id) + 1) % ids.length];
    $("episode").value = id; loadEpisode(id);
  }

  function initAnnotation() {
    $$(".verdict").forEach((b) => b.addEventListener("click", () => saveVerdict(b.dataset.verdict)));
    $("ann-next").addEventListener("click", nextEpisode);
    $("ann-prev").addEventListener("click", previousEpisode);
    bindField("ann-default-prompt", metaSaver("default_prompt", "Default prompt"));
    bindField("ann-group-dig", metaSaver("group_prompts", "Prompt for dig", {group: "dig"}));
    bindField("ann-group-ball", metaSaver("group_prompts", "Prompt for ball", {group: "ball"}));
    bindField("ann-prompt", saveEpisodePrompt);
  }

  // ------------------------------------------------------------------ UI wiring
  function msg(id, text) { const m = $(id); m.textContent = text; m.hidden = !text; }

  function applyCamSource() {
    const e = S.ep, v = $("cam-video"), slamMode = S.camSource === "slam";
    $("cam-slam").hidden = !slamMode;
    v.hidden = slamMode;
    msg("cam-msg", "");
    if (slamMode) {
      v.pause(); v.removeAttribute("src"); v.load(); S.drift.cam = null;
      drawSlam();
      return;
    }
    const src = e?.media["video.mp4"];
    if (!src) { msg("cam-msg", "The original video.mp4 is not in the data root for this episode."); return; }
    v.src = ctx.url(src);
    v.addEventListener("loadedmetadata", () => {
      const f = camFollower();
      if (f) S.playing ? playFollower(f) : seekFollower(f);
    }, {once: true});
    if (!e.original_video_synced)
      msg("cam-msg", "No frames.csv found for this video; showing it unsynced (offset 0).");
  }
  // The server renders frames only when there is no replay video of this track: replays made with the
  // earlier two-joint twin (re-solved by the server).
  function applySim() {
    const e = S.ep, v = $("sim-video");
    S.serverRender = !e?.fk.twin?.video;
    v.hidden = S.serverRender; $("sim-img").hidden = !S.serverRender;
    msg("sim-msg", "");
    if (S.serverRender) {
      v.pause(); v.removeAttribute("src"); v.load();
      requestSimFrame();
      return;
    }
    const src = e?.media[e.fk.twin.video];
    if (!src) { msg("sim-msg", "Replay video missing."); return; }
    v.src = ctx.url(src);
    v.addEventListener("loadedmetadata", () => {
      const f = simFollower();
      if (f) S.playing ? playFollower(f) : seekFollower(f);
    }, {once: true});
  }

  // ------------------------------------------------------------------ episodes
  async function fetchEpisode(id) {
    const r = await fetch(ctx.url(`episode?id=${encodeURIComponent(id)}`), {cache: "no-store"});
    if (!r.ok) throw new Error(`Could not load episode ${id}`);
    const e = await r.json();
    for (const f of Object.values(e.fk)) {
      f._q = (f.joints || e.joints || JOINTS).map((_, i) => f.q.map((row) => row[i]));
      f._tip = [0, 1, 2].map((i) => f.tip.map((row) => row[i]));
    }
    return e;
  }

  async function loadEpisode(id, back = false) {
    if (S.playing) pause();
    const from = S.ep?.id;
    let e;
    try { e = await fetchEpisode(id); } catch (err) { msg("sim-msg", err.message); return; }
    if (disposed) return;
    if (from && from !== id && !back) HISTORY.push(from);
    S.ep = e; S.t = 0; S.drift = {cam: null, sim: null}; slam.key = null;
    $("scrub").max = String(e.duration_s);
    $("scrub").step = "0.01";
    $("camera-source").querySelector('option[value="original"]').textContent =
      e.media["video.mp4"] ? "Original video" : "Original video (not in data root)";
    if (e.media["joints.png"]) $("joints-png").src = ctx.url(e.media["joints.png"]);
    else $("joints-png").removeAttribute("src");
    if (ctx.params().get("episode") !== id) ctx.setParams({episode: id});
    applyCamSource(); applySim(); render();
    showAnnotation(true);
  }

  function episodeLabel(e) {
    return `${MARK[A.verdicts[e.recording]] || "·"} ${shortId(e.id)}  ${e.group || "dig"}  ${Number(e.duration_s).toFixed(1)} s`;
  }
  // Episodes of the chosen source (all of them when the index offers no choice).
  function shownEpisodes(idx = S.index) {
    const eps = idx?.episodes || [];
    return idx?.sources && S.source ? eps.filter((e) => e.source === S.source) : eps;
  }
  // Source picker (index sources).
  function applyIndexExtras(idx) {
    const recs = idx.sources || [];
    $("source-field").hidden = !recs.length;
    if (recs.length) {
      const sel = $("source"), count = (id) => idx.episodes.filter((e) => e.source === id).length;
      if (!recs.some((r) => r.id === S.source)) S.source = (recs.find((r) => count(r.id)) || recs[0]).id;
      sel.textContent = "";
      for (const r of recs) {
        const o = document.createElement("option");
        o.value = r.id; o.textContent = `${r.name} (${count(r.id)})`;
        sel.appendChild(o);
      }
      sel.value = S.source;
    }
  }
  // Rebuild the selector grouped by label (dig first, then the rest alphabetically).
  function fillEpisodeSelect(episodes, keep) {
    const sel = $("episode");
    sel.textContent = "";
    episodes = shownEpisodes({...S.index, episodes});
    const groups = new Map();
    for (const e of episodes) {
      const g = e.group || "dig";
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(e);
    }
    const names = [...groups.keys()].sort((a, b) => (a === "dig" ? -1 : b === "dig" ? 1 : a.localeCompare(b)));
    for (const g of names) {
      const og = document.createElement("optgroup");
      og.label = `${g} (${groups.get(g).length})`;
      for (const e of groups.get(g)) {
        const o = document.createElement("option");
        o.value = e.id;
        o.textContent = episodeLabel(e);
        o.title = `${e.id}${e.task ? " - " + e.task : ""}`;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    if (keep && episodes.some((e) => e.id === keep)) sel.value = keep;
  }

  // Overall error: every episode's IK / twin path vs its SLAM path, pooled (path_error_summary).
  async function refreshOverallError() {
    const fmtAll = (x) => !x ? "-"
      : `tip RMSE ${fmt(x.tip?.rmse_mm, 1)} mm`
        + (x.tip_umi ? ` (${fmt(x.tip_umi.rmse_mm, 1)} mm UMI)` : "")
        + (x.angle ? ` · bucket angle RMSE ${fmt(x.angle.rmse_deg, 2)}°` : "")
        + ` · median ep ${fmt(x.tip?.median_episode_rmse_mm, 1)} mm`
        + (x.angle ? ` / ${fmt(x.angle.median_episode_rmse_deg, 2)}°` : "")
        + ` · worst ep ${fmt(x.tip?.worst_episode_rmse_mm, 1)} mm (${x.tip?.worst_episode})`
        + ` · ${x.episodes} eps, ${x.samples} samples`;
    try {
      const s = await api("path_error_summary");
      $("r-all-ik").textContent = fmtAll(s.ik);
      $("r-all-twin").textContent = fmtAll(s.twin);
      S.overall = s;
      renderAccuracy();
    } catch { /* keep the last value */ }
  }

  // IK accuracy to SLAM data (Annotate panel): overall and this episode's RMSE, coloured by the good /
  // medium / bad bands (thirds of the per-episode RMSE, from the server).
  let accSig = "";
  function renderAccuracy() {
    const o = S.overall;
    if (!o || (!o.ik && !o.twin)) return;
    const pe = S.ep?.path_error;
    const cols = [["ik", "tip", "mm"], ["ik", "angle", "deg"], ["twin", "tip", "mm"], ["twin", "angle", "deg"]];
    const epVal = (m, k) => pe?.[m]?.[k === "tip" ? "tip_rmse_mm" : "angle_rmse_deg"];
    const band = (m, k, v) => {
      const b = o[m]?.[k]?.bands;
      return v == null || !b ? "" : v <= b.good_max ? "good" : v <= b.medium_max ? "medium" : "bad";
    };
    const cell = (m, k, v) => `<td class="${band(m, k, v)}">${v == null ? "-" : fmt(v, 1)}</td>`;
    const valueRow = (label, get) => `<tr><td>${label}</td>${cols.map(([m, k, u]) => cell(m, k, get(m, k, u))).join("")}</tr>`;
    const bandRow = (name, txt) => `<tr class="band"><td class="${name}">${name}</td>${cols.map(([m, k]) => {
      const b = o[m]?.[k]?.bands;
      return `<td class="${name}">${b ? txt(b) : "-"}</td>`;
    }).join("")}</tr>`;
    const html = valueRow("all eps", (m, k, u) => o[m]?.[k]?.[`rmse_${u}`])
      + valueRow("this ep", (m, k) => epVal(m, k))
      + bandRow("good", (b) => `≤ ${fmt(b.good_max, 1)}`)
      + bandRow("medium", (b) => `${fmt(b.good_max, 1)}–${fmt(b.medium_max, 1)}`)
      + bandRow("bad", (b) => `> ${fmt(b.medium_max, 1)}`);
    if (html === accSig) return;
    accSig = html;
    $("acc-body").innerHTML = html;
    $("ann-overall").hidden = false;
  }

  const fetchIndex = () => api("index");

  // Pick up replays added while the page is open (the server rescans on each index request).
  async function refreshIndex() {
    if (disposed || !ctx.active()) return;
    try {
      const idx = await fetchIndex();
      const sig = (x) => x.episodes.map((e) => `${e.id}|${e.group}|${e.duration_s}`).join(",");
      refreshLabels(true);                         // edit-safe: never touches fields being edited
      if (S.index && sig(idx) === sig(S.index)) return;
      S.index = idx;
      applyIndexExtras(idx);
      refreshOverallError();
      fillEpisodeSelect(idx.episodes, S.ep && S.ep.id);
      if (!S.ep && shownEpisodes(idx).length) await loadEpisode($("episode").value);
    } catch { /* keep the current list */ }
  }

  // An ?episode= deep link (back / forward, or a link into this view) selects that episode.
  function followParams() {
    if (!S.index || !ctx.active()) return;
    const id = ctx.params().get("episode");
    const e = S.index.episodes.find((x) => x.id === id);
    if (S.ep && id !== S.ep.id && e) {
      if (S.index.sources && e.source !== S.source) { S.source = e.source; applyIndexExtras(S.index); fillEpisodeSelect(S.index.episodes, id); }
      $("episode").value = id; loadEpisode(id);
    }
  }

  async function init() {
    initAnnotation();
    S.index = await fetchIndex();
    if (disposed) return;
    const want = ctx.params().get("episode") || "";
    S.source = S.index.episodes.find((e) => e.id === want)?.source || null;
    applyIndexExtras(S.index);
    await refreshLabels();
    fillEpisodeSelect(S.index.episodes);
    refreshOverallError();
    const shown = shownEpisodes();
    if (!shown.length) { msg("sim-msg", "No episodes found (run python -m policy reconstruct / labels for the app's source configs)."); return; }
    const sel = $("episode");
    sel.value = shown.some((e) => e.id === want) ? want : shown[0].id;
    await loadEpisode(sel.value);
  }

  // ------------------------------------------------------------------ events
  $("episode").addEventListener("change", (ev) => loadEpisode(ev.target.value));
  // Switching source keeps the recording when the other source has it.
  $("source").addEventListener("change", (ev) => {
    S.source = ev.target.value;
    const rec = S.ep?.recording, shown = shownEpisodes();
    const next = shown.find((e) => e.recording === rec) || shown[0];
    fillEpisodeSelect(S.index.episodes, next?.id);
    if (next) loadEpisode(next.id);
    else msg("sim-msg", "No replays for this source yet (python -m policy labels <config>).");
  });
  $("play").addEventListener("click", () => (S.playing ? pause() : play()));
  $("scrub").addEventListener("input", (ev) => seek(Number(ev.target.value)));
  $("speed").addEventListener("change", (ev) => setSpeed(Number(ev.target.value)));
  $("camera-source").addEventListener("change", (ev) => { S.camSource = ev.target.value; applyCamSource(); });
  let dragging = false;
  $("timeline").addEventListener("pointerdown", (ev) => { dragging = true; $("timeline").setPointerCapture(ev.pointerId); timelineSeek(ev); });
  $("timeline").addEventListener("pointermove", (ev) => { if (dragging) timelineSeek(ev); });
  $("timeline").addEventListener("pointerup", () => { dragging = false; });

  const onResize = () => { drawTimeline(); drawSlam(); };
  const onKey = (ev) => {
    if (!ctx.active() || !S.ep) return;
    if (ev.target.closest?.("input[type=text], textarea, select")) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const k = ev.key.toLowerCase();
    if (k === "g" || k === "b") { ev.preventDefault(); saveVerdict({g: "good", b: "bad"}[k]); }
    else if (k === "n") { ev.preventDefault(); nextEpisode(); }
    else if (k === "p") { ev.preventDefault(); previousEpisode(); }
    else if (ev.key === " ") { ev.preventDefault(); S.playing ? pause() : play(); }
    else if (ev.key === "ArrowRight") { ev.preventDefault(); if (S.playing) pause(); seek(S.t + STEP_S); }
    else if (ev.key === "ArrowLeft") { ev.preventDefault(); if (S.playing) pause(); seek(S.t - STEP_S); }
  };
  window.addEventListener("resize", onResize);
  window.addEventListener("focus", refreshIndex);
  window.addEventListener("popstate", followParams);
  document.addEventListener("keydown", onKey);
  const indexTimer = setInterval(refreshIndex, INDEX_REFRESH_MS);

  // Hidden: stop playback (views stay mounted); shown again: redraw and pick up new replays.
  ctx.onHide(() => {
    if (S.playing) pause();
  });
  ctx.onShow(() => {
    render();
    followParams();
    refreshIndex();
  });

  init().catch((err) => msg("sim-msg", `Failed to load: ${err.message}`));
  requestAnimationFrame(tick);

  return () => {
    disposed = true;
    S.playing = false;
    for (const v of [$("cam-video"), $("sim-video")]) { v.pause(); v.removeAttribute("src"); v.load(); }
    clearInterval(indexTimer); clearTimeout(toastTimer);
    if (simReq.url) URL.revokeObjectURL(simReq.url);
    window.removeEventListener("resize", onResize);
    window.removeEventListener("focus", refreshIndex);
    window.removeEventListener("popstate", followParams);
    document.removeEventListener("keydown", onKey);
  };
}
