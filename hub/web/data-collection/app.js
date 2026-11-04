// Live dashboard page: camera preview, IMU readings and charts, plus the recorder (recording.js).
//
// An ES module shared by the Pi's own page (index.html -> pi.js) and the hub's view
// (hub/web/views/umi/view.js): mount(root, ctx) renders into root and returns a cleanup function.
// ctx.url(p) turns a Pi path ("api/status", "preview.jpg") into the URL to fetch, ctx.href("dashboard") gives
// the link to this page, ctx.active() says whether the page is visible, and
// ctx.onShow / ctx.onHide fire when that changes for reasons other than document visibility.
import {LatestPoll} from "./latest-poll.js";
import {mount as mountRecorder} from "./recording.js";

const MARKUP = href => `
  <header class="ui-header">
    <a class="brand" href="${href("dashboard")}" data-umi-link="dashboard" aria-label="Bucket lab home"><span class="ui-mark" aria-hidden="true"></span><h1 class="ui-title">Bucket lab</h1></a>
    <div class="ui-header-end"><span class="network-label"><span class="ui-dot ok"></span> Local connection</span><span id="connection" class="connection" role="status"><i></i>Connecting</span><button id="pause" class="button secondary"><span id="pause-icon">Ⅱ</span><span id="pause-label">Pause view</span></button></div>
  </header>
  <main>
    <section class="card recorder-card" aria-label="Recording controls">
      <div class="recorder-main">
        <div class="record-summary"><span id="record-state" class="record-state" role="status">Connecting</span><strong id="record-time">00:00:00</strong><span id="record-counts">Checking the recorder on your Pi</span></div>
        <div class="record-actions"><label for="record-label">Session name <span>optional</span><input id="record-label" maxlength="80" placeholder="Dirt digging" autocomplete="off"></label><button id="record-start" class="button record-start btn-primary" disabled><span aria-hidden="true">●</span> Start recording</button><button id="record-stop" class="button record-stop btn-danger" disabled><span aria-hidden="true">■</span> Stop &amp; save</button></div>
      </div>
      <p id="record-hint" class="record-hint">Video and motion readings save on the Pi. Recording continues if you close this tab.</p>
      <div id="record-error" class="record-error" role="alert" hidden></div>
      <div id="record-warning" class="record-warning" role="status" hidden></div>
      <div id="record-progress-wrap" class="record-progress-wrap" hidden><label for="record-progress" id="record-progress-label">Saving video</label><progress id="record-progress" max="1" value="0"></progress></div>
      <div class="record-bottom"><span id="record-storage">Checking storage</span></div>
    </section>
    <section class="monitor-grid" aria-label="Live camera and sensor monitor">
      <article class="card camera-card">
        <div class="card-heading"><div><h2>Camera</h2></div><span id="camera-status" class="device-state">Connecting</span></div>
        <div class="camera-stage" id="camera-stage">
          <img id="camera" alt="Live view from the bucket’s USB camera">
          <div class="camera-placeholder" id="camera-placeholder"><svg viewBox="0 0 48 48" aria-hidden="true"><path d="M6 15h9l4-6h10l4 6h9v24H6z"/><circle cx="24" cy="26" r="8"/></svg><h3 id="camera-message">Waiting for the camera</h3><p id="camera-detail">Your live view will appear here.</p></div>
          <div class="camera-overlay" id="camera-overlay" hidden><span class="live-tag"><i></i><span id="view-mode">LIVE</span></span><span class="frame-counter" id="frames">0 frames</span></div>
          <span class="crosshair crosshair-tl"></span><span class="crosshair crosshair-tr"></span><span class="crosshair crosshair-bl"></span><span class="crosshair crosshair-br"></span>
        </div>
        <div class="camera-toolbar"><div class="camera-meta"><strong id="preview-size">360p preview</strong><span class="meta-divider"></span><span id="camera-rate">— fps viewing</span><span id="transfer-time" title="Request and image download time, not camera-to-IMU synchronization">— ms transfer</span></div><div class="camera-tools"><button class="icon-button" id="snapshot" title="Save a full-resolution camera image" aria-label="Save a full-resolution camera image" disabled><svg viewBox="0 0 24 24"><path d="M12 3v12m-4-4 4 4 4-4M5 16v5h14v-5"/></svg></button><button class="icon-button" id="fullscreen" title="Fullscreen camera" aria-label="Fullscreen camera"><svg viewBox="0 0 24 24"><path d="M8 3H3v5m13-5h5v5M3 16v5h5v-5"/></svg></button></div></div>
      </article>
      <aside class="readings">
        <article class="card readings-card"><div class="card-heading"><div><h2>Acceleration</h2></div><span class="unit">g</span></div><div class="axis-rows"><div><span class="axis x">X</span><strong id="ax">—</strong><span>g</span></div><div><span class="axis y">Y</span><strong id="ay">—</strong><span>g</span></div><div><span class="axis z">Z</span><strong id="az">—</strong><span>g</span></div></div><div class="reading-note"><span>Includes gravity</span><strong id="accel-norm">— g total</strong></div></article>
        <article class="card readings-card"><div class="card-heading"><div><h2>Angular velocity</h2></div><span class="unit">°/s</span></div><div class="axis-rows"><div><span class="axis x">X</span><strong id="gx">—</strong><span>°/s</span></div><div><span class="axis y">Y</span><strong id="gy">—</strong><span>°/s</span></div><div><span class="axis z">Z</span><strong id="gz">—</strong><span>°/s</span></div></div><div class="reading-note"><span id="imu-status">Waiting for IMU</span><strong id="imu-rate">— Hz</strong></div></article>
      </aside>
    </section>
    <div id="notice" class="notice" role="status" hidden></div>
    <p id="time-alignment" class="reading-note" role="status">Checking time correction…</p>
    <section class="card recordings-card" aria-labelledby="saved-heading"><div class="card-heading"><h2 id="saved-heading">Saved recordings</h2><span class="unit">Stored on your Pi</span></div><p class="recordings-help">Download the MP4 to watch. Download video + IMU data for episode splitting and SLAM.</p><p id="record-history-error" class="recordings-help" role="status" hidden></p><div id="record-history"><p class="recordings-empty">Loading recordings…</p></div></section>
    <section class="charts-grid" aria-label="Recent motion graphs">
      <article class="card chart-card"><div class="chart-heading"><div><h2>Acceleration over time</h2><p>Last 15 seconds <span>·</span> g</p></div><div class="legend"><span class="x">X</span><span class="y">Y</span><span class="z">Z</span></div></div><div class="chart-wrap"><canvas id="accel-chart" aria-label="Live X, Y, and Z acceleration chart"></canvas><div class="chart-empty" id="accel-empty">Motion readings will appear here</div></div><div class="time-axis"><span>−15 s</span><span>−10 s</span><span>−5 s</span><span>Now</span></div></article>
      <article class="card chart-card"><div class="chart-heading"><div><h2>Rotation over time</h2><p>Last 15 seconds <span>·</span> degrees per second</p></div><div class="legend"><span class="x">X</span><span class="y">Y</span><span class="z">Z</span></div></div><div class="chart-wrap"><canvas id="gyro-chart" aria-label="Live X, Y, and Z angular velocity chart"></canvas><div class="chart-empty" id="gyro-empty">Motion readings will appear here</div></div><div class="time-axis"><span>−15 s</span><span>−10 s</span><span>−5 s</span><span>Now</span></div></article>
    </section>
    <footer><div><span class="footer-dot"></span> Raspberry Pi 4 <span>·</span> BerryIMUv3 <span>·</span> USB camera</div><div>Live preview <span>·</span> <span id="recording-footer">Checking recording</span> <span>·</span> <span id="last-update">Waiting for data</span></div></footer>
  </main>`;

export function mount(root, ctx) {
  root.innerHTML = MARKUP(ctx.href);
  const $ = id => root.querySelector("#" + id);
  const history = [];
  const token = name => getComputedStyle(root).getPropertyValue(name).trim();
  const MAX_FRAME_AGE = 700;
  let paused = false, latest = null, lastEvent = 0, lastSample = 0, viewTime = 0;
  let frameURL = null, frameSequence = "", frameTime = 0, frameAge = 0;
  let running = false, disposed = false;
  const displayed = [];
  const fmt = (value, digits) => Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}` : "—";
  const visible = () => !disposed && ctx.active();

  function badge(text, kind = "") {
    const badge = $("connection");
    badge.className = `connection ${kind}`;
    badge.replaceChildren(document.createElement("i"), document.createTextNode(text));
  }
  function frameFresh(now = performance.now()) {
    return frameTime > 0 && now - frameTime + frameAge < MAX_FRAME_AGE;
  }
  function updateCamera() {
    if (paused || !visible()) return;
    const fresh = frameFresh();
    $("camera").classList.toggle("visible", fresh);
    $("camera-placeholder").hidden = fresh;
    $("camera-overlay").hidden = !fresh;
    $("snapshot").disabled = !fresh;
    $("camera-status").textContent = fresh ? "Live" : latest?.camera.live ? "Catching up" : "Waiting";
    $("camera-status").className = `device-state ${fresh ? "live" : ""}`;
    if (!fresh) {
      $("camera-rate").textContent = "— fps viewing";
      $("transfer-time").textContent = "Waiting for fresh video";
      $("camera-message").textContent = latest?.camera.live ? "Waiting for fresh video" : "Waiting for the camera";
      $("camera-detail").textContent = latest?.camera.error || "The connection is slow. Old images are skipped automatically.";
    }
    const imuFresh = latest?.imu.live && performance.now() - lastEvent < 700;
    badge(fresh && imuFresh ? "Live" : fresh || imuFresh ? "Partially connected" : "Reconnecting", fresh && imuFresh ? "online" : "offline");
  }

  const preview = new LatestPoll({
    url: () => ctx.url(`preview.jpg?after=${encodeURIComponent(frameSequence)}`),
    async read(response) {
      const age = Number(response.headers.get("X-Frame-Age-Ms"));
      const sequence = response.headers.get("X-Frame-Sequence");
      if (!sequence || !Number.isFinite(age) || age < 0) throw new Error("Missing frame timing");
      const url = URL.createObjectURL(await response.blob());
      try {
        const image = new Image();
        image.src = url;
        await image.decode();
        return {url, sequence, age};
      } catch (error) {
        URL.revokeObjectURL(url);
        throw error;
      }
    },
    onValue(frame, elapsed) {
      // Full round-trip time is a conservative bound; browser and Pi clocks differ.
      if (frame.age + elapsed >= MAX_FRAME_AGE) throw new Error("Skipping an old image");
      const previousURL = frameURL;
      frameURL = frame.url;
      frameSequence = frame.sequence;
      $("camera").src = frameURL;
      if (previousURL) URL.revokeObjectURL(previousURL);
      frameTime = performance.now();
      frameAge = frame.age + elapsed;
      displayed.push(frameTime);
      while (displayed.length && displayed[0] < frameTime - 2000) displayed.shift();
      const fps = displayed.length > 1 ? (displayed.length - 1) * 1000 / (frameTime - displayed[0]) : 0;
      updateCamera();
      $("camera-rate").textContent = fps ? `${fps.toFixed(1)} fps viewing` : "Live preview";
      $("transfer-time").textContent = `${Math.round(elapsed)} ms transfer`;
      $("frames").textContent = `Frame ${Number(frame.sequence).toLocaleString()}`;
    },
    onError: updateCamera,
    dispose: frame => URL.revokeObjectURL(frame.url),
  });

  function clearReadings() {
    ["ax", "ay", "az", "gx", "gy", "gz"].forEach(id => $(id).textContent = "—");
    $("accel-norm").textContent = "— g total";
    $("imu-rate").textContent = "— Hz";
  }
  function render(state) {
    latest = state;
    viewTime = state.server_monotonic_s;
    const cam = state.camera, imu = state.imu;
    const alignment = state.time_alignment;
    $("time-alignment").textContent = alignment?.applied
      ? `Time correction active: ${alignment.offset_ms >= 0 ? "+" : ""}${alignment.offset_ms.toFixed(1)} ms for recorded camera timestamps.`
      : alignment?.error || "No time correction installed.";
    root.classList.toggle("offline", !imu.live);
    $("preview-size").title = `${cam.fps.toFixed(1)} fps capture at 720p. Snapshots save the full resolution.`;
    updateCamera();
    if (imu.live && imu.sample) {
      const sample = imu.sample;
      ["ax", "ay", "az"].forEach((id, i) => $(id).textContent = fmt(sample.accel[i], 3));
      ["gx", "gy", "gz"].forEach((id, i) => $(id).textContent = fmt(sample.gyro[i], 2));
      $("accel-norm").textContent = `${Math.hypot(...sample.accel).toFixed(3)} g total`;
      $("imu-rate").textContent = `${Math.round(imu.hz)} Hz`;
      $("imu-status").textContent = "IMU connected";
      if (sample.read_monotonic_ns !== lastSample) {
        lastSample = sample.read_monotonic_ns;
        history.push({t: viewTime, accel: sample.accel, gyro: sample.gyro});
      }
    } else {
      clearReadings();
      $("imu-status").textContent = "Waiting for IMU";
    }
    while (history.length && history[0].t < viewTime - 15) history.shift();
    const messages = [];
    if (!imu.live) messages.push(imu.error || "No fresh motion readings. Check the IMU connection.");
    if (imu.sample?.clipped) messages.push("Motion exceeded the sensor range. Move the rig more slowly for this check.");
    $("notice").hidden = messages.length === 0;
    $("notice").textContent = messages.join(" ");
    $("last-update").textContent = cam.live || imu.live ? "Receiving data" : "Waiting for data";
    $("accel-empty").hidden = history.length > 1;
    $("gyro-empty").hidden = history.length > 1;
  }
  function lostConnection() {
    if (paused || !visible()) return;
    root.classList.add("offline");
    $("time-alignment").textContent = "Reconnecting to check time correction…";
    clearReadings();
    $("imu-status").textContent = "Reconnecting";
    $("last-update").textContent = "Waiting for motion data";
    updateCamera();
  }
  const readings = new LatestPoll({
    url: () => ctx.url("api/status"),
    interval: 100,
    read: response => response.json(),
    onValue(state, elapsed) {
      if (elapsed >= 700) throw new Error("Skipping delayed readings");
      if (state.imu.age_s !== null && state.imu.age_s * 1000 + elapsed >= 700) {
        state.imu.live = false;
        state.imu.sample = null;
      }
      lastEvent = performance.now();
      render(state);
    },
    onError: lostConnection,
  });

  function stopView() {
    running = false;
    preview.stop();
    readings.stop();
  }
  function startView() {
    if (paused || !visible() || running) return;
    running = true;
    frameTime = 0;
    lastEvent = 0;
    frameSequence = "";
    displayed.length = 0;
    history.length = 0;
    lastSample = 0;
    lostConnection();
    preview.start();
    readings.start();
  }
  const sync = () => visible() ? startView() : stopView();
  $("pause").addEventListener("click", () => {
    paused = !paused;
    root.classList.toggle("paused", paused);
    $("pause-label").textContent = paused ? "Resume view" : "Pause view";
    $("pause-icon").textContent = paused ? "▷" : "Ⅱ";
    $("view-mode").textContent = paused ? "PAUSED" : "LIVE";
    if (paused) {
      stopView();
      badge("View paused");
      $("last-update").textContent = "View paused";
      $("snapshot").disabled = true;
      $("camera-status").textContent = "Paused";
      $("camera-status").className = "device-state";
      $("camera-rate").textContent = "View paused";
      $("transfer-time").textContent = "";
    } else startView();
  });
  $("snapshot").addEventListener("click", async () => {
    try {
      const response = await fetch(ctx.url("snapshot.jpg"), {signal: AbortSignal.timeout(5000)});
      if (!response.ok) throw new Error("No fresh camera image. Wait for the camera to reconnect.");
      const blob = await response.blob(), url = URL.createObjectURL(blob);
      const link = Object.assign(document.createElement("a"), {href: url, download: `bucket-${new Date().toISOString().replaceAll(":", "-")}.jpg`});
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      $("notice").hidden = false;
      $("notice").textContent = error.message;
    }
  });
  $("fullscreen").addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await $("camera-stage").requestFullscreen();
    } catch (_) {
      $("notice").hidden = false;
      $("notice").textContent = "Fullscreen is unavailable in this browser. Open the dashboard in Safari or Chrome.";
    }
  });

  function draw(id, key, minimum) {
    const canvas = $(id), rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(rect.width*dpr) || canvas.height !== Math.round(rect.height*dpr)) {
      canvas.width = Math.round(rect.width*dpr); canvas.height = Math.round(rect.height*dpr);
    }
    const ctx = canvas.getContext("2d"), w = rect.width, h = rect.height, left = 35, top = 8, bottom = h-8;
    ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
    let max = minimum;
    for (const point of history) for (const v of point[key]) max = Math.max(max, Math.abs(v)*1.15);
    max = max <= 2 ? Math.ceil(max*2)/2 : Math.ceil(max/10)*10;
    const colors = [token("--x"), token("--y"), token("--z")], grid = token("--line"), axisLine = token("--line-strong"), label = token("--muted");
    ctx.font = `10px ${token("--font-display") || "sans-serif"}`; ctx.textAlign="right";
    [1,.5,0,-.5,-1].forEach(level => {
      const y = top + (1-level)/2*(bottom-top);
      ctx.strokeStyle = level === 0 ? axisLine : grid;
      ctx.lineWidth = 1; ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w,y);ctx.stroke();
      if (level===1 || level===0 || level===-1) {
        ctx.fillStyle=label;ctx.fillText((max*level).toFixed(max<10?1:0),left-8,y+3);
      }
    });
    for (let axis=0;axis<3;axis++) {
      ctx.strokeStyle=colors[axis];ctx.lineWidth=1.65;ctx.lineJoin="round";ctx.beginPath();
      let previous = null;
      for (const point of history) {
        const x=left+(point.t-(viewTime-15))/15*(w-left), y=top+(1-point[key][axis]/max)/2*(bottom-top);
        if (previous===null || point.t-previous>.3) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        previous=point.t;
      }
      ctx.stroke();
    }
  }
  let lastDraw = 0, frame = 0;
  function animate(now) {
    if (disposed) return;
    if (now-lastDraw>45 && visible()) {draw("accel-chart","accel",1.5);draw("gyro-chart","gyro",30);lastDraw=now;}
    frame = requestAnimationFrame(animate);
  }
  const ticker = setInterval(() => {
    if (paused || !visible()) return;
    if (!lastEvent || performance.now() - lastEvent >= 700) lostConnection();
    updateCamera();
  }, 100);
  document.addEventListener("visibilitychange", sync);
  ctx.onShow(sync);
  ctx.onHide(sync);
  const recorder = mountRecorder(root, ctx);
  startView(); frame = requestAnimationFrame(animate);
  return () => {
    disposed = true;
    stopView();
    recorder();
    clearInterval(ticker);
    cancelAnimationFrame(frame);
    document.removeEventListener("visibilitychange", sync);
    if (frameURL) URL.revokeObjectURL(frameURL);
  };
}
