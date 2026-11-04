// Recorder card and saved-recordings list of the dashboard page (app.js renders their markup).
// mount(root, ctx) wires them up and returns a cleanup function; see app.js for ctx.
import {LatestPoll} from "./latest-poll.js";

export function mount(root, ctx) {
  const el = id => root.querySelector("#" + id);
  let disposed = false;
  let current = null, known = false, pending = false, lastActionError = "", reconcile = false, historyKey = "";
  const clock = seconds => {
    const n = Math.max(0, Math.floor(seconds || 0));
    return [Math.floor(n/3600), Math.floor(n/60)%60, n%60].map(v => String(v).padStart(2, "0")).join(":");
  };
  const message = (id, value) => {el(id).textContent = value; el(id).hidden = !value;};
  function show() {
    const state = current || {}, mode = state.recording_mode;
    const recording = state.status === "recording", saving = state.status === "saving";
    const devices = state.devices || {}, camera = devices.camera || {}, imu = devices.imu || {};
    const ready = camera.live && imu.live && camera.timing?.clock === "CLOCK_MONOTONIC";
    el("record-start").disabled = !known || pending || !ready || recording || saving || state.status === "unavailable";
    el("record-stop").disabled = !known || pending || !recording;
    el("record-label").disabled = recording || saving || pending;
    const names = {idle:"Ready to record", recording:"Recording on Pi", saving:"Saving on Pi", ready:state.error ? "Stopped with an error" : "Saved on Pi", error:"Could not finish saving", unavailable:"Recorder unavailable"};
    let title = known ? names[state.status] || "Checking recorder" : "Connection lost";
    if (known && state.status === "idle" && !ready) title = "Waiting for camera and IMU";
    if (pending) title = "Sending request…";
    el("record-state").textContent = title;
    el("record-state").dataset.state = !known ? "unknown" : state.error ? "error" : state.status;
    el("record-time").textContent = clock(state.elapsed_s);
    el("record-counts").textContent = `${(state.frames || 0).toLocaleString()} images · ${(state.imu_samples || 0).toLocaleString()} motion samples${mode === "calibration" && recording ? " · Calibration" : ""}`;
    el("recording-footer").textContent = title;
    let hint = "Video and motion readings save on the Pi. Recording continues if you close this tab.";
    if (recording) hint = mode === "calibration" ? "A time calibration is recording. It will stop automatically after 45 seconds." : "Recording until you press Stop & save. Pausing the preview does not pause recording.";
    if (saving) hint = state.raw_data_saved ? "Original video and motion data are saved. Preparing the MP4 can take a while for a long session." : "Saving captured data. Keep the Pi powered on.";
    if (state.status === "ready") hint = state.video_status === "ready" ? "Your video is saved. Download it from Saved recordings below." : "Download the captured data from Saved recordings below.";
    el("record-hint").textContent = hint;
    const deviceErrors = [];
    if (!camera.live) deviceErrors.push(`Camera: ${camera.error || "no fresh images. Check its USB cable."}`);
    if (!imu.live) deviceErrors.push(`IMU: ${imu.error || "no fresh readings. Check its connection."}`);
    let problem = state.error || (known && !saving && !recording ? deviceErrors.join(" ") : "");
    if (!known) problem = "Lost connection to the Pi. Recording may still be running there. Reconnect to check; this page cannot confirm that recording has stopped.";
    message("record-error", [lastActionError, problem].filter(Boolean).join(" "));
    message("record-warning", state.warning || "");
    el("record-progress-wrap").hidden = !saving;
    el("record-progress-label").textContent = state.raw_data_saved ? `Preparing video · ${Math.round((state.video_progress || 0)*100)}%` : "Saving captured data…";
    el("record-progress").value = state.video_progress || 0;
    el("record-storage").textContent = state.free_bytes == null ? "Free space is checked before recording" : `${(state.free_bytes/1024**3).toFixed(1)} GB free on Pi · ${((state.bytes_written || 0)/1024**3).toFixed(2)} GB captured`;
  }
  const status = new LatestPoll({url:()=>ctx.url("api/recordings/status"), interval:500, timeout:3000,
    read:r=>r.json(), onValue(value) {current=value; known=true; if (reconcile) {lastActionError=""; reconcile=false;} show();},
    onError() {known=false; show();},
  });
  async function action(name) {
    if (pending) return;
    pending = true; lastActionError = ""; status.stop(); show();
    try {
      const response = await fetch(ctx.url(`api/recordings/${name}`), {method:"POST", headers:{"Content-Type":"application/json", "X-UMI-Action":"recording"},
        body:JSON.stringify(name === "start" ? {label:el("record-label").value} : {id:current.id}), signal:AbortSignal.timeout(10000)});
      const result = await response.json();
      if (!response.ok) {lastActionError=result.error || "The Pi could not complete this request."; return;}
      current = {...result, devices:current?.devices}; known = true;
    } catch(error) {
      known = false; reconcile = true;
      lastActionError = error.name === "TimeoutError" || error.name === "TypeError" ? "Could not confirm the request. Checking the Pi before enabling controls again." : error.message;
    } finally {pending=false; show(); if (!disposed && ctx.active()) status.start();}
  }
  el("record-start").addEventListener("click", ()=>action("start"));
  el("record-stop").addEventListener("click", ()=>action("stop"));
  function history(entries) {
    message("record-history-error", "");
    const key = JSON.stringify(entries);
    if (key === historyKey) return;
    historyKey = key;
    const fragment = document.createDocumentFragment();
    if (!entries.length) {
      const empty=document.createElement("p"); empty.className="recordings-empty"; empty.textContent="Your first recording will appear here after you start."; fragment.append(empty);
    }
    for (const item of entries) {
      const row=document.createElement("div"), info=document.createElement("div"), title=document.createElement("strong"), details=document.createElement("small"), links=document.createElement("div");
      row.className="recording-row"; info.className="recording-info"; links.className="recording-links";
      title.textContent=item.label || (item.recording_mode === "task" ? "Untitled session" : "Time calibration");
      details.textContent=`${item.id} · ${clock(item.elapsed_s)} · ${(item.frames || 0).toLocaleString()} images`;
      info.append(title,details);
      if (item.error || item.warning) {const problem=document.createElement("span"); problem.className="recording-problem"; problem.textContent=[item.error,item.warning].filter(Boolean).join(" "); info.append(problem);}
      const link = (kind, label) => {const a=document.createElement("a"); a.className="button"; a.href=ctx.url(`api/recordings/download?id=${encodeURIComponent(item.id)}&kind=${kind}`); a.textContent=label; a.download=""; links.append(a);};
      if (item.video_status === "ready") link("video", "Download MP4");
      if (item.raw_data_saved || item.status === "error") link("data", "Video + IMU data");
      if (item.status === "recording" || item.status === "saving") {const note=document.createElement("span"); note.textContent=item.status === "recording" ? "Recording…" : "Saving…"; links.append(note);}
      row.append(info,links); fragment.append(row);
    }
    el("record-history").replaceChildren(fragment);
  }
  const saved = new LatestPoll({url:()=>ctx.url("api/recordings"), interval:2000, timeout:3000, read:r=>r.json(), onValue:history,
    onError() {message("record-history-error", "Could not refresh saved recordings. Existing files stay on the Pi.");},
  });
  const sync = () => {
    if (!disposed && ctx.active()) {status.start(); saved.start();}
    else {status.stop(); saved.stop();}
  };
  document.addEventListener("visibilitychange", sync);
  ctx.onShow(sync);
  ctx.onHide(sync);
  sync();
  return () => {disposed = true; sync(); document.removeEventListener("visibilitychange", sync);};
}
