// GPIO bridge view (hub app "gpio", digital_twin/hardware/gpio.py): manual pin control, pin names and
// session recordings over the bridge websocket (/api/gpio/ws), using the ESP32's JSON protocol
// (rc-control/src/main.cpp).

const PAGE = `
<header class="ui-header">
  <span class="ui-mark" aria-hidden="true"></span>
  <h1 class="ui-title">GPIO control <span class="ui-sub">ESP32 · simulated button presses</span></h1>
  <div class="ui-header-end">
    <span class="chip ui-tag" id="bridgeChip"><i></i>Bridge</span>
    <span class="chip ui-tag" id="espChip" title=""><i></i>ESP32</span>
  </div>
</header>

<main>
  <section class="rec-bar" id="recBar" aria-label="Recording">
    <button type="button" class="rec-btn" id="recBtn" disabled>
      <i class="rec-dot"></i><span id="recBtnText">Start recording</span>
    </button>
    <input id="recName" type="text" maxlength="80" placeholder="Session name (optional)" autocomplete="off">
    <div class="rec-status" id="recStatus">Not recording</div>
  </section>

  <section class="toolbar" aria-label="Pulse settings">
    <div class="field">
      <label for="gpio-ms">Pulse</label>
      <div class="ms-input">
        <input id="gpio-ms" type="number" min="1" max="60000" step="1" value="200" inputmode="numeric">
        <span>ms</span>
      </div>
    </div>
    <div class="presets" role="group" aria-label="Pulse presets">
      <button type="button" data-ms="100">100</button>
      <button type="button" data-ms="200">200</button>
      <button type="button" data-ms="500">500</button>
      <button type="button" data-ms="1000">1s</button>
      <button type="button" data-ms="3000">3s</button>
    </div>
    <div class="spacer"></div>
    <button type="button" class="btn primary btn-primary" id="trigAll">Trigger all</button>
    <button type="button" class="btn" id="releaseAll">Release all</button>
  </section>

  <section class="grid" id="grid" aria-live="polite">
    <p class="empty" id="empty">Waiting for the ESP32…</p>
  </section>

  <section class="log-wrap">
    <div class="log-head">
      <h2>Activity</h2>
      <span class="hint">Keys <kbd>1</kbd>–<kbd>9</kbd> trigger pins in on-screen order · hold <kbd>Shift</kbd> + key to hold pressed</span>
    </div>
    <ol class="log" id="log"></ol>
  </section>

  <section class="recordings">
    <div class="log-head">
      <h2>Recordings</h2>
      <span class="hint">Saved in <code>rc-control/recordings/</code> · one event per line</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Session</th><th>Started</th><th class="num">Length</th><th class="num">Events</th><th>Download</th></tr></thead>
        <tbody id="recList"><tr><td colspan="5" class="muted">No recordings yet</td></tr></tbody>
      </table>
    </div>
  </section>
</main>`;

// ---- Grouping by function --------------------------------------------------
//
// Labels like "Big arm up" / "Big arm down" are split into a group ("Big arm")
// and a direction ("up"), so opposite controls sit side by side.

const DIRECTIONS = {
  forward: { rank: 0, arrow: "↑" }, fwd: { rank: 0, arrow: "↑" },
  back: { rank: 1, arrow: "↓" }, backward: { rank: 1, arrow: "↓" }, reverse: { rank: 1, arrow: "↓" },
  left: { rank: 2, arrow: "←" }, right: { rank: 3, arrow: "→" },
  up: { rank: 4, arrow: "↑" }, down: { rank: 5, arrow: "↓" },
  in: { rank: 6, arrow: "→←" }, close: { rank: 6, arrow: "→←" },
  out: { rank: 7, arrow: "←→" }, open: { rank: 7, arrow: "←→" },
  cw: { rank: 8, arrow: "↻" }, ccw: { rank: 9, arrow: "↺" },
};
// Common misspellings, so "drive rigth" still groups with "Drive".
const DIRECTION_ALIASES = {
  rigth: "right", rihgt: "right", rght: "right", rigt: "right", lft: "left", leftt: "left",
  foward: "forward", forwad: "forward", forwards: "forward", backwards: "backward", bakward: "backward",
  dwn: "down",
};
// Rough front-to-back order of an excavator's controls; anything else sorts after.
const GROUP_ORDER = ["drive", "track", "base", "turn", "swing", "big arm", "boom", "small arm", "stick", "arm", "bucket", "gripper", "claw"];

const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);

function parseLabel(label) {
  const words = label.trim().split(/\s+/);
  const raw = words.at(-1)?.toLowerCase() ?? "";
  const last = DIRECTION_ALIASES[raw] ?? raw;
  if (words.length > 1 && Object.hasOwn(DIRECTIONS, last)) {
    return { group: words.slice(0, -1).join(" "), dir: last, ...DIRECTIONS[last] };
  }
  return { group: label.trim(), dir: "", rank: 99, arrow: "" };
}

function groupRank(title) {
  const t = title.toLowerCase();
  const i = GROUP_ORDER.findIndex((k) => t.includes(k));
  return i < 0 ? GROUP_ORDER.length : i;
}

function fmtDuration(ms) {
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  const mmss = `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  return h ? `${h}:${mmss}` : mmss;
}

export async function mount(root, ctx) {
  ctx.css("style.css");
  root.innerHTML = PAGE;
  const $ = (id) => root.querySelector("#" + id);
  const grid = $("grid");
  const logEl = $("log");
  const msInput = $("gpio-ms");

  const cards = new Map();
  const order = []; // pins in on-screen order (drives keys 1-9 and "trigger all")
  const firmwareOrder = [];
  const timers = [];
  let disposed = false;

  let ws = null;
  let bridgeUp = false;
  let espUp = false;
  let nextId = 1;
  const sent = new Map(); // id -> description, for log lines

  // ---- Helpers ---------------------------------------------------------------

  function log(text, isError = false) {
    const li = document.createElement("li");
    if (isError) li.className = "err";
    const t = document.createElement("time");
    t.textContent = new Date().toLocaleTimeString();
    li.append(t, text);
    logEl.prepend(li);
    while (logEl.children.length > 200) logEl.lastChild.remove();
  }

  function pulseMs() {
    const v = Math.round(Number(msInput.value));
    return Number.isFinite(v) && v >= 1 ? Math.min(v, 60000) : 200;
  }

  // Friendly pin names ("Drive forward"), shared by all clients via the bridge.
  let labels = {};

  function label(pin) {
    if (pin === "all") return "all pins";
    return labels[pin] || cards.get(pin)?.info.name || `GPIO${pin}`;
  }

  function send(cmd, pin, extra = {}) {
    if (!ws || ws.readyState !== WebSocket.OPEN || !espUp) return;
    const id = nextId++;
    const desc = cmd === "trigger" ? `trigger ${label(pin)} ${extra.ms}ms` : `${cmd} ${label(pin)}`;
    sent.set(id, desc);
    ws.send(JSON.stringify({ cmd, pin, id, ...extra }));
  }

  const trigger = (pin) => {
    const ms = pulseMs();
    // "all" means every pin, triggered one at a time.
    for (const p of pin === "all" ? order : [pin]) {
      send("trigger", p, { ms });
      animatePulse(p, ms);
    }
  };

  // ---- Rendering -------------------------------------------------------------

  function animatePulse(pin, ms) {
    const c = cards.get(pin);
    if (!c) return;
    const b = c.bar.firstElementChild;
    b.style.transition = "none";
    b.style.width = "100%";
    void b.offsetWidth; // restart the transition
    b.style.transition = `width ${ms}ms linear`;
    b.style.width = "0%";
  }

  function makeCard(info) {
    const el = document.createElement("article");
    el.className = "card";
    el.innerHTML = `
      <div class="card-head">
        <span class="arrow" aria-hidden="true"></span>
        <div class="titles">
          <button type="button" class="pin-name" title="Click to rename"></button>
          <div class="pin-meta">GPIO ${info.pin}</div>
        </div>
        <span class="key" title="Keyboard shortcut"></span>
      </div>
      <div class="level"><span class="txt">Released</span><span class="bar"><b></b></span></div>
      <div class="actions">
        <button type="button" class="btn primary trig">Trigger</button>
        <button type="button" class="btn hold">Hold</button>
      </div>`;
    const nameEl = el.querySelector(".pin-name");
    nameEl.onclick = () => editName(info.pin, nameEl);

    const hold = el.querySelector(".hold");
    el.querySelector(".trig").onclick = () => trigger(info.pin);

    let held = false;
    hold.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      held = true;
      hold.setPointerCapture(e.pointerId);
      hold.classList.add("active");
      send("press", info.pin, { autorelease: true });
    });
    const release = () => {
      if (!held) return;
      held = false;
      hold.classList.remove("active");
      send("release", info.pin);
    };
    hold.addEventListener("pointerup", release);
    hold.addEventListener("pointercancel", release);
    hold.addEventListener("lostpointercapture", release);

    firmwareOrder.push(info.pin);
    const card = {
      el,
      key: el.querySelector(".key"),
      arrow: el.querySelector(".arrow"),
      bar: el.querySelector(".bar"),
      txt: el.querySelector(".txt"),
      meta: el.querySelector(".pin-meta"),
      name: nameEl,
      hold,
      info,
    };
    cards.set(info.pin, card);
    return card;
  }

  function layout() {
    const groups = new Map();
    const unassigned = [];
    for (const pin of firmwareOrder) {
      const lbl = labels[pin];
      const fw = firmwareOrder.indexOf(pin);
      if (!lbl) {
        unassigned.push({ pin, rank: fw });
        continue;
      }
      const p = parseLabel(lbl);
      const key = p.group.toLowerCase();
      if (!groups.has(key)) groups.set(key, { title: cap(p.group), items: [] });
      groups.get(key).items.push({ pin, rank: p.rank * 100 + fw });
    }

    const sorted = [...groups.values()].sort(
      (a, b) => groupRank(a.title) - groupRank(b.title) || a.title.localeCompare(b.title),
    );
    const sections = [];
    order.length = 0;
    const addSection = (title, items, extra = "") => {
      const sec = document.createElement("section");
      sec.className = `group ${extra}`.trim();
      const head = document.createElement("header");
      const h = document.createElement("h3");
      h.textContent = title;
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = extra ? "click a name to assign it" : `${items.length} pin${items.length === 1 ? "" : "s"}`;
      head.append(h, count);
      const body = document.createElement("div");
      body.className = "group-cards";
      for (const it of items.sort((a, b) => a.rank - b.rank)) {
        body.append(cards.get(it.pin).el);
        order.push(it.pin);
      }
      sec.append(head, body);
      sections.push(sec);
    };
    for (const g of sorted) addSection(g.title, g.items);
    if (unassigned.length) addSection("Unassigned", unassigned, "unassigned");
    grid.replaceChildren(...sections);

    for (const c of cards.values()) c.key.textContent = "";
    order.forEach((pin, i) => (cards.get(pin).key.textContent = i < 9 ? String(i + 1) : ""));
    for (const c of cards.values()) renderName(c);
    applyButtonState();
  }

  function applyButtonState() {
    const ready = bridgeUp && espUp;
    root.querySelectorAll(".btn").forEach((b) => {
      b.disabled = !ready;
    });
  }

  function renderName(c) {
    const board = c.info.name ?? `GPIO ${c.info.pin}`;
    const custom = labels[c.info.pin];
    const p = custom ? parseLabel(custom) : null;
    const title = p?.dir ? cap(p.dir) : custom || board;
    if (!c.name.querySelector("input")) c.name.textContent = title;
    c.name.title = custom ? `${custom} (click to rename)` : "Click to name this pin";
    c.arrow.textContent = p?.arrow ?? "";
    c.el.classList.toggle("unnamed", !custom);
    c.meta.textContent = custom ? `${board} · GPIO ${c.info.pin}` : `GPIO ${c.info.pin} · unassigned`;
  }

  function editName(pin, el) {
    if (el.querySelector("input")) return;
    const input = document.createElement("input");
    input.className = "name-input";
    input.maxLength = 40;
    input.value = labels[pin] ?? "";
    input.placeholder = cards.get(pin)?.info.name ?? `GPIO ${pin}`;
    el.replaceChildren(input);
    input.focus();
    input.select();
    let done = false;
    const finish = (save) => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      input.remove();
      if (save && value !== (labels[pin] ?? "") && ws?.readyState === WebSocket.OPEN) {
        labels = { ...labels, [pin]: value };
        ws.send(JSON.stringify({ cmd: "set_label", pin, label: value }));
      }
      if (cards.size) layout();
    };
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") finish(true);
      else if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
    input.addEventListener("click", (e) => e.stopPropagation());
  }

  function render(pins) {
    $("empty")?.remove();
    let added = false;
    // Drop cards for pins the firmware no longer reports (e.g. after a reflash with fewer pins).
    const current = new Set(pins.map((p) => p.pin));
    for (const [pin, c] of cards) {
      if (current.has(pin)) continue;
      c.el.remove();
      cards.delete(pin);
      firmwareOrder.splice(firmwareOrder.indexOf(pin), 1);
      added = true;
    }
    for (const info of pins) {
      let c = cards.get(info.pin);
      if (!c) {
        c = makeCard(info);
        added = true;
      }
      const wasPressed = c.info.pressed;
      c.info = info;
      const high = info.mode === "push-high";
      c.el.classList.toggle("low", info.pressed);
      c.txt.textContent = info.pressed ? (high ? "Pressed · 3.3V" : "Pressed · GND")
                                       : (high ? "Released · 0V" : "Released");
      renderName(c);
      if (wasPressed && !info.pressed) {
        const b = c.bar.firstElementChild;
        b.style.transition = "none";
        b.style.width = "100%";
      }
    }
    if (added) layout();
  }

  // ---- Recording -------------------------------------------------------------

  const recBar = $("recBar");
  const recBtn = $("recBtn");
  const recName = $("recName");
  let recState = { active: false };
  let recSyncedAt = 0; // performance.now() when recState.elapsedMs was received

  function renderRecording() {
    const on = recState.active;
    recBar.classList.toggle("active", on);
    $("recBtnText").textContent = on ? "Stop recording" : "Start recording";
    recName.disabled = on;
    recBtn.disabled = !bridgeUp;
    if (recState.active) {
      const elapsed = recState.elapsedMs + (performance.now() - recSyncedAt);
      const lbl = recState.name ? ` · ${recState.name}` : "";
      $("recStatus").textContent = `REC ${fmtDuration(elapsed)} · ${recState.events} events${lbl}`;
    } else {
      $("recStatus").textContent = bridgeUp ? "Not recording" : "Bridge offline";
    }
  }
  timers.push(setInterval(() => recState.active && ctx.active() && renderRecording(), 250));

  recBtn.onclick = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(recState.active ? { cmd: "record_stop" } : { cmd: "record_start", name: recName.value }));
  };
  recName.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !recState.active) recBtn.click();
  });

  async function loadRecordings() {
    let list;
    try {
      list = await ctx.api("recordings");
    } catch {
      return;
    }
    if (disposed) return;
    const body = $("recList");
    body.replaceChildren();
    if (!list.length) {
      body.innerHTML = `<tr><td colspan="5" class="muted">No recordings yet</td></tr>`;
      return;
    }
    for (const r of list) {
      const tr = document.createElement("tr");
      const live = recState.active && recState.id === r.id;
      const name = document.createElement("td");
      name.textContent = r.name || "(untitled)";
      const sub = document.createElement("span");
      sub.className = "sub";
      sub.textContent = r.id;
      name.append(sub);
      if (live) name.insertAdjacentHTML("beforeend", `<span class="tag">recording</span>`);
      else if (!r.complete) name.insertAdjacentHTML("beforeend", `<span class="tag">interrupted</span>`);
      const started = document.createElement("td");
      started.textContent = r.startedAt ? new Date(r.startedAt).toLocaleString() : "?";
      const dur = document.createElement("td");
      dur.className = "num";
      dur.textContent = r.durationMs == null ? "?" : fmtDuration(r.durationMs);
      const ev = document.createElement("td");
      ev.className = "num";
      ev.textContent = String(r.events);
      const dl = document.createElement("td");
      const id = encodeURIComponent(r.id);
      const jsonl = document.createElement("a");
      jsonl.href = ctx.url(`recordings/${id}`);
      jsonl.download = "";
      jsonl.textContent = "JSONL";
      const csv = document.createElement("a");
      csv.href = ctx.url(`recordings/${id}?format=csv`);
      csv.download = "";
      csv.textContent = "CSV";
      dl.append(jsonl, csv);
      tr.append(name, started, dur, ev, dl);
      body.append(tr);
    }
  }

  function refreshControls() {
    renderRecording();
    const ready = bridgeUp && espUp;
    $("bridgeChip").classList.toggle("on", bridgeUp);
    $("espChip").classList.toggle("on", ready);
    applyButtonState();
    const empty = $("empty");
    if (empty) empty.textContent = !bridgeUp ? "Connecting to bridge…" : !espUp ? "Waiting for the ESP32…" : "Loading pins…";
  }

  // ---- Connection ------------------------------------------------------------

  function connect() {
    if (disposed) return;
    const sock = ctx.ws("ws");
    ws = sock;

    sock.onopen = () => {
      bridgeUp = true;
      refreshControls();
    };
    sock.onclose = () => {
      if (ws !== sock) return;
      if (bridgeUp) log("bridge disconnected, retrying…", true);
      bridgeUp = false;
      espUp = false;
      refreshControls();
      if (!disposed) timers.push(setTimeout(connect, 1500));
    };
    sock.onerror = () => sock.close();
    sock.onmessage = (ev) => {
      let m;
      try {
        m = JSON.parse(ev.data);
      } catch {
        return;
      }
      switch (m.type) {
        case "bridge":
          if (m.esp32 !== espUp) log(m.esp32 ? `ESP32 connected (${m.target})` : "ESP32 offline", !m.esp32);
          espUp = m.esp32;
          $("espChip").title = m.target;
          refreshControls();
          break;
        case "state":
          render(m.pins);
          break;
        case "ack":
          log(`ok · ${(m.id !== undefined && sent.get(m.id)) || m.cmd}`);
          if (m.id !== undefined) sent.delete(m.id);
          break;
        case "recording": {
          const wasActive = recState.active;
          const { type: _, ...st } = m;
          recState = st;
          recSyncedAt = performance.now();
          if (st.active && !wasActive) {
            log(`recording started · ${st.id}`);
            loadRecordings();
          }
          if (!st.active && wasActive) log("recording stopped");
          renderRecording();
          break;
        }
        case "recordings_changed":
          loadRecordings();
          break;
        case "labels":
          labels = m.labels;
          if (cards.size) layout();
          break;
        case "error":
          log(`error · ${m.error}${m.id !== undefined && sent.has(m.id) ? ` (${sent.get(m.id)})` : ""}`, true);
          if (m.id !== undefined) sent.delete(m.id);
          break;
      }
    };
  }

  // ---- Toolbar & keyboard ----------------------------------------------------

  const presets = root.querySelectorAll(".presets button");
  function syncPresets() {
    presets.forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.ms) === pulseMs())));
  }
  function savePulse() {
    try {
      localStorage.setItem("gpio.pulseMs", String(pulseMs()));
    } catch {}
  }
  presets.forEach((b) => (b.onclick = () => ((msInput.value = b.dataset.ms), syncPresets(), savePulse())));
  msInput.addEventListener("input", () => (syncPresets(), savePulse()));
  try {
    const saved = localStorage.getItem("gpio.pulseMs");
    if (saved) msInput.value = saved;
  } catch {}
  syncPresets();

  $("trigAll").onclick = () => trigger("all");
  $("releaseAll").onclick = () => send("release", "all");

  // Keyboard shortcuts act only while this view is the one on screen.
  const keyHolds = new Set();
  const releaseKeyHolds = () => {
    for (const pin of keyHolds) {
      cards.get(pin)?.hold.classList.remove("active");
      send("release", pin);
    }
    keyHolds.clear();
  };
  const onKeyDown = (e) => {
    if (!ctx.active()) return;
    const tag = e.target?.tagName;
    if (e.repeat || e.metaKey || e.ctrlKey || e.altKey || tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    const n = Number(e.code.replace("Digit", ""));
    const pin = order[n - 1];
    if (!e.code.startsWith("Digit") || pin === undefined) return;
    e.preventDefault();
    if (e.shiftKey) {
      keyHolds.add(pin);
      cards.get(pin)?.hold.classList.add("active");
      send("press", pin, { autorelease: true });
    } else {
      trigger(pin);
    }
  };
  const onKeyUp = (e) => {
    const pin = order[Number(e.code.replace("Digit", "")) - 1];
    if (pin === undefined || !keyHolds.delete(pin)) return;
    cards.get(pin)?.hold.classList.remove("active");
    send("release", pin);
  };
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", onKeyUp);
  window.addEventListener("blur", releaseKeyHolds);
  ctx.onHide(releaseKeyHolds);
  ctx.onShow(() => { renderRecording(); loadRecordings(); });

  refreshControls();
  loadRecordings();
  connect();

  return () => {
    disposed = true;
    releaseKeyHolds();
    document.removeEventListener("keydown", onKeyDown);
    document.removeEventListener("keyup", onKeyUp);
    window.removeEventListener("blur", releaseKeyHolds);
    for (const t of timers) { clearInterval(t); clearTimeout(t); }
    try { ws?.close(); } catch {}
  };
}
