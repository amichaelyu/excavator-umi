// Hub shell: app list + status, start/stop/settings, and client-side routing to each app's view.
//
// A view is an ES module at /web/views/<view>/view.js:
//
//   export async function mount(root, ctx) { ...; return () => { /* cleanup */ }; }
//
// root is a <section class="view" data-view="<view>" data-app="<id>"> that the view owns (query
// inside it, never document-wide: several views can be mounted at once, and one view module can be
// mounted for two apps). Keep state inside mount(), not at module level. Views stay mounted while
// hidden so they keep their state; ctx.onShow / ctx.onHide pause polling. A view's CSS is scoped
// under [data-view="<view>"] and loaded with ctx.css('style.css').
//
// ctx: { id, view, app, base, url(p), api(p, body?, opts?), events(p), ws(p), css(href),
//        params(), setParams(obj, {replace}), onShow(fn), onHide(fn), active(), navigate(href), hub }
const $ = (id) => document.getElementById(id);
const S = { apps: [], current: null, settingsOpen: false, views: new Map(), css: new Set() };

async function api(path, body, opts = {}) {
  const init = body === undefined ? { cache: 'no-store', ...opts }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), ...opts };
  const r = await fetch(path, init);
  const ct = r.headers.get('Content-Type') || '';
  const out = ct.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error((out && out.error) || r.statusText);
  return out;
}

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k.startsWith('on') ? e.addEventListener(k.slice(2), v) : e.setAttribute(k, v);
  e.append(...kids);
  return e;
}

const statusText = (a) => ({ running: 'running', starting: 'starting…', stopped: 'stopped',
  crashed: `failed: ${a.error}` }[a.status] || a.status);

async function act(id, action) {
  try { await api(`/api/hub/apps/${id}/${action}`, {}); }
  catch (e) { alert(`${id} ${action}: ${e.message}`); }
  await refresh();
}

// ---------------------------------------------------------------- routing
function navigate(href, replace = false) {
  history[replace ? 'replaceState' : 'pushState'](null, '', href);
  route();
}

function route() {
  const id = decodeURIComponent(location.pathname.split('/')[1] || '') || null;
  const prev = S.current;
  S.current = S.apps.length && id && !S.apps.some((a) => a.id === id) ? null : id;
  if (prev !== S.current) {
    const old = S.views.get(prev);
    if (old) { old.root.hidden = true; old.fire('hide'); }
    const now = S.views.get(S.current);
    if (now) { now.root.hidden = false; now.fire('show'); }
  }
  render();
}

document.addEventListener('click', (e) => {
  const a = e.target.closest('a[data-nav]');
  if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
  e.preventDefault();
  navigate(a.getAttribute('href'));
});
window.addEventListener('popstate', route);

// ---------------------------------------------------------------- views
function makeCtx(app, root) {
  const base = `/api/${app.id}`;
  const hooks = { show: [], hide: [] };
  const owned = [];
  const url = (p = '') => base + '/' + String(p).replace(/^\//, '');
  const ctx = {
    id: app.id, view: app.view, base, hub: { api, apps: () => S.apps, navigate },
    get app() { return S.apps.find((a) => a.id === app.id) || app; },
    url,
    api: (p, body, opts) => api(url(p), body, opts),
    events(p) { const es = new EventSource(url(p)); owned.push(() => es.close()); return es; },
    ws(p) {
      const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${url(p)}`);
      owned.push(() => ws.close()); return ws;
    },
    css(href) {
      const full = new URL(href, `${location.origin}/web/views/${app.view}/`).pathname;
      if (S.css.has(full)) return;
      S.css.add(full);
      document.head.append(el('link', { rel: 'stylesheet', href: full }));
    },
    params: () => new URLSearchParams(S.current === app.id ? location.search : (S.views.get(app.id)?.search || '')),
    setParams(obj, { replace = true } = {}) {
      const p = ctx.params();
      for (const [k, v] of Object.entries(obj)) (v === null || v === undefined || v === '') ? p.delete(k) : p.set(k, v);
      const q = p.toString();
      const v = S.views.get(app.id);
      if (v) v.search = q ? `?${q}` : '';
      if (S.current === app.id) history[replace ? 'replaceState' : 'pushState'](null, '', `/${app.id}${q ? `?${q}` : ''}`);
    },
    onShow: (fn) => hooks.show.push(fn),
    onHide: (fn) => hooks.hide.push(fn),
    active: () => S.current === app.id && !document.hidden,
    navigate,
  };
  const fire = (k) => hooks[k].forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
  return { ctx, fire, dispose: () => owned.splice(0).forEach((f) => f()) };
}

async function mountView(app) {
  const root = el('section', { class: 'view', 'data-view': app.view, 'data-app': app.id });
  root.hidden = S.current !== app.id;
  $('views').append(root);
  const { ctx, fire, dispose } = makeCtx(app, root);
  const v = { root, fire, cleanup: null, dispose, search: S.current === app.id ? location.search : '' };
  S.views.set(app.id, v);
  try {
    const mod = await import(`/web/views/${app.view}/view.js`);
    v.cleanup = await mod.mount(root, ctx);
  } catch (e) {
    console.error(e);
    root.replaceChildren(el('pre', { class: 'ui-log' }, `View ${app.view} failed to load:\n${e.stack || e}`));
  }
}

function unmountView(id) {
  const v = S.views.get(id);
  if (!v) return;
  S.views.delete(id);
  try { v.cleanup && v.cleanup(); } catch (e) { console.error(e); }
  v.dispose();
  v.root.remove();
}

// Mount a view once its app is running; drop it when the app stops (a restart is a fresh service).
function syncViews() {
  for (const a of S.apps) {
    const has = S.views.has(a.id);
    if (a.status === 'running' && !has && (a.id === S.current || a.preload)) mountView(a);
    else if (has && a.status !== 'running' && a.status !== 'starting') unmountView(a.id);
  }
}

// ---------------------------------------------------------------- rendering
function renderNav() {
  const nav = $('nav'); nav.replaceChildren();
  for (const g of [...new Set(S.apps.map((a) => a.group || 'Other'))]) {
    nav.append(el('h3', {}, g));
    for (const a of S.apps.filter((x) => (x.group || 'Other') === g))
      nav.append(el('a', { href: `/${a.id}`, 'data-nav': '', class: a.id === S.current ? 'active' : '' },
        el('span', { class: `dot ${a.status}` }), a.name));
  }
}

function renderCards() {
  $('cards').replaceChildren(...S.apps.map((a) => {
    const live = a.status === 'running';
    const btn = live ? [el('button', { onclick: () => act(a.id, 'stop') }, 'Stop')]
      : [el('button', { class: 'btn-primary', onclick: () => act(a.id, 'start'), ...(a.status === 'starting' ? { disabled: '' } : {}) },
        a.status === 'starting' ? 'Starting…' : 'Start')];
    return el('div', { class: 'card ui-panel' },
      el('div', { class: 'top' }, el('span', { class: `dot ${a.status}` }), el('b', {}, a.name)),
      el('p', {}, a.description || ''),
      el('div', { class: 'row' }, ...btn, el('a', { href: `/${a.id}`, 'data-nav': '', class: 'btn' }, 'Open'),
        el('span', { class: 'meta' }, statusText(a))));
  }));
}

function renderApp() {
  const a = S.apps.find((x) => x.id === S.current);
  $('home').hidden = !!a; $('app').hidden = !a;
  if (!a) return;
  const live = a.status === 'running';
  document.title = `${a.name} · UMI Excavator hub`;
  $('a-dot').className = `dot ${a.status}`;
  $('a-name').textContent = a.name;
  $('a-status').textContent = statusText(a);
  $('a-start').disabled = live || a.status === 'starting';
  $('a-stop').disabled = !live;
  $('a-restart').disabled = !live;
  if (document.activeElement !== $('a-options')) $('a-options').value = JSON.stringify(a.options, null, 2);
  $('a-trace').hidden = !a.traceback;
  $('a-trace').textContent = a.traceback || '';
  const ph = $('a-placeholder');
  ph.hidden = S.views.has(a.id);
  $('views').hidden = !S.views.has(a.id);
  if (!S.views.has(a.id)) {
    ph.replaceChildren(el('div', {}, a.status === 'starting' ? 'Starting…' : statusText(a)),
      ...(a.status === 'starting' || live ? [] : [el('button', { class: 'btn-primary', onclick: () => act(a.id, 'start') }, `Start ${a.name}`)]));
  }
}

function render() {
  if (!S.current) document.title = 'UMI Excavator hub';
  syncViews(); renderNav(); renderCards(); renderApp();
}

async function refresh() {
  try { S.apps = (await api('/api/hub/apps')).apps; } catch (e) { return; }
  if (S.current && !S.apps.some((a) => a.id === S.current)) S.current = null;
  render();
}

$('a-start').onclick = () => act(S.current, 'start');
$('a-stop').onclick = () => act(S.current, 'stop');
$('a-restart').onclick = () => { unmountView(S.current); act(S.current, 'restart'); };
$('a-settings-btn').onclick = () => {
  S.settingsOpen = !S.settingsOpen; $('a-settings').hidden = !S.settingsOpen;
  $('a-settings-btn').setAttribute('aria-pressed', S.settingsOpen);
};
$('a-save').onclick = async () => {
  const a = S.apps.find((x) => x.id === S.current);
  try {
    await api(`/api/hub/apps/${a.id}/settings`, { options: JSON.parse($('a-options').value || '{}') });
    $('a-options').blur();
    if (a.status === 'running' && confirm(`Restart ${a.name} to apply?`)) { unmountView(a.id); await act(a.id, 'restart'); }
  } catch (e) { alert(e.message); }
  refresh();
};
document.addEventListener('visibilitychange', () => {
  const v = S.views.get(S.current);
  if (v) v.fire(document.hidden ? 'hide' : 'show');
});

await refresh();
route();
setInterval(refresh, 2000);
