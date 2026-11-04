// Hub shell: app nav + status and client-side routing to each app's view. Apps start with the hub and
// run until it exits, so the page has no start / stop controls. There is no overview page: / and unknown
// ids open the landing app (/api/hub/apps "landing": the apps.json entry marked "landing": true).
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
const S = { apps: [], landing: null, current: null, views: new Map(), css: new Set() };

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

// ---------------------------------------------------------------- routing
function navigate(href, replace = false) {
  history[replace ? 'replaceState' : 'pushState'](null, '', href);
  route();
}

function route() {
  const id = decodeURIComponent(location.pathname.split('/')[1] || '') || null;
  const known = (x) => S.apps.some((a) => a.id === x);
  if (S.apps.length && !known(id)) {
    history.replaceState(null, '', `/${known(S.landing) ? S.landing : S.apps[0].id}`);
    return route();
  }
  const prev = S.current;
  S.current = id;
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
function makeCtx(app) {
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
  const { ctx, fire, dispose } = makeCtx(app);
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

// Mount a view once its app is running; drop it if the app is no longer up (e.g. the hub restarted and
// the app then failed to start).
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

function renderApp() {
  const a = S.apps.find((x) => x.id === S.current);
  $('app').hidden = !a;
  if (!a) return;
  document.title = `${a.name} · UMI Excavator hub`;
  const ph = $('a-placeholder');
  ph.hidden = S.views.has(a.id);
  $('views').hidden = !S.views.has(a.id);
  if (!S.views.has(a.id)) {
    // A failed app stays down until the hub restarts: show why.
    ph.replaceChildren(el('div', {}, statusText(a)),
      ...(a.traceback ? [el('pre', { class: 'ui-log' }, a.traceback)] : []));
  }
}

function render() {
  if (!S.current) document.title = 'UMI Excavator hub';
  syncViews(); renderNav(); renderApp();
}

async function refresh() {
  try { ({ apps: S.apps, landing: S.landing } = await api('/api/hub/apps')); } catch { return; }
  route();     // also moves off an app that is no longer listed (the hub restarted with other --only)
}

document.addEventListener('visibilitychange', () => {
  const v = S.views.get(S.current);
  if (v) v.fire(document.hidden ? 'hide' : 'show');
});

await refresh();
setInterval(refresh, 2000);
