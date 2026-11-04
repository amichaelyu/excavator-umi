// UMI data collection (hub app pi-dashboard): the Pi's own page modules (hub/web/data-collection, served at
// /web/data-collection/) mounted with the hub's ctx, so their requests go through the hub's relay
// (/api/<id>/<pi path>, hub/apps/umi.py).
import {mount as dashboard} from '/web/data-collection/app.js';

const PAGES = { dashboard };

export async function mount(root, ctx) {
  for (const f of ['style.css', 'recording.css']) ctx.css(`/web/data-collection/${f}`);
  ctx.css('style.css');
  root.innerHTML = `<div class="umi-relay" role="status" hidden></div><div class="umi-page"></div>`;
  const banner = root.querySelector('.umi-relay');
  const host = root.querySelector('.umi-page');

  // Each page gets its own show/hide hooks so an unmounted page is never woken up again.
  let hooks = { show: [], hide: [] }, cleanup = null, page = null;
  ctx.onShow(() => { hooks.show.forEach((fn) => fn()); checkRelay(); });
  ctx.onHide(() => hooks.hide.forEach((fn) => fn()));
  const sub = {
    url: ctx.url,
    active: ctx.active,
    href: (name) => `/${ctx.id}${name === 'dashboard' ? '' : `?page=${name}`}`,
    onShow: (fn) => hooks.show.push(fn),
    onHide: (fn) => hooks.hide.push(fn),
  };

  function show(name) {
    name = PAGES[name] ? name : 'dashboard';
    if (name === page) return;
    if (cleanup) cleanup();
    hooks = { show: [], hide: [] };
    page = name;
    host.dataset.umiPage = name;
    host.replaceChildren();
    cleanup = PAGES[name](host, sub);
  }

  const onClick = (e) => {
    const a = e.target.closest('a[data-umi-link]');
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    const name = a.dataset.umiLink;
    ctx.setParams({ page: name === 'dashboard' ? null : name }, { replace: false });
    show(name);
  };
  root.addEventListener('click', onClick);
  // Back/forward between pages (the shell only routes between apps).
  const onPop = () => { if (ctx.active()) show(ctx.params().get('page')); };
  window.addEventListener('popstate', onPop);

  // Relay health: say why the page is empty instead of letting it spin on "Reconnecting".
  let relayTimer = 0;
  async function checkRelay() {
    clearTimeout(relayTimer);
    if (!ctx.active()) return;
    try {
      const s = await ctx.api('status');
      const msg = !s.reachable ? `The excavator Pi is not reachable (tried ${s.urls.join(', ')}).`
        : !s.token ? `Connected to ${s.upstream}, but no access token file was found (option token_files).`
          : s.authorized === false ? `${s.upstream} rejected the access token (check token_files).` : '';
      banner.textContent = msg;
      banner.hidden = !msg;
    } catch (e) {
      banner.textContent = `Relay: ${e.message}`;
      banner.hidden = false;
    }
    relayTimer = setTimeout(checkRelay, 5000);
  }

  show(ctx.params().get('page'));
  checkRelay();
  return () => {
    clearTimeout(relayTimer);
    root.removeEventListener('click', onClick);
    window.removeEventListener('popstate', onPop);
    if (cleanup) cleanup();
  };
}
