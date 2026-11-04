// Entry point of the Pi's own page (index.html): mounts app.js into <body> with a minimal ctx
// (same-origin relative URLs). The hub mounts the same module from hub/web/views/umi/view.js with its own ctx.
import {mount as dashboard} from "./app.js";

const hooks = {show: [], hide: []};
let pageActive = true;
const fire = kind => hooks[kind].forEach(fn => fn());
const ctx = {
  url: path => String(path).replace(/^\//, ""),
  href: () => "./",
  active: () => pageActive && !document.hidden,
  onShow: fn => hooks.show.push(fn),
  onHide: fn => hooks.hide.push(fn),
};
dashboard(document.body, ctx);
window.addEventListener("pagehide", () => {pageActive = false; fire("hide");});
window.addEventListener("pageshow", event => {
  if (!event.persisted) return;
  pageActive = true; fire("show");
});
