// ES module: the Pi pages (pi.js) and the hub views import it.
// The next request starts only after the previous response has been consumed.
// Stop/start invalidates late results even if an abort races with completion.
export class LatestPoll {
  constructor({url, read, onValue, onError, dispose = () => {}, interval = 50, timeout = 1000}) {
    Object.assign(this, {url, read, onValue, onError, dispose, interval, timeout});
    this.running = false;
    this.busy = false;
    this.generation = 0;
    this.failures = 0;
  }
  start() {
    if (this.running) return;
    this.running = true;
    this.generation++;
    this.failures = 0;
    if (!this.busy) this.step();
  }
  stop() {
    this.running = false;
    this.generation++;
    clearTimeout(this.timer);
    this.controller?.abort();
  }
  async step() {
    if (!this.running || this.busy) return;
    this.busy = true;
    const generation = this.generation, started = performance.now();
    const controller = new AbortController();
    this.controller = controller;
    const deadline = setTimeout(() => controller.abort(), this.timeout);
    let value = null;
    try {
      const response = await fetch(this.url(), {cache: "no-store", signal: controller.signal});
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      value = response.status === 204 ? null : await this.read(response);
      const elapsed = performance.now() - started;
      if (controller.signal.aborted || elapsed >= this.timeout) throw new Error("Response arrived too late");
      if (!this.running || generation !== this.generation) return;
      if (value !== null) {
        this.onValue(value, elapsed);
        value = null; // The recipient owns any image URL after accepting it.
      }
      this.failures = 0;
    } catch (error) {
      if (this.running && generation === this.generation) {
        this.failures++;
        this.onError(error);
      }
    } finally {
      if (value !== null) this.dispose(value);
      clearTimeout(deadline);
      this.busy = false;
      if (this.running) {
        const delay = this.failures ? Math.min(1000, 200 * this.failures) : Math.max(0, this.interval - (performance.now() - started));
        this.timer = setTimeout(() => this.step(), delay);
      }
    }
  }
}
