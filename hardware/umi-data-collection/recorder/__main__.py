"""Entry point: ``python -m recorder [--host H] [--port P] [--token-file F] [--data-dir D] [--no-hardware]``.

The data directory holds the recorder's state: ``recordings/``, ``imus.json`` and ``time-calibration.json``
(missing files read as absent). It is ``--data-dir``, else ``$UMI_DATA_DIR`` (``/data`` in the Docker image, the
Pi's ~/umi-dashboard), else hardware/umi-data-collection/ for local development.
"""
import argparse
import os
from pathlib import Path
import signal
import threading

from .recording import Recorder
from .server import Server, State, camera_worker, imu_worker, load_imu_config, preview_worker
from .time_alignment import TimeAlignment


def main():
    parser = argparse.ArgumentParser(prog="python -m recorder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--no-hardware", action="store_true", help="UI preview with devices explicitly offline")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("UMI_DATA_DIR") or Path(__file__).resolve().parents[1]),
                        help="recordings/, imus.json and time-calibration.json (default $UMI_DATA_DIR, else hardware/umi-data-collection/)")
    parser.add_argument("--imu-config", type=Path,
                        help="IMU list (default imus.json in the data directory; missing = the BerryIMUv3 on bus 1)")
    args = parser.parse_args()
    token = args.token_file.read_text().strip() if args.token_file else ""
    if args.host not in ("127.0.0.1", "localhost", "::1") and len(token) < 32:
        parser.error("A private access token file is required for LAN access.")
    state = State(TimeAlignment(args.data_dir / 'time-calibration.json'))
    state.configure_imus(*load_imu_config(args.imu_config or args.data_dir / 'imus.json'))
    server = Server((args.host, args.port), state, token)
    workers = []
    if args.no_hardware:
        state.error("camera", "This preview runs on your Mac. The live camera connects when the dashboard runs on the Pi.")
        state.error("imu", "Waiting for the dashboard to start on the Pi.")
        for name in state.extra_imus:
            state.extra_imu_error(name, "Waiting for the dashboard to start on the Pi.")
    else:
        state.recorder = Recorder(args.data_dir / 'recordings', health=state.snapshot)
        extra_imus = [(imu_worker, ({'name': name, 'bus': e['bus'], 'address': e['address'], 'model': e['model']},))
                      for name, e in state.extra_imus.items()]
        for target, extra in ((camera_worker, (args.device,)), (imu_worker, ()), (preview_worker, ()), *extra_imus):
            worker = threading.Thread(target=target, args=(state, *extra), daemon=True)
            worker.start()
            workers.append(worker)

    def stop(*_):
        if state.recorder:
            state.recorder.shutdown()
        state.stop.set()
        with state.condition:
            state.condition.notify_all()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Dashboard listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        state.stop.set()
        if state.recorder:
            state.recorder.shutdown()
        for worker in workers:
            worker.join(timeout=4)
        if state.recorder and state.recorder.thread:
            state.recorder.thread.join(timeout=20)
        server.server_close()


if __name__ == "__main__":
    main()
