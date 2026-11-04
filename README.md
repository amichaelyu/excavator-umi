# UMI for Excavators: Learning Scooping from a Tabletop Excavator

[[Paper]](https://www.yumichael.com/api/assets/michaelyu038385253749.pdf) [[Dataset]](https://huggingface.co/datasets/amichaelyu/umi-excavator) [[Model]](https://huggingface.co/amichaelyu/umi-excavator)

![System overview: handheld demonstration collection, pose reconstruction, kinematic retargeting in a MuJoCo twin, and π0.5 fine-tuning](assets/overview.png)

## Setup

Python is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                    # hub, digital twin, data tools, tests
uv sync --group training   # + JAX / OpenPI for π0.5 training (add --group cuda on a Linux GPU host)
```

Per-machine settings live in `configs/hub/local.json` (gitignored). Copy
[`configs/hub/local.example.json`](configs/hub/local.example.json) and set the Pi's URLs and token file
(`hosts.pi`) and the GPU box used for training (`hosts.lambda`). The hub's policy checkpoints live in
`configs/hub/local_checkpoints.json` (gitignored too); start it from
[`configs/hub/local_checkpoints.example.json`](configs/hub/local_checkpoints.example.json) or `{"checkpoints": []}`.

Your machine's wiring and calibration are per-machine too (gitignored), each with a tracked `.example.json`
template that holds the paper rig's values:

| File | What it holds | Updated by |
| --- | --- | --- |
| `configs/hardware/hardware_map.json` | ESP32 pin map, homing, joint ranges and speeds, link-IMU and slew calibration | **Hardware calibration** in the hub ([step 4](#4-rollout-with-the-hub)) |
| `configs/hardware/camera-calibration/camera.json` | Bucket camera intrinsics and camera-IMU extrinsics | `policy.processing.camera_calibration` ([step 1](#1-camera-calibration)) |
| `configs/hardware/robot_full4_calibrated.json` | The twin with your measured joint ranges and speeds | `python -m digital_twin.sim.build --calibrated` |

Each is created from its example the first time it is needed, so a fresh checkout runs as is; recalibrate to
replace the example values with your own. The tests always read the examples. `configs/hardware/robot.json` (the
uncalibrated E568 model), `configs/hardware/self_mask.json`, `configs/hub/apps.json` and `configs/policy/` are shared.

In a checkout from before these files moved into `configs/hardware/`, move your own copies (git doesn't move
ignored files; without them you get the example values again):

```sh
mkdir -p configs/hardware/camera-calibration
mv configs/hardware_map.json configs/robot_full4_calibrated.json configs/hardware/
mv configs/camera.json configs/hardware/camera-calibration/
mv configs/camera-calibration/* configs/hardware/camera-calibration/ 2>/dev/null   # raw calibration outputs, if any
```

## Layout

- `hardware/` ([README](hardware/README.md)):
  - `hardware/umi-data-collection/`: the recorder and dashboard that run on the Raspberry Pi (handheld bucket and excavator).
  - `hardware/rc-control/`: ESP32 firmware that presses the excavator's RC controller buttons over Bluetooth LE.
- `policy/`: camera calibration, ORB-SLAM3 reconstruction, twin retargeting, π0.5 training, evaluation and serving.
- `digital_twin/`: the MuJoCo twin, hardware calibration, joint-state estimation and the policy runner.
- `hub/`: one local web app for every UI (data collection, GPIO bridge, inference, data review).
- `configs/`: `hardware/` (hardware map, robot and self-mask configs, and `camera-calibration/` with the camera
  config), `hub/` (hub apps and local settings) and `policy/` (training configs).
- `calibration-target/`: the AprilGrid used for camera calibration.

## 1. Camera calibration

The camera calibration (fisheye lens and camera–IMU) is stored in `configs/hardware/camera-calibration/camera.json`.

1. Print [`calibration-target/aprilgrid.svg`](calibration-target/aprilgrid.svg) at full size. Measure the outer
   width of the black grid in millimetres.
2. Record a take with the handheld rig (see [step 2](#2-umi-data-collection)): move the camera slowly over the grid
   from several angles and distances, keeping the whole grid in view. Download the take's **Data** archive (`.tar`).
3. Fit the lens (OpenCV fisheye) and the camera–IMU rotation, time offset and gyro bias. This writes
   `configs/hardware/camera-calibration/camera.json` in place, which reconstruction uses; the raw outputs go to
   `configs/hardware/camera-calibration/<recording id>/`:

   ```sh
   uv run python -m policy.processing.camera_calibration <take>.tar --grid-mm <measured width>
   ```

   Add `--dry-run` to write only the raw outputs and leave `configs/hardware/camera-calibration/camera.json` unchanged.
   `configs/hardware/camera-calibration/camera.example.json` is the calibration of the paper's handheld rig.

## 2. UMI data collection

The handheld bucket carries a Raspberry Pi with an RGB camera and an IMU. The Pi runs the recorder in Docker
and serves a dashboard with a live preview and record controls.

1. Deploy the recorder to the Pi from your Mac. This copies the image's build context, builds it on the Pi, starts
   it with Docker Compose and saves the Pi's access token to
   `hardware/umi-data-collection/.devices/bucket/access-token`:

   ```sh
   bash hardware/umi-data-collection/deploy/deploy.sh pi@<pi-host>
   ```

   Use `--no-build` to restart with the image already on the Pi, and `--diagnose` for read-only Wi-Fi, power and
   container checks.
2. Start the hub and open the dashboard at <http://127.0.0.1:8888/pi-dashboard>. The hub relays the Pi's API with the
   access token from `hosts.pi` in `configs/hub/local.json`:

   ```sh
   uv run python -m hub
   ```

3. Record demonstrations: start a take, scoop with the handheld bucket, stop. Each take can be downloaded as
   **Data** (`metadata.json`, `frames.csv`, `imu.csv`, `video.mjpg`) and **Video** (MP4).
4. Put each take in `dataset/<id>/` as `recording.tar` (the Data archive), its `metadata.json` and `video.mp4`.
   The recordings root is `dataset/` unless `$UMI_DATA_DIR` or a config's `source.recordings[].data_root` says
   otherwise. To copy every take off the Pi at once with `rsync` and convert them to this layout, see
   [hardware/README.md](hardware/README.md#copy-every-take-off-the-pi).

To start from the paper's demonstrations instead, download the published dataset (same layout):

```sh
uvx --from huggingface_hub hf download amichaelyu/umi-excavator --repo-type dataset --local-dir dataset
```

## 3. Policy training

Every stage is driven by one config ([`configs/policy/dirt-orb.yaml`](configs/policy/dirt-orb.yaml)), validated by
`policy/configs/schema.py`:

```sh
uv run python -m policy <stage> <config> [--dry-run] [--host ubuntu@<gpu-box>] [--runner local|docker|lambda]
```

1. Check the config and print the plan (where each stage runs, its inputs and outputs):

   ```sh
   uv run python -m policy check dirt-orb
   ```

2. Run the stages in order, or all at once with `all`. Finished work is skipped:

   | Stage | What it does |
   | --- | --- |
   | `reconstruct` | ORB-SLAM3 map factors, KLT two-view constraints and IMU preintegration in a batch smoother give each recording's bucket trajectory (ORB-SLAM3 runs in the processing image: `docker compose build processing`) |
   | `labels` | Retargets each trajectory onto the twin (workspace fit, joint-space IK, contact-aware pitch, path retiming) and writes four-joint action labels |
   | `cohort` | Exports the training dataset, renders twin masks for CLOAK and freezes the train/holdout split |
   | `train` | Fine-tunes π0.5 on the GPU box (`hosts.lambda`); detached, check on it with `status` |
   | `eval` | Normalized joint-action MAE on the held-out session for each saved step |

   ```sh
   uv run python -m policy all dirt-orb
   uv run python -m policy status dirt-orb
   ```

   Review recordings next to their twin replay in the data review app at <http://127.0.0.1:8888/data-review>.
   Rejected episodes go in `source.exclude`.
3. Serve the chosen step (`serve.step`, default `best`) on the GPU box and register it with the hub. Then forward
   its port to your machine:

   ```sh
   uv run python -m policy serve dirt-orb --register
   ssh -N -L 8020:127.0.0.1:8020 ubuntu@<gpu-box>
   ```

   `--register` adds the endpoint to `configs/hub/local_checkpoints.json`. `python -m policy stop dirt-orb` removes
   the train and serve containers.

## 4. Rollout with the hub

The excavator carries a Raspberry Pi (running the same recorder as the handheld bucket), an RGB camera on the
bucket, and IMUs on the boom, stick and bucket. An ESP32 presses the RC controller's buttons, and the hub drives it
over Bluetooth LE (wiring, pins and protocol: [hardware/README.md](hardware/README.md#rc-control-esp32)).

![ESP32 DevKit on a breakout board, wired to the button pads of the excavator's RC controller board](assets/rc-control.jpg)

1. Flash the ESP32 firmware with PlatformIO:

   ```sh
   uvx platformio run -d hardware/rc-control -t upload
   ```

2. Deploy the recorder to the excavator's Pi as in [step 2](#2-umi-data-collection) and point `hosts.pi` in
   `configs/hub/local.json` at it.
3. Start the hub natively (on macOS Docker can't reach Bluetooth) and open <http://127.0.0.1:8888/> (it lands on
   UMI data collection; the sidebar has the other apps):

   ```sh
   uv run python -m hub
   ```

   - **GPIO bridge** (`/gpio`): connects to the ESP32 (`ble://gpio`); manual pin control. Pin names and joint
     directions are in `configs/hardware/hardware_map.json`.
   - **Inference** (`/inference`): runs the served π0.5 policies on the excavator.
4. Set up the checkpoint registry, `configs/hub/local_checkpoints.json`. It is per machine (gitignored) and lists
   the policy servers **Inference** can run. Create it on the hub machine from the example, or empty:

   ```sh
   cp configs/hub/local_checkpoints.example.json configs/hub/local_checkpoints.json   # or: echo '{"checkpoints": []}' > ...
   ```

   Then add a checkpoint in one of three ways:
   - `uv run python -m policy serve <config> --register` (see [step 3](#3-policy-training)) starts the server on
     the GPU box and writes its entry.
   - In **Inference**, open **Lambda endpoints**, enter a label, the forwarded URL and the prompt, and click
     **Check & add**. The hub connects once and only saves servers that are 4-joint CLOAK π0.5 policies.
   - Edit the file by hand. Each entry in `checkpoints` looks like this:

     ```json
     {
       "id": "cloak-dirt-good-cloak-1000",
       "task": "dirt",
       "mode": "cloak",
       "prompt": "scoop the dirt",
       "step": 1000,
       "gateway_url": "ws://127.0.0.1:8040",
       "gateway_env": "UMI_EXCAVATOR_CLOAK_DIRT_GOOD_CLOAK_URL"
     }
     ```

     | Field | Meaning |
     | --- | --- |
     | `id` | Unique name; also the label shown in the picker unless you add `label`. |
     | `task`, `mode` | Training task (`dirt`, `bottlecap`) and variant (`cloak`, `cloak_oov`, `baseline`, `noaug`, `pretrained`). |
     | `prompt` | The prompt the page fills in when you pick the checkpoint. |
     | `step` | Training step served. |
     | `gateway_url` | The policy server's websocket on this machine, i.e. the local end of the SSH forward (`ssh -N -L 8040:127.0.0.1:8040 ubuntu@<gpu-box>`). Each checkpoint needs its own port. |
     | `gateway_env` | Optional environment variable that overrides `gateway_url` when set. |
     | `label`, `default` | Optional: display name, and `true` to preselect this checkpoint. |

   The page reads the file on **Refresh checkpoints**; there is no need to restart the hub.
5. In **Inference**:
   1. Calibrate the arm under **Hardware calibration**: tick the prerequisites, then **Calibrate**. It moves the
      machine, fits the link IMUs, joint ranges, speeds and dead times, and merges them into the hardware map.
      (A simulated run, which sends nothing, is CLI only: `uv run python -m digital_twin.hardware.calibrate`.)
   2. Put the arm in the standard pose: a policy run does not home. Home it from the command line (with the hub
      running); this drives the stick, boom and bucket into their top stops:

      ```sh
      uv run python -m digital_twin.policy.runner --home-only
      ```

      It runs outside the Inference page, so the page's **Stop** and **Reset** can't stop it: press Ctrl-C (the
      runner releases every button on exit), or **Release all** on the GPIO page. Like a policy run, it needs the
      calibrated arm IMUs.

   3. The profile is **π0.5 · standard tabletop excavator** (`pi05_tabletop_excavator`, all four joints). Pick a
      registered checkpoint and the prompt (`scoop the dirt`). Check the live camera, the twin's joint
      estimate and the observation the policy will see.
   4. **Start** runs the policy on the machine right away. There is no dry run and no arming step. The policy's
      ten predicted targets per chunk run as speed-limited button pulses, and the boom, stick and bucket IMUs
      give the joint state. A run is refused if they are not calibrated or not live.
   5. **Stop** ends the job and releases every button. **Reset** also releases every button. Closing every
      Inference page during a job triggers an emergency stop. Each run is recorded under
      `artifacts/inference-runs/`.
