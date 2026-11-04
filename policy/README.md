# Policy training

Recordings in, served policy out: reconstruction, labelling, the training cohort, π0.5 fine-tuning, evaluation and
serving, all from one command line (`python -m policy`). See the [main README](../README.md) for the full pipeline.

- [`__main__.py`](__main__.py), [`stages.py`](stages.py): the command line and the stage pipeline.
- [`configs/schema.py`](configs/schema.py): every config key, validated when a config loads.
- [`processing/`](processing/): camera calibration and ORB-SLAM3 reconstruction (`slam/`).
- [`preprocessing.py`](preprocessing.py): recording selection, twin IK labels and the frozen train/holdout cohort.
- [`training/`](training/): the CLOAK / OpenPI π0.5 trainer, bundles and policy server (`cloak/`), the GPU box over
  SSH (`lambda_host.py`) and the training image (`docker/`).
- [`eval/`](eval/): offline holdout evaluation ([Evaluation](#evaluation)).

Training and serving run in Docker on a Linux GPU box reached over SSH (`hosts.lambda` in
`configs/hub/local.json`, or `--host`); the stages sync the code and data to it. The other stages run on your machine.

## Data

Training reads each take from `dataset/<id>/`: `recording.tar` (the take's **Data** archive), its `metadata.json`
and `video.mp4`. The recordings root is `dataset/` unless `$UMI_DATA_DIR` or a config's
`source.recordings[].data_root` says otherwise. Record your own takes
([hardware/README.md](../hardware/README.md#recording-demonstrations)), or download the paper's 85 dirt
demonstrations ([dataset card](https://huggingface.co/datasets/amichaelyu/umi-excavator)):

```sh
uvx --from huggingface_hub hf download amichaelyu/umi-excavator --repo-type dataset --local-dir dataset
```

## Train a policy

Every stage is driven by one config in [`configs/policy/`](../configs/policy/), validated by
[`configs/schema.py`](configs/schema.py). There is one per policy in the paper:

| Config | Policy | Serves on |
| --- | --- | --- |
| [`dirt-orb-cloak`](../configs/policy/dirt-orb-cloak.yaml) | π0.5 + CLOAK, the robot masked with the twin's self-mask | port 8020 |
| [`dirt-orb`](../configs/policy/dirt-orb.yaml) | π0.5 baseline without CLOAK | port 8021 |

They share the reconstruction, labels and train/holdout split, so those stages run once. The commands below use
`dirt-orb-cloak`.

```sh
uv run python -m policy <stage> <config> [--dry-run] [--host ubuntu@<gpu-box>] [--runner local|docker|lambda]
```

1. Check the config and print the plan (where each stage runs, its inputs and outputs):

   ```sh
   uv run python -m policy check dirt-orb-cloak
   ```

2. Run the stages in order, or all at once with `all`. Finished work is skipped:

   | Stage | What it does |
   | --- | --- |
   | `reconstruct` | ORB-SLAM3 map factors, KLT two-view constraints and IMU preintegration in a batch smoother give each recording's bucket trajectory (ORB-SLAM3 runs in the processing image: `docker compose build processing`) |
   | `labels` | Retargets each trajectory onto the twin (workspace fit, joint-space IK, contact-aware pitch, path retiming) and writes four-joint action labels |
   | `cohort` | Exports the training dataset, renders twin masks for CLOAK and freezes the train/holdout split |
   | `train` | Fine-tunes π0.5 on the GPU box (`hosts.lambda`); detached, check on it with `status` |
   | `eval` | Normalized joint-action MAE on the held-out recordings for each saved step ([Evaluation](#evaluation)) |

   ```sh
   uv run python -m policy all dirt-orb-cloak
   uv run python -m policy status dirt-orb-cloak
   ```

   Review recordings next to their twin replay in the data review app at <http://127.0.0.1:8888/data-review>.
   Rejected episodes go in `source.exclude`.

## Evaluation

The `eval` stage scores a training run offline, on the held-out recordings of its cohort:

```sh
uv run python -m policy eval dirt-orb-cloak               # the steps training evaluates (every evaluate.every steps, and the last)
uv run python -m policy eval dirt-orb-cloak --steps all   # every saved checkpoint; or list steps: --steps 500 1000
```

For each step it predicts an action chunk at every window of each held-out recording and compares it with the
reconstructed labels. Each report gives three predictors:

| Predictor | What it predicts |
| --- | --- |
| `policy` | The model's chunk |
| `repeat_state` | Holds the current joint position for the whole chunk |
| `training_mean` | The mean training action at every step |

Per predictor it reports per-joint MAE and RMSE in radians, and `normalized_mae`: the absolute error divided by each
joint's training-action std, averaged over steps, windows, joints and then recordings. A policy that has learned
something beats both baselines. By default it also scores `pi05_base` before fine-tuning (`eval-initial.json`).

Reports go to `dataset/runs/<config>/eval-<step>.json` and the stage prints a table, with `*` on the step with the
lowest holdout `policy` error. Training picks its step by the same rule (`selected_step` in the run's `result.json`),
which is what `serve.step: best` serves. Training writes the same reports as it saves checkpoints, so `eval` only
fills in the missing ones. It runs in the training image on the GPU box (`hosts.lambda`); `--runner local` runs it
in this Python, e.g. on a local GPU. The `evaluate` section of the config sets the steps, the split (`holdout`, or
`train` for a fit check) and whether to save per-window predictions; see [`eval/stage.py`](eval/stage.py).

Limits:

- This is offline imitation error against reconstructed labels, not task success. There are no measured joint
  angles to compare against, and lower offline error need not mean more successful rollouts (see the
  [results](../README.md#results)).
- The released checkpoints cannot be scored with `eval` as published. A bundle carries only its cohort's
  manifest, not the frames and labels, and evaluation checks the cohort against that manifest's SHA-256.
- Nothing scores physical success automatically. The hub records each rollout under `artifacts/inference-runs/`
  (frames, button pulses, log); success is judged by an operator.

## Serve

A served policy is a websocket on the GPU box (`127.0.0.1:<port>` there). Forward its port to the hub machine, then
add it to the hub's checkpoint registry ([Rollouts](../README.md#rollouts)).

### A trained run

Serve the chosen step (`serve.step`, default `best`) and register it with the hub:

```sh
uv run python -m policy serve dirt-orb-cloak --register
ssh -N -L 8020:127.0.0.1:8020 ubuntu@<gpu-box>
```

`--register` adds the endpoint to `configs/hub/local_checkpoints.json`. `python -m policy stop dirt-orb-cloak` removes
the train and serve containers.

### A released policy

To serve a [pre-trained model](../README.md#pre-trained-models) without training, run this on the GPU box (Docker
with the NVIDIA container toolkit), from a checkout of this repository. The first build of the training image takes
a while:

```sh
git clone https://github.com/amichaelyu/excavator-umi.git umi-excavator && cd umi-excavator
uvx --from huggingface_hub hf download amichaelyu/umi-excavator --include 'policy/cloak-1000/*' --local-dir dataset/models
docker build -f policy/training/docker/train.Dockerfile \
  --build-arg CLOAK_REV=f3d6ab37bb850bfe63430adf9f769e6eda4e10e3 -t umi-excavator-train .
mkdir -p ~/.cache/openpi
docker run --rm --ipc=host --gpus '"device=0"' --user "$(id -u):$(id -g)" -p 127.0.0.1:8020:8020 \
  -v "$PWD":/workspace -v "$HOME/.cache/openpi":/cache/openpi umi-excavator-train \
  python -m policy.training.cloak.bundle serve --backend /opt/cloak --bundle dataset/models/policy/cloak-1000 --port 8020
```

Use `policy/non-cloak-1000` for the baseline. `python -m policy.training.cloak.bundle verify --bundle <dir>` checks a
download against `bundle.json`, and `serve` runs the same check before loading. `CLOAK_REV` must match the bundle's
`upstream_revision`. Forward port 8020 as above and add the server in the hub with **Check & add**; a hand-written
registry entry uses `"task": "dirt"`, `"mode": "cloak"` (or `"baseline"`) and `"step": 1000`.
