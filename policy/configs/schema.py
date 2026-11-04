"""Load a run config (``configs/policy/<name>.yaml``), validate every section and derive every stage's paths.

One YAML file describes one trained policy. ``kind`` names the trainer (only ``cloak``, the default):

  cloak     π0.5 fine-tune (plain or CLOAK-masked) on handheld recordings, in Docker on a Lambda box:
            recordings -> reconstruct -> labels -> cohort -> train -> eval -> serve  (python -m policy <stage>)

Paths are relative to the repository root; outputs default to ``dataset/<kind>/<name>``. Every key and default
is below (SECTIONS / KEYS, the *_DEFAULTS tables; the ``evaluate`` section: EVALUATE_KEYS, check_evaluate).

This package is the leaf of policy/: it imports nothing else from the repository, and every stage imports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs' / 'policy'
METHODS = ('orbslam3_smoother',)
RECONSTRUCT_RUNNERS = ('local', 'docker', 'lambda')   # where policy.training.stages.reconstruct runs
PREPROCESS_RUNNERS = ('local', 'docker')              # labels / cohort: this Python or the umi-excavator-preprocess image
EVALUATE_RUNNERS = ('lambda', 'local')                # host handed to policy.eval.stage.run (None = local)
EVALUATE_SPLITS = ('holdout', 'train')                # recordings a report scores (policy.training.cloak.evaluator)
EVALUATE_STRESS = ('best', 'all', 'none')             # steps that get the synthetic edge-mask stress test
EVALUATE_KEYS = ('every', 'steps', 'pretrained', 'mask_seed', 'stress', 'split', 'predictions', 'runner', 'gpus',
                 'backend')
# Task -> the prompt the policy is trained (policy.preprocessing.export) and served with.
TASKS = {'dirt': 'scoop the dirt', 'bottlecap': 'scoop the bottle cap'}
# Policy action clock: action k of a chunk targets t_obs + k / POLICY_HZ (labels, training contract, runner). Not the
# twin's replay / IK rate (digital_twin.sim.kinematics.CONTROL_HZ = 20 Hz).
POLICY_HZ = 10
# Actions per chunk: the excavator client (digital_twin.policy.cloak) and the hub runner accept only 10-step chunks.
ACTION_HORIZON = 10
PROFILES = ('full4', 'full4_calibrated')
# Twin used for the IK labels -> joint space the policy is trained in. full4 is the prior twin
# (wider than the machine; release cloak-ablation-20261005). full4_calibrated has the measured
# ranges / speeds of the real machine (configs/hardware/robot_full4_calibrated.json; 20261013 retrain).
LABEL_PROFILES = {'full4': 'handheld_reconstructed', 'full4_calibrated': 'calibrated'}
PLANNERS = ('lookahead', 'ik', 'coupled')   # digital_twin.sim.replay --planner; the cohort stage needs lookahead
QUALITY_POLICIES = ('lenient', 'strict')
MODES = ('baseline', 'cloak', 'cloak_oov', 'noaug')
ACTION_REPRESENTATIONS = ('delta', 'absolute')
# Embodiment masks frozen into the cohort: all-empty with an out-of-view confirmation (baseline, cloak_oov), or
# rendered from the twin at each sample's labelled joint state (policy.training.cloak.twin_masks; cloak, noaug).
MASKS = ('out_of_view', 'twin')
MASKS_FOR_MODE = {'baseline': MASKS, 'cloak_oov': ('out_of_view',), 'cloak': ('twin',), 'noaug': ('twin',)}
# The recipe's `training` options (policy.training.cloak.experiment passes each as --<key> to the trainer). The one
# source of training defaults: `python -m policy.training.cloak train` uses them as its option defaults too.
TRAIN_DEFAULTS = {'mode': 'baseline', 'action_representation': 'delta', 'steps': 1000, 'batch': 32, 'fsdp': 1,
                  'seed': 20261005, 'horizon': 10, 'peak_lr': 5e-5, 'warmup': 25, 'save_interval': 250,
                  'keep_period': 250}
# Trainer options left out of the recipe unless the config sets them (the trainer's own defaults apply).
TRAIN_OPTIONAL = ('initial_checkpoint', 'microbatch', 'host_optimizer')
# train.wandb: Weights & Biases tracking (policy.training.cloak.wandb_run), not a trainer option and not part of the
# recipe. Off unless the config turns it on; `wandb: true` is {mode: online}. name and group default to the config name.
WANDB_MODES = ('online', 'offline', 'disabled')
# The evaluation's baselines (policy.eval.metrics), whose episode-mean normalized MAE wandb_run logs.
EVAL_BASELINES = ('policy', 'repeat_state', 'training_mean')
WANDB_DEFAULTS = {'project': 'umi-excavator', 'entity': None, 'mode': 'disabled', 'tags': [], 'name': None, 'group': None}
SECTIONS = {'name', 'kind', 'task', 'source', 'reconstruct', 'labels', 'cohort', 'train', 'evaluate', 'lambda', 'serve'}
# Allowed keys per section (evaluate: EVALUATE_KEYS, checked by check_evaluate).
KEYS = {'source': {'recordings', 'visibility', 'holdout', 'embodiment', 'exclude'},
        'reconstruct': {'method', 'workers', 'runner', 'output'},
        'labels': {'profile', 'planner', 'contact_height', 'pitch_clip', 'workers', 'runner', 'output'},
        'cohort': {'quality_policy', 'max_pitch_clip_fraction', 'max_pitch_clip_deg', 'masks', 'mask_config',
                   'mask_workers', 'runner', 'output'},
        'train': {*TRAIN_DEFAULTS, *TRAIN_OPTIONAL, 'run_dir', 'wandb'},
        'lambda': {'host', 'root', 'gpus', 'poll'},
        'serve': {'step', 'port', 'gpus'}}
REQUIRED_SOURCE = ('recordings', 'visibility')
CAMERA = 'configs/hardware/camera-calibration/camera.json'           # the camera calibration of every reconstruction (policy.data.camera)
RECORDING_KEYS = {'data_root', 'label', 'ids'}        # one source.recordings entry (policy.preprocessing.select)
KINDS = ('cloak',)


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict

    @property
    def name(self):
        return self.raw['name']

    @property
    def kind(self):
        return self.raw.get('kind', 'cloak')

    @property
    def task(self):
        return self.raw['task']

    def section(self, key):
        return self.raw.get(key) or {}

    def dataset(self, kind, key='output'):
        """``<section>.<key>`` if given, else ``dataset/<kind>/<name>``."""
        section = {'reconstructions': 'reconstruct', 'labels': 'labels', 'cohorts': 'cohort', 'runs': 'train'}[kind]
        return Path(self.section(section).get(key) or f'dataset/{kind}/{self.name}')

    @property
    def method(self):
        return self.section('reconstruct').get('method', 'orbslam3_smoother')

    @property
    def calibration(self):
        """Camera/IMU calibration of the reconstruction: configs/hardware/camera-calibration/camera.json (policy.processing.camera_calibration)."""
        return Path(CAMERA)

    def runner(self, stage):
        """Where a stage runs: reconstruct / labels / cohort from the config; train, eval, serve on Lambda."""
        if stage in ('reconstruct', 'labels', 'cohort'):
            return self.section(stage).get('runner', 'local')
        if stage == 'eval':
            return self.section('evaluate').get('runner', 'lambda')
        return 'lambda'

    @property
    def masks(self):
        """Embodiment masks in the cohort: ``cohort.masks``, else twin for cloak / noaug, out_of_view otherwise."""
        mode = self.section('train').get('mode', TRAIN_DEFAULTS['mode'])
        return self.section('cohort').get('masks') or ('twin' if mode in ('cloak', 'noaug') else 'out_of_view')

    @property
    def cohort_data(self):
        """Frozen training dataset: ``<cohort>/frozen/<method>`` (``frozen-twin`` with twin masks;
        policy.preprocessing.freeze)."""
        return self.dataset('cohorts') / ('frozen' if self.masks == 'out_of_view' else 'frozen-twin') / self.method

    @property
    def run_dir(self):
        return self.dataset('runs', 'run_dir')

    @property
    def training(self):
        options = {**TRAIN_DEFAULTS, **self.section('train')}
        options.pop('run_dir', None)
        options.pop('wandb', None)
        return options

    @property
    def wandb(self):
        """Resolved ``train.wandb``: every WANDB_DEFAULTS key; mode ``disabled`` unless the config turns it on."""
        return resolve_wandb(self.section('train').get('wandb'), self.name)

    def evaluate_steps(self):
        steps, every = self.training['steps'], self.section('evaluate').get('every', self.training['save_interval'])
        return sorted(set(range(every, steps + 1, every)) | {steps})

    def recipe(self):
        """The ``actathon.cloak.experiment.v1`` recipe ``policy.training.cloak.experiment`` runs."""
        labels = self.section('labels')
        return {'schema': 'actathon.cloak.experiment.v1', 'task_group': self.task,
                'data': str(self.cohort_data), 'run_dir': str(self.run_dir),
                'training': self.training, 'evaluate_steps': self.evaluate_steps(),
                'mask_seed': self.section('evaluate').get('mask_seed', self.training['seed']),
                'comparison': {'config': self.name, 'motion_source': self.method,
                               'masking': self.training['mode'], 'embodiment_masks': self.masks,
                               'label_profile': labels.get('profile', 'full4_calibrated'),
                               'contact_height_m': labels.get('contact_height'),
                               'pitch_clip': bool(labels.get('pitch_clip')),
                               'initialization': 'pi05_base, independent full fine-tune'}}


def resolve(path):
    """A config path as given (relative to the working directory or the repository root), or a name under
    configs/policy/ (``dirt-orb`` or ``dirt-orb.yaml``)."""
    path = Path(path)
    for candidate in (path, ROOT / path, CONFIGS / path, CONFIGS / f'{path}.yaml'):
        if candidate.is_file():
            return candidate
    names = ', '.join(sorted(p.stem for p in CONFIGS.glob('*.yaml')))
    raise FileNotFoundError(f'No config {path} (a path, or a name in configs/policy/: {names})')


def load(path):
    path = resolve(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: expected a mapping of sections')
    unknown = set(raw) - SECTIONS
    if unknown:
        raise ValueError(f'{path}: unknown sections {sorted(unknown)}')
    if not raw.get('name'):
        raise ValueError(f'{path}: name is required')
    for section in SECTIONS - {'name', 'kind', 'task'}:
        if raw.get(section) is not None and not isinstance(raw[section], dict):
            raise ValueError(f'{path}: {section} must be a mapping')
    cfg = Config(path.resolve(), raw)
    if cfg.kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}')
    if not raw.get('task'):
        raise ValueError(f'{path}: task is required')
    for section, keys in KEYS.items():
        extra = set(cfg.section(section)) - keys
        if extra:
            raise ValueError(f'unknown {section} options {sorted(extra)} (allowed: {sorted(keys)})')
    missing = [k for k in REQUIRED_SOURCE if not cfg.section('source').get(k)]
    if missing:
        raise ValueError(f'source needs {missing}')
    check_recordings(cfg.section('source'))
    _choice(cfg, 'reconstruct', 'method', METHODS, 'orbslam3_smoother')
    _choice(cfg, 'reconstruct', 'runner', RECONSTRUCT_RUNNERS, 'local')
    _choice(cfg, 'labels', 'runner', PREPROCESS_RUNNERS, 'local')
    _choice(cfg, 'cohort', 'runner', PREPROCESS_RUNNERS, 'local')
    _choice(cfg, 'labels', 'profile', PROFILES, 'full4_calibrated')
    _choice(cfg, 'labels', 'planner', PLANNERS, 'lookahead')
    _choice(cfg, 'cohort', 'quality_policy', QUALITY_POLICIES, 'lenient')
    _choice(cfg, 'cohort', 'masks', MASKS, cfg.masks)
    for section, key in (('reconstruct', 'workers'), ('labels', 'workers'), ('cohort', 'mask_workers'),
                         ('lambda', 'poll'), ('serve', 'port')):
        value = cfg.section(section).get(key)
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError(f'{section}.{key} must be a positive integer')
    training = cfg.training
    if training['mode'] not in MODES:
        raise ValueError(f'train.mode must be one of {MODES}')
    if training['action_representation'] not in ACTION_REPRESENTATIONS:
        raise ValueError(f'train.action_representation must be one of {ACTION_REPRESENTATIONS}')
    if cfg.masks not in MASKS_FOR_MODE[training['mode']]:
        raise ValueError(f'train.mode {training["mode"]} trains on {MASKS_FOR_MODE[training["mode"]]} masks, '
                         f'not cohort.masks {cfg.masks}')
    for key in ('steps', 'batch', 'fsdp', 'horizon', 'warmup', 'save_interval', 'keep_period', 'microbatch'):
        if key in training and (not isinstance(training[key], int) or training[key] < 1):
            raise ValueError(f'train.{key} must be a positive integer')
    if training['horizon'] != ACTION_HORIZON:
        raise ValueError(f'train.horizon must be {ACTION_HORIZON}: the excavator client and hub run '
                         f'{ACTION_HORIZON}-step chunks (digital_twin.policy.cloak)')
    if training['batch'] % training['fsdp']:
        raise ValueError('train.batch must divide over train.fsdp devices')
    gpus = cfg.section('lambda').get('gpus', [0])
    if len(gpus) < training['fsdp']:
        raise ValueError('lambda.gpus has fewer devices than train.fsdp')
    step = cfg.section('serve').get('step', 'best')
    if step != 'best' and not isinstance(step, int):
        raise ValueError('serve.step must be best or a saved step')
    check_evaluate(cfg.section('evaluate'))
    check_wandb(cfg.section('train').get('wandb'))
    return cfg


def _int(value, minimum):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def check_recordings(source) -> None:
    """``source.recordings``: a list of {data_root, label and / or ids}; ``holdout`` a fraction in [0, 1)."""
    recordings = source.get('recordings')
    if not isinstance(recordings, list) or not recordings:
        raise ValueError('source.recordings must be a list of {data_root, label | ids}')
    for spec in recordings:
        if not isinstance(spec, dict) or not spec.get('data_root'):
            raise ValueError('each source.recordings entry needs a data_root')
        extra = set(spec) - RECORDING_KEYS
        if extra:
            raise ValueError(f'unknown source.recordings options {sorted(extra)} (allowed: {sorted(RECORDING_KEYS)})')
        if 'label' not in spec and 'ids' not in spec:
            raise ValueError(f'source.recordings entry {spec["data_root"]} needs a label or ids')
        if 'ids' in spec and (not isinstance(spec['ids'], list) or not all(isinstance(i, str) for i in spec['ids'])):
            raise ValueError('source.recordings ids must be a list of recording ids')
    holdout = source.get('holdout', 0.2)
    if isinstance(holdout, bool) or not isinstance(holdout, (int, float)) or not 0 <= holdout < 1:
        raise ValueError('source.holdout must be a fraction in [0, 1)')
    if not isinstance(source.get('exclude', []), list):
        raise ValueError('source.exclude must be a list of recording ids')


def check_evaluate(section) -> None:
    """Validate an ``evaluate`` section (policy.eval.stage resolves it; the keys are EVALUATE_KEYS).
    ValueError names the offending key."""
    section = section or {}
    if not isinstance(section, dict):
        raise ValueError('evaluate must be a mapping')
    unknown = sorted(set(section) - set(EVALUATE_KEYS))
    if unknown:
        raise ValueError(f'unknown evaluate options {unknown}; known: {list(EVALUATE_KEYS)}')
    if 'every' in section and not _int(section['every'], 1):
        raise ValueError('evaluate.every must be a positive integer')
    if 'steps' in section:
        check_steps(section['steps'], 'evaluate.steps')
    for key in ('pretrained', 'predictions'):
        if key in section and not isinstance(section[key], bool):
            raise ValueError(f'evaluate.{key} must be true or false')
    if 'mask_seed' in section and not _int(section['mask_seed'], 0):
        raise ValueError('evaluate.mask_seed must be a non-negative integer')
    for key, allowed in (('stress', EVALUATE_STRESS), ('split', EVALUATE_SPLITS), ('runner', EVALUATE_RUNNERS)):
        if key in section and section[key] not in allowed:
            raise ValueError(f'evaluate.{key} must be one of {allowed}, not {section[key]!r}')
    if 'gpus' in section and not (isinstance(section['gpus'], list) and section['gpus']
                                  and all(_int(g, 0) for g in section['gpus'])):
        raise ValueError('evaluate.gpus must be a non-empty list of GPU indices')
    if 'backend' in section and not (isinstance(section['backend'], str) and section['backend']):
        raise ValueError('evaluate.backend must be a path')


def resolve_wandb(value, name):
    """``train.wandb`` (absent, true / false, or a mapping) with every default filled in."""
    value = {'mode': 'online'} if value is True else value or {}
    out = {**WANDB_DEFAULTS, **value, 'tags': list(value.get('tags') or [])}
    out['name'], out['group'] = out['name'] or name, out['group'] or name
    return out


def check_wandb(value) -> None:
    """Validate ``train.wandb``: true / false, or a mapping of WANDB_DEFAULTS keys."""
    if value is None or isinstance(value, bool):
        return
    if not isinstance(value, dict):
        raise ValueError('train.wandb must be true, false or a mapping {project, entity, mode, tags, name, group}')
    unknown = sorted(set(value) - set(WANDB_DEFAULTS))
    if unknown:
        raise ValueError(f'unknown train.wandb options {unknown}; known: {list(WANDB_DEFAULTS)}')
    if 'mode' in value and value['mode'] not in WANDB_MODES:
        raise ValueError(f'train.wandb.mode must be one of {WANDB_MODES}, not {value["mode"]!r}')
    if 'project' in value and not (isinstance(value['project'], str) and value['project']):
        raise ValueError('train.wandb.project must be a non-empty string')
    for key in ('entity', 'name', 'group'):
        if value.get(key) is not None and not (isinstance(value[key], str) and value[key]):
            raise ValueError(f'train.wandb.{key} must be a non-empty string (or unset)')
    tags = value.get('tags')
    if tags is not None and not (isinstance(tags, list)
                                 and all(isinstance(tag, str) and 0 < len(tag) <= 64 for tag in tags)):
        raise ValueError('train.wandb.tags must be a list of strings of 1-64 characters')


def check_steps(steps, where):
    """``'all'``, ``'recipe'`` or a list of distinct positive step numbers."""
    if steps in ('all', 'recipe'):
        return
    if not (isinstance(steps, list) and steps and all(_int(s, 1) for s in steps)):
        raise ValueError(f"{where} must be 'all', 'recipe' or a non-empty list of positive step numbers")
    if len(set(steps)) != len(steps):
        raise ValueError(f'{where} repeats a step')


def _choice(cfg, section, key, choices, default):
    if cfg.section(section).get(key, default) not in choices:
        raise ValueError(f'{section}.{key} must be one of {choices}')
