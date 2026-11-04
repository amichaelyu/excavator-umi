"""Learn motor corrections for frozen pi0.5 plans in MuJoCo.

The VLA remains frozen. Plans use real recorded observations, not simulated
camera feedback. This measures motion tracking, not soil collection success.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from gymnasium import spaces

from digital_twin.paths import CONFIG, SIM
from policy.training.rl.env import ExcavatorEnv


class PlanTrackingEnv(ExcavatorEnv):
    def __init__(self, plans, split='train', gain=5., lead=.10, render_mode=None):
        super().__init__(randomize=True, render_mode=render_mode, horizon=400)
        self.plans_root = Path(plans)
        manifest = json.loads((self.plans_root / 'manifest.json').read_text())
        self.rows = [e for e in manifest['episodes'] if e['split'] == split]
        if not self.rows:
            raise ValueError(f'No {split} plans')
        self.plans = {}
        for e in self.rows:
            p = self.plans_root / e['path']
            if hashlib.sha256(p.read_bytes()).hexdigest() != e['sha256']:
                raise ValueError(f'Plan hash mismatch: {p}')
            with np.load(p, allow_pickle=False) as a:
                t, q = a['time_s'].copy(), a['q'].copy()
            if q.shape != (len(t), 2) or not np.isfinite(q).all() or not np.all(np.diff(t) > 0):
                raise ValueError(f'Invalid plan: {p}')
            self.plans[e['id']] = (t, q)
        self.gain, self.lead = gain, lead
        self.observation_space = spaces.Box(-np.inf, np.inf, (24,), dtype=np.float32)
        self.reference = None
        self.history = deque([np.zeros(2) for _ in range(4)], maxlen=4)

    def desired(self, future=0.):
        return np.array([np.interp(self.elapsed * self.dt + future, self.times, self.reference[:, j])
                         for j in range(2)])

    def feedforward(self):
        return (self.desired(self.lead + self.dt) - self.desired(self.lead)) / self.dt

    def baseline(self):
        velocity = self.feedforward() + self.gain * (self.desired(self.lead) - self.data.qpos)
        return np.clip(velocity / self.speed, -1., 1.)

    def _obs(self):
        if self.reference is None:
            return np.zeros(24, dtype=np.float32)
        q = self.data.qpos
        return np.r_[q, self.data.qvel / self.speed,
                     np.concatenate([(self.desired(t) - q) / .15 for t in (0., .1, .2, .4)]),
                     np.concatenate(self.history), self.baseline(), self.feedforward() / self.speed].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.reference = None
        super().reset(seed=seed)
        eid = (options or {}).get('episode_id') or self.rows[int(self.np_random.integers(len(self.rows)))]['id']
        t, original = self.plans[eid]
        # Same joint limits and rate limiting for every controller. Keep the raw
        # plan for reporting; clipping is not counted as better VLA prediction.
        q = np.clip(original, self.bounds[:, 0] + .025, self.bounds[:, 1] - .025)
        for k in range(1, len(q)):
            dq = self.speed * .8 * (t[k] - t[k - 1])
            q[k] = q[k - 1] + np.clip(q[k] - q[k - 1], -dq, dq)
        self.episode_id, self.times, self.reference = eid, t, q
        self.original = original
        self.elapsed = 0
        self.data.qpos[:] = np.clip(q[0] + self.np_random.normal(0, .01, 2),
                                   self.bounds[:, 0] + .025, self.bounds[:, 1] - .025)
        self.data.qvel[:] = 0.
        self.target_q = self.data.qpos.copy()
        self.data.ctrl[:] = self.target_q
        self.velocity_scale = self.np_random.uniform(.55, 1.25, 2)
        self.delay = int(self.np_random.integers(0, 5))
        self.queue = deque([np.zeros(2) for _ in range(self.delay)])
        self.history = deque([np.zeros(2) for _ in range(4)], maxlen=4)
        self.last_action = np.zeros(2)
        self.horizon = int(np.ceil(t[-1] / self.dt))
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {'episode_id': eid, 'delay_s': self.delay * self.dt,
                             'speed_scale': self.velocity_scale.tolist()}

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError('Expected two finite residual commands')
        issued = np.clip(self.baseline() + .75 * np.clip(a, -1, 1), -1, 1)
        self.history.append(np.where(abs(issued) < .25, 0, np.sign(issued)))
        _, _, _, _, info = super().step(issued)
        error = np.abs(self.data.qpos - self.desired())
        raw = np.array([np.interp(self.elapsed * self.dt, self.times, self.original[:, j]) for j in range(2)])
        reward = -float(error.mean()) / .08 - .001 * float(np.square(a).mean())
        if info['unstable']:
            reward -= 20.
        info.update(tracking_mae_rad=float(error.mean()), joint_error_rad=error.tolist(),
                    raw_plan_mae_rad=float(np.abs(self.data.qpos - raw).mean()),
                    is_success=False, episode_id=self.episode_id)
        return self._obs(), reward, bool(info['unstable']), self.elapsed >= self.horizon, info


def evaluate(plans, policy=None, split='holdout', gain=5., lead=.1, seeds=(9000, 9001, 9002), limit=None):
    env = PlanTrackingEnv(plans, split, gain, lead)
    rows = []
    for e in env.rows[:limit]:
        for seed in seeds:
            obs, initial = env.reset(seed=seed, options={'episode_id': e['id']})
            errors, raw_errors, joints = [], [], []
            for _ in range(env.horizon):
                a = np.zeros(2) if policy is None else policy.predict(obs, deterministic=True)[0]
                obs, _, done, trunc, info = env.step(a)
                errors.append(info['tracking_mae_rad']); raw_errors.append(info['raw_plan_mae_rad'])
                joints.append(info['joint_error_rad'])
                if done or trunc:
                    break
            rows.append(dict(initial, seed=seed, mae_rad=float(np.mean(errors)),
                             raw_plan_mae_rad=float(np.mean(raw_errors)),
                             joint_mae_rad=np.mean(joints, axis=0).tolist(), unstable=bool(info['unstable'])))
    env.close()
    return {'mae_rad': float(np.mean([r['mae_rad'] for r in rows])),
            'raw_plan_mae_rad': float(np.mean([r['raw_plan_mae_rad'] for r in rows])),
            'unstable_episodes': sum(r['unstable'] for r in rows), 'episodes': rows}


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=300000)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.time()
    # Choose conventional controller parameters using training episodes only.
    tuning = []
    for gain in (3., 6., 10.):
        for lead in (0., .1, .2):
            result = evaluate(a.plans, split='train', gain=gain, lead=lead, seeds=(321,), limit=10)
            tuning.append({'gain': gain, 'lead_s': lead, 'mae_rad': result['mae_rad']})
            print('Baseline tuning', tuning[-1], flush=True)
    best = min(tuning, key=lambda r: r['mae_rad'])
    gain, lead = best['gain'], best['lead_s']
    (a.output / 'baseline_tuning.json').write_text(json.dumps(tuning, indent=2) + '\n')
    cfg = {'task': __doc__, 'steps_requested': a.steps, 'baseline': best, 'train_episodes': 47,
           'holdout_episodes': 7, 'seed': 27, 'workers': a.workers,
           'speed_scale_range': [.55, 1.25], 'delay_s_range': [0., .2],
           'mass_scale_range': [.8, 1.2], 'friction_scale_range': [.5, 1.5],
           'model_sha256': hashlib.sha256((SIM / 'models/arm2.xml').read_bytes()).hexdigest(),
           'robot_config_sha256': hashlib.sha256((CONFIG/'robot.json').read_bytes()).hexdigest(),
           'plans_manifest_sha256': hashlib.sha256((a.plans / 'manifest.json').read_bytes()).hexdigest()}
    (a.output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')

    class SaveBest(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 50000
            self.best = float('inf')
            self.history = []

        def _on_step(self):
            if self.num_timesteps >= self.next_eval:
                # Fixed validation seeds on training trajectories; holdout stays untouched.
                r = evaluate(a.plans, self.model, 'train', gain, lead, seeds=(707,), limit=10)
                self.history.append({'step': self.num_timesteps, 'mae_rad': r['mae_rad']})
                print('Controller validation', self.history[-1], flush=True)
                if r['mae_rad'] < self.best:
                    self.best = r['mae_rad']; self.model.save(a.output / 'best_controller')
                (a.output / 'validation.json').write_text(json.dumps(self.history, indent=2) + '\n')
                self.next_eval += 50000
            return True

    env = SubprocVecEnv([lambda: PlanTrackingEnv(a.plans, 'train', gain, lead) for _ in range(a.workers)],
                       start_method='spawn')
    model = PPO('MlpPolicy', env, n_steps=512, batch_size=256, n_epochs=6, learning_rate=3e-4,
                gamma=.98, gae_lambda=.95, ent_coef=.002,
                policy_kwargs={'net_arch': [128, 128], 'log_std_init': -.7}, seed=27, device='cpu', verbose=1)
    model.learn(total_timesteps=a.steps, callback=SaveBest())
    model.save(a.output / 'last_controller')
    steps = model.num_timesteps
    selected = a.output / 'best_controller.zip'
    if not selected.exists():
        selected = a.output / 'last_controller.zip'
    model = PPO.load(selected, device='cpu')
    baseline = evaluate(a.plans, gain=gain, lead=lead)
    residual = evaluate(a.plans, model, gain=gain, lead=lead)
    report = {'scope': __doc__, 'actual_steps': steps, 'runtime_s': time.time() - started,
              'selected_controller': selected.name, 'evaluation': '7 unseen real-observation plans x 3 paired dynamics seeds',
              'baseline': baseline, 'rl': residual,
              'mae_reduction_percent': 100 * (1 - residual['mae_rad'] / baseline['mae_rad']),
              'promote_controller': residual['mae_rad'] < baseline['mae_rad'] and residual['unstable_episodes'] == 0}
    (a.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('baseline', 'rl')}, indent=2), flush=True)
    env.close()


if __name__ == '__main__':
    main()
