"""Train PPO bucket-tip reaching in the twin; compare against seeded random/zero policies on unseen episodes."""
import argparse
import json
import time
import hashlib
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

from digital_twin.paths import CONFIG, SIM
from policy.training.rl.env import ExcavatorEnv

def evaluate(policy, episodes=50, profile='arm2', randomize=True):
    env=ExcavatorEnv(profile=profile,randomize=randomize)
    rows=[]
    for i in range(episodes):
        obs,info=env.reset(seed=10000+i);initial=info['distance_m'];total=0.
        rng=np.random.default_rng(20000+i)
        for step in range(env.horizon):
            if isinstance(policy,str) and policy=='random':a=rng.uniform(-1,1,env.n)
            elif isinstance(policy,str) and policy=='zero':a=np.zeros(env.n)
            else:a=policy.predict(obs,deterministic=True)[0]
            obs,r,done,trunc,info=env.step(a);total+=r
            if done or trunc:break
        rows.append({'seed':10000+i,'initial_distance_m':initial,'final_distance_m':info['distance_m'],
                     'success':bool(info['is_success']),'unstable':bool(info['unstable']),'return':float(total),'steps':step+1})
    env.close()
    return {'success_rate':float(np.mean([r['success'] for r in rows])),
            'mean_final_distance_m':float(np.mean([r['final_distance_m'] for r in rows])),
            'mean_initial_distance_m':float(np.mean([r['initial_distance_m'] for r in rows])),
            'unstable_episodes':int(sum(r['unstable'] for r in rows)),'episodes':rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=150000)
    p.add_argument('--profile',choices=['arm2'],default='arm2');p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--resume',action='store_true',help='Continue the existing checkpoint only when config/model hashes match')
    p.add_argument('--output',type=Path,help='run directory (default digital_twin/sim/outputs/rl/<profile>)')
    args=p.parse_args();torch.set_num_threads(2)
    out=args.output or SIM/'outputs/rl'/args.profile;out.mkdir(parents=True,exist_ok=True)
    config=(CONFIG/'robot.json').read_bytes()
    previous=json.loads((out/'evaluation.json').read_text()) if args.resume else None
    model_xml=(SIM/'models'/f'{args.profile}.xml').read_bytes()
    if previous and (previous['config_sha256']!=hashlib.sha256(config).hexdigest() or previous['model_sha256']!=hashlib.sha256(model_xml).hexdigest()):
        raise ValueError('Configuration changed; start a fresh training run instead of resuming.')
    (out/'training_config.json').write_bytes(config)
    (out/'training_model.xml').write_bytes(model_xml)
    env=DummyVecEnv([lambda:ExcavatorEnv(profile=args.profile,randomize=True) for _ in range(8)])
    model=PPO.load(out/'ppo.zip',env=env,device='cpu') if args.resume else PPO('MlpPolicy',env,n_steps=512,batch_size=256,n_epochs=8,learning_rate=3e-4,
              gamma=.98,gae_lambda=.95,ent_coef=.005,policy_kwargs={'net_arch':[128,128]},
              seed=27,device='cpu',verbose=1)
    started=time.time();model.learn(total_timesteps=args.steps,reset_num_timesteps=not args.resume);model.save(out/'ppo')
    report={'profile':args.profile,'requested_steps':args.steps,'actual_steps':model.num_timesteps,
            'training_seconds':time.time()-started+(previous['training_seconds'] if previous else 0),'seed':27,'task':'bucket-tip reach and hold within 15 mm for 0.4 s',
            'resumed_from_steps':previous['actual_steps'] if previous else 0,
            'config_sha256':hashlib.sha256(config).hexdigest(),'model_sha256':hashlib.sha256(model_xml).hexdigest(),
            'evaluation':f'{args.episodes} independent unseen seeds with domain randomization; simulator only',
            'rl':evaluate(model,args.episodes,args.profile),
            'random':evaluate('random',args.episodes,args.profile),
            'zero':evaluate('zero',args.episodes,args.profile)}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rl','random','zero']},indent=2))
    for name in ['rl','random','zero']:print(name,{k:v for k,v in report[name].items() if k!='episodes'})
    env.close()

if __name__=='__main__':main()
