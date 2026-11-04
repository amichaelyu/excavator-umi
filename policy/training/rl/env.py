"""Gymnasium bucket-tip reaching with delayed, ternary electric-motor commands.

No network or physical robot commands are issued. Parameters marked as priors
in configs/sim/robot.json are randomized; real transfer requires their identification.
"""
from collections import deque
import json
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from digital_twin.paths import CONFIG, SIM


class ExcavatorEnv(gym.Env):
    metadata = {'render_modes':['rgb_array'], 'render_fps':20}

    def __init__(self, profile='arm2', randomize=True, render_mode=None, horizon=160):
        if profile not in ['arm2']:
            raise ValueError('RL reach supports the stationary arm2 profile (boom + bucket).')
        self.profile=profile;self.randomize=randomize;self.render_mode=render_mode;self.horizon=horizon
        self.cfg=json.loads((CONFIG/'robot.json').read_text())
        self.names=['boom','bucket']           # = kinematics.JOINTS; slew and stick are welded in the model
        self.n=len(self.names)
        path=SIM/'models'/f'{profile}.xml'
        if render_mode:
            self.model=mujoco.MjModel.from_xml_path(str(path))
        else:
            xml=ET.parse(path).getroot()
            xml.remove(xml.find('asset'))
            for parent in xml.iter():
                for g in list(parent):
                    if g.tag=='geom' and g.get('mesh'):parent.remove(g)
            self.model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
        self.data=mujoco.MjData(self.model)
        self.frameskip=10;self.dt=self.model.opt.timestep*self.frameskip
        self.action_space=spaces.Box(-1,1,(self.n,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(self.n*4+9,),dtype=np.float32)
        self.bounds=np.array([self.model.joint(n).range.copy() for n in self.names])
        self.speed=np.array([self.cfg['joints'][n]['speed_rad_s'] for n in self.names])
        self.positive_speed=np.array([self.cfg['joints'][n].get('speed_positive_rad_s',self.cfg['joints'][n]['speed_rad_s']) for n in self.names])
        self.negative_speed=np.array([self.cfg['joints'][n].get('speed_negative_rad_s',self.cfg['joints'][n]['speed_rad_s']) for n in self.names])
        self.base_mass=self.model.body_mass.copy();self.base_inertia=self.model.body_inertia.copy()
        self.base_friction=self.model.dof_frictionloss.copy()
        self.renderer=None

    def _tip(self):return self.data.site('tooth_tip').xpos.copy()

    def _obs(self):
        q=self.data.qpos.copy();v=self.data.qvel.copy();tip=self._tip()
        if self.randomize:
            q+=self.np_random.normal(0,.002,self.n)
            v+=self.np_random.normal(0,.005,self.n)
        return np.r_[np.sin(q),np.cos(q),v,self.last_action,tip,self.goal,self.goal-tip].astype(np.float32)

    def _sample_q(self):
        # Stay inside visually plausible, collision-free workspace. Range values
        # are simulation priors, not measured hardware hard stops.
        lo=self.bounds[:,0]*.65;hi=self.bounds[:,1]*.65
        for _ in range(100):
            q=self.np_random.uniform(lo,hi)
            self.data.qpos[:]=q;mujoco.mj_forward(self.model,self.data)
            if self._tip()[2]>.035 and self.data.ncon==0:return q
        raise RuntimeError('No collision-free configuration in configured workspace')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        scale=self.np_random.uniform(.8,1.2) if self.randomize else 1.
        self.model.body_mass[:]=self.base_mass*scale
        self.model.body_inertia[:]=self.base_inertia*scale
        self.model.dof_frictionloss[:]=self.base_friction*(self.np_random.uniform(.5,1.5) if self.randomize else 1.)
        mujoco.mj_setConst(self.model,self.data)
        self.goal_q=self._sample_q();self.goal=self._tip()
        initial=self._sample_q();self.data.qpos[:]=initial;self.data.qvel[:]=0
        self.target_q=initial.copy();self.data.ctrl[:]=initial
        self.data.mocap_pos[0]=self.goal
        self.last_action=np.zeros(self.n)
        self.velocity_scale=self.np_random.uniform(.8,1.2,self.n) if self.randomize else np.ones(self.n)
        self.delay=int(self.np_random.integers(0,4)) if self.randomize else 1
        self.queue=deque([np.zeros(self.n) for _ in range(self.delay+1)])
        self.elapsed=0;self.hold=0
        mujoco.mj_forward(self.model,self.data)
        self.distance=float(np.linalg.norm(self._tip()-self.goal))
        return self._obs(),{'distance_m':self.distance,'goal':self.goal.copy()}

    def step(self, action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(self.n,) or not np.isfinite(action).all():raise ValueError('Expected finite action vector')
        action=np.clip(action,-1,1)
        command=np.where(abs(action)<.25,0,np.sign(action))
        self.queue.append(command);applied=self.queue.popleft()
        rate=np.where(applied>=0,self.positive_speed,self.negative_speed)
        self.target_q=np.clip(self.target_q+applied*rate*self.velocity_scale*self.dt,self.bounds[:,0]+.025,self.bounds[:,1]-.025)
        self.data.ctrl[:]=self.target_q
        mujoco.mj_step(self.model,self.data,nstep=self.frameskip)
        self.elapsed+=1
        distance=float(np.linalg.norm(self._tip()-self.goal))
        speed=float(np.linalg.norm(self.data.qvel))
        good=distance<.015 and speed<.12
        self.hold=self.hold+1 if good else 0
        # Potential shaping plus an absolute distance penalty and sustained hold.
        reward=40*(self.distance-distance)-2*distance-.002*np.square(command).sum()
        reward+=.5 if good else 0
        self.distance=distance;self.last_action=applied.copy()
        unstable=not np.isfinite(self.data.qpos).all() or np.max(abs(self.data.qvel))>20
        success=self.hold>=8
        if success:reward+=5.
        if unstable:reward-=10.
        info={'distance_m':distance,'is_success':success,'joint_names':self.names,
              'command':applied.copy(),'sim_time_s':float(self.data.time),'unstable':unstable}
        return self._obs(),float(reward),bool(success or unstable),self.elapsed>=self.horizon,info

    def render(self):
        if self.render_mode!='rgb_array':return None
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=480,width=640)
        self.renderer.update_scene(self.data,camera='overview')
        return self.renderer.render()

    def close(self):
        if self.renderer:self.renderer.close();self.renderer=None

if __name__=='__main__':
    from gymnasium.utils.env_checker import check_env
    env=ExcavatorEnv(randomize=False)
    check_env(env,skip_render_check=True)
    print('Gymnasium API check passed')
