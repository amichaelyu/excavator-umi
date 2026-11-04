import unittest
import json
import numpy as np
import mujoco
from policy.training.rl.env import ExcavatorEnv
from digital_twin.paths import CONFIG, MODELS
from digital_twin.sim.pivots import fk
from digital_twin.sim.kinematics import HARDWARE_JOINTS, full_chain_q

class PhysicsTests(unittest.TestCase):
    def test_articulated_fk_matches_mujoco(self):
        # the twin full4.xml keeps all four pivots articulated
        cfg=json.loads((CONFIG/'robot.json').read_text())
        model=mujoco.MjModel.from_xml_path(str(MODELS/'full4.xml'));data=mujoco.MjData(model)
        for q in np.random.default_rng(27).uniform(-.4,.4,(20,4)):
            data.qpos[:]=q;mujoco.mj_forward(model,data);rot,pos=fk(q,cfg)
            for i,n in enumerate(['base','slew','boom','stick','bucket']):
                np.testing.assert_allclose(data.body(n).xpos,pos[i],atol=1e-8)
                np.testing.assert_allclose(data.body(n).xmat.reshape(3,3),rot[i],atol=1e-8)
        # the RL model arm2.xml is that chain with slew and stick welded at their rest values
        cfg=json.loads((CONFIG/'robot.json').read_text())
        model=mujoco.MjModel.from_xml_path(str(MODELS/'arm2.xml'));data=mujoco.MjData(model)
        n=len(HARDWARE_JOINTS);self.assertEqual(model.nq,n);self.assertEqual(model.nu,n)
        for q in np.random.default_rng(28).uniform(-.4,.4,(20,n)):
            data.qpos[:]=q;mujoco.mj_forward(model,data);rot,pos=fk(full_chain_q(q),cfg)
            for i,n in enumerate(['base','slew','boom','stick','bucket']):
                np.testing.assert_allclose(data.body(n).xpos,pos[i],atol=1e-8)
                np.testing.assert_allclose(data.body(n).xmat.reshape(3,3),rot[i],atol=1e-8)

    def test_welded_joints_are_actually_fixed(self):
        # slew never moves relative to the base; the stick never moves relative to the boom
        env=ExcavatorEnv('arm2',randomize=False);env.reset(seed=2)
        def rel(a,b):
            Ra=env.data.body(a).xmat.reshape(3,3);return Ra.T@(env.data.body(b).xpos-env.data.body(a).xpos),Ra.T@env.data.body(b).xmat.reshape(3,3)
        initial={n:env.data.body(n).xpos.copy() for n in ['base','slew']}
        stick0=rel('boom','stick')
        for _ in range(120):env.step(np.array([1.,-1.]))
        for n,p in initial.items():np.testing.assert_array_equal(env.data.body(n).xpos,p)
        np.testing.assert_allclose(env.data.body('slew').xmat.reshape(3,3),np.eye(3),atol=1e-12)
        stick1=rel('boom','stick')
        np.testing.assert_allclose(stick1[0],stick0[0],atol=1e-9);np.testing.assert_allclose(stick1[1],stick0[1],atol=1e-9)
        self.assertGreater(abs(env.data.body('boom').xmat.reshape(3,3)[0,0]-1),1e-3)   # the boom itself moved
        self.assertTrue(np.isfinite(env.data.qpos).all());env.close()

    def test_randomized_control_stays_finite(self):
        rng=np.random.default_rng(27)
        for profile in ['arm2']:
            env=ExcavatorEnv(profile)
            for seed in range(5):
                obs,_=env.reset(seed=seed)
                for _ in range(160):
                    obs,_,done,truncated,info=env.step(rng.uniform(-1,1,env.n))
                    self.assertTrue(np.isfinite(obs).all());self.assertFalse(info['unstable'])
                    self.assertTrue(np.all(env.data.qpos>env.bounds[:,0]-.03))
                    self.assertTrue(np.all(env.data.qpos<env.bounds[:,1]+.03))
                    if done or truncated:break
            env.close()


    def test_invalid_commands_are_rejected(self):
        env=ExcavatorEnv();env.reset(seed=1)
        with self.assertRaises(ValueError):env.step([np.nan,0])
        with self.assertRaises(ValueError):env.step([0])
        env.close()


if __name__=='__main__':unittest.main()
