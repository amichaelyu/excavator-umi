import unittest
import json
import numpy as np
import mujoco
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
        # the two-joint model arm2.xml is that chain with slew and stick welded at their rest values
        cfg=json.loads((CONFIG/'robot.json').read_text())
        model=mujoco.MjModel.from_xml_path(str(MODELS/'arm2.xml'));data=mujoco.MjData(model)
        n=len(HARDWARE_JOINTS);self.assertEqual(model.nq,n);self.assertEqual(model.nu,n)
        for q in np.random.default_rng(28).uniform(-.4,.4,(20,n)):
            data.qpos[:]=q;mujoco.mj_forward(model,data);rot,pos=fk(full_chain_q(q),cfg)
            for i,n in enumerate(['base','slew','boom','stick','bucket']):
                np.testing.assert_allclose(data.body(n).xpos,pos[i],atol=1e-8)
                np.testing.assert_allclose(data.body(n).xmat.reshape(3,3),rot[i],atol=1e-8)


if __name__=='__main__':unittest.main()
