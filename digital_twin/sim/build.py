"""Segment the supplied textured scan and build articulated MuJoCo models.

The twin (``full4``; ``full4_calibrated`` with the measured ranges / speeds) articulates the
whole chain: slew (base turn), boom (the big arm), stick and bucket (the wrist); it is the only
model digital_twin.sim.kinematics, replays, goto_pose and the viewers use. ``arm2`` articulates only the two
joints the real machine drives - ``boom`` and ``bucket`` - with slew and stick welded (the hardware map's
``locked_joints``, kinematics.locked_joints()); it is the RL training environment (policy.training.rl.env).
"""
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

from ..paths import CONFIG, SIM, hardware_map, set_hardware_map
from .kinematics import locked_joints   # the hardware map's locked_joints: where arm2 welds slew, stick
from .pivots import transform
from . import pins


def fmt(v):
    return ' '.join(f'{float(x):.7g}' for x in v)

def segment(cfg):
    mesh = trimesh.load(SIM/'assets/scan/9_26_2026.obj', process=False)
    c = mesh.triangles_center
    # Anatomical cuts in the scan rest pose. Hoses and cylinder geometry are
    # attached to the closest parent; this is an MVP rigid-link approximation.
    robot = (c[:,1] > .073) & (c[:,0] > -.17) & (c[:,0] < .315) & (c[:,2] > -.13) & (c[:,2] < .085)
    labels = np.full(len(c), -1)
    labels[robot] = 1
    labels[robot & (c[:,1] < .108)] = 0
    arm = robot & ((c[:,1] > .215) | ((c[:,1]>.13)&(c[:,0]>-.035)&(c[:,0]<.04)&(c[:,2]>-.018)&(c[:,2]<.039)))
    labels[arm] = 2
    # Diagonal boundary at the arm separates the two touching links.
    labels[arm & (c[:,0] > .172 + .18*(c[:,1]-.395))] = 3
    labels[arm & (c[:,0] > .215) & (c[:,1] < .284)] = 4
    out = SIM/'assets/links';out.mkdir(parents=True,exist_ok=True)
    texture = SIM/'assets/scan/textures/f5329bb7ed3ebef713db473eee4c2f5b.jpg'
    image=Image.open(texture).convert('RGB');image.thumbnail((2048,2048));image.save(out/'albedo.png')
    origins = {'base':cfg['scan_origin'], **cfg.get('scan_frames',cfg['scan_pivots'])}
    counts = {}
    for i,name in enumerate(['base','slew','boom','stick','bucket']):
        part = mesh.submesh([np.where(labels==i)[0]], append=True, repair=False)
        part.vertices = transform(part.vertices, cfg)-transform(origins[name], cfg)
        # Export OBJ with UVs; MuJoCo uses an explicit shared texture material.
        text = trimesh.exchange.obj.export_obj(part, include_texture=True, return_texture=False, write_texture=False)
        (out/f'{name}.obj').write_text(text.rstrip()+'\n')
        counts[name] = {'faces':len(part.faces),'vertices':len(part.vertices),'bounds_local_m':part.bounds.tolist()}
    (SIM/'outputs/mesh_segments.json').write_text(json.dumps(counts,indent=2)+'\n')
    return counts


CONTROL_JOINTS = ['boom','bucket']          # what the hardware can move (= kinematics.HARDWARE_JOINTS)
FULL_JOINTS = ['slew','boom','stick','bucket']
PROFILES = {
    # the twin (digital_twin.sim.kinematics): slew spins, boom / stick / bucket free
    'full4':        {'active': FULL_JOINTS},
    # calibrated twin: measured boom / bucket / stick ranges (anchored at the homing stops) and
    # speeds, continuous slew; configs/sim/robot_full4_calibrated.json
    'full4_calibrated': {'active': FULL_JOINTS, 'config': 'robot_full4_calibrated.json'},
    # RL training environment (policy.training.rl.env): slew and stick welded at kinematics.locked_joints()
    'arm2':         {'active': CONTROL_JOINTS},
}

def coupled_config(ccfg, hw):
    """calibrated_config() + stick range measured (anchored at its top stop) + slew continuous
    (range_deg null). Channel rates live in hardware_map.json ``coupling`` (commands.ChannelModel)."""
    import copy
    out = copy.deepcopy(ccfg)
    cal = hw.get('calibration', {}); st = out['joints']['stick']
    r = (cal.get('ranges', {}).get('stick') or {}).get('range_measured_rad')
    top = float(hw['homing']['joints']['stick']['home_value_rad'])
    if r:
        st['range_deg'] = [round(float(np.degrees(top - r)), 6), round(float(np.degrees(top)), 6)]
        st['calibration'] = f'range measured ({np.degrees(r):.1f} deg), anchored at the top stop {np.degrees(top):.1f} deg'
    else:
        print('WARNING: no measured stick range; using the prior')
    # Stick speeds: the decoupled stick pins' measured speeds (calibration.speeds.stick), else the coupling
    # channels (per stick direction; on the old wiring A / B = stick + / - also yawed the base).
    sp = ((cal.get('speeds', {}).get('stick') or {}).get('speed_measured_rad_s') or {})
    meas = {k: float(sp[k]) for k in ('positive', 'negative') if (sp.get(k) or 0) > 0}
    ch = hw.get('coupling', {}).get('channels', {})
    up, down = ch.get('positive', {}).get('stick_deg_s'), ch.get('negative', {}).get('stick_deg_s')
    if meas:
        st['speed_positive_rad_s'] = meas.get('positive', meas.get('negative'))
        st['speed_negative_rad_s'] = meas.get('negative', meas.get('positive'))
        st['speed_rad_s'] = min(st['speed_positive_rad_s'], st['speed_negative_rad_s'])
        st['calibration'] += ', speed measured (calibration.speeds)'
    elif up and down:
        st['speed_positive_rad_s'], st['speed_negative_rad_s'] = float(np.radians(abs(up))), float(np.radians(abs(down)))
    sl = out['joints']['slew']
    sl['range_deg'] = None          # continuous (the base turns freely)
    # Slew speeds per direction from the base-turn calibration (hardware_map slew.yaw_deg_s: left = +slew),
    # as policy.runner.slew_config uses them.
    y = (hw.get('slew') or {}).get('yaw_deg_s') or {}
    yaw = {k: abs(float(y[w])) for k, w in (('positive', 'left'), ('negative', 'right')) if y.get(w)}
    if yaw:
        sl['speed_positive_rad_s'] = float(np.radians(yaw.get('positive', yaw.get('negative'))))
        sl['speed_negative_rad_s'] = float(np.radians(yaw.get('negative', yaw.get('positive'))))
        sl['speed_rad_s'] = min(sl['speed_positive_rad_s'], sl['speed_negative_rad_s'])
        sl['calibration'] = 'speed measured (hardware_map slew)'
    out['coupled_from'] = {'note': 'Generated by python -m digital_twin.sim.build --calibrated (full4_calibrated); '
                                   'slew/stick move only together through the stick channels (hardware_map coupling).'}
    return out

def calibrated_config(cfg, hw):
    """robot.json with the boom/bucket ranges and speeds replaced by the live calibration
    (the hardware map's ``calibration``): range = measured, anchored at the homing stop
    (boom top +30 deg -> [30 - range, 30]; bucket in-stop -40 deg -> [-40, -40 + range]); speeds =
    measured per direction. Joints without a measurement keep their priors."""
    import copy
    from .joint_config import real_joint_config
    real, src = real_joint_config(hw, None, warn=lambda m: print('WARNING:', m))
    out = copy.deepcopy(cfg)
    for n in CONTROL_JOINTS:
        j = out['joints'][n]
        j['range_deg'] = [round(float(np.degrees(real[n]['lo'])), 6), round(float(np.degrees(real[n]['hi'])), 6)]
        j['speed_positive_rad_s'], j['speed_negative_rad_s'] = real[n]['v_pos'], real[n]['v_neg']
        j['speed_rad_s'] = min(real[n]['v_pos'], real[n]['v_neg'])
        j['calibration'] = f'measured ({src[n]})'
    out['calibrated_from'] = {'hardware_map': 'configs/hardware_map.json calibration',
                              'note': 'Generated by python -m digital_twin.sim.build --calibrated; do not edit by hand. '
                                      'Anchored at the homing stops (no absolute stop angles were measured).'}
    return out

def build(cfg, profile='full4'):
    active = PROFILES[profile]['active']
    m = ET.Element('mujoco',model='E568_'+profile)
    welded = locked_joints()
    ET.SubElement(m,'compiler',angle='radian',meshdir='../assets/links',texturedir='../assets/links',autolimits='true')
    ET.SubElement(m,'option',timestep='.005',gravity='0 0 -9.81',integrator='implicitfast',cone='elliptic')
    vis=ET.SubElement(m,'visual');ET.SubElement(vis,'global',offwidth='960',offheight='720')
    ET.SubElement(vis,'quality',shadowsize='2048')
    default=ET.SubElement(m,'default')
    ET.SubElement(default,'joint',damping='.04',armature='.004',frictionloss='.015')
    ET.SubElement(default,'geom',friction='.8 .01 .001',condim='3',solref='.01 1')
    a=ET.SubElement(m,'asset')
    ET.SubElement(a,'texture',name='scan_texture',type='2d',file='albedo.png')
    ET.SubElement(a,'material',name='scan_material',texture='scan_texture',specular='.1',shininess='.1')
    for name in ['base','slew','boom','stick','bucket']:
        ET.SubElement(a,'mesh',name=name,file=name+'.obj',inertia='shell')
    w=ET.SubElement(m,'worldbody')
    ET.SubElement(w,'light',pos='0 -1 2',dir='0 0 -1',diffuse='.9 .9 .9')
    ET.SubElement(w,'light',pos='1 1 1.5',dir='-1 -1 -1',diffuse='.5 .5 .5')
    ET.SubElement(w,'geom',name='ground',type='plane',size='2 2 .1',rgba='.24 .27 .30 1')
    ET.SubElement(w,'camera',name='overview',pos='.85 -1 .65',xyaxes='.78 .62 0 -.26 .33 .91')
    ET.SubElement(w,'camera',name='side',pos='.20 -1 .30',xyaxes='1 0 0 0 0 1',fovy='42')
    target=ET.SubElement(w,'body',name='target',mocap='true',pos='.30 -.04 .18')
    ET.SubElement(target,'geom',type='sphere',size='.012',rgba='.1 1 .4 .6',contype='0',conaffinity='0')
    base=ET.SubElement(w,'body',name='base',pos='0 0 .001')
    ET.SubElement(base,'geom',type='mesh',mesh='base',material='scan_material',contype='0',conaffinity='0',group='1',mass='0')
    for y in [-.065,.065]:
        ET.SubElement(base,'geom',type='box',pos=f'-.02 {y} .025',size='.115 .018 .025',rgba='.08 .08 .08 0',group='3')
    actuators=ET.SubElement(m,'actuator')
    p={k:transform(v,cfg) for k,v in cfg['scan_pivots'].items()}
    frames={k:transform(v,cfg) for k,v in cfg.get('scan_frames',cfg['scan_pivots']).items()}
    parent=base; previous=np.zeros(3)
    for name in ['slew','boom','stick','bucket']:
        joint_cfg=cfg['joints'][name]
        body=ET.SubElement(parent,'body',name=name,pos=fmt(frames[name]-previous))
        axis=[0,0,1] if name=='slew' else [0,-1,0]
        weld=0.0 if name in active else welded.get(name,0.0)
        if weld:
            # Welded joint (the hardware map's locked_joints): rotate the body about its
            # pivot exactly as the hinge would at q=weld, so arm2 == full4 at that value.
            R=Rotation.from_rotvec(np.asarray(axis,float)*weld);a=p[name]-frames[name]
            fine=lambda v:' '.join(f'{float(x):.12g}' for x in v)
            body.set('pos',fine(frames[name]-previous+a-R.apply(a)))
            qx,qy,qz,qw=R.as_quat();body.set('quat',fine([qw,qx,qy,qz]))
        if name in active:
            if joint_cfg.get('range_deg') is None:          # continuous joint (full4_calibrated slew)
                ET.SubElement(body,'joint',name=name,type='hinge',axis=fmt(axis),pos=fmt(p[name]-frames[name]),limited='false')
                ET.SubElement(actuators,'position',name=name,joint=name,kp='35',kv='1.2',ctrllimited='false',forcerange=fmt([-joint_cfg['force_nm'],joint_cfg['force_nm']]))
            else:
                ET.SubElement(body,'joint',name=name,type='hinge',axis=fmt(axis),pos=fmt(p[name]-frames[name]),range=fmt(np.deg2rad(joint_cfg['range_deg'])))
                ET.SubElement(actuators,'position',name=name,joint=name,kp='35',kv='1.2',ctrlrange=fmt(np.deg2rad(joint_cfg['range_deg'])),forcerange=fmt([-joint_cfg['force_nm'],joint_cfg['force_nm']]))
        ET.SubElement(body,'geom',type='mesh',mesh=name,material='scan_material',contype='0',conaffinity='0',group='1',mass='0')
        if name=='slew':
            ET.SubElement(body,'geom',name='cab_collision',type='box',pos='-.04 .025 .045',size='.08 .055 .045',mass=str(joint_cfg['mass_kg']),rgba='.1 .1 .1 0',group='3')
        elif name in ['boom','stick']:
            end=p['stick' if name=='boom' else 'bucket']-frames[name]
            ET.SubElement(body,'geom',name=name+'_collision',type='capsule',fromto=fmt(np.r_[p[name]-frames[name],end]),size='.012',mass=str(joint_cfg['mass_kg']),rgba='.4 .4 .4 0',group='3')
        else:
            # Separate plates leave the scoop concave (a mesh convex hull would close its mouth).
            end=p['tip']-frames['bucket'];width=.030
            for y in [-width,width]:
                ET.SubElement(body,'geom',type='box',pos=fmt(end*.45+[0,y,0]),size='.025 .002 .027',mass='.01',rgba='.4 .4 .4 0',group='3')
            ET.SubElement(body,'geom',type='box',pos=fmt(end*.8),size='.024 .030 .003',mass='.015',rgba='.4 .4 .4 0',group='3')
            ET.SubElement(body,'geom',type='box',pos=fmt([.012,0,-.023]),size='.003 .030 .025',mass='.015',rgba='.4 .4 .4 0',group='3')
            ET.SubElement(body,'site',name='tooth_tip',pos=fmt(end),size='.003',rgba='1 .2 .1 1')
            ET.SubElement(body,'camera',name='bucket_camera',pos='0 0 .025',xyaxes='0 -1 0 .7 0 .7',fovy='75')
        parent=body;previous=frames[name]
    contact=ET.SubElement(m,'contact')
    for b1,b2 in [('base','slew'),('slew','boom'),('boom','stick'),('stick','bucket'),('slew','stick')]:
        ET.SubElement(contact,'exclude',body1=b1,body2=b2)
    out=SIM/'models';out.mkdir(exist_ok=True)
    path=out/f'{profile}.xml'
    ET.indent(m);ET.ElementTree(m).write(path,encoding='unicode')
    return path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models-only',action='store_true',help='Reuse shipped link meshes; original scan is only needed to re-segment')
    parser.add_argument('--calibrated',action='store_true',help='Only (re)write configs/sim/robot_full4_calibrated.json from hardware_map.json calibration/coupling and build models/full4_calibrated.xml')
    parser.add_argument('--hardware-map',help='hardware map to build from (default: digital_twin.paths.hardware_map(), $UMI_EXCAVATOR_HARDWARE_MAP or the tracked configs/hardware_map.json)')
    args=parser.parse_args()
    if args.hardware_map:
        set_hardware_map(args.hardware_map)
    print('hardware map:',hardware_map())
    cfg=json.loads((CONFIG/'robot.json').read_text())
    if args.calibrated:
        hw=pins.read(hardware_map())
        kcfg=coupled_config(calibrated_config(cfg,hw),hw)
        (CONFIG/PROFILES['full4_calibrated']['config']).write_text(json.dumps(kcfg,indent=2)+'\n')
        path=build(kcfg,'full4_calibrated')
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(path))
        print(path.name,'nq',model.nq,'nu',model.nu,{n:([round(float(np.degrees(x)),2) for x in model.jnt_range[model.joint(n).id]] if model.jnt_limited[model.joint(n).id] else 'continuous') for n in FULL_JOINTS})
        return
    (SIM/'outputs').mkdir(exist_ok=True)
    counts=segment(cfg) if not args.models_only else {name:{'source':'existing link OBJ'} for name in ['base','slew','boom','stick','bucket']}
    paths=[]
    for p in PROFILES:
        if 'config' in PROFILES[p]:                 # full4_calibrated: only from its own config
            f=CONFIG / PROFILES[p]['config']
            if f.exists(): paths.append(build(json.loads(f.read_text()),p))
            continue
        paths.append(build(cfg,p))
    lengths={a+'_'+b:float(np.linalg.norm(transform(cfg['scan_pivots'][a],cfg)-transform(cfg['scan_pivots'][b],cfg))) for a,b in [('boom','stick'),('stick','bucket'),('bucket','tip')]}
    (SIM/'outputs/geometry.json').write_text(json.dumps({'lengths_m':lengths,'provenance':cfg['provenance'],'meshes':counts},indent=2)+'\n')
    import mujoco
    for path in paths:
        model=mujoco.MjModel.from_xml_path(str(path));print(path.name,'nq',model.nq,'nu',model.nu)

if __name__=='__main__':main()
