"""Franka Panda (robosuite, MuJoCo) with a Robotiq 2F-85 gripper, seen by a camera in front of the robot, with the
V-JEPA 2-AC state and action conventions:
  state  = [x, y, z, roll, pitch, yaw, closedness] of the end effector in the robot base frame (extrinsic xyz Euler,
           gripper pointing down gives roll near +-pi), closedness 0 open .. 1 closed; z is measured from the table
           top (the DROID arms stand on the table; robosuite's pedestal puts the base 0.112 m above it);
  action = target displacement of the end effector [dx, dy, dz, 0, 0, 0, dgrip], applied by the OSC controller as a
           tracked target for one control period (default 0.5 s), orientation held at the episode's start orientation.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402
import robosuite as suite  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

CAM = "frontview"


def look_at_quat(pos, target, up=(0, 0, 1)):
    """MuJoCo camera quaternion (w, x, y, z): the camera looks along its -z axis with +y up."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    mat = np.stack([r, u, -f], axis=1)          # columns: camera x, y, z in world
    x, y, z, w = R.from_matrix(mat).as_quat()
    return np.array([w, x, y, z])


class FrankaReach:
    def __init__(self, size=256, control_freq=20, period=0.5, cam_offset=(1.05, 0.25, 0.28), cam_target=(0.45, 0.0, 0.25),
                 gripper="Robotiq85Gripper", env_name="Lift"):
        self.env = suite.make(env_name, robots="Panda", gripper_types=gripper, has_renderer=False,
                              has_offscreen_renderer=True, use_camera_obs=True, camera_names=CAM,
                              camera_heights=size, camera_widths=size, control_freq=control_freq, horizon=100000,
                              ignore_done=True, initialization_noise=None, hard_reset=False)
        self.steps_per_action = int(round(period * control_freq))
        self.cam_offset, self.cam_target = np.array(cam_offset), np.array(cam_target)
        self.obs = None

    # ---------------------------------------------------------------- frames
    @property
    def sim(self):
        return self.env.sim

    def _base(self):
        robot = self.env.robots[0]
        bid = self.sim.model.body_name2id(robot.robot_model.root_body)
        return self.sim.data.body_xpos[bid].copy(), self.sim.data.body_xmat[bid].reshape(3, 3).copy()

    def _place_camera(self):
        p, Rb = self._base()
        cid = self.sim.model.camera_name2id(CAM)
        pos = p + Rb @ self.cam_offset
        self.sim.model.cam_pos[cid] = pos
        self.sim.model.cam_quat[cid] = look_at_quat(pos, p + Rb @ self.cam_target)
        self.sim.model.cam_fovy[cid] = 60.0

    def reset(self, seed, start=(0.50, 0.0, 0.26)):
        """Start: end effector at `start` (state coordinates), gripper open, table top dark like the DROID tables."""
        np.random.seed(seed)
        self.obs = self.env.reset()
        self._place_camera()
        m = self.sim.model
        m.geom_rgba[m.geom_name2id("table_visual")] = [0.12, 0.12, 0.13, 1.0]
        self.obs = self._step_raw(np.zeros(self.env.action_dim), render=True)
        p, Rb = self._base()
        self.rot0 = R.from_matrix(Rb) * R.from_euler("xyz", [np.pi, 0, 0])   # gripper straight down
        self.z_off = p[2] - self.env.table_offset[2]
        tgt = np.array(start, float) - [0, 0, self.z_off]
        self.move_to(p + Rb @ tgt, iters=8)
        return self.frame(), self.state()

    def _step_raw(self, a, render=True):
        o, _, _, _ = self.env.step(a)
        return o

    def frame(self):
        return self.obs[CAM + "_image"][::-1].copy()          # robosuite renders upside down

    def _eef_rot_world(self):
        q = self.obs["robot0_eef_quat"]                       # x, y, z, w in world
        return R.from_quat(q)

    def eef_world(self):
        return self.obs["robot0_eef_pos"].copy()

    def gripper_closedness(self):
        return float(np.clip(self.obs["robot0_gripper_qpos"][0] / 0.8, 0, 1))

    def state(self):
        p, Rb = self._base()
        pos = Rb.T @ (self.eef_world() - p) + [0, 0, getattr(self, "z_off", 0.0)]
        rot = R.from_matrix(Rb.T) * self._eef_rot_world()
        eul = rot.as_euler("xyz")
        if eul[0] > 0:                                         # roll near +-pi: report it as in DROID (near -pi)
            eul[0] -= 2 * np.pi
        return np.concatenate([pos, eul, [self.gripper_closedness()]]).astype(np.float32)

    # ---------------------------------------------------------------- control
    def apply(self, action, grip_cmd=None):
        """action [7] in the model convention. Tracks the target position for one control period."""
        p, Rb = self._base()
        target = self.eef_world() + Rb @ np.asarray(action[:3], float)
        close = (self.state()[6] + action[6]) > 0.5 if grip_cmd is None else grip_cmd
        ctrl = self.env.robots[0].part_controllers["right"]
        pmax = float(np.atleast_1d(ctrl.output_max)[0])
        rmax = float(np.atleast_1d(ctrl.output_max)[3])
        for _ in range(self.steps_per_action):
            dp = (target - self.eef_world()) / pmax
            rerr = (self.rot0 * self._eef_rot_world().inv()).as_rotvec() / rmax
            a = np.zeros(self.env.action_dim)
            a[:3] = np.clip(dp, -1, 1)
            a[3:6] = np.clip(rerr, -1, 1)
            a[6] = 1.0 if close else -1.0
            self.obs = self._step_raw(a)
        return self.frame(), self.state()

    def move_to(self, target_world, iters=6):
        p, Rb = self._base()
        for _ in range(iters):
            d = Rb.T @ (target_world - self.eef_world())
            self.apply(np.concatenate([d, np.zeros(4)]), grip_cmd=False)
        return self.frame(), self.state()

    def project(self, pts_world):
        """Pixel (row, col) in the 256 frame (after the vertical flip) of world points [N, 3]."""
        cid = self.sim.model.camera_name2id(CAM)
        pos = self.sim.data.cam_xpos[cid]
        mat = self.sim.data.cam_xmat[cid].reshape(3, 3)
        fovy = np.deg2rad(self.sim.model.cam_fovy[cid])
        h = w = self.obs[CAM + "_image"].shape[0]
        f = 0.5 * h / np.tan(fovy / 2)
        pc = (np.asarray(pts_world) - pos) @ mat              # camera coords: x right, y up, -z forward
        u = w / 2 + f * pc[:, 0] / -pc[:, 2]
        v = h / 2 - f * pc[:, 1] / -pc[:, 2]
        return np.stack([v, u], 1)

    def gripper_relevance(self, sigma=2.0):
        """[256] Gaussian bump (grid units) around the projected end effector."""
        v, u = self.project(self.eef_world()[None])[0] / 16.0
        gh, gw = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        r = np.exp(-((gh + 0.5 - v) ** 2 + (gw + 0.5 - u) ** 2) / (2 * sigma ** 2))
        return r.flatten().astype(np.float32)
