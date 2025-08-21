from typing import Any, NamedTuple

import gym
from gym import spaces

import franka_env
import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep

import os
import cv2
import torch
from scipy.spatial.transform import Rotation as R

from robot_utils.franka.utils import (
    triangulate_points,
    pixel2d_to_3d,
    rigid_transform_3D,
)
from robot_utils.franka.gripper_points import extrapoints, Tshift
from robot_utils.franka.utils import pixelkey2camera

import zmq
import io
import base64
from PIL import Image
from pathlib import Path

def serialize_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(
        self,
        env,
        task_name,
        object_labels,
        calib_path,
        width=256,
        height=256,
        use_robot=False,
        max_episode_len=300,
        max_state_dim=100,
        pixel_keys=["pixels0"],
        use_robot_points=True,
        num_robot_points=9,
        use_object_points=True,
        num_object_points=8,
        points_cfg=None,
        use_gt_depth=False,
        use_depth_anything=False,   # NEW parameter for Depth Anything mode
        point_dim=2,
        reset_flag = True,
        use_pin_points=False,
    ):
        self._env = env
        self._task_name = task_name
        self._object_labels = object_labels
        self._height, self._width = height, width
        self.use_robot = use_robot
        self._max_episode_len = max_episode_len
        self._max_state_dim = max_state_dim
        self._pixel_keys = pixel_keys
        self._device = "cpu"
        self._use_gt_depth = use_gt_depth
        self._point_dim = point_dim
        self._use_depth_anything = use_depth_anything  # store the new flag
        self.reset_flag = reset_flag

        # If using Depth Anything, instantiate the depth model.
        if self._use_depth_anything:
            from point_utils.depth import Depth
            self._depth_model = Depth("/home/bobby/Point-Policy/Depth-Anything-V2", 'cuda')
            

        # track vars
        self._use_robot_points = use_robot_points
        self._num_robot_points = num_robot_points
        self._use_object_points = use_object_points
        self._num_object_points = num_object_points
        self._use_pin_points = use_pin_points

        # ------------------------------------------------------------------
        # pin-point servers (only when _use_pin_points=True)
        # ------------------------------------------------------------------
        if self.use_robot and self._use_pin_points:
            self._robopoint_addr = os.getenv("ROBOPOINT_ADDR", "tcp://localhost:5558")
            self._mast3r_addr    = os.getenv("MAST3R_ADDR",    "tcp://localhost:5559")

            self._zmq_ctx = zmq.Context.instance()   # reuse one context
            # sockets are created lazily on first call
            self._robopoint_sock = None
            self._mast3r_sock    = None

        if self.use_robot and self._use_object_points and not self._use_pin_points:
            if points_cfg is None:
                raise ValueError(
                    "points_cfg must be provided when use_object_points=True"
                )
            # init points class if using object points
            from point_utils.points_class import PointsClass

            points_cfg["task_name"] = task_name
            points_cfg["pixel_keys"] = self._pixel_keys
            points_cfg["object_labels"] = object_labels
            self._points_class = PointsClass(**points_cfg)
            self._des_object = os.environ.get("DES_OBJECT",
                                  self._object_labels[0]
                                  if self._object_labels else "")
            print(f'we are using {self._des_object} as the desired object')

        # calibration data
        assert calib_path is not None
        # print(calib_path)
        # exit()
        self.calibration_data = np.load(calib_path, allow_pickle=True).item()
        self._camera_names = list(self.calibration_data.keys())
        self.camera_projections = {}
        for camera_name in self._camera_names:
            intrinsic = self.calibration_data[camera_name]["int"]
            intrinsic = np.concatenate((intrinsic, np.zeros((3, 1))), axis=1)
            extrinsic = self.calibration_data[camera_name]["ext"]
            self.camera_projections[camera_name] = intrinsic @ extrinsic

        obs = self._env.reset(reset_flag=self.reset_flag)  # maybe change here to control whether reset at the beginning
        # import ipdb; ipdb.set_trace()
        if self.use_robot:
            pixels = obs[self._pixel_keys[0]]
            self.observation_space = spaces.Box(
                low=0, high=255, shape=pixels.shape, dtype=pixels.dtype
            )

            # Action spec
            action_spec = self._env.action_space
            self._action_spec = specs.Array(
                shape=action_spec.shape, dtype=action_spec.dtype, name="action"
            )
            # Observation spec
            robot_state = obs["features"]
            self._obs_spec = {}
            for pixel_key in self._pixel_keys:
                self._obs_spec[pixel_key] = specs.BoundedArray(
                    shape=obs[pixel_key].shape,
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name=pixel_key,
                )
            self._obs_spec["proprioceptive"] = specs.BoundedArray(
                shape=robot_state.shape,
                dtype=np.float32,
                minimum=-np.inf,
                maximum=np.inf,
                name="proprioceptive",
            )
        else:
            pixels, features = obs["pixels"], obs["features"]
            self.observation_space = spaces.Box(
                low=0, high=255, shape=pixels.shape, dtype=pixels.dtype
            )

            # Action spec
            action_spec = self._env.action_space
            self._action_spec = specs.Array(
                shape=action_spec.shape, dtype=action_spec.dtype, name="action"
            )

            # Observation spec
            self._obs_spec = {}
            for pixel_key in self._pixel_keys:
                self._obs_spec[pixel_key] = specs.BoundedArray(
                    shape=pixels.shape,
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name=pixel_key,
                )
            self._obs_spec["proprioceptive"] = specs.BoundedArray(
                shape=features.shape,
                dtype=np.float32,
                minimum=-np.inf,
                maximum=np.inf,
                name="proprioceptive",
            )
        self._obs_spec["features"] = specs.BoundedArray(
            shape=(self._max_state_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="features",
        )

        self.render_image = None
        self.prev_gripper_points = None

        # amount for shifting the points in robot base frame
        self.Tshift = Tshift

    def reset(self, **kwargs):
        self._step = 0
        obs = self._env.reset(**kwargs)
        self.prev_gripper_state = -1  # Default open gripper

        self._current_pose = obs["features"]

        observation = {}

        # point tracker init
        robot_points, robot_points_3d = self.get_pixel_on_robot()
        self.init_track_points(obs, robot_points, robot_points_3d)

        for pixel_key in self._pixel_keys:
            observation[pixel_key] = obs[pixel_key]

        if self._point_dim == 2:
            # pixels and point tracks
            for pixel_key in self._pixel_keys:
                observation[f"point_tracks_{pixel_key}"] = self._track_pts[pixel_key]

        # Get 3d points from 3D depth or 2D triangulation
        elif self._point_dim == 3 and not self._use_gt_depth:
            P, pts = [], []
            for pixel_key in self._pixel_keys:
                camera_name = pixelkey2camera[pixel_key]
                P.append(self.camera_projections[camera_name])
                pt2d = self._track_pts[pixel_key]
                pts.append(pt2d)

            pts3d = triangulate_points(P, pts)[:, :3]
            pts3d[: self._num_robot_points] = robot_points_3d
            for pixel_key in self._pixel_keys:
                observation[f"point_tracks_{pixel_key}"] = np.array(pts3d)
        elif self._point_dim == 3 and self._use_gt_depth:
            for pixel_key in self._pixel_keys:
                camera_name = pixelkey2camera[pixel_key]
                pt2d = self._track_pts[pixel_key]
                # Use Depth Anything if enabled; otherwise, use the provided depth map.
                if self._use_depth_anything:
                    # import ipdb; ipdb.set_trace()
                    depth_map = self._depth_model.get_depth(obs[pixel_key])
                else:
                    depth_key = f"depth{pixel_key[-1]}"
                    depth_map = obs[depth_key]
                # Compute depth for each point
                depths = []
                for pt in pt2d:
                    x, y = pt.astype(int)
                    depths.append(depth_map[y, x])
                depths = np.array(depths) / 1000.0  # convert to meters
                extr = self.calibration_data[camera_name]["ext"]
                intr = self.calibration_data[camera_name]["int"]
                pt3d = pixel2d_to_3d(pt2d, depths, intr, extr)
                observation[f"point_tracks_{pixel_key}"] = pt3d

        observation["features"] = self._current_pose
        observation["goal_achieved"] = False
        self.observation = observation
        return observation

    # def step(self, action):
    #     self._step += 1
    #     robot_action = self.point2action(action)
    #     print("Robot action:", robot_action)
    #     obs, reward, done, info = self._env.step(robot_action)

    #     self._current_pose = obs["features"]

    #     observation = {}
    #     for pixel_key in self._pixel_keys:
    #         observation[pixel_key] = obs[pixel_key]

    #     # robot points
    #     robot_points, robot_points_3d = self.get_pixel_on_robot()
    #     self.prev_gripper_points = robot_points_3d
    #     for pixel_key in self._pixel_keys:
    #         robot_point = robot_points[pixel_key]
    #         current_track = robot_point

    #         if self._use_object_points:
    #             self._points_class.add_to_image_list(
    #                 obs[pixel_key][:, :, ::-1], pixel_key
    #             )
    #             self._points_class.track_points(pixel_key)
    #             object_pts = self._points_class.get_points_on_image(pixel_key).numpy()[0]
    #             current_track = np.concatenate([current_track, object_pts], axis=0)

    #         self._track_pts[pixel_key] = current_track
    #         observation[f"point_tracks_{pixel_key}"] = current_track

    #     # Get 3d points from 3D depth or 2D triangulation
    #     if self._point_dim == 3:
    #         if not self._use_gt_depth:
    #             P, pts = [], []
    #             for pixel_key in self._pixel_keys:
    #                 camera_name = pixelkey2camera[pixel_key]
    #                 P.append(self.camera_projections[camera_name])
    #                 pt2d = self._track_pts[pixel_key]
    #                 pts.append(pt2d)

    #             pts3d = triangulate_points(P, pts)[:, :3]
    #             pts3d[: self._num_robot_points] = robot_points_3d
    #             for pixel_key in self._pixel_keys:
    #                 observation[f"point_tracks_{pixel_key}"] = np.array(pts3d)
    #         else:
    #             for pixel_key in self._pixel_keys:
    #                 camera_name = pixelkey2camera[pixel_key]
    #                 pt2d = self._track_pts[pixel_key]
    #                 if self._use_depth_anything:
    #                     depth_map = self._depth_model.get_depth(obs[pixel_key])
    #                 else:
    #                     depth_key = f"depth{pixel_key[-1]}"
    #                     depth_map = obs[depth_key]
    #                 depths = []
    #                 for pt in pt2d:
    #                     x, y = pt.astype(int)
    #                     depths.append(depth_map[y, x])
    #                 depths = np.array(depths) / 1000.0  # convert to meters
    #                 extr = self.calibration_data[camera_name]["ext"]
    #                 intr = self.calibration_data[camera_name]["int"]
    #                 pt3d = pixel2d_to_3d(pt2d, depths, intr, extr)
    #                 observation[f"point_tracks_{pixel_key}"] = pt3d

    #     observation["features"] = self._current_pose
    #     observation["goal_achieved"] = done

    #     if self._step >= self._max_episode_len:
    #         done = True
    #     done = done | observation["goal_achieved"]

    #     self.observation = observation

    #     return observation, reward, done, info
    
    # def step(self, action):
    #     # ------------------------------------------------------------------
    #     # 1) Environment step
    #     # ------------------------------------------------------------------
    #     self._step += 1

    #     robot_action = self.point2action(action)
    #     print("Robot action:", robot_action)
    #     obs, reward, done, info = self._env.step(robot_action)

    #     self._current_pose = obs["features"]

    #     # ------------------------------------------------------------------
    #     # 2) Build raw observation dict (RGB frames only for now)
    #     # ------------------------------------------------------------------
    #     observation = {}
    #     for pixel_key in self._pixel_keys:
    #         observation[pixel_key] = obs[pixel_key]

    #     # ------------------------------------------------------------------
    #     # 3) 2‑D point tracks (robot + optional objects) for every camera
    #     # ------------------------------------------------------------------
    #     robot_points, robot_points_3d = self.get_pixel_on_robot()          # ← 2‑D & 3‑D
    #     self.prev_gripper_points = robot_points_3d

    #     for pixel_key in self._pixel_keys:
    #         robot_point = robot_points[pixel_key]                          # (N_r,2)
    #         current_track = robot_point

    #         if self._use_object_points:
    #             self._points_class.add_to_image_list(obs[pixel_key][:, :, ::-1],
    #                                                 pixel_key)
    #             self._points_class.track_points(pixel_key)
    #             object_pts = self._points_class.get_points_on_image(pixel_key)\
    #                                             .numpy()[0]               # (N_o,2)
    #             current_track = np.concatenate([current_track, object_pts], axis=0)

    #         self._track_pts[pixel_key] = current_track
    #         observation[f"point_tracks_{pixel_key}"] = current_track       # (N,2)

    #     # ------------------------------------------------------------------
    #     # 4) Lift to 3‑D if requested
    #     # ------------------------------------------------------------------
    #     if self._point_dim == 3:
    #         if not self._use_gt_depth:
    #             Ps, pts2d_all = [], []
    #             for pixel_key in self._pixel_keys:
    #                 camera_name = pixelkey2camera[pixel_key]
    #                 Ps.append(self.camera_projections[camera_name])        # 4×4
    #                 pts2d_all.append(self._track_pts[pixel_key])           # (N,2)

    #             pts3d = triangulate_points(Ps, pts2d_all)[:, :3]           # (N,3)
    #             pts3d[: self._num_robot_points] = robot_points_3d          # keep GT
    #             for pixel_key in self._pixel_keys:
    #                 observation[f"point_tracks_{pixel_key}"] = np.array(pts3d)

    #         else:  # depth supervision per camera
    #             for pixel_key in self._pixel_keys:
    #                 camera_name = pixelkey2camera[pixel_key]
    #                 pt2d = self._track_pts[pixel_key]

    #                 if self._use_depth_anything:
    #                     depth_map = self._depth_model.get_depth(obs[pixel_key])
    #                 else:
    #                     depth_key = f"depth{pixel_key[-1]}"
    #                     depth_map = obs[depth_key]

    #                 depths = [depth_map[int(pt[1]), int(pt[0])]
    #                         for pt in pt2d]                              # (N,)
    #                 depths = np.array(depths) / 1000.0                     # → m

    #                 extr = self.calibration_data[camera_name]["ext"]
    #                 intr = self.calibration_data[camera_name]["int"]
    #                 pt3d = pixel2d_to_3d(pt2d, depths, intr, extr)         # (N,3)
    #                 observation[f"point_tracks_{pixel_key}"] = pt3d

    #     # ------------------------------------------------------------------
    #     # 5) Book‑keeping flags
    #     # ------------------------------------------------------------------
    #     observation["features"] = self._current_pose
    #     observation["goal_achieved"] = done

    #     if self._step >= self._max_episode_len:
    #         done = True
    #     done = done | observation["goal_achieved"]

    #     self.observation = observation

    #     # ==================================================================
    #     # 6) DEBUG VISUALISATION  – re‑project 3‑D points onto every camera
    #     #    Comment‑out this whole block when you no longer need the PNGs.
    #     # ==================================================================
    #     import cv2
    #     from pathlib import Path
    #     debug_dir = Path("debug_imgs")
    #     debug_dir.mkdir(exist_ok=True)

    #     for pixel_key in self._pixel_keys:
    #         img = self.observation[pixel_key].copy()

    #         pts3d = self.observation[f"point_tracks_{pixel_key}"]
    #         if pts3d.ndim != 2 or pts3d.shape[1] != 3:       # skip if still 2‑D
    #             continue

    #         camera_name = pixelkey2camera[pixel_key]
    #         P = self.camera_projections[camera_name]         # robot‑base → camera
    #         R, t = P[:3, :3], P[:3, 3]
    #         rvec, _ = cv2.Rodrigues(R)

    #         K = self.calibration_data[camera_name]["int"]
    #         D = np.zeros(5)                                  # assume no distortion

    #         pts2d, _ = cv2.projectPoints(pts3d.astype(np.float32),
    #                                     rvec, t, K, D)      # (N,1,2)

    #         for (x, y) in pts2d[:, 0].astype(int):
    #             cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

    #         cv2.imwrite(str(debug_dir /
    #                         f"step{self._step:04d}_{pixel_key}.png"),
    #                     img)

    #     import ipdb; ipdb.set_trace()   # ← handy breakpoint
    #     # ==================================================================

    #     # ------------------------------------------------------------------
    #     # 7) Return
    #     # ------------------------------------------------------------------
    #     return observation, reward, done, info

    def step(self, action):
        # ------------------------------------------------------------------
        # 1) 环境执行一步
        # ------------------------------------------------------------------
        self._step += 1

        robot_action = self.point2action(action)
        print("Robot action:", robot_action)
        obs, reward, done, info = self._env.step(robot_action)

        self._current_pose = obs["features"]

        # ------------------------------------------------------------------
        # 2) 组装 observation（先放 RGB 帧）
        # ------------------------------------------------------------------
        observation = {pk: obs[pk] for pk in self._pixel_keys}

        # ------------------------------------------------------------------
        # 3) 2-D 轨迹（机器人 + 目标物体）
        # ------------------------------------------------------------------
        robot_pts_2d, robot_pts_3d = self.get_pixel_on_robot()
        self.prev_gripper_points = robot_pts_3d

        for pk in self._pixel_keys:
            cur_track = robot_pts_2d[pk]

            if self._use_object_points and not self._use_pin_points:
                # 使用对象关键点
                self._points_class.add_to_image_list(obs[pk][:, :, ::-1], pk)
                self._points_class.track_points(pk)
                obj_pts = self._points_class.get_points_on_image(pk).numpy()[0]
                cur_track = np.concatenate([cur_track, obj_pts], axis=0)
            # --- B) pin-point 流程（PointClass 已经被绕开）---------
            elif self._use_pin_points:
                # 取上一帧保存的目标点，或者首帧 pin-point
                pin_pts = self._track_pts[pk][-self._num_object_points :]
                # 也可以在此处调用 MASt3R / CoTracker 对 pin_pts 做前向跟踪，
                # 若暂时不跟踪，也至少把上一帧坐标沿用下来：
                cur_track = np.concatenate([cur_track, pin_pts], axis=0)

            self._track_pts[pk] = cur_track
            observation[f"point_tracks_{pk}"] = cur_track

        # ------------------------------------------------------------------
        # 4) 提升到 3-D（如果需要）
        # ------------------------------------------------------------------
        if self._point_dim == 3:
            if not self._use_gt_depth:
                # (a) 三角测量
                Ps, pts2d_all = [], []
                for pk in self._pixel_keys:
                    cam = pixelkey2camera[pk]
                    Ps.append(self.camera_projections[cam])   # K·[R|t]
                    pts2d_all.append(self._track_pts[pk])

                pts3d = triangulate_points(Ps, pts2d_all)[:, :3]
                pts3d[: self._num_robot_points] = robot_pts_3d
                for pk in self._pixel_keys:
                    observation[f"point_tracks_{pk}"] = pts3d
            else:
                # (b) 深度图反投影
                for pk in self._pixel_keys:
                    cam = pixelkey2camera[pk]
                    pt2d = self._track_pts[pk]

                    depth_map = (
                        self._depth_model.get_depth(obs[pk])
                        if self._use_depth_anything
                        else obs[f"depth{pk[-1]}"]
                    )

                    depths = np.array([depth_map[int(y), int(x)] for x, y in pt2d]) / 1000.0
                    extr = self.calibration_data[cam]["ext"]
                    intr = self.calibration_data[cam]["int"]
                    pt3d = pixel2d_to_3d(pt2d, depths, intr, extr)
                    observation[f"point_tracks_{pk}"] = pt3d

        # ------------------------------------------------------------------
        # 5) 其它标志位
        # ------------------------------------------------------------------
        observation["features"] = self._current_pose
        observation["goal_achieved"] = done

        if self._step >= self._max_episode_len:
            done = True
        done = done | observation["goal_achieved"]

        self.observation = observation

        # # ==================================================================
        # # 6) DEBUG 可视化 —— 把 3-D 点重新投影到各相机
        # # ==================================================================
        # import cv2
        # from pathlib import Path
        # debug_dir = Path("debug_imgs")
        # debug_dir.mkdir(exist_ok=True)

        # for pk in self._pixel_keys:
        #     img = self.observation[pk].copy()
        #     pts3d = self.observation[f"point_tracks_{pk}"]

        #     # 只在已经是 3-D 时可视化
        #     if pts3d.ndim != 2 or pts3d.shape[1] != 3:
        #         continue

        #     cam = pixelkey2camera[pk]

        #     # —— 修 正 处 —— 直接使用外参，不再重复乘 K
        #     extr = self.calibration_data[cam]["ext"]     # 4×4, world → cam
        #     R_wc = extr[:3, :3]
        #     t_wc = extr[:3, 3]
        #     rvec, _ = cv2.Rodrigues(R_wc)

        #     K = self.calibration_data[cam]["int"]        # 内参
        #     D = np.zeros(5)                              # 若有畸变可替换

        #     pts2d, _ = cv2.projectPoints(
        #         pts3d.astype(np.float32), rvec, t_wc, K, D
        #     )
        #     pts2d = pts2d[:, 0]

        #     print(f"[{pk}] x {pts2d[:,0].min():.1f}~{pts2d[:,0].max():.1f}, "
        #         f"y {pts2d[:,1].min():.1f}~{pts2d[:,1].max():.1f}")

        #     for x_f, y_f in pts2d:
        #         x, y = int(round(x_f)), int(round(y_f))
        #         if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        #             cv2.circle(img, (x, y), 6, (0, 0, 255), -1)   # 红点
        #         else:
        #             print(f"⚠️  {pk} 投影点 ({x_f:.1f},{y_f:.1f}) 越界，已跳过")

        #     cv2.imwrite(str(debug_dir / f"step{self._step:04d}_{pk}.png"), img)

        # import ipdb; ipdb.set_trace()
        # ==================================================================

        # ------------------------------------------------------------------
        # 7) 返回
        # ------------------------------------------------------------------
        return observation, reward, done, info




    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        return cv2.resize(self._env.render("rgb_array"), (width, height))

    def get_pixel_on_robot(self):
        # get current gripper pose in robot base frame
        pos = self._current_pose[:3]
        ori = self._current_pose[3:7]  # in quat
        T_g_b = np.eye(4)
        T_g_b[:3, :3] = R.from_quat(ori).as_matrix()
        T_g_b[:3, 3] = pos

        # shift the points in robot base frame
        T_g_b = T_g_b @ self.Tshift

        # add extra points
        points3d = [T_g_b[:3, 3]]
        gripper_state = self._current_pose[-1]
        for idx, Tp in enumerate(extrapoints):
            if gripper_state == 1 and idx in [0, 1]:
                Tp = Tp.copy()
                Tp[1, 3] = 0.015 if idx == 0 else -0.015
            pt = T_g_b @ Tp
            pt = pt[:3, 3]
            points3d.append(pt[:3])
        points3d = np.array(points3d)

        pixel_poses = {}
        for pixel_key in self._pixel_keys:
            if pixel_key == "pixels51":
                continue

            camera_name = pixelkey2camera[pixel_key]
            # import ipdb; ipdb.set_trace()

            P = self.calibration_data[camera_name]["ext"]
            K = self.calibration_data[camera_name]["int"]
            D = self.calibration_data[camera_name]["dist_coeff"]

            r, t = P[:3, :3], P[:3, 3]
            r, _ = cv2.Rodrigues(r)
            points2d, _ = cv2.projectPoints(points3d, r, t, K, D)
            points2d = points2d[:, 0]

            pixel_poses[pixel_key] = points2d

        return pixel_poses, points3d

    def init_track_points(self, obs, robot_points, robot_points_3d):
        self.prev_gripper_points = robot_points_3d

        self.base_robot_points = np.array(robot_points_3d)
        # orientation of the robot at the 0th step
        # self.robot_base_orientation = R.from_rotvec([np.pi, 0, 0]).as_matrix()
        self.robot_base_orientation = R.from_quat(self._current_pose[3:7]).as_matrix() # (x, y, z, w) -> (R)

        # grid_pts = None
        self._track_pts = {}
        for pixel_key in self._pixel_keys:
            points = []

            robot_pts = torch.tensor(
                robot_points[pixel_key], device=self._device
            ).float()[None]
            if self._use_robot_points:
                points.append(robot_pts)
            else:
                points[0][:, -len(robot_pts[0]) :] = robot_pts

            # # modify to use server to get object points
            # if self._use_pin_points:
            #     # --------------------------------------------------------------
            #     # (A) 通过 pin-points 获取对象关键点
            #     # --------------------------------------------------------------
            #     ref_pk = self._pixel_keys[0]                # 选第 1 号相机做参考
            #     print(f"Using {ref_pk} as reference camera for pin-pointing")
            #     height, width = obs[ref_pk].shape[:2]

            #     # ------ 1) RoboPoint on ref camera ------
            #     if pixel_key == ref_pk:
            #         sock = self._get_socket("robopoint")
            #         image_rgb = Image.fromarray(obs[ref_pk][..., ::-1])
            #         req = {
            #             "image": serialize_image(image_rgb),
            #             "image_path": "",
            #             "prompt":
            #                 "Please pinpoint 3 points on the target "
            #                     "object. Return a list of (x,y) normalized "
            #                     "between 0 and 1.",
            #             }
            #         sock.send_json(req)
            #         rep = sock.recv_json()
            #         ref_pts_norm = rep["points"][: self._num_object_points]

            #         # 存到 PointsClass
            #         pix_pts = [
            #             [0.0, *self._norm2pix(pt, width, height)[::-1]]
            #             for pt in ref_pts_norm
            #         ]
            #         tensor = torch.tensor(pix_pts, dtype=torch.float32)
            #         self._points_class.semantic_similar_points[
            #             f"{ref_pk}_{self._des_object}"
            #         ] = tensor

            #         # 留给下一个分支使用
            #         self._ref_image_b64 = serialize_image(image_rgb)
            #         self._ref_pts_norm  = ref_pts_norm

            #     # ------ 2) MASt3R on other cameras ------
            #     else:
            #         sock = self._get_socket("mast3r")
            #         image_rgb = Image.fromarray(obs[pixel_key][..., ::-1])
            #         req = {
            #             "ref_image":   self._ref_image_b64,
            #             "target_image": serialize_image(image_rgb),
            #             "points":       self._ref_pts_norm,
            #         }
            #         sock.send_json(req)
            #         rep = sock.recv_json()
            #         tgt_pts_norm = rep["target_points"][: self._num_object_points]

            #         h, w = image_rgb.size[1], image_rgb.size[0]
            #         pix_pts = [
            #             [0.0, *self._norm2pix(pt, w, h)[::-1]]
            #             for pt in tgt_pts_norm
            #         ]
            #         tensor = torch.tensor(pix_pts, dtype=torch.float32)
            #         self._points_class.semantic_similar_points[
            #             f"{pixel_key}_{self._des_object}"
            #         ] = tensor

            #     self._points_class.add_to_image_list(
            #         obs[pixel_key][:, :, ::-1], pixel_key
            #     )                
            if self._use_pin_points:
                # 1) 选 reference 相机
                ref_pk  = self._pixel_keys[0]
                height, width = obs[ref_pk].shape[:2]

                # ---------- A) 参考相机：RoboPoint ----------
                if pixel_key == ref_pk:
                    sock = self._get_socket("robopoint")
                    img  = Image.fromarray(obs[ref_pk][..., ::-1])
                    sock.send_json({
                        "image":  serialize_image(img),
                        "image_path": "",
                        "prompt": "Please pinpoint 3 points on the empty space under the bottle so bottle can be placed. Return a list of (x,y) normalized "
                    })
                    rep = sock.recv_json()
                    ref_pts_norm = rep["points"][: self._num_object_points]   # (N,2)
                    print(f"Pin-pointed {ref_pts_norm} on {ref_pk}")
                    # import ipdb; ipdb.set_trace()
                    # # CURRENT HACK TO BYPASS ROBOPOINT
                    # ref_pts_norm = [[0.4, 0.8593758]]

                    # # 记录给其它相机用
                    # self._ref_image_b64 = serialize_image(img)
                    # self._ref_pts_norm  = ref_pts_norm
                    # 用当前相机的尺寸，保持 (x,y) 顺序，不要 [::-1]
                    # import ipdb; ipdb.set_trace()
                    h, w = obs[ref_pk].shape[:2]              # numpy: (H,W,3)
                    pix_pts = [list(self._norm2pix(pt, w, h)) for pt in ref_pts_norm]  # [[x,y]]

                    self._ref_image_b64 = serialize_image(img)
                    self._ref_pts_norm  = ref_pts_norm

                    # 转成像素坐标
                    # pix_pts = [[0.0, *self._norm2pix(pt, width, height)[::-1]]
                    #         for pt in ref_pts_norm]
                    # import ipdb; ipdb.set_trace()
                    # pix_pts = [list(self._norm2pix(pt, width, height)[::-1])
                    #            for pt in ref_pts_norm]             # [[x, y]]

                # ---------- B) 其它相机：MASt3R ----------
                else:
                    sock = self._get_socket("mast3r")
                    img  = Image.fromarray(obs[pixel_key][..., ::-1])
                    # convert self._ref_pts_norm to pixel coordinates
                    pixel_coords = [
                        self._norm2pix(pt, width, height) for pt in self._ref_pts_norm
                    ]
                    sock.send_json({
                        "ref_image":    self._ref_image_b64,
                        "target_image": serialize_image(img),
                        "ref_points":      pixel_coords,  # (N,2)
                    })
                    rep = sock.recv_json()
                    tgt_pts = rep["target_points"][: self._num_object_points]
                    # convert to normalized coordinates
                    tgt_pts_norm = [
                        [pt[0] / width, pt[1] / height] for pt in tgt_pts
                    ]  # (N,2)
                    # # HACK
                    # tgt_pts_norm = [[0.27, 0.81]]

                    # import ipdb; ipdb.set_trace()

                    # import ipdb; ipdb.set_trace()

                    # h, w = img.size[1], img.size[0]
                    # # pix_pts = [[0.0, *self._norm2pix(pt, w, h)[::-1]]
                    # #         for pt in tgt_pts_norm]
                    # pix_pts = [list(self._norm2pix(pt, width, height)[::-1])
                    #            for pt in ref_pts_norm]             # [[x, y]]

                    # 用“目标相机”的尺寸，换算“目标相机”的点；不要 [::-1]
                    h, w = obs[pixel_key].shape[:2]
                    pix_pts = [list(self._norm2pix(pt, w, h)) for pt in tgt_pts_norm]  # [[x,y]]

                # === 关键：直接 append 到轨迹 ===
                # import ipdb; ipdb.set_trace()
                object_pts = torch.tensor(pix_pts, dtype=torch.float32)[None]   # shape (1,N,D)
                points.append(object_pts)                                       # <<<<<<<<
            elif self._use_object_points:
                # start server
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                socket.connect("tcp://100.96.11.47:6000")

                frame = obs[pixel_key]
                image = Image.fromarray(frame[..., ::-1])
                serialized_image = serialize_image(image)

                self._points_class.reset_episode()
                self._points_class.add_to_image_list(frame[:, :, ::-1], pixel_key)
                for object_label in self._object_labels:
                    request = {
                        "image": serialized_image,
                        "image_path": "",
                        # "query": f"Get the bounding box of the {object_label} in the image",
                        # "query": f"Get the bounding box of the bottle in the image",
                        "query": f"Get the bounding box of the {self._des_object} in the image",
                    }
                    socket.send_json(request)
                    response = socket.recv_json()
                    bbox = response["result"]
                    bbox = bbox[:-1] if len(bbox) == 5 else bbox
                    # make sure bbox is a list of int
                    bbox = [int(x) for x in bbox]
                    print(f"bbox: {bbox}")
                    print(f"bbox type: {type(bbox)}")
                    self._points_class.find_semantic_similar_points(
                        pixel_key, object_label, bbox
                    )
                    print(f'points: {self._points_class.semantic_similar_points}')
                self._points_class.track_points(pixel_key, is_first_step=True)
                self._points_class.track_points(pixel_key)
                object_pts = self._points_class.get_points_on_image(pixel_key)
                points.append(object_pts)
                # plot
                self._points_class.plot_image(pixel_key)

            self._track_pts[pixel_key] = torch.cat(points, dim=1)[0].numpy()

            # for debugging
            # import ipdb; ipdb.set_trace()
            R_const = self.robot_base_orientation                        
            R_runtime = R.from_quat(self._current_pose[3:7]).as_matrix()

            delta = R_const.T @ R_runtime
            print("ΔEuler runtime (deg):",
                R.from_matrix(delta).as_euler('xyz', degrees=True))


    def point2action(self, action):
        """
        Action is a dict with 10 points corresponding to each camera frame.
        """
        points, projection_matrices = [], []

        for pixel_key in self._pixel_keys:
            if pixel_key == "pixels51":
                continue
            robot_pts_end_idx = self._num_robot_points if self._use_robot_points else 0
            future_tracks = action[f"future_tracks_{pixel_key}"][
                :robot_pts_end_idx, : self._point_dim
            ]

            camera_name = pixelkey2camera[pixel_key]
            extrinsic = self.calibration_data[camera_name]["ext"]
            intrinsic = self.calibration_data[camera_name]["int"]
            intrinsic = np.concatenate([intrinsic, np.zeros((3, 1))], axis=1)
            projection_matrix = intrinsic @ extrinsic

            points.append(future_tracks)
            projection_matrices.append(projection_matrix)

        if self._point_dim == 2:
            points3d = triangulate_points(projection_matrices, points)[:, :3]
        elif self._point_dim == 3:
            points3d = np.mean(points, axis=0)

        robot_pos, ori = self.compute_action_from_3dpoints(points3d)
        gripper_state = self.compute_gripper(action)
        robot_action = self.compute_robot_action(robot_pos, ori, gripper_state)

        return robot_action

    def compute_action_from_3dpoints(self, points3d):
        robot_pos = points3d[0, :3]
        ori, _ = rigid_transform_3D(self.base_robot_points, points3d)
        ori = ori @ self.robot_base_orientation
        return robot_pos, ori

    def compute_gripper(self, action):
        gripper_state = action["gripper"][:1]

        if self.prev_gripper_state == -1 and gripper_state > -0.3:
            gripper_state = 1
        elif self.prev_gripper_state == 1 and gripper_state < 0.6:
            gripper_state = -1
        else:
            gripper_state = self.prev_gripper_state
        self.prev_gripper_state = gripper_state

        gripper_state = np.array([gripper_state])
        return gripper_state

    def compute_robot_action(self, target_position, target_orientation, gripper):
        """
        Return absolute actions
        """
        T_target = np.eye(4)
        T_target[:3, :3] = target_orientation
        T_target[:3, 3] = target_position

        # T_target = T_eef @ Tshift -> get T_eef
        T_eef = T_target @ np.linalg.inv(self.Tshift)
        target_position = T_eef[:3, 3]
        target_orientation = T_eef[:3, :3]

        # convert orientation from rotation matrix to quaternion
        target_orientation = R.from_matrix(target_orientation).as_quat()

        return np.concatenate([target_position, target_orientation, gripper])
    
    # ---------------------------  helpers  --------------------------------
    def _get_socket(self, name: str):
        if name == "robopoint":
            if self._robopoint_sock is None:
                self._robopoint_sock = self._zmq_ctx.socket(zmq.REQ)
                self._robopoint_sock.connect(self._robopoint_addr)
            return self._robopoint_sock
        elif name == "mast3r":
            if self._mast3r_sock is None:
                self._mast3r_sock = self._zmq_ctx.socket(zmq.REQ)
                self._mast3r_sock.connect(self._mast3r_addr)
            return self._mast3r_sock
        else:
            raise ValueError(name)

    @staticmethod
    def _norm2pix(pt_norm, width, height):
        x_n, y_n = pt_norm
        return int(x_n * width), int(y_n * height)


    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

        # Action spec
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.Array(
            shape=wrapped_action_spec.shape, dtype=dtype, name="action"
        )

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        step_type = StepType.LAST if done else StepType.MID

        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def point2action(self, action):
        return self._env.point2action(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return TimeStep(
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self, time_step, observation=None, action=None, reward=None, discount=None
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def point2action(self, action):
        return self._env.point2action(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(
    task_name,
    object_labels,
    action_repeat,
    height,
    width,
    max_episode_len,
    max_state_dim,
    calib_path,
    eval,  # True means use_robot=True
    pixel_keys,
    use_robot_points,
    num_robot_points,
    use_object_points,
    num_object_points,
    points_cfg,
    use_gt_depth,
    use_depth_anything,  # NEW parameter passed through make()
    point_dim,
    reset_flag=True,
    use_pin_points=False
):
    env = gym.make(
        "Franka-v1",
        height=height,
        width=width,
        use_robot=eval,
        use_gt_depth=use_gt_depth,
    )

    # apply wrappers
    env = RGBArrayAsObservationWrapper(
        env,
        task_name,
        object_labels,
        calib_path=calib_path,
        height=height,
        width=width,
        use_robot=eval,
        max_episode_len=max_episode_len,
        max_state_dim=max_state_dim,
        pixel_keys=pixel_keys,
        use_robot_points=use_robot_points,
        num_robot_points=num_robot_points,
        use_object_points=use_object_points,
        num_object_points=num_object_points,
        points_cfg=points_cfg,
        use_gt_depth=use_gt_depth,
        use_depth_anything=use_depth_anything,  # pass new flag to wrapper
        point_dim=point_dim,
        reset_flag=reset_flag,
        use_pin_points=use_pin_points,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = ExtendedTimeStepWrapper(env)

    return [env], [task_name]
