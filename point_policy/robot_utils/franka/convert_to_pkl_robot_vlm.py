import sys

sys.path.append("../../")

import re
import yaml
import argparse
import pickle as pkl
from pathlib import Path
import cv2
import numpy as np
from pandas import read_csv
from scipy.spatial.transform import Rotation as R

from gripper_points import extrapoints, Tshift

import torch
from point_utils.points_class import PointsClass
from utils import (
    camera2pixelkey,
    pixel2d_to_3d_torch,
    triangulate_points,
)

import zmq
import io
import base64
from PIL import Image

# =============================================================
#                      Helper utilities
# =============================================================

def serialize_image(image):
    """Serialize a PIL.Image to base‑64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_number(s: str) -> float:
    """Extract a float no matter whether it is wrapped in np.float32/64()."""
    s = s.strip().strip("[]")
    m = re.match(r"np\.float(?:32|64)\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)", s)
    if m:
        return float(m.group(1))
    m = re.match(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)
    if m:
        return float(m.group(0))
    raise ValueError(f"Cannot extract number from '{s}'")

# =============================================================
#                      Argument parsing
# =============================================================

parser = argparse.ArgumentParser("Convert processed robot data into a pkl file")
parser.add_argument("--data_dir", required=True)
parser.add_argument("--calib_path", required=True)
parser.add_argument("--task_names", nargs="+", type=str)
parser.add_argument("--num_demos", type=int)
parser.add_argument("--process_points", type=bool, default=True)
parser.add_argument("--use_gt_depth", type=bool, default=False)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
CALIB_PATH = Path(args.calib_path)
TASK_NAMES = args.task_names
NUM_DEMOS = args.num_demos
PROCESS_POINTS = args.process_points
USE_GT_DEPTH = args.use_gt_depth
USE_DEPTH_ANYTHING = False

# =============================================================
#                     Global parameters
# =============================================================

camera_indices = [2, 5]
original_img_size = (640, 480)
crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)
save_img_size = (256, 256)

# -------------------------------------------------------------
#   Leave empty for robot‑only mode; include labels otherwise.
# -------------------------------------------------------------
object_labels: list[str] = []  # e.g. ["bottle"]
OBJECT_POINTS_ENABLED = len(object_labels) > 0

PROCESSED_DATA_PATH = DATA_DIR / "processed_data"
SAVE_DATA_PATH = DATA_DIR / "expert_demos" / "franka_env"
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

if TASK_NAMES is None:
    TASK_NAMES = [d.name for d in PROCESSED_DATA_PATH.iterdir() if d.is_dir()]

# =============================================================
#                   Calibration + Point tracker
# =============================================================

calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()

if PROCESS_POINTS and OBJECT_POINTS_ENABLED:
    with open("../../cfgs/suite/points_cfg.yaml") as fp:
        cfg = yaml.safe_load(fp)
    cfg["dift_path"] = f"{cfg['root_dir']}/{cfg['dift_path']}"
    cfg["cotracker_checkpoint"] = f"{cfg['root_dir']}/{cfg['cotracker_checkpoint']}"
    cfg["task_name"] = TASK_NAMES[0]
    cfg["pixel_keys"] = [camera2pixelkey[f"cam_{cid}"] for cid in camera_indices]
    cfg["object_labels"] = object_labels
    points_class = PointsClass(**cfg)
else:
    points_class = None

# =============================================================
#                      Main processing loop
# =============================================================

for TASK_NAME in TASK_NAMES:
    ds_path = PROCESSED_DATA_PATH / TASK_NAME
    pkl_path = SAVE_DATA_PATH / f"{TASK_NAME}.pkl"

    if pkl_path.exists():
        prior = pkl.load(open(pkl_path, "rb"))
        observations = prior["observations"]
        max_cartesian, min_cartesian = prior["max_cartesian"], prior["min_cartesian"]
        max_gripper, min_gripper = prior["max_gripper"], prior["min_gripper"]
    else:
        observations = []
        max_cartesian = min_cartesian = None
        max_gripper = min_gripper = None

    traj_dirs = sorted([d for d in ds_path.iterdir() if d.is_dir()])

    for traj_idx, traj in enumerate(traj_dirs):
        if NUM_DEMOS is not None and int(traj.name.split("_")[-1]) >= NUM_DEMOS:
            continue
        print(f"Processing {TASK_NAME} → {traj.name} ({traj_idx+1}/{len(traj_dirs)})")

        obs: dict[str, np.ndarray | torch.Tensor] = {}

        # ------------------- RGB frames --------------------
        vid_dir = traj / "videos"
        if not vid_dir.exists():
            print("  ↳ Missing videos, skipping")
            continue
        for cid in camera_indices:
            cap = cv2.VideoCapture(str(vid_dir / f"camera{cid}.mp4"))
            frames = []
            while cap.isOpened():
                ret, fr = cap.read()
                if not ret:
                    break
                h, w, _ = fr.shape
                fr = fr[int(h*crop_h[0]):int(h*crop_h[1]), int(w*crop_w[0]):int(w*crop_w[1])]
                fr = cv2.resize(fr, save_img_size)
                frames.append(fr)
            obs[f"pixels{cid}"] = np.asarray(frames)

        # ------------------- Depth (optional) --------------
        if not USE_DEPTH_ANYTHING and USE_GT_DEPTH:
            depth_dir = traj / "depth"
            if not depth_dir.exists():
                print("  ↳ Missing depth, skipping")
                continue
            depth_frames = {cid: pkl.load(open(depth_dir / f"depth{cid}.pkl", "rb")) for cid in camera_indices}

        # ------------------- Robot states ------------------
        state = read_csv(traj / "states.csv")
        cart = np.array([[extract_number(x) for x in row.strip("[]").split(",")] for row in state["pose_aa"].values], dtype=np.float32)
        grip = state["gripper_state"].values.astype(np.float32)
        obs["cartesian_states"], obs["gripper_states"] = cart, grip
        cmd_cart = np.array([[extract_number(x) for x in row.strip("[]").split(",")] for row in state["cmd_pose_aa"].values], dtype=np.float32)
        cmd_grip = state["cmd_gripper_state"].values.astype(np.float32)
        obs["cmd_cartesian_states"], obs["cmd_gripper_states"] = cmd_cart, cmd_grip

        # ------------------- Robot keypoints ---------------
        for cid in camera_indices:
            px_key = camera2pixelkey[f"cam_{cid}"]
            obs[f"robot_tracks_{px_key}"] = []
            obs[f"gt_robot_tracks_3d_{px_key}"] = []
            # Prepare **object placeholders** immediately
            # obs[f"object_tracks_{px_key}"] = np.empty((0, 0, 2), np.float32)
            # obs[f"object_tracks_3d_{px_key}"] = np.empty((0, 0, 3), np.float32)

        for t, (pos, rot_vec) in enumerate(zip(cart[:, :3], cart[:, 3:])):
            T_g_b = np.eye(4)
            T_g_b[:3, :3] = R.from_rotvec(rot_vec).as_matrix()
            T_g_b[:3, 3] = pos
            T_g_b = T_g_b @ Tshift
            pts3d = [T_g_b[:3, 3]]
            for idx, Tp in enumerate(extrapoints):
                Tp_l = Tp.copy()
                if grip[t] == 1 and idx in (0, 1):
                    Tp_l[1, 3] = 0.015 if idx == 0 else -0.015
                pts3d.append((T_g_b @ Tp_l)[:3, 3])
            pts3d = np.stack(pts3d)

            for cid in camera_indices:
                cam_name = f"cam_{cid}"
                px_key = camera2pixelkey[cam_name]
                ext = calibration_data[cam_name]["ext"]
                K = calibration_data[cam_name]["int"]
                D = calibration_data[cam_name]["dist_coeff"]
                rvec, _ = cv2.Rodrigues(ext[:3, :3])
                tvec = ext[:3, 3]
                pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, D)
                pts2d = pts2d[:, 0]
                w0, h0 = original_img_size
                pts2d[:, 0] -= int(w0*crop_w[0]); pts2d[:, 1] -= int(h0*crop_h[0])
                wc = int(w0*(crop_w[1]-crop_w[0])); hc = int(h0*(crop_h[1]-crop_h[0]))
                pts2d[:, 0] = pts2d[:, 0]*save_img_size[0]/wc
                pts2d[:, 1] = pts2d[:, 1]*save_img_size[1]/hc
                obs[f"robot_tracks_{px_key}"].append(pts2d)
                obs[f"gt_robot_tracks_3d_{px_key}"].append(pts3d)

        # ------------------- Object / hand tracking --------
        if PROCESS_POINTS and OBJECT_POINTS_ENABLED:
            # (kept identical to previous revision – omitted here for brevity)
            ...

        # ------------------- Normalisation ranges ----------
        if max_cartesian is None:
            max_cartesian, min_cartesian = cart.max(0), cart.min(0)
        else:
            max_cartesian = np.maximum(max_cartesian, cart.max(0))
            min_cartesian = np.minimum(min_cartesian, cart.min(0))
        if max_gripper is None:
            max_gripper, min_gripper = grip.max(), grip.min()
        else:
            max_gripper = max(max_gripper, grip.max())
            min_gripper = min(min_gripper, grip.min())

        # ------------------- Triangulation (no depth) ------
        if not USE_GT_DEPTH:
            for cid in camera_indices:
                px_key = camera2pixelkey[f"cam_{cid}"]
                obs[f"robot_tracks_3d_{px_key}"] = []
                if OBJECT_POINTS_ENABLED:
                    obs[f"object_tracks_3d_{px_key}"] = []

            ref_px = camera2pixelkey[f"cam_{camera_indices[0]}"]
            n_frames = len(obs[f"robot_tracks_{ref_px}"])
            for f in range(n_frames):
                Pmats, rob2d, obj2d = [], [], []
                for cid in camera_indices:
                    cam_name = f"cam_{cid}"
                    px_key = camera2pixelkey[cam_name]
                    ext = calibration_data[cam_name]["ext"]
                    intr = calibration_data[cam_name]["int"]
                    Pmats.append(np.concatenate([intr, np.zeros((3,1))],1) @ ext)
                    # ---------------- un‑crop ----------------
                    w0,h0 = original_img_size
                    wc=int(w0*(crop_w[1]-crop_w[0])); hc=int(h0*(crop_h[1]-crop_h[0]))
                    r2d = obs[f"robot_tracks_{px_key}"][f]
                    x = r2d[:,0]*wc/save_img_size[0] + w0*crop_w[0]
                    y = r2d[:,1]*hc/save_img_size[1] + h0*crop_h[0]
                    rob2d.append(np.column_stack((x.astype(int),y.astype(int))))
                    if OBJECT_POINTS_ENABLED:
                        o2d = obs[f"object_tracks_{px_key}"][f]
                        x = o2d[:,0]*wc/save_img_size[0] + w0*crop_w[0]
                        y = o2d[:,1]*hc/save_img_size[1] + h0*crop_h[0]
                        obj2d.append(np.column_stack((x.astype(int),y.astype(int))))
                rob3d = triangulate_points(Pmats, rob2d)[:,:3]
                for cid in camera_indices:
                    px_key = camera2pixelkey[f"cam_{cid}"]
                    obs[f"robot_tracks_3d_{px_key}"].append(rob3d)
                if OBJECT_POINTS_ENABLED:
                    obj3d = triangulate_points(Pmats, obj2d)[:,:3]
                    for cid in camera_indices:
                        px_key = camera2pixelkey[f"cam_{cid}"]
                        obs[f"object_tracks_3d_{px_key}"].append(obj3d)
            for cid in camera_indices:
                px_key = camera2pixelkey[f"cam_{cid}"]
                obs[f"robot_tracks_3d_{px_key}"] = np.asarray(obs[f"robot_tracks_3d_{px_key}"])
                if OBJECT_POINTS_ENABLED:
                    obs[f"object_tracks_3d_{px_key}"] = np.asarray(obs[f"object_tracks_3d_{px_key}"])

        # ------------------- Final conversions -------------
        for cid in camera_indices:
            px_key = camera2pixelkey[f"cam_{cid}"]
            obs[f"robot_tracks_{px_key}"] = np.asarray(obs[f"robot_tracks_{px_key}"])
            obs[f"gt_robot_tracks_3d_{px_key}"] = np.asarray(obs[f"gt_robot_tracks_3d_{px_key}"])
            # # guarantee placeholders exist & are numpy
            # # import ipdb; ipdb.set_trace()
            # if f"object_tracks_{px_key}" not in obs:
            #     obs[f"object_tracks_{px_key}"] = np.empty((0,0,2), np.float32)
            # if f"object_tracks_3d_{px_key}" not in obs:
            #     obs[f"object_tracks_3d_{px_key}"] = np.empty((0,0,3), np.float32)

        observations.append(obs)

    # ------------------- Persist task pkl ------------------
    pkl.dump({
        "observations": observations,
        "max_cartesian": max_cartesian,
        "min_cartesian": min_cartesian,
        "max_gripper": max_gripper,
        "min_gripper": min_gripper,
    }, open(pkl_path, "wb"))

print("All tasks processed ✔")
