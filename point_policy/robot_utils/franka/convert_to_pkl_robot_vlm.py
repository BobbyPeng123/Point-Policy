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
    # triangulate_points,  <-- no longer used
)

# Create the parser
parser = argparse.ArgumentParser(
    description="Convert processed robot data into a pkl file"
)

# Add the arguments
parser.add_argument("--data_dir", type=str, help="Path to the data directory")
parser.add_argument("--calib_path", type=str, help="Path to the calibration file")
parser.add_argument("--task_names", nargs="+", type=str, help="List of task names")
parser.add_argument(
    "--num_demos", type=int, default=None, help="Number of demonstrations to process"
)
parser.add_argument(
    "--process_points", type=bool, default=False, help="Process human/robot points"
)
parser.add_argument(
    "--use_gt_depth", type=bool, default=True, help="Use ground truth depth"
)

args = parser.parse_args()
DATA_DIR = Path(args.data_dir)
CALIB_PATH = Path(args.calib_path)
task_names = args.task_names
NUM_DEMOS = args.num_demos
process_points = args.process_points
use_gt_depth = args.use_gt_depth

# Use two camera views
camera_indices = [1, 2]
original_img_size = (640, 480)
crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)
save_img_size = (256, 256)
# Define separate semantic labels for objects and robot
object_labels = ["objects"]
robot_labels = ["robot"]

PROCESSED_DATA_PATH = Path(DATA_DIR) / "processed_data"
SAVE_DATA_PATH = Path(DATA_DIR) / "expert_demos" / "franka_env"
print(f"Processed data path: {PROCESSED_DATA_PATH}")
print(f"Save data path: {SAVE_DATA_PATH}")

if save_img_size is None:
    save_img_size = (
        int(original_img_size[0] * (crop_w[1] - crop_w[0])),
        int(original_img_size[1] * (crop_h[1] - crop_h[0])),
    )

if task_names is None:
    task_names = [x.name for x in PROCESSED_DATA_PATH.iterdir() if x.is_dir()]

SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Calibration data
calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()

# Load p3po config for tracking and create two instances:
if process_points:
    with open("../../cfgs/suite/points_cfg.yaml") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        root_dir, dift_path, cotracker_checkpoint = (
            cfg["root_dir"],
            cfg["dift_path"],
            cfg["cotracker_checkpoint"],
        )
        cfg["dift_path"] = f"{root_dir}/{dift_path}"
        cfg["cotracker_checkpoint"] = f"{root_dir}/{cotracker_checkpoint}"
        # Use the task name from the first task (or override as needed)
        cfg["task_name"] = task_names[0]
        cfg["num_points"] = cfg.get("num_points", 5)
        # Modified to use two camera views:
        cfg["pixel_keys"] = [camera2pixelkey[f"cam_{idx}"] for idx in camera_indices]
    
    # Create separate configs:
    object_cfg = cfg.copy()
    object_cfg["object_labels"] = object_labels
    object_cfg["num_points"] = 5

    robot_cfg = cfg.copy()
    robot_cfg["object_labels"] = robot_labels
    robot_cfg["num_points"] = 8

    # Instantiate two separate PointsClass objects:
    points_class_obj = PointsClass(**object_cfg)
    points_class_robot = PointsClass(**robot_cfg)


def extract_number(s):
    s = s.strip().strip("[]")
    match = re.match(r"np\.float(?:32|64)\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)", s)
    if match:
        return float(match.group(1))
    else:
        match = re.match(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"Cannot extract number from '{s}'")


for TASK_NAME in task_names:
    DATASET_PATH = Path(f"{PROCESSED_DATA_PATH}/{TASK_NAME}")

    if (SAVE_DATA_PATH / f"{TASK_NAME}.pkl").exists():
        print(f"Data for {TASK_NAME} already exists. Appending to it...")
        input("Press Enter to continue...")
        data = pkl.load(open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "rb"))
        observations = data["observations"]
        max_cartesian = data["max_cartesian"]
        min_cartesian = data["min_cartesian"]
        max_gripper = data["max_gripper"]
        min_gripper = data["min_gripper"]
    else:
        observations = []
        max_cartesian, min_cartesian = None, None
        max_gripper, min_gripper = None, None

    dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
    for i, data_point in enumerate(sorted(dirs)):
        print(f"Processing data point {i+1}/{len(dirs)}")

        if NUM_DEMOS is not None and int(str(data_point).split("_")[-1]) >= NUM_DEMOS:
            print(f"Skipping data point {data_point}")
            continue

        observation = {}

        # Process images from two camera views
        image_dir = data_point / "videos"
        if not image_dir.exists():
            print(f"Data point {data_point} is incomplete")
            continue

        for save_idx, idx in enumerate(camera_indices):
            video_path = image_dir / f"camera{idx}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Video {video_path} could not be opened")
                continue

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape
                frame = frame[int(h * crop_h[0]): int(h * crop_h[1]),
                              int(w * crop_w[0]): int(w * crop_w[1])]
                frame = cv2.resize(frame, save_img_size)
                frames.append(frame)
            observation[f"pixels{idx}"] = np.array(frames)

        # Process depth (if using gt depth)
        if use_gt_depth:
            depth_dir = data_point / "depth"
            if not depth_dir.exists():
                print(f"Data point {data_point} is incomplete (no depth)")
                continue

            depth_frames = {}
            for idx in camera_indices:
                depth_file = depth_dir / f"depth{idx}.pkl"
                with open(depth_file, "rb") as f:
                    depth = pkl.load(f)  # depth in meters
                depth_frames[idx] = depth

        # Process state data as before
        state_csv_path = data_point / "states.csv"
        state = read_csv(state_csv_path)
        cartesian_states = state["pose_aa"].values
        cartesian_states = np.array(
            [
                np.array([extract_number(x) for x in pose.strip("[]").split(",")])
                for pose in cartesian_states
            ],
            dtype=np.float32,
        )
        gripper_states = state["gripper_state"].values.astype(np.float32)
        observation["cartesian_states"] = cartesian_states.astype(np.float32)
        observation["gripper_states"] = gripper_states.astype(np.float32)
        cmd_cartesian_states = state["cmd_pose_aa"].values
        cmd_cartesian_states = np.array(
            [
                np.array([extract_number(x) for x in pose.strip("[]").split(",")])
                for pose in cmd_cartesian_states
            ],
            dtype=np.float32,
        )
        cmd_gripper_states = state["cmd_gripper_state"].values.astype(np.float32)
        observation["cmd_cartesian_states"] = cmd_cartesian_states.astype(np.float32)
        observation["cmd_gripper_states"] = cmd_gripper_states.astype(np.float32)

        if process_points:
            mark_every = 8
            save = True

            # -------------------------------
            # Process object and robot tracks for each camera view via p3po
            # -------------------------------
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]

                frames_obj = observation[pixel_key]
                # Convert frames from BGR to RGB
                frames_obj = [frame[..., ::-1] for frame in frames_obj]

                # Object tracking with points_class_obj
                points_class_obj.reset_episode()
                points_class_obj.add_to_image_list(frames_obj[0], pixel_key)
                for obj_label in object_labels:
                    points_class_obj.find_semantic_similar_points(pixel_key, obj_label)
                try:
                    points_class_obj.track_points(pixel_key, last_n_frames=mark_every, is_first_step=True)
                except Exception as e:
                    print(f"Error in tracking object points for {pixel_key}: {e}")
                    save = False
                    continue
                points_class_obj.track_points(pixel_key, last_n_frames=mark_every, one_frame=(mark_every == 1))
                points_class_obj.plot_image(pixel_key)
                object_points_list = []
                object_points = points_class_obj.get_points_on_image(pixel_key)
                object_points_list.append(object_points[0])  # torch.Size([5, 2])

                # Get calibration matrices for projection
                P_obj = calibration_data[camera_name]["ext"]  # 4x4
                K_obj = calibration_data[camera_name]["int"]  # 3x3

                # Process depth for the first frame
                if use_gt_depth:
                    depth = depth_frames[cam_idx][0]
                    points_class_obj.set_depth(depth, pixel_key, original_img_size, save_img_size, (crop_h, crop_w))
                else:
                    points_class_obj.get_depth(pixel_key)
                points_with_depth = points_class_obj.get_points(pixel_key)
                depths_obj = points_with_depth[:, :, -1]
                obj_points_3d_list = []
                obj_points3d = pixel2d_to_3d_torch(object_points[0], depths_obj[0], K_obj, P_obj)
                obj_points_3d_list.append(obj_points3d)

                for idx, image in enumerate(frames_obj[1:]):
                    print(f"Object Traj: {i}, Frame: {idx}, Image: {pixel_key}")
                    points_class_obj.add_to_image_list(image, pixel_key)
                    if use_gt_depth:
                        depth = depth_frames[cam_idx][idx]
                        points_class_obj.set_depth(depth, pixel_key, original_img_size, save_img_size, (crop_h, crop_w))
                    else:
                        print("Using depthanything!")
                        points_class_obj.get_depth(pixel_key, last_n_frames=mark_every)
                    if (idx + 1) % mark_every == 0 or idx == (len(frames_obj) - 2):
                        to_add = mark_every - (idx + 1) % mark_every
                        if to_add < mark_every:
                            for j in range(to_add):
                                points_class_obj.add_to_image_list(image, pixel_key)
                        else:
                            to_add = 0
                        points_class_obj.track_points(pixel_key, last_n_frames=mark_every, one_frame=(mark_every == 1))
                        points_class_obj.plot_image(pixel_key)
                        object_points = points_class_obj.get_points_on_image(pixel_key, last_n_frames=mark_every)
                        for j in range(mark_every - to_add):
                            object_points_list.append(object_points[j])
                        points_with_depth = points_class_obj.get_points(pixel_key, last_n_frames=mark_every)
                        for j in range(mark_every - to_add):
                            depth = points_with_depth[j, :, -1]
                            pt3d = pixel2d_to_3d_torch(object_points[j], depth, K_obj, P_obj)
                            obj_points_3d_list.append(pt3d)

                observation[f"object_tracks_{pixel_key}"] = torch.stack(object_points_list).numpy()
                observation[f"object_tracks_3d_{pixel_key}"] = torch.stack(obj_points_3d_list).numpy()

                # -------------------------------
                # Process robot tracks via p3po (using only "robot")
                # -------------------------------
                points_class_robot.reset_episode()
                points_class_robot.add_to_image_list(frames_obj[0], pixel_key)
                for r_label in robot_labels:
                    points_class_robot.find_semantic_similar_points(pixel_key, r_label)
                try:
                    points_class_robot.track_points(pixel_key, last_n_frames=mark_every, is_first_step=True)
                except Exception as e:
                    print(f"Error in tracking robot points for {pixel_key}: {e}")
                    save = False
                    continue
                points_class_robot.track_points(pixel_key, last_n_frames=mark_every, one_frame=(mark_every == 1))
                points_class_robot.plot_image(pixel_key)
                robot_points_list = []
                robot_points = points_class_robot.get_points_on_image(pixel_key)
                robot_points_list.append(robot_points[0])

                # Get calibration matrices for robot projection (same as object)
                P_robot = calibration_data[camera_name]["ext"]
                K_robot = calibration_data[camera_name]["int"]

                if use_gt_depth:
                    depth = depth_frames[cam_idx][0]
                    points_class_robot.set_depth(depth, pixel_key, original_img_size, save_img_size, (crop_h, crop_w))
                else:
                    points_class_robot.get_depth(pixel_key)
                points_with_depth = points_class_robot.get_points(pixel_key)
                depths_robot = points_with_depth[:, :, -1]
                robot_points_3d_list = []
                r_points3d = pixel2d_to_3d_torch(robot_points[0], depths_robot[0], K_robot, P_robot)
                robot_points_3d_list.append(r_points3d)

                for idx, image in enumerate(frames_obj[1:]):
                    print(f"Robot Traj: {i}, Frame: {idx}, Image: {pixel_key}")
                    points_class_robot.add_to_image_list(image, pixel_key)
                    if use_gt_depth:
                        depth = depth_frames[cam_idx][idx]
                        points_class_robot.set_depth(depth, pixel_key, original_img_size, save_img_size, (crop_h, crop_w))
                    else:
                        points_class_robot.get_depth(pixel_key, last_n_frames=mark_every)
                    if (idx + 1) % mark_every == 0 or idx == (len(frames_obj) - 2):
                        to_add = mark_every - (idx + 1) % mark_every
                        if to_add < mark_every:
                            for j in range(to_add):
                                points_class_robot.add_to_image_list(image, pixel_key)
                        else:
                            to_add = 0
                        points_class_robot.track_points(pixel_key, last_n_frames=mark_every, one_frame=(mark_every == 1))
                        points_class_robot.plot_image(pixel_key)
                        robot_points = points_class_robot.get_points_on_image(pixel_key, last_n_frames=mark_every)
                        for j in range(mark_every - to_add):
                            robot_points_list.append(robot_points[j])
                        points_with_depth = points_class_robot.get_points(pixel_key, last_n_frames=mark_every)
                        for j in range(mark_every - to_add):
                            depth = points_with_depth[j, :, -1]
                            r_pt3d = pixel2d_to_3d_torch(robot_points[j], depth, K_robot, P_robot)
                            robot_points_3d_list.append(r_pt3d)

                observation[f"robot_tracks_{pixel_key}"] = torch.stack(robot_points_list).numpy()
                observation[f"robot_tracks_3d_{pixel_key}"] = torch.stack(robot_points_3d_list).numpy()

            # End process_points block

        observations.append(observation)

        # (Optional normalization update for cartesian and gripper states)
        if max_cartesian is None:
            max_cartesian = np.max(cartesian_states, axis=0)
            min_cartesian = np.min(cartesian_states, axis=0)
        else:
            max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
            min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))
        if max_gripper is None:
            max_gripper = np.max(gripper_states)
            min_gripper = np.min(gripper_states)
        else:
            max_gripper = np.maximum(max_gripper, np.max(gripper_states))
            min_gripper = np.minimum(min_gripper, np.min(gripper_states))

    # Save data to a pickle file
    data = {
        "observations": observations,
        "max_cartesian": max_cartesian,
        "min_cartesian": min_cartesian,
        "max_gripper": max_gripper,
        "min_gripper": min_gripper,
    }
    with open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "wb") as f:
        pkl.dump(data, f)

print("Processing complete.")
