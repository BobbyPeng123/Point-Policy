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
from pathlib import Path

def serialize_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

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
    "--process_points", type=bool, default=False, help="Process human hand points"
)
parser.add_argument(
    "--use_gt_depth", type=bool, default=False, help="Use ground truth depth"
)

args = parser.parse_args()
DATA_DIR = Path(args.data_dir)
CALIB_PATH = Path(args.calib_path)
task_names = args.task_names
NUM_DEMOS = args.num_demos
process_points = args.process_points
use_gt_depth = args.use_gt_depth
use_depth_anything = False

# import ipdb; ipdb.set_trace()

camera_indices = [1, 2]
# camera_indices = [2]
original_img_size = (640, 480)
crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)
save_img_size = (256, 256)
object_labels = [
    "empty_space",
    # "bottle",
]

PROCESSED_DATA_PATH = Path(DATA_DIR) / "processed_data"
SAVE_DATA_PATH = Path(DATA_DIR) / "expert_demos" / "franka_env"

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

# Load p3po config for tracking
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
        cfg["task_name"] = task_names[0]
        cfg["pixel_keys"] = [
            camera2pixelkey[f"cam_{cam_idx}"] for cam_idx in camera_indices
        ]
        cfg["object_labels"] = object_labels

        # import ipdb; ipdb.set_trace()

    points_class = PointsClass(**cfg)



def extract_number(s):
    s = s.strip()
    # Remove any leading/trailing brackets or whitespace
    s = s.strip("[]")
    # Match 'np.float32(number)' or 'np.float64(number)'
    match = re.match(r"np\.float(?:32|64)\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)", s)
    if match:
        return float(match.group(1))
    else:
        # Match plain numbers, including negatives and decimals
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

        # Process images
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

                # crop the image
                h, w, _ = frame.shape
                frame = frame[
                    int(h * crop_h[0]) : int(h * crop_h[1]),
                    int(w * crop_w[0]) : int(w * crop_w[1]),
                ]

                frame = cv2.resize(frame, save_img_size)
                frames.append(frame)

            observation[f"pixels{idx}"] = np.array(frames)

        # Process depth
        if not use_depth_anything and use_gt_depth:
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


        state_csv_path = data_point / "states.csv"
        state = read_csv(state_csv_path)

        # Parsing cartesian pose data
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
            # Robot Tracks
            for cam_idx in camera_indices:
                if cam_idx == 51:
                    continue
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"robot_tracks_{pixel_key}"] = []
                # if save_tracks_3d:
                observation[f"gt_robot_tracks_3d_{pixel_key}"] = []

            for timestep, state in enumerate(cartesian_states):
                pos = state[:3]
                ori = state[3:]
                T_g_b = np.eye(4)
                T_g_b[:3, :3] = R.from_rotvec(ori).as_matrix()
                T_g_b[:3, 3] = pos

                # shift the point
                T_g_b = T_g_b @ Tshift

                # add extra points
                points3d = [T_g_b[:3, 3]]
                for idx, Tp in enumerate(extrapoints):
                    if gripper_states[timestep] == 1 and idx in [0, 1]:
                        Tp = Tp.copy()
                        Tp[1, 3] = 0.015 if idx == 0 else -0.015
                    pt = T_g_b @ Tp
                    pt = pt[:3, 3]
                    points3d.append(pt)
                points3d = np.array(points3d)

                # for pixel_key, camera_name in pixelkey2camera.items():
                for cam_idx in camera_indices:
                    if cam_idx == 51:
                        continue
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]

                    P = calibration_data[camera_name]["ext"]  # 4x4
                    K = calibration_data[camera_name]["int"]  # 3x3
                    D = calibration_data[camera_name]["dist_coeff"]  # 5

                    r, t = P[:3, :3], P[:3, 3]
                    r, _ = cv2.Rodrigues(r)
                    points2d, _ = cv2.projectPoints(points3d, r, t, K, D)
                    points2d = points2d[:, 0]

                    # consider cropping params
                    w, h = original_img_size
                    points2d[:, 0] = points2d[:, 0] - int(w * crop_w[0])
                    points2d[:, 1] = points2d[:, 1] - int(h * crop_h[0])
                    img_w, img_h = int(w * (crop_w[1] - crop_w[0])), int(
                        h * (crop_h[1] - crop_h[0])
                    )
                    points2d[:, 0] = points2d[:, 0] * save_img_size[0] / img_w
                    points2d[:, 1] = points2d[:, 1] * save_img_size[1] / img_h

                    observation[f"robot_tracks_{pixel_key}"].append(points2d)

                    # store 3D points in robot base frame
                    # check robot points ###
                    # import ipdb; ipdb.set_trace()
                    observation[f"gt_robot_tracks_3d_{pixel_key}"].append(points3d)

            # Human hand tracks
            # import ipdb; ipdb.set_trace()
            mark_every = 8
            save = True
            for cam_idx in camera_indices:
                # if cam_idx == 1: continue
                if save == False:
                    break
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]

                if pixel_key == "pixels1":
                    context = zmq.Context()
                    socket = context.socket(zmq.REQ)
                    socket.connect("tcp://localhost:5558")

                    frames = observation[pixel_key]
                    # CV2 reads in BGR format, so we need to convert to RGB
                    frames = [frame[..., ::-1] for frame in frames]

                    # send image to server
                    frame = frames[0]
                    # image = Image.fromarray(frame[..., ::-1])  # CV2 reads in BGR format
                    image = Image.fromarray(frame)  # CV2 reads in BGR format
                    serialized_image = serialize_image(image)
                    serialized_image_cam1 = serialized_image

                    # get bbox of object
                    for object_label in object_labels:
                        request = {
                            "image": serialized_image,
                            # "image": "",
                            "image_path": "",
                            # "image_path": "/home/bobby/RoboPoint/images/cam_2_rgb_video_1.png",
                            "prompt": f"Please pinpoint 3 points in the empty space on the black blanket. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image.",
                        }
                        print("send request")
                        socket.send_json(request)
                        response = socket.recv_json()

                        points = response["points"]
                        # only save the first two points
                        # points = points[:2]
                        # only save the first one
                        points = points[:1]

                        # # hack
                        # points = [[0.475, 0.892], [0.492, 0.89]]

                        print(points) # normalized [[0.406, 0.819], [0.422, 0.827]]
                        # print(response["raw_text"])

                        # annotated_image_b64 = response["annotated_image"]
                        # # Save the annotated image. Decode the base64 string and open it with PIL.
                        # img_data = base64.b64decode(annotated_image_b64)
                        # annotated_img = Image.open(io.BytesIO(img_data))
                        # save_path = "annotated_image.png"
                        # annotated_img.save(save_path)
                        # print(f"Annotated image saved to {save_path}")

                        # Convert the response points (assumed normalized) to pixel coordinates.
                        height, width = save_img_size[0], save_img_size[1]
                        pixel_points = []
                        cam1_points = []
                        for pt in points:
                            # pt is expected to be [x, y] with values between 0 and 1
                            x_norm, y_norm = pt
                            x_pixel = int(x_norm * width)
                            y_pixel = int(y_norm * height)
                            # For example, you might ignore the first coordinate or set it to zero.
                            # Here we store [dummy, y, x] if that is what you had before.
                            # pixel_points.append([0.0, float(y_pixel), float(x_pixel)])
                            pixel_points.append([0.0, float(x_pixel), float(y_pixel)])
                            cam1_points.append([float(x_norm), float(y_norm)])
                        print(cam1_points) # format: [[0.0, 103.0, 209.0], [0.0, 108.0, 211.0]]
                        points_tensor = torch.tensor(pixel_points, dtype=torch.float32)
                        # # Extract only the x and y coordinates (discard the dummy or extra column)
                        # points_tensor = points_tensor[:, 1:]  # now shape is (N, 2)
                        # Add a batch dimension so that later torch.cat works as expected.
                        # points_tensor = points_tensor.unsqueeze(0)  # now shape is (1, N, 2)

                        # IMPORTANT: Instead of replacing the whole dictionary, update the existing dict!
                        points_class.semantic_similar_points[f"{pixel_key}_{object_label}"] = points_tensor

                        # height, width = frame.shape[:2]
                        # pixel_points = []
                        # for pt in points:
                        #     # Assuming pt is [x, y] with values between 0 and 1
                        #     x_norm, y_norm = pt
                        #     x_pixel = int(x_norm * width)
                        #     y_pixel = int(y_norm * height)
                        #     pixel_points.append([0.0, float(y_pixel), float(x_pixel)])
                        # points_tensor = torch.tensor(pixel_points, dtype=torch.float32)
                        points_class.add_to_image_list(frames[0], pixel_key)
                        # points_class.semantic_similar_points = {f"{pixel_key}_{object_label}": points_tensor}
                        print(f"points_class.semantic_similar_points: {points_class.semantic_similar_points}")
                        # points_class.tracks[pixel_key] = points_class.semantic_similar_points
                elif pixel_key == "pixels2":
                    context = zmq.Context()
                    socket_mast3r = context.socket(zmq.REQ)
                    socket_mast3r.connect("tcp://localhost:5559")

                    frames = observation[pixel_key]
                    # CV2 reads in BGR format, so we need to convert to RGB
                    frames = [frame[..., ::-1] for frame in frames]

                    # send image to server
                    frame = frames[0]
                    # image = Image.fromarray(frame[..., ::-1])  # CV2 reads in BGR format
                    image = Image.fromarray(frame)  # CV2 reads in BGR format
                    serialized_image = serialize_image(image)
                    print(cam1_points)
                    for object_label in object_labels:
                        # Use the cam1 points (normalized) from the earlier response.
                        request = {
                            "ref_image": serialized_image_cam1,
                            # "ref_image_path": "dust3r/croco/assets/Chateau1.png",
                            "target_image": serialized_image,
                            # "target_image_path": "dust3r/croco/assets/Chateau2.png",
                            "points": cam1_points,
                            # "points": [
                            #     [100, 100],
                            #     [200, 150],
                            #     [300, 250],
                            #     [400, 300],
                            #     [250, 400]
                            # ],
                        }
                        print("Sending request to MASt3R server for cam2")
                        socket_mast3r.send_json(request)
                        response = socket_mast3r.recv_json()
                        points = response["target_points"]
                        print(f"Received corresponding points from MASt3R (cam2) for {object_label}: {points}")

                        # Convert normalized points to pixel coordinates for cam2.
                        height, width = save_img_size[1], save_img_size[0]
                        pixel_points = []
                        for pt in points:
                            x_norm, y_norm = pt
                            x_pixel = int(x_norm * width)
                            y_pixel = int(y_norm * height)
                            pixel_points.append([0.0, float(x_pixel), float(y_pixel)])
                        # cam2_points[object_label] = pixel_points

                        points_tensor = torch.tensor(pixel_points, dtype=torch.float32)
                        points_class.semantic_similar_points[f"{pixel_key}_{object_label}"] = points_tensor
                        points_class.add_to_image_list(frames[0], pixel_key)
                try:
                    points_class.track_points(
                        pixel_key, last_n_frames=mark_every, is_first_step=True
                    )
                # exit()
                except:
                    print(f"Error in tracking hand points for {pixel_key}")
                    save = False
                    continue

                # import ipdb; ipdb.set_trace() # len(points_class.semantic_similar_points[f"{pixel_key}_{object_label}"])

                points_class.track_points(
                    pixel_key, last_n_frames=mark_every, one_frame=(mark_every == 1)
                )

                points_list = []
                points = points_class.get_points_on_image(pixel_key)
                points_list.append(points[0])

                if use_gt_depth:
                    points_3d_list = []
                    # if use_gt_depth:
                    if not use_depth_anything:
                        depth = depth_frames[cam_idx][0]
                        points_class.set_depth(
                            depth,
                            pixel_key,
                            original_img_size,
                            save_img_size,
                            (crop_h, crop_w),
                        )
                    else:
                        points_class.get_depth(pixel_key, save_img_size, save_img_size, (crop_h, crop_w), last_n_frames=mark_every)
                        # import ipdb; ipdb.set_trace()
                    points_with_depth = points_class.get_points(pixel_key)
                    depths = points_with_depth[:, :, -1]

                    P = calibration_data[camera_name]["ext"]  # 4x4
                    K = calibration_data[camera_name]["int"]  # 3x3
                    points3d = pixel2d_to_3d_torch(points[0], depths[0], K, P)
                    points_3d_list.append(points3d)

                for idx, image in enumerate(frames[1:]):
                    print(f"Traj: {i}, Frame: {idx}, Image: {pixel_key}")
                    points_class.add_to_image_list(image, pixel_key)
                    print("added")

                    # if use_gt_depth:
                    if not use_depth_anything and use_gt_depth:
                        depth = depth_frames[cam_idx][idx]
                        points_class.set_depth(
                            depth,
                            pixel_key,
                            original_img_size,
                            save_img_size,
                            (crop_h, crop_w),
                        )
                        print("set depth")
                    elif use_depth_anything:
                        # get depth from the previous frame
                        points_class.get_depth(
                            pixel_key,
                            save_img_size,
                            save_img_size,
                            (crop_h, crop_w),
                            last_n_frames=mark_every
                        )
                        print("got depth")

                    if (idx + 1) % mark_every == 0 or idx == (len(frames) - 2):
                        to_add = mark_every - (idx + 1) % mark_every
                        if to_add < mark_every:
                            for j in range(to_add):
                                points_class.add_to_image_list(image, pixel_key)
                        else:
                            to_add = 0

                        points_class.track_points(
                            pixel_key,
                            last_n_frames=mark_every,
                            one_frame=(mark_every == 1),
                        )

                        print("tracked points")

                        points = points_class.get_points_on_image(
                            pixel_key, last_n_frames=mark_every
                        )

                        # # plot image
                        # import ipdb; ipdb.set_trace()
                        points_class.plot_image(
                        pixel_key,
                        )


                        for j in range(mark_every - to_add):
                            points_list.append(points[j])

                        if use_gt_depth:
                            # if not use_gt_depth:
                            #     points_class.get_depth(last_n_frames=mark_every)
                            #     # import ipdb; ipdb.set_trace()

                            points_with_depth = points_class.get_points(
                                pixel_key, last_n_frames=mark_every
                            )
                            for j in range(mark_every - to_add):
                                depth = points_with_depth[j, :, -1]
                                # points3d = pixel2d_to_3d_camera_frame_torch(
                                #     points[j], depth, K
                                # )
                                points3d = pixel2d_to_3d_torch(points[j], depth, K, P)
                                points_3d_list.append(points3d)

                observation[f"object_tracks_{pixel_key}"] = torch.stack(
                    points_list
                ).numpy()
                if use_gt_depth:
                    observation[f"object_tracks_3d_{pixel_key}"] = torch.stack(
                        points_3d_list
                    ).numpy()
                    # check object points ###
                # import ipdb; ipdb.set_trace()
                points_class.reset_episode()

            if save == False:
                continue

            for cam_idx in camera_indices:
                if cam_idx == 51:
                    continue
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"robot_tracks_{pixel_key}"] = np.array(
                    observation[f"robot_tracks_{pixel_key}"]
                )
                # if save_tracks_3d:
                observation[f"gt_robot_tracks_3d_{pixel_key}"] = np.array(
                    observation[f"gt_robot_tracks_3d_{pixel_key}"]
                )

        # Update max and min cartesian values for normalization
        if max_cartesian is None:
            max_cartesian = np.max(cartesian_states, axis=0)
            min_cartesian = np.min(cartesian_states, axis=0)
        else:
            max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
            min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))

        # Update max and min gripper values for normalization
        if max_gripper is None:
            max_gripper = np.max(gripper_states)
            min_gripper = np.min(gripper_states)
        else:
            max_gripper = np.maximum(max_gripper, np.max(gripper_states))
            min_gripper = np.minimum(min_gripper, np.min(gripper_states))

        if not use_gt_depth:
            """
            Triangulate 3D points from 2D points when gt_depth is not available
            """
            # import ipdb; ipdb.set_trace()
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"object_tracks_3d_{pixel_key}"] = []
                observation[f"robot_tracks_3d_{pixel_key}"] = []
            # import ipdb; ipdb.set_trace()
            for t_idx in range(
                len(observation[f"object_tracks_{pixel_key}"])
            ):  # for each frame
                P, pts_obj, pts_robot = [], [], []
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]

                    extr = calibration_data[camera_name]["ext"]
                    intr = calibration_data[camera_name]["int"]
                    intr = np.concatenate([intr, np.zeros((3, 1))], axis=1)
                    P.append(intr @ extr)

                    # For object points
                    pt2d_obj = observation[f"object_tracks_{pixel_key}"][t_idx]
                    # compute point_h in original image
                    point_h = pt2d_obj[:, 1]
                    h_orig, h_curr = original_img_size[1], save_img_size[1]
                    h_orig_cropped = h_orig * (crop_h[1] - crop_h[0])
                    point_h_orig = (
                        point_h / h_curr
                    ) * h_orig_cropped + h_orig * crop_h[0]
                    point_h_orig = point_h_orig.astype(np.int32)
                    # compute point_w in original image
                    point_w = pt2d_obj[:, 0]
                    w_orig, w_curr = original_img_size[0], save_img_size[0]
                    w_orig_cropped = w_orig * (crop_w[1] - crop_w[0])
                    point_w_orig = (
                        point_w / w_curr
                    ) * w_orig_cropped + w_orig * crop_w[0]
                    point_w_orig = point_w_orig.astype(np.int32)
                    # Store points
                    pt2d_obj = np.column_stack((point_w_orig, point_h_orig))
                    pts_obj.append(pt2d_obj)

                    # For robot points
                    pt2d_robot = observation[f"robot_tracks_{pixel_key}"][t_idx]
                    # compute point_h in original image
                    point_h = pt2d_robot[:, 1]
                    h_orig, h_curr = original_img_size[1], save_img_size[1]
                    h_orig_cropped = h_orig * (crop_h[1] - crop_h[0])
                    point_h_orig = (
                        point_h / h_curr
                    ) * h_orig_cropped + h_orig * crop_h[0]
                    point_h_orig = point_h_orig.astype(np.int32)
                    # compute point_w in original image
                    point_w = pt2d_robot[:, 0]
                    w_orig, w_curr = original_img_size[0], save_img_size[0]
                    w_orig_cropped = w_orig * (crop_w[1] - crop_w[0])
                    point_w_orig = (
                        point_w / w_curr
                    ) * w_orig_cropped + w_orig * crop_w[0]
                    point_w_orig = point_w_orig.astype(np.int32)
                    # Store points
                    pt2d_robot = np.column_stack((point_w_orig, point_h_orig))
                    pts_robot.append(pt2d_robot)

                pts3d_obj = triangulate_points(P, pts_obj)
                pts3d_robot = triangulate_points(P, pts_robot)
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]
                    observation[f"object_tracks_3d_{pixel_key}"].append(
                        pts3d_obj[:, :3]
                    )
                    observation[f"robot_tracks_3d_{pixel_key}"].append(
                        pts3d_robot[:, :3]
                    )
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"object_tracks_3d_{pixel_key}"] = np.array(
                    observation[f"object_tracks_3d_{pixel_key}"]
                )
                observation[f"robot_tracks_3d_{pixel_key}"] = np.array(
                    observation[f"robot_tracks_3d_{pixel_key}"]
                )

        observations.append(observation)
        # import ipdb; ipdb.set_trace()

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
