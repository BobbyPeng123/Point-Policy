# import pickle as pkl
# import numpy as np
# import imageio
# from pathlib import Path
# import cv2

# DATA_DIR = Path("/home/bobby/data/expert_demos/franka_env")
# TASK_NAME = "place_bottle_from_fridge_left_robot"
# plot_pts = True

# DATA_PATH = DATA_DIR / f"{TASK_NAME}.pkl"
# SAVE_DIR = Path(f"./videos/{TASK_NAME}")
# pixel_keys = ["pixels4", "pixels6"]
# original_image_size = (640, 480)
# k = 1  # number of track points to plot per frame
# traj_indices = None

# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# # Read data
# with open(DATA_PATH, "rb") as f:
#     data = pkl.load(f)

# if traj_indices is None:
#     traj_indices = [i for i in range(len(data["observations"]))]

# for traj_idx in traj_indices:
#     print(f"Processing traj_idx: {traj_idx}")
#     for pixel_key in pixel_keys:
#         point_track_key = (
#             f"robot_tracks_{pixel_key}"
#             if "human" not in TASK_NAME
#             else f"human_tracks_{pixel_key}"
#         )
#         object_track_key = f"object_tracks_{pixel_key}"

#         # Extract images and point tracks
#         frames = data["observations"][traj_idx][pixel_key]
#         frames = np.array(frames)

#         if plot_pts and pixel_key != "pixels51":
#             point_tracks = data["observations"][traj_idx][point_track_key]
#             point_tracks = np.array(point_tracks)
#             object_tracks = data["observations"][traj_idx][object_track_key]
#             object_tracks = np.array(object_tracks)
#             point_tracks = np.concatenate([point_tracks, object_tracks], axis=1)

#             # Color for each point
#             num_points = point_tracks.shape[1]
#             colors = np.zeros((num_points, 3))
#             third = num_points // 3
#             colors[:third, 0] = 255
#             colors[third : 2 * third, 1] = 255
#             colors[2 * third :, 2] = 255

#         save_frames = []
#         for i, frame in enumerate(frames):
#             frame = frame[..., [2, 1, 0]].copy()
#             if plot_pts and pixel_key != "pixels51":
#                 for j, points in enumerate(point_tracks[max(0, i - k) : i + 1]):
#                     # points = points[3:4]
#                     for l, point in enumerate(points):
#                         point = point.astype(int)
#                         point[0] = int(
#                             point[0] * frame.shape[1] / original_image_size[0]
#                         )
#                         point[1] = int(
#                             point[1] * frame.shape[0] / original_image_size[1]
#                         )
#                         frame = cv2.circle(
#                             frame, tuple(point), 2, colors[l].tolist(), -1
#                         )
#             save_frames.append(frame)

#         # Save the video
#         save_frames = np.array(save_frames).astype(np.uint8)
#         save_path = SAVE_DIR / f"{TASK_NAME}_traj{traj_idx}_{pixel_key}.mp4"
#         imageio.mimwrite(save_path, save_frames, fps=20)

import pickle as pkl
import numpy as np
import imageio
from pathlib import Path
import cv2

"""Robust video‑saving utility that handles empty `object_tracks_*` placeholders
and prints a friendly path without crashing when the video directory lies
outside the current working directory.
"""

# ---------------- User parameters ----------------
DATA_DIR   = Path("/home/bobby/data/expert_demos/franka_env")
TASK_NAME  = "open_fridge_door_right_robot"
PLOT_PTS   = True
PIXEL_KEYS = ["pixels2", "pixels5"]
ORIG_SIZE  = (640, 480)  # original frame resolution used during demo collection
HISTORY_K  = 1           # number of previous frames whose points we overlay
TRAJ_IDX   = None        # list[int] | None → process all trajectories if None
FPS        = 20

# ---------------- Internal paths -----------------
DATA_PATH = DATA_DIR / f"{TASK_NAME}.pkl"
SAVE_DIR  = Path("videos") / TASK_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Helper functions --------------

def ensure_frame_len(arr: np.ndarray, n: int, n_dim: int = 2) -> np.ndarray:
    """Return *arr* if its first dimension equals *n*; else an empty placeholder
    of shape (n, 0, n_dim) with the same dtype."""
    if arr.size == 0 or arr.shape[0] != n:
        return np.empty((n, 0, n_dim), dtype=arr.dtype)
    return arr


def nice_path(p: Path) -> str:
    """Return `p` relative to CWD if possible, otherwise its absolute form."""
    try:
        return str(p.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(p)

# ---------------- Load data ----------------------
with open(DATA_PATH, "rb") as f:
    data = pkl.load(f)

if TRAJ_IDX is None:
    TRAJ_IDX = list(range(len(data["observations"])))

# ---------------- Main loop ---------------------
for t_idx in TRAJ_IDX:
    print(f"\n=== Processing trajectory {t_idx} ===")
    obs = data["observations"][t_idx]

    for px_key in PIXEL_KEYS:
        frames = np.asarray(obs[px_key])  # (T, H, W, 3)
        T = frames.shape[0]

        # Select correct key for agent/human tracks
        robot_or_human_key = "robot_tracks_" if "human" not in TASK_NAME else "human_tracks_"
        tracks_key   = f"{robot_or_human_key}{px_key}"
        object_key   = f"object_tracks_{px_key}"

        if PLOT_PTS and px_key != "pixels51":
            # import ipdb; ipdb.set_trace()
            robot_tr  = np.asarray(obs[tracks_key])  # (T, N_r, 2)
            # obj_tr    = np.asarray(obs[object_key])  # maybe (0, 0, 2)
            # obj_tr    = ensure_frame_len(obj_tr, T)
            # pts_all   = np.concatenate([robot_tr, obj_tr], axis=1)
            pts_all = robot_tr

            n_pts = pts_all.shape[1]
            colors = np.zeros((n_pts, 3), dtype=np.uint8)
            if n_pts:
                third = max(1, n_pts // 3)
                colors[:third, 0] = 255
                colors[third:2*third, 1] = 255
                colors[2*third:, 2] = 255
        else:
            pts_all = None

        out_frames = []
        for f_idx, fr in enumerate(frames):
            fr = fr[..., ::-1].copy()  # BGR→RGB
            if pts_all is not None:
                start = max(0, f_idx - HISTORY_K)
                for pts in pts_all[start:f_idx+1]:
                    for pi, pt in enumerate(pts):
                        x = int(pt[0] * fr.shape[1] / ORIG_SIZE[0])
                        y = int(pt[1] * fr.shape[0] / ORIG_SIZE[1])
                        cv2.circle(fr, (x, y), 2, colors[pi].tolist(), -1)
            out_frames.append(fr)

        out_arr = np.asarray(out_frames, dtype=np.uint8)
        save_path = SAVE_DIR / f"{TASK_NAME}_traj{t_idx}_{px_key}.mp4"
        imageio.mimwrite(save_path, out_arr, fps=FPS)
        print(f"  → Saved {nice_path(save_path)}")
