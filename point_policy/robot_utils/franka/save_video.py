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
TASK_NAME  = "place_bottle_from_the_fridge_left_robot"  # e.g. "pick_bottle_from_the_fridge_left_robot"
PLOT_PTS   = True
PIXEL_KEYS = ["pixels4", "pixels6"]
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
# import ipdb; ipdb.set_trace()  # Debugging breakpoint
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
            obj_tr    = np.asarray(obs[object_key])  # maybe (0, 0, 2)
            obj_tr    = ensure_frame_len(obj_tr, T)
            pts_all   = np.concatenate([robot_tr, obj_tr], axis=1)
            # pts_all = robot_tr

            n_pts = pts_all.shape[1]
            colors = np.zeros((n_pts, 3), dtype=np.uint8)
            if n_pts:
                third = max(1, n_pts // 3)
                colors[:third, 0] = 255
                colors[third:2*third, 1] = 255
                colors[2*third:, 2] = 255
        else:
            pts_all = None

        # print("frame shape (h, w):", frames.shape[:2])
        # print("ORIG_SIZE :", ORIG_SIZE)
        # print("min/max pt:", robot_tr.min(), robot_tr.max())
        # exit()

        out_frames = []
        for f_idx, fr in enumerate(frames):
            fr = fr[..., ::-1].copy()  # BGR→RGB
            if pts_all is not None:
                start = max(0, f_idx - HISTORY_K)
                for pts in pts_all[start:f_idx+1]:
                    for pi, pt in enumerate(pts):
                        # x = int(pt[0] * fr.shape[1] / ORIG_SIZE[0])
                        # y = int(pt[1] * fr.shape[0] / ORIG_SIZE[1])
                        x = int(pt[0])
                        y = int(pt[1])
                        cv2.circle(fr, (x, y), 2, colors[pi].tolist(), -1)
            out_frames.append(fr)

        out_arr = np.asarray(out_frames, dtype=np.uint8)
        save_path = SAVE_DIR / f"{TASK_NAME}_traj{t_idx}_{px_key}.mp4"
        imageio.mimwrite(save_path, out_arr, fps=FPS)
        print(f"  → Saved {nice_path(save_path)}")
