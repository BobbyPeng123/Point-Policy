# # chain_eval_franka.py
# """Automatic task‑manager for the Franka manipulation stack.

# * Listens for high‑level task instructions from a remote LLM server
#   over ZMQ (REQ/REP).
# * Captures multi‑view RGB snapshots via ``CameraCaptureFranka``.
# * Launches an evaluation policy (``eval_point_track.py``) with the
#   appropriate task name and model checkpoint.
# * Periodically sends RGB frames to the server so it can judge whether
#   the current sub‑task is complete.

# This is the Franka drop‑in replacement for the original ``chain_eval.py``
# from the xArm environment.
# """

# from __future__ import annotations

# import argparse
# import base64
# import io
# import os
# import subprocess
# import sys
# import time
# # import numpy as np
# from pathlib import Path
# from typing import Dict, List

# import cv2
# import zmq
# from PIL import Image

# # -----------------------------------------------------------------------------
# # Local imports ‑ adjust PYTHONPATH if necessary
# # -----------------------------------------------------------------------------

# # Assume ``get_image_franka.py`` lives in the same directory or is importable
# try:
#     from get_image import CameraCaptureFranka  # type: ignore
# except ImportError as exc:  # pragma: no cover
#     print(
#         "[ERROR] Could not import CameraCaptureFranka. "
#         "Make sure get_image_franka.py is in PYTHONPATH."
#     )
#     raise exc

# # -----------------------------------------------------------------------------
# # Constants and utilities
# # -----------------------------------------------------------------------------

# SERVER_ADDR = "tcp://100.96.11.47:6000"  # Endpoint for the LLM‑based task server

# # Modify this table to map server‑provided *task strings* to your local
# # Franka policy checkpoints.
# TASK2CKPT: Dict[str, str] = {
#     "test": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.07/point_policy/deterministic/125804_hidden_dim_256/snapshot/50000.pt",
#     "place_bottle": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.03/point_policy/deterministic/010549_hidden_dim_256/snapshot/50000.pt",
# }

# # -----------------------------------------------------------------------------
# # Helper functions
# # -----------------------------------------------------------------------------

# def serialize_image(pil_img: Image.Image) -> str:
#     """Base‑64‑encode a PIL image in PNG format for ZMQ transmission."""
#     buf = io.BytesIO()
#     pil_img.save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


# def np2pil(rgb: "np.ndarray") -> Image.Image:
#     """Convert an OpenCV image (BGR) to a PIL RGB image."""
#     return Image.fromarray(rgb[:, :, ::-1])


# # -----------------------------------------------------------------------------
# # Main control‑loop class
# # -----------------------------------------------------------------------------

# class ChainEvalFranka:
#     def __init__(
#         self,
#         cam_ids: List[int],
#         task_prompt: str,
#         server_addr: str = SERVER_ADDR,
#     ) -> None:
#         self.task_prompt = task_prompt
#         self.context = zmq.Context()
#         self.socket = self.context.socket(zmq.REQ)
#         self.socket.connect(server_addr)

#         print("[✓] Task‑manager connected to", server_addr)

#         # Camera interface
#         self.camera = CameraCaptureFranka(cam_ids=cam_ids)

#         # Subprocess handle for eval_point_track.py
#         self.eval_process: subprocess.Popen | None = None

#         # Control flags
#         self.initialized = False
#         self.task_complete = True  # triggers first sub‑task request

#     # ------------------------------------------------------------------
#     # ZMQ request helpers
#     # ------------------------------------------------------------------
#     def _send_request(self, query: str, images: List[str]) -> dict:
#         """Send a JSON request to the LLM server and wait for reply."""
#         self.socket.send_json({"image": images, "image_path": "", "query": query})
#         reply: dict = self.socket.recv_json()
#         return reply

#     # ------------------------------------------------------------------
#     # Image capture helpers
#     # ------------------------------------------------------------------
#     def _capture_and_serialize(self) -> List[str]:
#         captures = self.camera.capture_images()
#         # capture_images returns List[Tuple[int, np.ndarray]]
#         serialized: List[str] = []
#         for cam_id, img in captures:
#             pil_img = np2pil(img)
#             serialized.append(serialize_image(pil_img))
#         return serialized

#     # ------------------------------------------------------------------
#     # Eval‑policy management helpers
#     # ------------------------------------------------------------------
#     def _stop_eval(self):
#         if self.eval_process and self.eval_process.poll() is None:
#             print("[~] Terminating existing eval process…")
#             self.eval_process.terminate()
#             self.eval_process.wait()
#             print("[✓] Eval process terminated.")
#         self.eval_process = None

#     def _start_eval(self, task_name: str, model_path: str):
#         """Launch eval_point_track.py with the correct CLI arguments."""
#         print(f"[→] Starting eval_point_track.py for task '{task_name}' → {model_path}")
#         self.eval_process = subprocess.Popen(
#             [
#                 sys.executable,
#                 "eval_point_track.py",
#                 "agent=point_policy",
#                 "suite=point_policy",
#                 "dataloader=point_policy",
#                 "eval=true",
#                 "suite.use_robot_points=true",
#                 "suite.use_object_points=true",
#                 "experiment=eval_point_policy",
#                 f"suite/task/franka_env={task_name}",
#                 f"bc_weight={model_path}",
#             ]
#         )

#     # ------------------------------------------------------------------
#     # Main loop
#     # ------------------------------------------------------------------
#     def run(self):
#         print("[✓] Task‑manager initialised. Prompt:", self.task_prompt)
#         serialize_images: List[str] = []

#         while True:
#             # ------------------------------------------------------
#             # One‑time initialisation handshake
#             # ------------------------------------------------------
#             if not self.initialized:
#                 print("[→] Sending Initialise message to server…")
#                 reply = self._send_request(f"Initialize: {self.task_prompt}", serialize_images)
#                 print("[✓] Server replied:", reply)
#                 self.initialized = True
#                 continue  # proceed to next iteration

#             # ------------------------------------------------------
#             # Ask for next sub‑task when previous completed
#             # ------------------------------------------------------
#             if self.task_complete:
#                 self.task_complete = False
#                 serialize_images = self._capture_and_serialize()
#                 print("[→] Requesting next sub‑task…")
#                 task_reply = self._send_request("Next task", serialize_images)
#                 print("[✓] Server task reply:", task_reply)

#                 raw_task = task_reply.get("result", "").lower()
#                 if not raw_task or raw_task == "stop":
#                     print("[✓] All tasks completed. Exiting.")
#                     break

#                 # Expect format "[task_name, desired_object]"
#                 try:
#                     inside_brackets = raw_task.strip()[1:-1]
#                     task_name, desired_object = [s.strip() for s in inside_brackets.split(",")][:2]
#                 except Exception as exc:
#                     print("[!] Could not parse task string:", raw_task, "—", exc)
#                     self.task_complete = True
#                     continue

#                 # Map to checkpoint path
#                 model_path = TASK2CKPT.get(task_name)
#                 if model_path is None:
#                     print(f"[!] Unknown task '{task_name}'. Add entry to TASK2CKPT.")
#                     self.task_complete = True
#                     continue
#                 if not Path(model_path).exists():
#                     print(f"[!] Checkpoint not found: {model_path}")
#                     self.task_complete = True
#                     continue

#                 # Stop previous eval & launch new one
#                 self._stop_eval()
#                 self._start_eval(task_name, model_path)

#                 # Bookkeeping
#                 self.current_task_name = task_name  # type: ignore[attr-defined]

#             # ------------------------------------------------------
#             # Periodically send judge frames
#             # ------------------------------------------------------
#             frame_counter = 0
#             while not self.task_complete:
#                 frame_counter += 1
#                 time.sleep(0.02)  # ~50 Hz loop; tune as needed

#                 if frame_counter % 200 == 0:  # Roughly every 4 s
#                     judge_imgs = self._capture_and_serialize()
#                     print("[→] Sending judge images for", self.current_task_name)
#                     judge_reply = self._send_request(f"Judge: {self.current_task_name}", judge_imgs)
#                     print("[✓] Judge reply:", judge_reply)
#                     if judge_reply.get("result") == "1":
#                         print(f"[✓] Task '{self.current_task_name}' complete!")
#                         self.task_complete = True
#                         self._stop_eval()
#                         break

#         # Final cleanup
#         self._stop_eval()
#         self.camera.close()
#         print("[✓] Shutdown complete.")


# # -----------------------------------------------------------------------------
# # CLI entry‑point
# # -----------------------------------------------------------------------------

# def parse_args():
#     p = argparse.ArgumentParser(description="Chain‑of‑Thought Task Manager for Franka")
#     p.add_argument("prompt", type=str, help="High‑level natural‑language task prompt sent to the server.")
#     p.add_argument(
#         "--cam_ids", type=int, nargs="+", default=[5], help="IDs of Franka cameras to use (default: 0)"
#     )
#     p.add_argument(
#         "--server", type=str, default=SERVER_ADDR, help=f"ZMQ server address (default: {SERVER_ADDR})"
#     )
#     return p.parse_args()


# def main():
#     args = parse_args()
#     mgr = ChainEvalFranka(cam_ids=args.cam_ids, task_prompt=args.prompt, server_addr=args.server)
#     mgr.run()


# if __name__ == "__main__":
#     main()

"""Automatic task‑manager for the Franka manipulation stack.

Changes vs. the original template
---------------------------------
* Uses the *same* TASK_MODELS mapping as `manual_eval.py`, so every
  task entry keeps **model‑checkpoint**, **hand** (left/right) and
  **use_object_point** in one place.
* The chosen hand is propagated to `eval_point_track.py` via a
  `--hand=<left|right>` CLI flag **and** the `HAND` environment
  variable, so downstream code agrees on which arm is active.
* Automatically toggles `suite.use_object_points` and, when disabled,
  also sets `suite.task_make_fn.points_cfg=null` to skip dummy point
  loading – mirroring the logic from the manual script.
* Extra sanity checks & clearer console logs.

Run this script on the Franka control PC. It connects to your remote
LLM server over ZMQ, requests the next sub‑task, launches the
appropriate evaluation policy, and periodically sends camera frames so
the LLM can decide when the sub‑task is finished.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2  # noqa: F401 – imported for potential debugging, not directly used
import zmq
from PIL import Image

# -----------------------------------------------------------------------------
# Local imports – adjust PYTHONPATH if necessary
# -----------------------------------------------------------------------------

try:
    from get_image import CameraCaptureFranka  # type: ignore
except ImportError as exc:  # pragma: no cover
    msg = (
        "[ERROR] Could not import CameraCaptureFranka. "
        "Make sure get_image.py is importable or in the same directory."
    )
    raise ImportError(msg) from exc

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SERVER_ADDR = "tcp://100.96.11.47:6000"  # ZMQ endpoint for the LLM‑based task server

# Task → (checkpoint, hand, use_object_point) lookup table – keep in sync with
# `manual_eval.py` so human & automatic modes behave identically.
TASK_MODELS: Dict[str, Dict[str, object]] = {
    "test": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.07/point_policy/deterministic/125804_hidden_dim_256/snapshot/50000.pt",
        "hand": "right",
        "use_object_point": True,
    },
    "place_bottle": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.03/point_policy/deterministic/010549_hidden_dim_256/snapshot/50000.pt",
        "hand": "right",
        "use_object_point": True,
    },
    "pick_bottle_from_fridge_right_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.08/point_policy/deterministic/212839_hidden_dim_256/snapshot/50000.pt",
        "hand": "right",
        "use_object_point": True,
    },
    "pick_bottle_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.12/point_policy/deterministic/061129_hidden_dim_256/snapshot/50000.pt",
        "hand": "left",
        "use_object_point": True,
    },
    "place_bottle_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.12/point_policy/deterministic/061325_hidden_dim_256/snapshot/10000.pt",
        "hand": "left",
        "use_object_point": False,
    },
    "pick_bottle_from_fridge_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.16/point_policy/deterministic/080420_hidden_dim_256/snapshot/10000.pt",
        "hand": "left",
        "use_object_point": True,
    },
    "place_bottle_from_fridge_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.15/point_policy/deterministic/071919_hidden_dim_256/snapshot/10000.pt",
        "hand": "left",
        "use_object_point": False,
    },
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _serialize_image(pil_img: Image.Image) -> str:
    """Base‑64 encode a PIL image (PNG) for ZMQ transmission."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _cv2_to_pil(bgr_img):
    """Convert BGR (OpenCV) numpy array → PIL RGB image."""
    return Image.fromarray(bgr_img[:, :, ::-1])


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class ChainEvalFranka:
    """High‑level task‑manager talking to an LLM server via ZMQ."""

    # --------------------------- life‑cycle helpers ---------------------------

    def __init__(
        self,
        cam_ids: List[int],
        task_prompt: str,
        server_addr: str = SERVER_ADDR,
    ) -> None:
        self.task_prompt = task_prompt

        # ZMQ setup – REQ/REP pattern
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(server_addr)
        print(f"[✓] Connected to LLM server @ {server_addr}")

        # Camera interface (multi‑view)
        self._cam = CameraCaptureFranka(cam_ids=cam_ids)

        # Subprocess handle for eval_point_track.py
        self._eval_proc: subprocess.Popen | None = None

        # Driver state
        self._initialised = False
        self._task_complete = True  # triggers first sub‑task request
        self._current_task = ""

    # -------------------------- eval‑policy control --------------------------

    def _stop_eval(self):
        if self._eval_proc and self._eval_proc.poll() is None:
            print("[~] Terminating running eval process …", flush=True)
            self._eval_proc.terminate()
            self._eval_proc.wait()
            print("[✓] Eval process terminated.")
        self._eval_proc = None

    def _start_eval(self, task_name: str, model: str, hand: str, use_object_point: bool):
        """Spawn eval_point_track.py with the right CLI flags/env vars."""
        if use_object_point:
            obj_flag = "true"
            extra = []  # no need to null points_cfg
        else:
            obj_flag = "false"
            extra = ["suite.task_make_fn.points_cfg=null"]

        cmd = [
            sys.executable,
            "eval_point_track.py",
            f"--hand={hand}",  # NEW – propagate arm choice
            "agent=point_policy",
            "suite=point_policy",
            "dataloader=point_policy",
            "eval=true",
            "suite.use_robot_points=true",
            f"suite.use_object_points={obj_flag}",
            *extra,
            "experiment=eval_point_policy",
            "suite.task_make_fn.reset_flag=False",
            f"suite/task/franka_env={task_name}",
            f"bc_weight={model}",
        ]

        env = {**os.environ, "HAND": hand}
        print("[→] Launching eval:", " ".join(cmd))
        self._eval_proc = subprocess.Popen(cmd, env=env)

    # ---------------------------- ZMQ helpers -----------------------------

    def _send_req(self, query: str, images: List[str]):
        self._sock.send_json({"image": images, "image_path": "", "query": query})
        return self._sock.recv_json()

    # ---------------------- camera capture helper ------------------------

    def _capture(self) -> List[str]:
        serialised: List[str] = []
        for _cid, bgr in self._cam.capture_images():
            serialised.append(_serialize_image(_cv2_to_pil(bgr)))
        return serialised

    # ----------------------------- main loop -----------------------------

    def run(self):
        print("[✓] Chain‑eval manager started. Prompt:", self.task_prompt)
        imgs: List[str] = []  # first init has no frames

        while True:
            # ----------------------------------------------------------
            # One‑time handshake so the LLM knows our global goal
            # ----------------------------------------------------------
            if not self._initialised:
                print("[→] Sending INIT to server …")
                reply = self._send_req(f"Initialize: {self.task_prompt}", imgs)
                print("[✓] Server replied:", reply)
                self._initialised = True
                continue  # proceed with loop

            # ----------------------------------------------------------
            # Request next task once previous is finished
            # ----------------------------------------------------------
            if self._task_complete:
                imgs = self._capture()
                print("[→] Asking for next sub‑task …")
                reply = self._send_req("Next task", imgs)
                print("[✓] Task reply:", reply)

                raw = reply.get("result", "").lower()
                if raw in ("", "stop"):
                    print("[✓] All tasks finished. Shutting down.")
                    break

                # Expect the server to send something like "[task_name, desired_obj]"
                try:
                    inside = raw.strip()[1:-1]
                    task_name, _desired = [s.strip() for s in inside.split(",", 1)]
                except Exception as exc:
                    print("[!] Failed to parse task string:", raw, "—", exc)
                    self._task_complete = True  # ask again in next loop
                    continue

                # Look‑up checkpoint, hand, etc.
                print("task_name")
                exit()
                cfg = TASK_MODELS.get(task_name)
                if cfg is None:
                    avail = ", ".join(TASK_MODELS)
                    print(f"[!] Unknown task '{task_name}'. Available: {avail}")
                    self._task_complete = True
                    continue

                model = cfg["model"]  # type: ignore[index]
                hand = cfg["hand"]  # type: ignore[index]
                use_obj = cfg["use_object_point"]  # type: ignore[index]

                if not Path(model).exists():
                    print(f"[!] Checkpoint missing: {model}")
                    self._task_complete = True
                    continue

                # (Re)launch eval
                self._stop_eval()
                self._start_eval(task_name, model, hand, use_obj)

                # Book‑keeping
                self._current_task = task_name
                self._task_complete = False

            # ----------------------------------------------------------
            # Inner loop – send judge frames until LLM says "done"
            # ----------------------------------------------------------
            frame_idx = 0
            while not self._task_complete:
                time.sleep(0.02)  # ≈50 Hz robot control rate
                frame_idx += 1

                if frame_idx % 200 == 0:  # ~every 4 s (@50 Hz)
                    judge_imgs = self._capture()
                    print(f"[→] Judge frames for '{self._current_task}' → server")
                    verdict = self._send_req(f"Judge: {self._current_task}", judge_imgs)
                    print("[✓] Judge reply:", verdict)

                    if verdict.get("result") == "1":
                        print(f"[✓] Task '{self._current_task}' complete!")
                        self._task_complete = True
                        self._stop_eval()
                        break

        # ------------------------------------------------------------------
        # Clean shutdown
        # ------------------------------------------------------------------
        self._stop_eval()
        self._cam.close()
        print("[✓] Shutdown complete.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Chain‑of‑thought task manager for Franka")
    p.add_argument("prompt", help="High‑level natural‑language prompt sent to the LLM server")
    p.add_argument("--cam_ids", type=int, nargs="+", default=[6], help="Franka camera IDs (default: 5)")
    p.add_argument("--server", default=SERVER_ADDR, help="ZMQ server address")
    return p.parse_args()


def main():
    args = _parse_args()
    mgr = ChainEvalFranka(cam_ids=args.cam_ids, task_prompt=args.prompt, server_addr=args.server)
    mgr.run()


if __name__ == "__main__":
    main()

