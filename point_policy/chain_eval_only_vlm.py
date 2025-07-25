# """chain_eval_only_test_vlm.py – Lightweight VLM tester
# =====================================================

# **Purpose**: Verify the *task‑planner* and *completion judge* parts of
# your Vision‑Language Model (VLM) stack without starting any robot or
# policy. You move objects in the workspace manually while the script:

# 1. Connects to the remote ZMQ server.
# 2. Sends the high‑level prompt via an `Initialize:` request.
# 3. Repeatedly requests `Next task` and prints the server‑proposed
#    sub‑task (e.g. `[pick_bottle_left_robot, coke]`).
# 4. Periodically captures RGB frames from **CameraCaptureFranka** (which
#    *must* be importable) and sends them to the server with `Judge:`
#    until the server replies `{result:"1"}`.

# No fallback webcams, no robot control – ideal for quick, safe VLM
# regression tests.
# """

# from __future__ import annotations

# import argparse
# import base64
# import io
# import time
# from typing import List

# import numpy as np
# import zmq
# from PIL import Image

# # ---------------------------------------------------------------------------
# # Camera interface – MUST be available
# # ---------------------------------------------------------------------------
# try:
#     from get_image import CameraCaptureFranka  # type: ignore
# except ImportError as exc:  # pragma: no cover
#     raise ImportError(
#         "CameraCaptureFranka is required but could not be imported. "
#         "Ensure get_image.py is on PYTHONPATH and provides the class."
#     ) from exc

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------

# SERVER_ADDR = "tcp://100.96.11.47:6000"  # default ZMQ endpoint

# # ---------------------------------------------------------------------------
# # Helper functions
# # ---------------------------------------------------------------------------

# def _serialize_image(pil: Image.Image) -> str:
#     """PNG‑encode → base64 string for JSON transport."""
#     buf = io.BytesIO()
#     pil.save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


# def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
#     """OpenCV BGR → PIL RGB."""
#     return Image.fromarray(bgr[:, :, ::-1])

# # ---------------------------------------------------------------------------
# # Main driver
# # ---------------------------------------------------------------------------

# class VLMOnlyTester:
#     def __init__(self, cam_ids: List[int], prompt: str, server: str):
#         self.prompt = prompt

#         # ZMQ REQ/REP socket
#         ctx = zmq.Context()
#         self._sock = ctx.socket(zmq.REQ)
#         self._sock.connect(server)
#         print(f"[✓] Connected to VLM server @ {server}")

#         # Multi‑view camera
#         self._cam = CameraCaptureFranka(cam_ids=cam_ids)

#         # State flags
#         self._initialised = False
#         self._task_complete = True
#         self._current_task = ""

#     # ---------------- ZMQ helpers ----------------
#     def _rpc(self, query: str, images: List[str]):
#         self._sock.send_json({"image": images, "image_path": "", "query": query})
#         return self._sock.recv_json()

#     # -------------- image capture ---------------
#     def _capture(self) -> List[str]:
#         serialised: List[str] = []
#         for _cid, bgr in self._cam.capture_images():
#             serialised.append(_serialize_image(_bgr_to_pil(bgr)))
#         return serialised

#     # ---------------- main loop -----------------
#     def run(self):
#         print("[✓] Tester started. Prompt:", self.prompt)
#         imgs: List[str] = []  # first INIT has no frames

#         while True:
#             # 1) Handshake
#             if not self._initialised:
#                 print("[→] INIT …")
#                 rep = self._rpc(f"Initialize: {self.prompt}", imgs)
#                 print("[✓] Server:", rep)
#                 self._initialised = True
#                 continue

#             # 2) Ask for next sub‑task when previous finished
#             if self._task_complete:
#                 import ipdb; ipdb.set_trace()
#                 imgs = self._capture()
#                 print("[→] Next task …")
#                 rep = self._rpc("Next task", imgs)
#                 print("[✓] Task reply:", rep)

#                 raw = rep.get("result", "").lower()
#                 if raw in ("", "stop"):
#                     print("[✓] All tasks completed. Bye.")
#                     break

#                 # Expect "[task_name, desired_obj]"
#                 try:
#                     inner = raw.strip()[1:-1]
#                     task_name, _ = [s.strip() for s in inner.split(",", 1)]
#                 except Exception as e:  # pragma: no cover
#                     print("[!] Failed to parse task string:", raw, "—", e)
#                     self._task_complete = True
#                     continue

#                 self._current_task = task_name
#                 print(f"[→] >>> New task: {task_name} <<<")
#                 self._task_complete = False

#             # 3) Judge loop – ~5 Hz (every 0.2 s) to reduce bandwidth
#             frame = 0
#             while not self._task_complete:
#                 time.sleep(0.2)
#                 frame += 1
#                 if frame % 5 == 0:  # every 1 s
#                     judge_imgs = self._capture()
#                     verdict = self._rpc(f"Judge: {self._current_task}", judge_imgs)
#                     print("[✓] Judge:", verdict)
#                     if verdict.get("result") == "1":
#                         print(f"[✓] Task '{self._current_task}' complete.")
#                         self._task_complete = True
#                         break

#         # Clean‑up
#         self._cam.close()
#         print("[✓] Shutdown complete.")

# # ---------------------------------------------------------------------------
# # CLI entry‑point
# # ---------------------------------------------------------------------------

# def _parse_args():
#     ap = argparse.ArgumentParser(description="VLM‑only evaluation loop (no robot)")
#     ap.add_argument("prompt", help="High‑level natural‑language goal")
#     ap.add_argument("--cam_ids", type=int, nargs="+", default=[4], help="Camera IDs (default: 5)")
#     ap.add_argument("--server", default=SERVER_ADDR, help="ZMQ server address")
#     return ap.parse_args()


# def main():
#     args = _parse_args()
#     tester = VLMOnlyTester(cam_ids=args.cam_ids, prompt=args.prompt, server=args.server)
#     tester.run()


# if __name__ == "__main__":
#     main()
"""chain_eval_only_vlm_franka.py
=================================

Franka **VLM‑only** evaluation loop (mirrors your previous
`chain_eval_only_vlm_xarm.py` semantics):

* **No** policy / controller subprocess is launched.
* **No** `Initialize:` / `Next task` protocol – instead each new cycle
  sends a single request whose query is:

      "Robot task prompt: <HIGH_LEVEL_PROMPT>"

* Server is expected to return a JSON with a `result` string containing
  a bracketed sub‑task description, e.g.:

      "[pick_bottle_left_robot, coke]"

* Script parses out `(task_name, des_object)` and then periodically
  sends judge queries of the form:

      "Judge: <task_name> <des_object>"

  (Maintaining the pattern from your xArm version where `policy_name`
  was used; here we just re‑use `task_name` since no policy launches.)

* Completion condition: server reply `result` string contains the
  substring `"judge: task completed"` (case‑insensitive match).

Adjust camera IDs / server endpoint via CLI arguments.

Example:
    python chain_eval_only_vlm_franka.py \\
        "get pink bottle out of fridge and get blue bottle out of the fridge." \\
        --cam_ids 5 6 \\
        --server tcp://100.96.11.47:6000
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys
import time
from typing import List, Tuple

import numpy as np
import zmq
from PIL import Image

# ---------------------------------------------------------------------------
# Required camera interface
# ---------------------------------------------------------------------------
try:
    from get_image import CameraCaptureFranka  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CameraCaptureFranka is required but was not importable. Ensure get_image.py is on PYTHONPATH."  # noqa: E501
    ) from exc

DEFAULT_SERVER = "tcp://100.96.11.47:6000"

# ---------------------------------------------------------------------------
# Image serialization helpers
# ---------------------------------------------------------------------------

def _serialize_pil(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1])  # BGR → RGB


def _capture_and_encode(cam) -> List[str]:
    encoded: List[str] = []
    for _cid, frame in cam.capture_images():  # frame is BGR (OpenCV)
        encoded.append(_serialize_pil(_bgr_to_pil(frame)))
    return encoded

# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

class FrankaVLMOnly:
    def __init__(self, prompt: str, cam_ids: List[int], server: str):
        self.prompt = prompt
        self.cam = CameraCaptureFranka(cam_ids=cam_ids)

        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.connect(server)
        print(f"[✓] Connected to server @ {server}")

        self.task_complete = True  # trigger first fetch
        self.current_task: str = ""
        self.current_object: str = ""

    # --------------- ZMQ RPC ---------------
    def _rpc(self, query: str, images: List[str]):
        self.sock.send_json({"image": images, "image_path": "", "query": query})
        return self.sock.recv_json()

    # ------------- Task parsing -------------
    @staticmethod
    def _parse_task(result_str: str) -> Tuple[str, str]:
        """Extract task name and desired object from a result string.

        Expected forms:
            "[task_name, object]"  or  "[task_name]"
        Returns (task_name, object_or_empty)
        """
        # Lower‑case & strip
        s = result_str.strip().lower()
        # Find first bracket pair
        m = re.search(r"\[(.*?)\]", s)
        if not m:
            raise ValueError(f"No bracketed task segment in: {result_str}")
        inside = m.group(1)
        if "," in inside:
            task_name, obj = [p.strip() for p in inside.split(",", 1)]
        else:
            task_name, obj = inside.strip(), ""
        return task_name, obj

    # --------------- Main run ---------------
    def run(self):
        print("[✓] Starting VLM‑only loop. Prompt:", self.prompt)

        while True:
            if self.task_complete:
                self.task_complete = False
                print("[→] Capturing images for new sub‑task …")
                imgs = _capture_and_encode(self.cam)
                query = f"Robot task prompt: {self.prompt}"
                print("[→] Sending task request: 'Robot task prompt:' …")
                reply = self._rpc(query, imgs)
                print("[✓] Server task reply:", reply)
                result_str = reply.get("result", "")
                if not result_str:
                    print("[!] Empty result string – retrying next loop.")
                    self.task_complete = True
                    continue
                result_lower = result_str.lower()
                if "stop" in result_lower and "[" not in result_lower:
                    print("[✓] Server indicated STOP. Exiting.")
                    break
                try:
                    task_name, obj = self._parse_task(result_str)
                except Exception as e:
                    print("[!] Parse failure:", e)
                    self.task_complete = True
                    continue

                if task_name in ("", "stop"):
                    print("[✓] All tasks complete.")
                    break

                self.current_task = task_name
                self.current_object = obj
                print("********************************************************")
                print(f"   Task: {task_name}    Object: {obj}")
                print("********************************************************")
                print("Proceed to manipulate the scene manually …")

            # Judge loop – high frequency capture, slower judge requests
            frame = 0
            while not self.task_complete:
                frame += 1
                time.sleep(0.01)  # 100 Hz internal tick
                # Every N ticks send judge images
                if frame % 500 == 0:  # ≈ every 20 s (2000 * 0.01)
                    judge_imgs = _capture_and_encode(self.cam)
                    judge_query = f"Judge: {self.current_task} {self.current_object}".strip()
                    print(f"[→] Sending JUDGE for '{self.current_task}' …")
                    verdict = self._rpc(judge_query, judge_imgs)
                    print("[✓] Judge reply:", verdict)
                    verdict_result = verdict.get("result", "").lower()
                    if "judge: task completed" in verdict_result:
                        print(f"[✓] Task '{self.current_task}' judged COMPLETE.")
                        self.task_complete = True
                        break

        # Cleanup
        self.cam.close()
        print("[✓] Shutdown complete.")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="Franka VLM only chain eval (no robot / no policy)")
    ap.add_argument("prompt", help="High‑level natural language goal")
    ap.add_argument("--cam_ids", type=int, nargs="+", default=[6], help="Camera IDs for capture (default: 5)")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="ZMQ server address")
    return ap.parse_args()


def main():
    args = _parse_args()
    driver = FrankaVLMOnly(prompt=args.prompt, cam_ids=args.cam_ids, server=args.server)
    driver.run()


if __name__ == "__main__":
    main()
