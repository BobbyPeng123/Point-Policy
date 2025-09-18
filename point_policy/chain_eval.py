#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  Chain‑of‑thought task manager (robot + policy) – *Robot task prompt* 协议版
# ──────────────────────────────────────────────────────────────────────────────
#
# 轮廓：
#   1. 连接远端 LLM (ZMQ)。
#   2. 当上一子任务完成，抓图 → 发送
#        "Robot task prompt: <high‑level prompt>"
#      服务器返回形如 "[task_name, desired_object]".
#   3. 根据 TASK_MODELS 查找 checkpoint / hand / use_object_point，
#      启动或切换 eval_point_track.py。
#   4. 每 ~4 s 发送一次 Judge 帧：
#        "Judge: <task_name> <desired_object>"
#      若返回字符串包含 "judge: task completed" → 该子任务结束，继续下一轮。
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import zmq
from PIL import Image

# ────────────────────────── Camera interface ────────────────────────────────
try:
    from get_image import CameraCaptureFranka  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CameraCaptureFranka is required. Make sure get_image.py is importable."
    ) from exc

# ─────────────────────────── Configuration ──────────────────────────────────
SERVER_ADDR = "tcp://100.96.11.47:6000"

PLACING_LEFT_FLAG = True 

TASK_MODELS: Dict[str, Dict[str, object]] = {
    # keep in sync with manual_eval.py
    "open_fridge_door_right_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.24/point_policy/deterministic/093048_hidden_dim_256/snapshot/10000.pt",
        "hand": "right",
        "use_object_point": False,
        "reset_flag": True,
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
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.25/point_policy/deterministic/084458_hidden_dim_256/snapshot/10000.pt",
        "hand": "left",
        "use_object_point": True,
        "reset_flag": True,
    },
    "pick_bottle_from_the_fridge_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.08.15/point_policy/deterministic/222054_hidden_dim_256/snapshot/15000.pt",
        "hand": "left",
        "use_object_point": True,
        "reset_flag": True,
    },
    "place_bottle_from_the_fridge_left_robot": {
        # "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.08.16/point_policy/deterministic/115405_hidden_dim_256/snapshot/5000.pt",
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.08.18/point_policy/deterministic/110015_hidden_dim_256/snapshot/5000.pt",
        "hand": "left",
        "use_object_point": True,
        "reset_flag": False,
        "use_pin_point": True,
    },
    "place_bottle_from_the_fridge_left_pos_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.08.16/point_policy/deterministic/121910_hidden_dim_256/snapshot/5000.pt",
        "hand": "left",
        "use_object_point": True,
        "reset_flag": False,
        "use_pin_point": True,
    },
    "open_the_fridge_door_right_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.30/point_policy/deterministic/010031_hidden_dim_256/snapshot/10000.pt",
        "hand": "right",
        "use_object_point": False,
        "reset_flag": True,
    },
    "pick_bottle_from_fridge_and_place_the_bottle_right_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.30/point_policy/deterministic/030818_hidden_dim_256/snapshot/10000.pt",
        "hand": "right",
        "use_object_point": True,
        "reset_flag": True,
    },
    "pick_bottle_from_side_door_of_fridge_and_place_the_bottle_right_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.30/point_policy/deterministic/044206_hidden_dim_256/snapshot/10000.pt",
        "hand": "right",
        "use_object_point": True,
        "reset_flag": True,
    },
    "put_bowl_into_oven_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.09.10/point_policy/deterministic/190412_hidden_dim_256/snapshot/20000.pt",
        "hand": "left",
        "use_object_point": True,
        "reset_flag": True,
    },
    "close_oven_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.09.10/point_policy/deterministic/163533_hidden_dim_256/snapshot/15000.pt",
        "hand": "left",
        "use_object_point": True,
        "reset_flag": True,
    },
    "put_bowl_into_basket_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.08.24/point_policy/deterministic/165331_hidden_dim_256/snapshot/20000.pt",
        "hand": "left",
        "use_object_point": True,
        "reset_flag": True,
    },
    "pick_bread_to_bowl_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.09.12/point_policy/deterministic/211940_hidden_dim_256/snapshot/20000.pt",
        "hand": "left",
        "use_object_point": True,
        "reset_flag": True,
    },
}

# ─────────────────────────── Helpers ────────────────────────────────────────
def _serialize_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bgr_to_pil(bgr):
    return Image.fromarray(bgr[:, :, ::-1])  # BGR → RGB


def _parse_task(raw: str) -> Tuple[str, str]:
    global PLACING_LEFT_FLAG
    """Extract task and object from '[task, obj]' (obj optional)."""
    inner = raw.strip()[1:-1]
    parts = [s.strip() for s in inner.split(",", 1)]
    task = parts[0]
    obj = parts[1] if len(parts) == 2 else ""
    # if "place_bottle_from_fridge" in task: task = "place_bottle_from_fridge_left_robot"
    # elif "pick_bottle_from_fridge" in task: task = "pick_bottle_from_fridge_left_robot"
    if "pick_bottle" in task and "side_door" in task: 
        task = "pick_bottle_from_side_door_of_fridge_and_place_the_bottle_right_robot"
    elif "pick_bottle" in task:
        task = "pick_bottle_from_the_fridge_left_robot"
    elif "open_fridge_door" in task: task = "open_fridge_door_right_robot"
    elif "open_the_fridge_door" in task: task = "open_the_fridge_door_right_robot"
    elif "place_bottle" in task:
        if PLACING_LEFT_FLAG:
            task = "place_bottle_from_the_fridge_left_pos_left_robot"
        else:
            task = "place_bottle_from_the_fridge_left_robot"
        
        PLACING_LEFT_FLAG = not PLACING_LEFT_FLAG
    elif "bread" in task:
        task = "pick_bread_to_bowl_left_robot"
    elif "put_bowl_into_oven" in task:
        task = "put_bowl_into_oven_left_robot"
    elif "close_oven" in task:
        task = "close_oven_left_robot"

    return task, obj

# ─────────────────────────── Main class ─────────────────────────────────────
class ChainEvalFranka:
    def __init__(self, cam_ids: List[int], prompt: str, server_addr: str = SERVER_ADDR):
        self.prompt = prompt
        self.cam = CameraCaptureFranka(cam_ids=cam_ids)

        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.connect(server_addr)
        print(f"[✓] Connected to LLM server @ {server_addr}")

        self.eval_proc: subprocess.Popen | None = None
        self.task_complete = True
        self.current_task = ""
        self.current_des_obj = ""
        self.max_judge_rounds = 10  # max rounds of judging before giving up
        self.judge_rounds = 0  # current round of judging
        # ── NEW: Judge not before ──
        self.judge_not_before: float = 0.0

        # ─────────── replay helper (special case for close_oven_left_robot) ─────
    def _start_policy(self, task: str):
        """
        Launch a fixed demonstration replay instead of the learned policy.
        """
        if task == "close_oven_left_robot":
            cmd = [
                "python",
                "/home/bobby/Point-Policy/Franka-Teach/replay_given_state.py",
                "--file",
                "/home/bobby/data/processed_data/close_oven_left_robot/demonstration_0/states.csv",
        ]
        elif task == "put_cup_into_basket_left_robot":
            cmd = [
                "python",
                "/home/bobby/Point-Policy/Franka-Teach/replay_given_state.py",
                "--file",
                "/home/bobby/data/processed_data/put_cup_into_basket_left_robot/demonstration_9/states.csv", # 9 for  right, 0 for left, 58 for front middle, 7 middle
        ]
        elif task == "pick_plate_from_rack_left_robot":
            cmd = [
                "python",
                "/home/bobby/Point-Policy/Franka-Teach/replay_given_state.py",
                "--file",
                "/home/bobby/data/processed_data/pick_plate_from_rack_left_robot/demonstration_2/states.csv", # right, 32 for middle, 10 for left
            ]
        # print("[→] Launching replay:\n  " + " ".join(cmd))
        # Run from the Franka-Teach repo so relative imports work, if any
        self.eval_proc = subprocess.Popen(cmd, cwd="/home/bobby/Point-Policy/Franka-Teach")
        # Replays typically spin up quickly; allow judge after a short delay
        self.judge_not_before = time.time() + 8
        print("[→] Judge will start after 8s (replay mode).")

    # ───────────── ZMQ ─────────────
    def _rpc(self, query: str, images: List[str]):
        self.sock.send_json({"image": images, "image_path": "", "query": query})
        return self.sock.recv_json()

    # ───────────── Camera ──────────
    def _capture(self) -> List[str]:
        enc: List[str] = []
        for _cid, frame in self.cam.capture_images():
            enc.append(_serialize_pil(_bgr_to_pil(frame)))
        return enc

    # ───────────── reset helper ─────────────
    def _reset_robot(self, hand: str):
        """
        Block until reset.py completes. Mirrors the right/left branches
        in manual_eval.launch_reset (defaults tailored for right arm).
        """
        if hand == "left":
            pixel_keys = "[pixels4,pixels6]"
            franka_env = "pick_bottle_left_robot"
        else:
            pixel_keys = "[pixels2,pixels5]"
            franka_env = "open_fridge_door_right_robot"

        cmd = [
            "python",
            "reset.py",
            f"--hand={hand}",
            "agent=point_policy",
            "suite=point_policy",
            "dataloader=point_policy",
            "eval=true",
            "suite.use_robot_points=true",
            "suite.use_object_points=false",
            f"suite.pixel_keys={pixel_keys}",
            "suite.task_make_fn.points_cfg=null",
            "suite.task_make_fn.reset_flag=True",
            f"suite/task/franka_env={franka_env}",
        ]
        env = {**os.environ, "HAND": hand}
        print("[→] Launching reset:\n  " + " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)

    # ─────────── eval helpers ──────
    def _stop_eval(self):
        if self.eval_proc and self.eval_proc.poll() is None:
            print("[~] Terminating running eval process…")
            self.eval_proc.terminate()
            self.eval_proc.wait()
            print("[✓] Eval process terminated.")
        self.eval_proc = None

    def _start_eval(self, task: str, model: str, hand: str, obj: str, use_obj_pt: bool, reset_flag: bool = False, use_pin_pt: bool = False):
        obj_flag = "true" if use_obj_pt else "false"
        pin_point_flag = "true" if use_pin_pt else "false"
        extra = ["suite.task_make_fn.points_cfg=null"] if use_pin_pt == use_obj_pt else []
        if use_pin_pt:
            extra.append("suite.num_object_points=1")

        print(f"extra: {extra}, use_obj_pt: {use_obj_pt}, use_pin_pt: {use_pin_pt}")
        # exit()
        # extra = []
        # if use_pin_pt:
        #     if use_obj_pt:
        #         extra = ["suite.task_make_fn.points_cfg=null"]
        # else:
        #     if not use_obj_pt:
        #         extra = ["suite.task_make_fn.points_cfg=null"]
        # extra = [] if use_obj_pt else ["suite.task_make_fn.points_cfg=null"]

        if hand == "left":
            pixel_keys = "[pixels4,pixels6]"
            calib = "/home/bobby/Point-Policy/calib/calib_46_left_robot.npy"
        else:
            pixel_keys = "[pixels2,pixels5]"
            calib = "/home/bobby/Point-Policy/calib/calib_25_right_robot.npy"

        cmd = [
            "python",
            "eval_point_track.py",
            f"--hand={hand}",
            "agent=point_policy",
            "suite=point_policy",
            "dataloader=point_policy",
            "eval=true",
            "suite.use_robot_points=true",
            f"suite.use_object_points={obj_flag}",
            f"suite.use_pin_points={pin_point_flag}",
            f"suite.pixel_keys={pixel_keys}",
            f"suite.task_make_fn.calib_path={calib}",
            "experiment=eval_point_policy",
            f"suite.task_make_fn.reset_flag={reset_flag}",
            f"suite/task/franka_env={task}",
            f"bc_weight={model}",
            *extra,
        ]

        # env = {**os.environ, "HAND": hand, "DES_OBJECT": obj}
        env = {
            **os.environ,
            "HAND": hand,
            "DES_OBJECT": obj,
            "DES_OBJECTS": obj,  # 支持 "orange bottle, blue basket" / "A; B" / '["A","B"]'
        }
        print("[→] Launching eval:\n  " + " ".join(cmd))
        self.eval_proc = subprocess.Popen(cmd, env=env)

        # ── NEW: 根据 extra 是否包含 points_cfg=null 决定 Judge 延迟 ──
        has_points_cfg_null = any(s.startswith("suite.task_make_fn.points_cfg=null") for s in extra)
        delay = 12 if has_points_cfg_null else 60
        self.judge_not_before = time.time() + delay
        print(f"[→] Judge will start after {delay}s (points_cfg_null={has_points_cfg_null}).")

    # ───────────── Main loop ───────
    def run(self):
        print("[✓] Chain‑eval started with prompt:", self.prompt)

        while True:
            # ── Ask for new sub‑task ──
            if self.task_complete:
                imgs = self._capture()
                reply = self._rpc(f"Robot task prompt: {self.prompt}", imgs)
                print("[✓] Task reply:", reply)

                result = reply.get("result", "").lower()
                if result in ("", "stop") or ("stop" in result and "[" not in result):
                    print("[✓] Server indicated STOP. Exiting.")
                    break

                print("[→] Parsing task from reply:", result)

                try:
                    task, des_obj = _parse_task(result)
                except Exception as e:
                    print("[!] Parse failed:", e, "for", result)
                    continue

                cfg = TASK_MODELS.get(task)
                if cfg is None:
                    print(f"[!] Unknown task '{task}'. Available: {', '.join(TASK_MODELS)}")
                    continue

                # model, hand, use_obj_pt, reset_flag = cfg["model"], cfg["hand"], cfg["use_object_point"], cfg["reset_flag"]  # type: ignore[index]

                # if not Path(model).exists():
                #     print(f"[!] Checkpoint missing: {model}")
                #     continue

                # # launch / switch policy
                # self._stop_eval()
                # self._start_eval(task, model, hand, des_obj, use_obj_pt, reset_flag, use_pin_pt=cfg.get("use_pin_point", False))

                # ── SPECIAL CASE: close_oven_left_robot → run Franka-Teach replay ──
                if task == "close_oven_left_robot" or task == "put_cup_into_basket_left_robot" or task == "pick_plate_from_rack_left_robot":
                    self._stop_eval()
                    time.sleep(20)
                    self._start_policy(task)
                else:
                    model, hand, use_obj_pt, reset_flag = (
                        cfg["model"],
                        cfg["hand"],
                        cfg["use_object_point"],
                        cfg["reset_flag"],
                    )  # type: ignore[index]
                    if not Path(model).exists():
                        print(f"[!] Checkpoint missing: {model}")
                        continue
                    # launch / switch policy
                    self._stop_eval()
                    self._start_eval(
                        task,
                        model,
                        hand,
                        des_obj,
                        use_obj_pt,
                        reset_flag,
                        use_pin_pt=cfg.get("use_pin_point", False),
                    )

                self.current_task = task
                self.current_des_obj = des_obj
                self.judge_rounds = 0
                self.task_complete = False
                # print(f"[→] >>> New task: {task}, object: {des_obj} <<<")
                if task == "close_oven_left_robot":
                    print(f"[→] >>> New task (REPLAY): {task} <<<")
                else:
                    print(f"[→] >>> New task: {task}, object: {des_obj} <<<")

            # ── Judge loop ──
            tick = 0
            while not self.task_complete:
                time.sleep(0.02)  # ~50 Hz
                tick += 1
                if time.time() < self.judge_not_before:
                    continue
                if tick % 700 == 0:  # 每约 10 s
                    j_imgs = self._capture()
                    verdict = self._rpc(
                        f"Judge: {self.current_task} {self.current_des_obj}".strip(),
                        j_imgs,
                    )
                    print("[✓] Judge reply:", verdict)
                    if "judge: task completed" in verdict.get("result", "").lower():
                        print(f"[✓] Task '{self.current_task}' completed.")
                        time.sleep(8)  # 等待 8 秒，确保 eval.py 完成任务
                        self.task_complete = True
                        self._stop_eval()
                        # ── NEW: if we just opened the fridge with the right arm, reset it now ──
                        if ("open_fridge_door" in self.current_task and "right" in self.current_task):
                            print("[→] Resetting robot (right arm) after opening fridge door…")
                            time.sleep(5)  # wait a bit before resetting
                            self._reset_robot("right")
                            print("[✓] Robot reset complete.")
                        # ──────────────────────────────────────────────────────────────────────
                        elif ("put_bowl_into_oven" in self.current_task) or ("put_bowl_into_basket" in self.current_task) or ("pick_bread_to_bowl" in self.current_task) or ("put_cup_into_basket" in self.current_task):
                            print("[→] Resetting robot (left arm)")
                            time.sleep(5)  # wait a bit before resetting
                            self._reset_robot("left")
                            print("[✓] Robot reset complete.")
                        break
                    else:
                        self.judge_rounds += 1
                        print(f"[→] Judge round {self.judge_rounds}/{self.max_judge_rounds} for '{self.current_task}'.")
                        if self.judge_rounds >= self.max_judge_rounds:
                            print(f"[!] Max judge rounds reached ({self.max_judge_rounds}). Forcing completion.")
                            self._stop_eval()
                            # 与真实完成时保持一致的后置动作（含右手机器人的重置）
                            if self.current_task in ("open_fridge_door_right_robot", "open_the_fridge_door_right_robot"):
                                try:
                                    print("[→] Auto-resetting right robot after forced completion…")
                                    self._reset_robot("right")
                                    print("[✓] Right robot reset complete.")
                                except subprocess.CalledProcessError as e:
                                    print(f"[!] Right-arm reset failed (returncode={e.returncode}); continuing.")
                            self.task_complete = True
                            break

        # ── Cleanup ──
        self._stop_eval()
        self.cam.close()
        print("[✓] Shutdown complete.")

# ─────────────────────────── CLI ────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="Chain‑eval manager (Robot task prompt protocol)")
    # ap.add_argument("--prompt", type=str, default="Get the bottle with Korean letters on it and coke bottle out from fridge",
    #                 help="High‑level prompt for the robot task")
    # ap.add_argument("--prompt", type=str, default="Get purple bottle from the side door of the fridge",
    #                 help="High‑level prompt for the robot task")
    # ap.add_argument("--prompt", type=str, default="Get the orange bottle and green bottle out from fridge, and get pink bottle from the side door of the fridge",
    #                 help="High‑level prompt for the robot task")
    # ap.add_argument("--prompt", type=str, default="Get the orange bottle and green bottle out from fridge",
    #                 help="High‑level prompt for the robot task")
    # ap.add_argument("--prompt", type=str, default="Get the orange bottle out from fridge(no need to open it first)",
    #                 help="High‑level prompt for the robot task")
    # ap.add_argument("--prompt", type=str, default="Clean up the table and place the plate on table",
    #                 help="High‑level prompt for the robot task")
    ap.add_argument("--prompt", type=str, default="Bake the round bagel in the oven",
                    help="High‑level prompt for the robot task")
    ap.add_argument("--cam_ids", type=int, nargs="+", default=[6], help="Camera IDs (default: 6)")
    ap.add_argument("--server", default=SERVER_ADDR, help="ZMQ server address")
    return ap.parse_args()

def main():
    args = _parse_args()
    mgr = ChainEvalFranka(cam_ids=args.cam_ids, prompt=args.prompt, server_addr=args.server)
    mgr.run()

if __name__ == "__main__":
    main()
