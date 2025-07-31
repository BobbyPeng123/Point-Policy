# #!/usr/bin/env python3

# import os
# import subprocess

# def main():
#     eval_process = None  # To track the running eval_point_track.py process

#     print("Task Manager Initialized")
#     print("Enter task name to start evaluation or 'reset' to reset the robot or 'stop' to terminate the ongoing evaluation.")

#     while True:
#         user_input = input("Enter a task name or 'stop': ").strip()

#         if user_input.lower() == "stop":
#             if eval_process and eval_process.poll() is None:
#                 print("Stopping the ongoing evaluation process...")
#                 eval_process.terminate()
#                 eval_process.wait()
#                 print("Evaluation process terminated.")
#             else:
#                 print("No evaluation process is currently running.")
#             continue
        
#         # # TODO: complete the reset function
#         # if user_input.lower() == "reset":
#         #     try:
#         #         print("Resetting the robot...")
#         #         subprocess.run(["python", "reset.py", "suite=xarm_env_reset"])
#         #         print("Robot reset successfully.")
#         #     except Exception as e:
#         #         print(f"Failed to reset the robot: {e}")
#         #     continue

#         if user_input:
#             task_name, des_object = user_input.split(',')[0].strip(), user_input.split(',')[1].strip()

#             # Manually map tasks to model paths
#             if task_name == 'test':
#                 model_path = '/home/bobby/Point-Policy/point_policy/exp_local/2025.07.07/point_policy/deterministic/125804_hidden_dim_256/snapshot/50000.pt'
#                 hand = "right"
#             elif task_name == 'place_bottle':
#                 model_path = '/home/bobby/Point-Policy/point_policy/exp_local/2025.07.03/point_policy/deterministic/010549_hidden_dim_256/snapshot/50000.pt'
#                 hand = "right"
#             elif task_name == 'pick_bottle_from_fridge_right_robot':
#                 model_path = '/home/bobby/Point-Policy/point_policy/exp_local/2025.07.08/point_policy/deterministic/212839_hidden_dim_256/snapshot/50000.pt'
#                 hand = "right"
#             elif task_name == 'pick_bottle_left_robot':
#                 model_path = '/home/bobby/Point-Policy/point_policy/exp_local/2025.07.12/point_policy/deterministic/061129_hidden_dim_256/snapshot/50000.pt'
#                 hand = "left"
#             elif task_name == 'place_bottle_left_robot':
#                 model_path = '/home/bobby/Point-Policy/point_policy/exp_local/2025.07.12/point_policy/deterministic/061325_hidden_dim_256/snapshot/50000.pt'
#                 hand = "left"
#             else:
#                 print(f"Unknown task name: {task_name}")
#                 continue

#             if not os.path.exists(model_path):
#                 print(f"Model file not found: {model_path}")
#                 continue

#             # import yaml
#             # info_dict = {
#             #     "task_name": task_name,
#             #     "model_path": model_path,
#             #     "desired_object": des_object
#             # }
#             # with open('/home/bobby/Point-Policy/point_policy/current_info.yaml', 'w') as f:
#             #     yaml.dump(info_dict, f)

#             if eval_process and eval_process.poll() is None:
#                 print("Stopping the existing evaluation process...")
#                 eval_process.terminate()
#                 eval_process.wait()
#                 print("Previous evaluation process terminated.")

#             try:
#                 print(f"Starting eval_point_track.py for task: {task_name}, model: {model_path}")
#                 eval_process = subprocess.Popen(
#                     [
#                         "python",
#                         "eval_point_track.py",
#                         "agent=point_policy",
#                         "suite=point_policy",
#                         "dataloader=point_policy",
#                         "eval=true",
#                         "suite.use_robot_points=true",
#                         "suite.use_object_points=true",
#                         "experiment=eval_point_policy",
#                         f"suite/task/franka_env={task_name}",
#                         f"bc_weight={model_path}",
#                     ]
#                 )
#             except Exception as e:
#                 print(f"Failed to start eval_point_track.py: {e}")
#         else:
#             print("Invalid input. Please enter a task name or 'stop'.")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""Manual task‑launcher for point‑policy evaluation.

Changes compared to the original:
1. Central `TASK_MODELS` dict keeps model‑path + hand info together.
2. The selected hand is forwarded to `eval_point_track.py` via a
   `--hand=` CLI flag *and* the `HAND` environment variable so every
   downstream module sees the same choice.
3. Minor input‑sanity checks and clearer messaging.
"""

import os
import subprocess
import sys
from pathlib import Path

# ───────────────────────── Task ↔ (model, hand) map ──────────────────────────
TASK_MODELS = {
    "open_fridge_door_right_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.24/point_policy/deterministic/093048_hidden_dim_256/snapshot/10000.pt",
        "hand": "right",
        "use_object_point": False,
    },
    "open_the_fridge_door_right_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.30/point_policy/deterministic/010031_hidden_dim_256/snapshot/10000.pt",
        "hand": "right",
        "use_object_point": False,
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
    "pick_bottle_from_fridge_and_place_the_bottle_right_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.30/point_policy/deterministic/030818_hidden_dim_256/snapshot/10000.pt",
        "hand": "right",
        "use_object_point": True,
    },
    "pick_bottle_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.12/point_policy/deterministic/061129_hidden_dim_256/snapshot/50000.pt",
        "hand": "left",
        "use_object_point": True,
    },
    "pick_bottle_from_side_door_of_fridge_and_place_the_bottle_right_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.30/point_policy/deterministic/044206_hidden_dim_256/snapshot/10000.pt",
        "hand": "right",
        "use_object_point": True,
    },
    "place_bottle_left_robot": {
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.12/point_policy/deterministic/061325_hidden_dim_256/snapshot/10000.pt",
        "hand": "left",
        "use_object_point": False,
    },
    "pick_bottle_from_fridge_left_robot": {
        # "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.14/point_policy/deterministic/064254_hidden_dim_256/snapshot/10000.pt",
        # "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.16/point_policy/deterministic/080420_hidden_dim_256/snapshot/10000.pt",
        "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.25/point_policy/deterministic/084458_hidden_dim_256/snapshot/10000.pt",
        "hand": "left",
        "use_object_point": True,
    },
    "place_bottle_from_fridge_left_robot": {
        # "model": "/home/bobby/Point-Policy/point_policy/exp_local/2025.07.26/point_policy/deterministic/081032_hidden_dim_256/snapshot/10000.pt",
        "model": '/home/bobby/Point-Policy/point_policy/exp_local/2025.07.26/point_policy/deterministic/125617_hidden_dim_256/snapshot/10000.pt',
        "hand": "left",
        "use_object_point": False,
    },
}


def launch_eval(task_name: str, model_path: str, hand: str, des_object: str, use_object_point=True):
    """Spawn `eval_point_track.py` with the correct flags/env."""
    if use_object_point:
        use_object = "true"
        if hand == 'left':
            cmd = [
                "python",
                "eval_point_track.py",
                f"--hand={hand}",  # NEW: propagate chosen arm
                "agent=point_policy",
                "suite=point_policy",
                "dataloader=point_policy",
                "eval=true",
                "suite.use_robot_points=true",
                f"suite.use_object_points={use_object}",
                "suite.pixel_keys=[pixels4,pixels6]",
                "suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_46_left_robot.npy",
                "experiment=eval_point_policy",
                'suite.task_make_fn.reset_flag=True',
                f"suite/task/franka_env={task_name}",
                f"bc_weight={model_path}",
            ]
        elif hand == 'right':
            cmd = [
                "python",
                "eval_point_track.py",
                f"--hand={hand}",  # NEW: propagate chosen arm
                "agent=point_policy",
                "suite=point_policy",
                "dataloader=point_policy",
                "eval=true",
                "suite.use_robot_points=true",
                f"suite.use_object_points={use_object}",
                "suite.pixel_keys=[pixels2,pixels5]",
                "suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_25_right_robot.npy",
                "experiment=eval_point_policy",
                'suite.task_make_fn.reset_flag=True',
                f"suite/task/franka_env={task_name}",
                f"bc_weight={model_path}",
            ]
    else:
        use_object = 'false'
        if hand == 'left':
            cmd = [
                "python",
                "eval_point_track.py",
                f"--hand={hand}",  # NEW: propagate chosen arm
                "agent=point_policy",
                "suite=point_policy",
                "dataloader=point_policy",
                "eval=true",
                "suite.use_robot_points=true",
                f"suite.use_object_points={use_object}",
                "suite.pixel_keys=[pixels4,pixels6]",
                "suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_46_left_robot.npy",
                'suite.task_make_fn.points_cfg=null',
                "experiment=eval_point_policy",
                'suite.task_make_fn.reset_flag=True',
                f"suite/task/franka_env={task_name}",
                f"bc_weight={model_path}",
            ]
        elif hand == 'right':
            cmd = [
                "python",
                "eval_point_track.py",
                f"--hand={hand}",  # NEW: propagate chosen arm
                "agent=point_policy",
                "suite=point_policy",
                "dataloader=point_policy",
                "eval=true",
                "suite.use_robot_points=true",
                f"suite.use_object_points={use_object}",
                "suite.pixel_keys=[pixels2,pixels5]",
                "suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_25_right_robot.npy",
                'suite.task_make_fn.points_cfg=null',
                "experiment=eval_point_policy",
                'suite.task_make_fn.reset_flag=True',
                f"suite/task/franka_env={task_name}",
                f"bc_weight={model_path}",
            ]

    # env = {**os.environ, "HAND": hand}  # also expose via env var
    env = {**os.environ, "HAND": hand, "DES_OBJECT": des_object} 
    print("Launching eval with command:\n  " + " ".join(cmd))
    return subprocess.Popen(cmd, env=env)

def launch_reset(hand: str):
    if hand == 'left':
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
            "suite.pixel_keys=[pixels4,pixels6]",
            "suite.task_make_fn.points_cfg=null",
            'suite.task_make_fn.reset_flag=True',
            'suite/task/franka_env=pick_bottle_left_robot',
        ]
    elif hand == 'right':
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
            "suite.pixel_keys=[pixels2,pixels5]",
            "suite.task_make_fn.points_cfg=null",
            'suite.task_make_fn.reset_flag=True',
            'suite/task/franka_env=open_fridge_door_right_robot',
        ]

    env = {**os.environ, "HAND": hand}  # also expose via env var
    print("Launching eval with command:\n  " + " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    eval_process: subprocess.Popen | None = None

    print("Task Manager Initialized")
    print("Enter '<task_name>,<desired_object>' to start evaluation, 'stop' to terminate the ongoing evaluation, or 'reset' (TODO).\n")

    while True:
        try:
            user_input = input("Enter a command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting task manager…")
            if eval_process and eval_process.poll() is None:
                print("Stopping running evaluation before exit…")
                eval_process.terminate()
                eval_process.wait()
            break

        if not user_input:
            print("Please enter a non‑empty command.")
            continue

        # ───────────────────────── handle special commands ─────────────────────────
        if user_input.lower() == "stop":
            if eval_process and eval_process.poll() is None:
                print("Stopping the ongoing evaluation process…")
                eval_process.terminate()
                eval_process.wait()
                print("Evaluation process terminated.")
            else:
                print("No evaluation process is currently running.")
            continue

         # ──────────────────────── reset, left|right ────────────────────────
        if user_input.lower().startswith("reset"):
            try:
                _, hand_raw = [p.strip().lower() for p in user_input.split(",", 1)]
            except ValueError:
                print("reset, left or reset, right")
                continue

            if hand_raw not in ("left", "right"):
                print("only 'left' or 'right', what you typed is:", hand_raw)
                continue

            # 若评估进程仍在运行，先安全中止
            if eval_process and eval_process.poll() is None:
                print("terminating eval process")
                eval_process.terminate()
                eval_process.wait()
                print("terminated")

            print(f"resetting（{hand_raw} hand）……")
            try:
                launch_reset(hand_raw)      # 阻塞等待复位完成
                print("✔ reset done。")
            except subprocess.CalledProcessError as e:
                print(f"reset failed {e.returncode}：{e}")
            continue

        # ───────────────────────── parse task + desired object ─────────────────────
        if "," not in user_input:
            print("Input should be '<task_name>,<desired_object>'. Try again.")
            continue

        task_name, des_object = [p.strip() for p in user_input.split(",", 1)]

        if task_name not in TASK_MODELS:
            print(f"Unknown task name: '{task_name}'. Available: {', '.join(TASK_MODELS)}")
            continue

        model_path = TASK_MODELS[task_name]["model"]
        hand = TASK_MODELS[task_name]["hand"]
        use_object = TASK_MODELS[task_name]["use_object_point"]

        if not Path(model_path).exists():
            print(f"Model file not found: {model_path}")
            continue

        # Stop any existing eval.
        if eval_process and eval_process.poll() is None:
            print("Stopping the existing evaluation process…")
            eval_process.terminate()
            eval_process.wait()
            print("Previous evaluation process terminated.")

        # Launch new evaluation.
        eval_process = launch_eval(task_name, model_path, hand, des_object, use_object)


if __name__ == "__main__":
    main()
