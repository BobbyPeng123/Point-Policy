#!/usr/bin/env python3

import os
import subprocess

def main():
    eval_process = None  # To track the running eval_point_track.py process

    print("Task Manager Initialized")
    print("Enter task name to start evaluation or 'reset' to reset the robot or 'stop' to terminate the ongoing evaluation.")

    while True:
        user_input = input("Enter a task name or 'stop': ").strip()

        if user_input.lower() == "stop":
            if eval_process and eval_process.poll() is None:
                print("Stopping the ongoing evaluation process...")
                eval_process.terminate()
                eval_process.wait()
                print("Evaluation process terminated.")
            else:
                print("No evaluation process is currently running.")
            continue
        
        # # TODO: complete the reset function
        # if user_input.lower() == "reset":
        #     try:
        #         print("Resetting the robot...")
        #         subprocess.run(["python", "reset.py", "suite=xarm_env_reset"])
        #         print("Robot reset successfully.")
        #     except Exception as e:
        #         print(f"Failed to reset the robot: {e}")
        #     continue

        if user_input:
            task_name, des_object = user_input.split(',')[0].strip(), user_input.split(',')[1].strip()

            # Manually map tasks to model paths
            if task_name == '0302_pick_bottle':
                model_path = '/home/bobby/Point-Policy/point_policy/exp_local/2025.03.04/point_policy/deterministic/015859_hidden_dim_256/snapshot/50000.pt'
            else:
                print(f"Unknown task name: {task_name}")
                continue

            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                continue

            import yaml
            info_dict = {
                "task_name": task_name,
                "model_path": model_path,
                "desired_object": des_object
            }
            with open('/home/bobby/Point-Policy/point_policy/current_info.yaml', 'w') as f:
                yaml.dump(info_dict, f)

            if eval_process and eval_process.poll() is None:
                print("Stopping the existing evaluation process...")
                eval_process.terminate()
                eval_process.wait()
                print("Previous evaluation process terminated.")

            try:
                print(f"Starting eval_point_track.py for task: {task_name}, model: {model_path}")
                eval_process = subprocess.Popen(
                    [
                        "python",
                        "eval_point_track.py",
                        "agent=point_policy",
                        "suite=point_policy",
                        "dataloader=point_policy",
                        "eval=true",
                        "suite.use_robot_points=true",
                        "suite.use_object_points=true",
                        "experiment=eval_point_policy",
                        f"suite/task/franka_env={task_name}",
                        f"bc_weight={model_path}",
                    ]
                )
            except Exception as e:
                print(f"Failed to start eval_point_track.py: {e}")
        else:
            print("Invalid input. Please enter a task name or 'stop'.")

if __name__ == "__main__":
    main()
