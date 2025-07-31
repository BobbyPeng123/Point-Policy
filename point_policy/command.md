python convert_to_pkl_robot.py --data_dir /home/bobby/data --calib_path /home/bobby/Point-Policy/calib/calib_46_left_robot.npy --task_names pick_bottle_from_fridge_left_robot

python convert_to_pkl_robot_vlm.py --data_dir /home/bobby/data --calib_path /home/bobby/Point-Policy/calib/calib_25_right_robot.npy --task_names open_the_fridge_door_right_robot

HYDRA_FULL_ERROR=1 python train.py agent=point_policy suite=point_policy dataloader=point_policy eval=false suite.use_robot_points=true suite.use_object_points=false suite/task/franka_env=open_the_fridge_door_right_robot suite.task_make_fn.points_cfg=null experiment=point_policy

python train.py agent=point_policy suite=point_policy dataloader=point_policy eval=false suite.use_robot_points=true suite.use_object_points=true suite/task/franka_env=pick_bottle_from_fridge_and_place_the_bottle_right_robot experiment=point_policy

python eval_point_track.py --hand=right agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=false suite.task_make_fn.points_cfg=null experiment=eval_point_policy suite.task_make_fn.reset_flag=True suite/task/franka_env=open_fridge_door_right_robot bc_weight=/home/bobby/Point-Policy/point_policy/exp_local/2025.07.24/point_policy/deterministic/093048_hidden_dim_256/snapshot/10000.pt