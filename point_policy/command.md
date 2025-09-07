python process_data_robot.py --data_dir /home/bobby/data --task_names put_bowl_into_basket_new_left_robot

python convert_to_pkl_robot.py --data_dir /home/bobby/data --calib_path /home/bobby/Point-Policy/calib/calib_46_left_robot.npy --task_names put_bowl_into_basket_small_left_robot

python convert_to_pkl_robot_vlm.py --data_dir /home/bobby/data --calib_path /home/bobby/Point-Policy/calib/calib_46_right_robot.npy --task_names open_the_fridge_door_right_robot

python convert_to_pkl_robot_target_point.py --data_dir /home/bobby/data --calib_path /home/bobby/Point-Policy/calib/calib_46_left_robot.npy --task_names place_bottle_from_the_fridge_left_robot

HYDRA_FULL_ERROR=1 python train.py agent=point_policy suite=point_policy dataloader=point_policy eval=false suite.use_robot_points=true suite.use_object_points=false suite/task/franka_env=open_the_fridge_door_right_robot suite.task_make_fn.points_cfg=null experiment=point_policy

python train.py agent=point_policy suite=point_policy dataloader=point_policy eval=false suite.use_robot_points=true suite.use_object_points=true suite/task/franka_env=put_bowl_into_oven_left_robot experiment=point_policy

python eval_point_track.py --hand=right agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=false suite.task_make_fn.points_cfg=null experiment=eval_point_policy suite.task_make_fn.reset_flag=True suite/task/franka_env=open_fridge_door_right_robot bc_weight=/home/bobby/Point-Policy/point_policy/exp_local/2025.07.24/point_policy/deterministic/093048_hidden_dim_256/snapshot/10000.pt

HYDRA_FULL_ERROR=1 python eval_point_track.py --hand=right agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=false suite.pixel_keys=[pixels2,pixels5] suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_25_right_robot.npy suite.task_make_fn.points_cfg=null experiment=eval_point_policy suite.task_make_fn.reset_flag=True suite/task/franka_env=test bc_weight=/home/bobby/Point-Policy/point_policy/exp_local/2025.08.04/point_policy/deterministic/061924_hidden_dim_256/snapshot/10000.pt

python eval_point_track.py --hand=right agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=true suite.use_pin_points=true suite.num_object_points=1 suite.pixel_keys=[pixels2,pixels5] suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_25_right_robot.npy suite.task_make_fn.points_cfg=null experiment=eval_point_policy suite.task_make_fn.reset_flag=False suite/task/franka_env=test2 bc_weight=/home/bobby/Point-Policy/point_policy/exp_local/2025.08.04/point_policy/deterministic/055732_hidden_dim_256/snapshot/10000.pt

CUDA_VISIBLE_DEVICES=2 python eval_point_track.py --hand=left agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=true suite.pixel_keys=[pixels4,pixels6] suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_46_left_robot.npy experiment=eval_point_policy suite.task_make_fn.reset_flag=True suite/task/franka_env=pick_bottle_from_the_fridge_left_robot bc_weight=/home/bobby/Point-Policy/point_policy/exp_local/2025.08.15/point_policy/deterministic/222054_hidden_dim_256/snapshot/15000.pt

python eval_point_track.py --hand=left agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=true suite.use_pin_points=true suite.num_object_points=1 suite.pixel_keys=[pixels4,pixels6] suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_46_left_robot.npy suite.task_make_fn.points_cfg=null experiment=eval_point_policy suite.task_make_fn.reset_flag=False suite/task/franka_env=place_bottle_from_the_fridge_left_robot bc_weight=/home/bobby/Point-Policy/point_policy/exp_local/2025.08.16/point_policy/deterministic/100700_hidden_dim_256/snapshot/5000.pt

python train.py agent=point_policy suite=point_policy dataloader=point_policy eval=false suite.use_robot_points=true suite.use_object_points=true suite.use_pin_points=true suite.num_object_points=1 suite.task_make_fn.points_cfg=null suite/task/franka_env=place_bottle_from_the_fridge_left_robot experiment=point_policy

/home/bobby/Point-Policy/point_policy/exp_local/2025.08.16/point_policy/deterministic/100700_hidden_dim_256/snapshot/5000.pt 
python eval_point_track.py --hand=left agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=true suite.use_pin_points=true suite.num_object_points=1 suite.pixel_keys=[pixels4,pixels6] suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_46_left_robot.npy suite.task_make_fn.points_cfg=null experiment=eval_point_policy suite.task_make_fn.reset_flag=False suite/task/franka_env=place_bottle_from_the_fridge_left_robot bc_weight=/home/bobby/Point-Policy/point_policy/exp_local/2025.08.16/point_policy/deterministic/115405_hidden_dim_256/snapshot/5000.pt  right place work

CUDA_VISIBLE_DEVICES=2 python eval_point_track.py --hand=left agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=true suite.pixel_keys=[pixels4,pixels6] suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_46_left_robot.npy experiment=eval_point_policy suite.task_make_fn.reset_flag=True suite/task/franka_env=put_bowl_into_oven_left_robot bc_weight=/home/bobby/Point-Policy/point_policy/exp_local/2025.08.23/point_policy/deterministic/000516_hidden_dim_256/snapshot/10000.pt

python eval_point_track.py --hand=left agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=true suite.use_pin_points=true suite.num_object_points=1 suite.pixel_keys=[pixels4,pixels6] suite.task_make_fn.calib_path=/home/bobby/Point-Policy/calib/calib_46_left_robot.npy suite.task_make_fn.points_cfg=null experiment=eval_point_policy suite.task_make_fn.reset_flag=False suite/task/franka_env=place_bottle_from_the_fridge_left_pos_left_robot bc_weight=/home/bobby/Point-Policy/point_policy/exp_local/2025.08.16/point_policy/deterministic/121910_hidden_dim_256/snapshot/5000.pt

scp /home/bobby/Point-Policy/point_policy/robot_utils/franka/output_image.jpg aadhithya@100.96.11.47:/home/aadhithya/bobby_wks/

python train.py agent=baku suite=baku dataloader=baku eval=false suite/task/franka_env=put_bowl_into_basket_small_left_robot suite.gt_depth=false experiment=baku

docker exec -it openpi-dev bash

uv run scripts/train.py pi0_libero_low_mem_finetune   --exp-name=libero_pi0_lora_v1 --overwrite   --num_workers=16 --batch_size=48 --save_interval=500

tmux new -s openpi

tmux attach -t openpi