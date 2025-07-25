#!/usr/bin/env python3

import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import numpy as np
import sys, os, warnings

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

hand = "right"                       # default
for arg in sys.argv:
    if arg.startswith("--hand="):
        hand = arg.split("=", 1)[1]
        sys.argv.remove(arg)         # hide it from Hydra
        break
os.environ["HAND"] = hand


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class Workspace:
    def __init__(self, cfg):
        # import ipdb; ipdb.set_trace()
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )

        try:
            if self.cfg.suite.use_object_points:
                import yaml

                cfg_path = f"{cfg.root_dir}/point_policy/cfgs/suite/points_cfg.yaml"
                with open(cfg_path) as stream:
                    try:
                        points_cfg = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
                    root_dir, dift_path, cotracker_checkpoint = (
                        points_cfg["root_dir"],
                        points_cfg["dift_path"],
                        points_cfg["cotracker_checkpoint"],
                    )
                    points_cfg["dift_path"] = f"{root_dir}/{dift_path}"
                    points_cfg[
                        "cotracker_checkpoint"
                    ] = f"{root_dir}/{cotracker_checkpoint}"
                self.cfg.suite.task_make_fn.points_cfg = points_cfg
        except:
            pass

        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

@hydra.main(config_path="cfgs", config_name="config_eval")
def main(cfg):
    workspace = Workspace(cfg)


if __name__ == "__main__":
    main()