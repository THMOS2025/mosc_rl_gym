# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import os
from typing import Tuple
from datetime import datetime
from isaacgym import gymapi
from isaacgym import gymutil

from humanoid.algo import VecEnv
from humanoid.algo import OnPolicyRunner

from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from humanoid.envs.custom.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

from omegaconf import DictConfig
from omegaconf import OmegaConf

def recursive_override_class_cfg(class_cfg, override_dict, path=""):
    # 将 DictConfig 转换为普通字典
    if hasattr(override_dict, '_content'):
        override_dict = OmegaConf.to_container(override_dict, resolve=True)
    
    for k, v in override_dict.items():
        current_path = f"{path}.{k}" if path else k
        
        if isinstance(v, dict):
            if hasattr(class_cfg, k):
                target_obj = getattr(class_cfg, k)
                recursive_override_class_cfg(target_obj, v, current_path)
        else:
            if hasattr(class_cfg, k):
                setattr(class_cfg, k, v)
            
class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        print("making env...")
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        self.env_cfg_for_wandb = env_cfg
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = class_to_dict(train_cfg)
        env_cfg_dict = class_to_dict(self.env_cfg_for_wandb)
        all_cfg = {**train_cfg_dict, **env_cfg_dict}
        
        runner_class = eval(train_cfg_dict["runner_class_name"])
        runner = runner_class(env, all_cfg, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path, load_optimizer=False)
        return runner, train_cfg




    def build_default_args(self, args, hydra_cfg):
        # 基本参数
        args.pipeline = "gpu"
        args.graphics_device_id = 0

        # 物理引擎互斥组
        args.flex = False
        args.physx = True

        args.num_threads = 0
        args.subscenes = 0
        args.slices = None

        # 解析 sim_device
        args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(hydra_cfg.sim_device)
        pipeline = args.pipeline.lower()

        assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), \
            f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
        args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

        # 逻辑处理
        if args.sim_device_type != 'cuda' and args.flex:
            print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
            args.sim_device = 'cuda:0'
            args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(hydra_cfg.sim_device)

        if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
            print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
            args.pipeline = 'CPU'
            args.use_gpu_pipeline = False

        # 默认物理引擎
        args.physics_engine = gymapi.SIM_PHYSX
        args.use_gpu = (args.sim_device_type == 'cuda')


        if args.slices is None:
            args.slices = args.subscenes

        return args


    def make_env_hydra(self, name, hydra_cfg=None, env_cfg=None):
        """
        hydra_cfg: DictConfig或dict，优先级高于env_cfg
        env_cfg: class-based config，如 LeggedRobotCfg
        """
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)  # class-based默认config
        recursive_override_class_cfg(env_cfg, hydra_cfg)
        # import pdb;pdb.set_trace()
        set_seed(env_cfg.seed)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}

        env_cfg = self.build_default_args(env_cfg, hydra_cfg)
            
        
        sim_params = parse_sim_params(env_cfg, sim_params)
        # import pdb;pdb.set_trace()
        # 4. 实例化环境
        env = self.get_task_class(name)(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device=getattr(hydra_cfg, "sim_device", "cuda:0") if hydra_cfg else "cuda:0",
            headless=getattr(hydra_cfg, "headless", False) if hydra_cfg else False
        )
        self.env_cfg_for_wandb = env_cfg
        return env, env_cfg
    
    def make_alg_runner_hydra(self, env, name=None, hydra_cfg=None, train_cfg=None, log_root="default"):
        """
        hydra_cfg: DictConfig或dict，优先级高于train_cfg
        train_cfg: class-based config，如 LeggedRobotCfgPPO
        """
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _, train_cfg = self.get_cfgs(name)  # class-based默认train_cfg
        recursive_override_class_cfg(train_cfg, hydra_cfg)
        # 日志路径
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        # 初始化Runner
        train_cfg_dict = class_to_dict(train_cfg)
        env_cfg_dict = class_to_dict(self.env_cfg_for_wandb)
        all_cfg = {**train_cfg_dict, **env_cfg_dict}
        runner_class = eval(train_cfg_dict["runner_class_name"])
        rl_device = getattr(hydra_cfg, "rl_device", "cuda:0") if hydra_cfg else "cuda:0"
        runner = runner_class(env, all_cfg, log_dir, device=rl_device)
        # resume模型
        resume = train_cfg.runner.resume
        if resume:
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path, load_optimizer=False)
        return runner, train_cfg
    
    
    
# make global task registry
task_registry = TaskRegistry()
