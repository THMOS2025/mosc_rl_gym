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
import hydra
from humanoid.envs import *
from humanoid.utils import get_args, task_registry

def print_config_simple(obj, name="config", indent=0):
    """
    简化版配置打印
    """
    if indent == 0:
        print("=" * 60)
        print(f"{name} 结构:")
        print("=" * 60)
    
    prefix = "  " * indent
    print(f"{prefix}{name}: {type(obj).__name__}")
    
    # 获取所有非私有、非方法属性
    for attr_name in sorted(dir(obj)):
        if not attr_name.startswith('_'):
            try:
                attr_value = getattr(obj, attr_name)
                if not callable(attr_value):
                    # 判断是否需要递归
                    if (hasattr(attr_value, '__dict__') or 
                        isinstance(attr_value, type) or
                        (hasattr(attr_value, '__module__') and 'config' in str(attr_value.__module__).lower())):
                        # 递归打印
                        print_config_simple(attr_value, attr_name, indent + 1)
                    else:
                        # 直接打印值
                        if isinstance(attr_value, (list, tuple)) and len(attr_value) > 5:
                            display_value = f"[{attr_value[0]}, ..., {attr_value[-1]}] (长度: {len(attr_value)})"
                        else:
                            display_value = str(attr_value)
                        print(f"{prefix}  {attr_name}: {display_value}")
            except:
                continue
    
    if indent == 0:
        print("=" * 60)


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "cfg")
@hydra.main(config_path=config_path, config_name="config")
def main(cfg):
    env, env_cfg = task_registry.make_env_hydra(name=cfg.task, hydra_cfg=cfg)
    print_config_simple(env_cfg, "env_cfg")
    ppo_runner, train_cfg = task_registry.make_alg_runner_hydra(env=env, name=cfg.task, hydra_cfg=cfg)
    print_config_simple(train_cfg, "train_cfg")
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    
if __name__ == '__main__':
    # args = get_args()
    # train(args)
    main()
