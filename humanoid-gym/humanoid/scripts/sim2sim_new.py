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
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
try:
    from humanoid import LEGGED_GYM_ROOT_DIR
except ImportError:
    # 如果您没有一个名为humanoid的模块，或者LEGGED_GYM_ROOT_DIR没有在其中定义
    # 您需要在这里手动设置LEGGED_GYM_ROOT_DIR的路径
    # 例如: LEGGED_GYM_ROOT_DIR = "/path/to/your/legged_gym"
    # 为了代码能顺利运行，我们暂时将其设置为当前目录
    print("Warning: 'humanoid' module not found. Setting 'LEGGED_GYM_ROOT_DIR' to the current working directory.")
    LEGGED_GYM_ROOT_DIR = os.getcwd()
import torch
from datetime import datetime
import time
import argparse

USD_JOINT_NAMES = ['b_Lh','Lh_Ll','Ll_Ll1','Ll1_Ll2','Ll2_La','La_Lf', 
                   'b_Rh','Rh_Rl','Rl_Rl1','Rl1_Rl2','Rl2_Ra','Ra_Rf']

def euler_to_quaternion(rpy):
    """
    将欧拉角 (roll, pitch, yaw) 转换为四元数 (w, x, y, z).
    """
    roll, pitch, yaw = rpy
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data,cfg):
    '''从MuJoCo数据结构中提取观测值'''
    name_list = USD_JOINT_NAMES
    q = np.zeros((cfg.env.num_actions), dtype=np.double)
    dq = np.zeros((cfg.env.num_actions), dtype=np.double)
    for i in range(cfg.env.num_actions):
        q[i] = data.sensor(name_list[i] + '_joint_p').data.astype(np.double)
        dq[i] = data.sensor(name_list[i] + '_joint_v').data.astype(np.double)

    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v_base = data.sensor('linear-velocity').data.astype(np.double)
    omega_base = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, omega_base, quat, v_base)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''从位置指令计算力矩'''
    return (target_q - q) * kp + (target_dq - dq) * kd

def load_data_from_log(log_dir):
    """从日志文件中加载数据"""
    joint_pos_path = os.path.join(log_dir, 'joint_pos.txt')
    joint_vel_path = os.path.join(log_dir, 'joint_vel.txt')
    imu_path = os.path.join(log_dir, 'base_ang_eul.txt')
    joint_act_path = os.path.join(log_dir, 'joint_act.txt')
    # 新增: 定义 input_imu.txt 的路径
    imu_ang_vel_path = os.path.join(log_dir, 'input_imu.txt')

    joint_pos = np.loadtxt(joint_pos_path, delimiter=',') if os.path.exists(joint_pos_path) else None
    joint_vel = np.loadtxt(joint_vel_path, delimiter=',') if os.path.exists(joint_vel_path) else None
    imu_rpy = np.loadtxt(imu_path, delimiter=',') if os.path.exists(imu_path) else None
    joint_act = np.loadtxt(joint_act_path, delimiter=',') if os.path.exists(joint_act_path) else None
    # 新增: 加载角速度数据
    imu_ang_vel = np.loadtxt(imu_ang_vel_path, delimiter=',') if os.path.exists(imu_ang_vel_path) else None
    
    # 新增: 返回角速度数据
    return joint_pos, joint_vel, imu_rpy, joint_act, imu_ang_vel

def run_mujoco(policy, cfg, mode='policy', log_dir=None, disable_gravity=False):
    """
    使用提供的策略和配置运行Mujoco仿真。

    Args:
        policy: 用于控制仿真的策略。
        cfg: 包含仿真设置的配置对象。
        mode: 'policy', 'observation', 'action', or 'replay'
        log_dir: 包含日志文件的目录路径
        disable_gravity (bool): 如果为True，则禁用仿真中的重力。
    """
    # --- 关键时间参数说明 ---
    # 物理仿真步长: dt = 0.001s
    # 高层控制/日志记录降采样率: decimation = 20
    # 因此，高层控制命令的更新周期以及日志数据的记录间隔为:
    # high_level_interval = dt * decimation = 0.001s * 20 = 0.02s (50Hz)
    # 这与从文件读取的数据序列间隔相匹配。
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt

    if disable_gravity:
        model.opt.gravity[:] = 0
        print("Gravity has been DISABLED for this simulation.")
    else:
        print(f"Gravity is ENABLED. Default value: {model.opt.gravity}")
    
    data = mujoco.MjData(model)

    # 新增: 为 imu_ang_vel_log 初始化
    joint_pos_log, joint_vel_log, imu_rpy_log, joint_act_log, imu_ang_vel_log = None, None, None, None, None
    if log_dir:
        # 新增: 接收返回的角速度数据
        joint_pos_log, joint_vel_log, imu_rpy_log, joint_act_log, imu_ang_vel_log = load_data_from_log(log_dir)

    # 新增: 在 'observation' 模式下检查 imu_ang_vel_log 是否存在
    if mode == 'observation' and (joint_pos_log is None or joint_vel_log is None or imu_rpy_log is None or imu_ang_vel_log is None):
        raise ValueError("在 'observation' 模式下, 'joint_pos.txt', 'joint_vel.txt', 'base_ang_eul.txt' 和 'input_imu.txt' 必须存在。")
    if mode == 'action' and joint_act_log is None:
        raise ValueError("在 'action' 模式下, 'joint_act.txt' 必须存在。")
    if mode == 'replay' and (joint_pos_log is None or imu_rpy_log is None):
        raise ValueError("在 'replay' 模式下, 'joint_pos.txt' 和 'base_ang_eul.txt' 必须存在。")

    for joint_name, value in cfg.robot_config.init_joint_pos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            qpos_index = model.jnt_qposadr[joint_id]
            data.qpos[qpos_index] = value
            
    action_offset = np.array([cfg.robot_config.init_joint_pos.get(joint, 0.0) for joint in USD_JOINT_NAMES])
    mujoco.mj_forward(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_target_q = np.zeros_like(target_q)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    log_idx = 0

    if mode in ['observation', 'action', 'replay']:
        # 新增: 将 imu_ang_vel_log 添加到 valid_logs 列表
        valid_logs = [log for log in [joint_pos_log, joint_vel_log, imu_rpy_log, joint_act_log, imu_ang_vel_log] if log is not None]
        if not valid_logs:
            print(f"警告: 在 '{mode}' 模式下没有提供任何日志文件。仿真将不会运行。")
            num_steps = 0
        else:
            num_steps = min(len(log) for log in valid_logs)
        sim_duration_steps = num_steps * cfg.sim_config.decimation
    else:
        num_steps = 1000
        sim_duration_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)

    for _step in tqdm(range(sim_duration_steps), desc=f"Simulating in {mode} mode..."):
        # --- Replay 模式逻辑 ---
        if mode == 'replay':
            # 在每个高层控制周期（0.02s）的开始，更新机器人的状态
            if _step % cfg.sim_config.decimation == 0:
                if log_idx < len(joint_pos_log):
                    # 1. 设置关节位置
                    for i, joint_name in enumerate(cfg.robot_config.name_list):
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        if joint_id != -1:
                            qpos_adr = model.jnt_qposadr[joint_id]
                            data.qpos[qpos_adr] = joint_pos_log[log_idx, i]
                    
                    # 2. 设置基座方向
                    if log_idx < len(imu_rpy_log):
                        # 日志文件提供的是世界坐标系下的RPY（Roll, Pitch, Yaw）欧拉角
                        rpy = imu_rpy_log[log_idx]
                        # 将RPY欧拉角转换为MuJoCo所需的[w, x, y, z]格式的四元数
                        quat = euler_to_quaternion(rpy)
                        # 设置基座姿态
                        data.qpos[0:4] = quat

                    mujoco.mj_forward(model, data)
                    log_idx += 1
            
            viewer.render()
            # 减慢回放速度以匹配数据的真实时间间隔(0.02s)
            # 这里在每个物理仿真步长后都稍作停顿，以获得平滑的视觉效果
            time.sleep(cfg.sim_config.dt) 
            continue

        # --- Policy, Observation, Action 模式逻辑 ---
        if mode == 'observation' and log_idx < len(joint_pos_log):
            q = joint_pos_log[log_idx]
            dq = joint_vel_log[log_idx]
            eu_ang = imu_rpy_log[log_idx]
            # 修改: 从加载的日志文件中读取 omega_base
            omega_base = imu_ang_vel_log[log_idx]
            quat = euler_to_quaternion(eu_ang)
        else:
            q, dq, omega_base, quat, v_base = get_obs(data, cfg)
        
        q = q[-cfg.env.num_actions:] 
        dq = dq[-cfg.env.num_actions:]

        # 高层控制循环，频率为 1 / (dt * decimation) = 50Hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            if mode != 'observation':
                 eu_ang = quaternion_to_euler_array(quat)
                 eu_ang[eu_ang > math.pi] -= 2 * math.pi

            # ------------------- 策略输入构建 -------------------
            obs_parts = []
            obs_parts.append(np.array([math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)])) 
            obs_parts.append(np.array([math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)]))
            obs_parts.append(np.array([cmd.vx]))
            obs_parts.append(np.array([cmd.vy]))
            obs_parts.append(np.array([cmd.az]))
            obs_parts.append((- q - action_offset) * cfg.normalization.obs_scales.dof_pos)
            obs_parts.append(- dq * cfg.normalization.obs_scales.dof_vel)
            obs_parts.append(action) # 上一个动作
            obs_parts.append(omega_base)
            obs_parts.append(eu_ang) 
            obs = np.expand_dims(np.concatenate(obs_parts, axis=-1), axis=0).astype(np.float32)

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            
            # ------------------- 动作计算 -------------------
            if mode == 'action':
                if log_idx < len(joint_act_log):
                    target_q = joint_act_log[log_idx]
                else:
                    target_q = action_offset
            else: # policy 和 observation 模式
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)  
                target_q = action * cfg.control.action_scale + action_offset
            
            alpha = 0.8
            if _step > 0:
                target_q = alpha * target_q + (1 - alpha) * last_target_q
            last_target_q = target_q.copy()
            
            # 如果使用日志文件，则增加索引
            if log_dir and log_idx < num_steps:
                log_idx += 1

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
                    
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)

        data.ctrl = tau
                        
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
    viewer.close()

class cmd:
    vx = 0.0
    vy = 0.0
    az = 0.0

class Sim2simCfg():
    class env:
        frame_stack = 5
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        num_actions = 12
        
    class normalization:
        class obs_scales:
            lin_vel = 2. 
            ang_vel = 1. 
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
        clip_observations = 18.
        clip_actions = 18.
        
    class sim_config:
        mujoco_model_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/robots/MOSC0516/MOSC_rotaion_test.xml')
        sim_duration = 20.0
        dt = 0.001
        decimation = 20

    class rewards:
        cycle_time = 0.8
    
    class robot_config:
        name_list = USD_JOINT_NAMES
        kps = np.array([100.0, 100.0,100.0, 100.0, 50.0, 24.0,
                        100.0, 100.0,100.0, 100.0, 50.0, 24.0], dtype=np.double) 
        kds = np.array([2.0, 2.0, 2.0, 2.0,1.5,0.3,
                        2.0, 2.0, 2.0, 2.0,1.5,0.3], dtype=np.double) 
        init_joint_pos = {
            "b_Lh": 0.3, "Lh_Ll": 0.0, "Ll_Ll1": 0.0, "Ll1_Ll2": -0.6, "Ll2_La": 0.3, "La_Lf": 0.0,
            "b_Rh": -0.3, "Rh_Rl": 0.0, "Rl_Rl1": 0.0, "Rl1_Rl2": 0.6, "Rl2_Ra": -0.3, "Ra_Rf": 0.0,
        }
        tau_limit = np.array([60.0] * 10 + [10.0] * 2, dtype=np.double)
        
    class control:
        action_scale = 0.25

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--run_name', type=str, required=True, help='Run to load from.')
    parser.add_argument('--mode', type=str, default='policy', choices=['policy', 'observation', 'action', 'replay'], help='Simulation mode.')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory containing the log files for observation, action, or replay modes.')
    parser.add_argument('--disable_gravity', action='store_true', help='Disable gravity in the simulation.')
    
    args = parser.parse_args()
    
    model_path = Sim2simCfg.sim_config.mujoco_model_path
    if not os.path.exists(model_path):
        print(f"错误: MuJoCo 模型文件未找到: {model_path}")
        print("请确保 'LEGGED_GYM_ROOT_DIR' 环境变量已正确设置, 或者在脚本中手动修改路径。")
        exit()

    policy = None
    if args.mode in ['policy', 'observation']:
        policy_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/MOSC/exported/policies/policy_" + args.run_name + ".pt")
        if not os.path.exists(policy_path):
            print(f"错误: 策略文件未找到: {policy_path}")
            print("请确保策略文件存在于正确的路径。")
            exit()
            
        policy = torch.jit.load(policy_path)
        print(f"从 {policy_path} 加载策略")

    run_mujoco(policy, Sim2simCfg(), mode=args.mode, log_dir=args.log_dir, disable_gravity=args.disable_gravity)