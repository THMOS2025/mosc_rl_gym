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
    print("警告: 'humanoid' 模块未找到。将 'LEGGED_GYM_ROOT_DIR' 设置为当前工作目录。")
    LEGGED_GYM_ROOT_DIR = os.getcwd()
import torch
from datetime import datetime
import time
import argparse

USD_JOINT_NAMES = ['b_Lh','Lh_Ll','Ll_Ll1','Ll1_Ll2','Ll2_La','La_Lf', 
                   'b_Rh','Rh_Rl','Rl_Rl1','Rl1_Rl2','Rl2_Ra','Ra_Rf']

# --- 核心功能函数 ---

def euler_to_quaternion(rpy):
    """将欧拉角 (roll, pitch, yaw) 转换为四元数 (w, x, y, z)."""
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
    """将四元数 (x, y, z, w) 转换为欧拉角 (roll, pitch, yaw)"""
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
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
    imu_path = os.path.join(log_dir, 'input_imu.txt')
    joint_act_path = os.path.join(log_dir, 'joint_act.txt')
    joint_pos = np.loadtxt(joint_pos_path, delimiter=',') if os.path.exists(joint_pos_path) else None
    joint_vel = np.loadtxt(joint_vel_path, delimiter=',') if os.path.exists(joint_vel_path) else None
    imu_rpy = np.loadtxt(imu_path, delimiter=',') if os.path.exists(imu_path) else None
    joint_act = np.loadtxt(joint_act_path, delimiter=',') if os.path.exists(joint_act_path) else None
    return joint_pos, joint_vel, imu_rpy, joint_act

# --- 新增：用于生成验证日志的函数 ---
def generate_logs_for_spin(cfg, output_dir="spin_logs", duration=10.0, total_spin_revolutions=1.0):
    """为原地旋转运动生成日志文件。"""
    print(f"正在于目录 '{output_dir}' 中生成用于验证的日志文件...")
    os.makedirs(output_dir, exist_ok=True)
    high_level_interval = cfg.sim_config.dt * cfg.sim_config.decimation
    num_log_steps = int(duration / high_level_interval)
    num_actions = cfg.env.num_actions
    
    # 1. 关节位置数据 (保持静止)
    static_joint_pos = np.array([cfg.robot_config.init_joint_pos[name] for name in cfg.robot_config.name_list])
    joint_pos_data = np.tile(static_joint_pos, (num_log_steps, 1))
    
    # 2. IMU 数据 (生成旋转)
    imu_rpy_data = np.zeros((num_log_steps, 3))
    total_angle = total_spin_revolutions * 2 * math.pi
    for i in range(num_log_steps):
        yaw_angle = (i / num_log_steps) * total_angle
        imu_rpy_data[i, :] = [0.0, 0.0, yaw_angle] # Roll, Pitch, Yaw
        
    # 3. 虚拟的速度和动作数据
    joint_vel_data = np.zeros((num_log_steps, num_actions))
    joint_act_data = np.zeros((num_log_steps, num_actions))
    
    # 4. 保存文件
    np.savetxt(os.path.join(output_dir, 'joint_pos.txt'), joint_pos_data, delimiter=',')
    np.savetxt(os.path.join(output_dir, 'input_imu.txt'), imu_rpy_data, delimiter=',')
    np.savetxt(os.path.join(output_dir, 'joint_vel.txt'), joint_vel_data, delimiter=',')
    np.savetxt(os.path.join(output_dir, 'joint_act.txt'), joint_act_data, delimiter=',')
    print("日志文件生成成功！")

# --- 主要仿真运行函数 ---
def run_mujoco(policy, cfg, mode='policy', log_dir=None, disable_gravity=False):
    """使用提供的策略和配置运行Mujoco仿真。"""
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    if disable_gravity:
        model.opt.gravity[:] = 0
        print("重力已在此次仿真中被禁用。")
    
    data = mujoco.MjData(model)
    joint_pos_log, joint_vel_log, imu_rpy_log, joint_act_log = None, None, None, None
    if log_dir:
        joint_pos_log, joint_vel_log, imu_rpy_log, joint_act_log = load_data_from_log(log_dir)

    # 检查日志文件是否存在
    if mode == 'observation' and (joint_pos_log is None or joint_vel_log is None or imu_rpy_log is None):
        raise ValueError("在 'observation' 模式下, 'joint_pos.txt', 'joint_vel.txt' 和 'input_imu.txt' 必须存在。")
    if mode == 'action' and joint_act_log is None:
        raise ValueError("在 'action' 模式下, 'joint_act.txt' 必须存在。")
    if mode == 'replay' and (joint_pos_log is None or imu_rpy_log is None):
        raise ValueError(f"在 'replay' 模式下, 'joint_pos.txt' 和 'input_imu.txt' 必须在目录 '{log_dir}' 中存在。")

    # 设置初始关节位置
    for joint_name, value in cfg.robot_config.init_joint_pos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            data.qpos[model.jnt_qposadr[joint_id]] = value
            
    action_offset = np.array([cfg.robot_config.init_joint_pos.get(joint, 0.0) for joint in USD_JOINT_NAMES])
    mujoco.mj_forward(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # 初始化变量
    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_target_q = np.zeros_like(target_q)
    hist_obs = deque(maxlen=cfg.env.frame_stack)
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    log_idx = 0

    # 确定仿真步数
    if mode in ['observation', 'action', 'replay']:
        num_steps = min(len(log) for log in [joint_pos_log, imu_rpy_log] if log is not None)
        sim_duration_steps = num_steps * cfg.sim_config.decimation
    else:
        sim_duration_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)

    # --- 主循环 ---
    for _step in tqdm(range(sim_duration_steps), desc=f"Simulating in {mode} mode..."):
        # --- Replay 模式逻辑 ---
        if mode == 'replay':
            if _step % cfg.sim_config.decimation == 0 and log_idx < len(joint_pos_log):
                # 1. 设置关节位置
                for i, joint_name in enumerate(cfg.robot_config.name_list):
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id != -1:
                        data.qpos[model.jnt_qposadr[joint_id]] = joint_pos_log[log_idx, i]
                # 2. 设置基座方向
                if log_idx < len(imu_rpy_log):
                    rpy = imu_rpy_log[log_idx]
                    quat = euler_to_quaternion(rpy)
                    data.qpos[0:4] = quat
                mujoco.mj_forward(model, data)
                log_idx += 1
            viewer.render()
            time.sleep(cfg.sim_config.dt) 
            continue

        # --- Policy, Observation, Action 模式逻辑 ---
        if mode == 'observation' and log_idx < len(joint_pos_log):
            q, dq, eu_ang = joint_pos_log[log_idx], joint_vel_log[log_idx], imu_rpy_log[log_idx]
            omega_base = np.zeros(3)
        else:
            q, dq, omega_base, quat, v_base = get_obs(data, cfg)
        
        q, dq = q[-cfg.env.num_actions:], dq[-cfg.env.num_actions:]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            if mode != 'observation':
                 eu_ang = quaternion_to_euler_array(quat)
                 eu_ang[eu_ang > math.pi] -= 2 * math.pi
            
            # 构建策略输入
            obs_parts = [
                np.array([math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)]),
                np.array([math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)]),
                np.array([cmd.vx, cmd.vy, cmd.az]),
                (q - action_offset) * cfg.normalization.obs_scales.dof_pos,
                dq * cfg.normalization.obs_scales.dof_vel,
                action, omega_base, eu_ang
            ]
            obs = np.expand_dims(np.concatenate(obs_parts, axis=-1), axis=0).astype(np.float32)
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            hist_obs.append(obs)
            policy_input = np.concatenate(list(hist_obs), axis=-1).astype(np.float32)
            
            # 计算动作
            if mode == 'action':
                target_q = joint_act_log[log_idx] if log_idx < len(joint_act_log) else action_offset
            else:
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)  
                target_q = action * cfg.control.action_scale + action_offset
            
            # 平滑动作
            alpha = 0.8
            target_q = alpha * target_q + (1 - alpha) * last_target_q if _step > 0 else target_q
            last_target_q = target_q.copy()
            
            if log_dir and log_idx < num_steps: log_idx += 1

        # PD 控制
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        data.ctrl = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
                        
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
    viewer.close()

# --- 配置和主执行逻辑 ---
class cmd:
    vx, vy, az = 0.3, 0.0, 0.0

class Sim2simCfg():
    class env:
        frame_stack, num_single_obs, num_actions = 5, 47, 12
        num_observations = int(frame_stack * num_single_obs)
    class normalization:
        class obs_scales:
            lin_vel, ang_vel, dof_pos, dof_vel, quat = 2., 1., 1., 0.05, 1.
        clip_observations, clip_actions = 18., 18.
    class sim_config:
        mujoco_model_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/robots/MOSC0516/MOSC_OL_up_hand.xml')
        sim_duration, dt, decimation = 20.0, 0.001, 20
    class rewards:
        cycle_time = 0.8
    class robot_config:
        name_list = USD_JOINT_NAMES
        kps = np.array([100, 100, 100, 100, 50, 24, 100, 100, 100, 100, 50, 24], dtype=np.double) 
        kds = np.array([2.0, 2.0, 2.0, 2.0, 1.5, 0.3, 2.0, 2.0, 2.0, 2.0, 1.5, 0.3], dtype=np.double) 
        init_joint_pos = {
            "b_Lh": 0.3, "Lh_Ll": 0.0, "Ll_Ll1": 0.0, "Ll1_Ll2": -0.6, "Ll2_La": 0.3, "La_Lf": 0.0,
            "b_Rh": -0.3, "Rh_Rl": 0.0, "Rl_Rl1": 0.0, "Rl1_Rl2": 0.6, "Rl2_Ra": -0.3, "Ra_Rf": 0.0,
        }
        tau_limit = np.array([60.0] * 10 + [10.0] * 2, dtype=np.double)
    class control:
        action_scale = 0.25

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='机器人仿真与验证脚本')
    parser.add_argument('--run_name', type=str, help='要加载的策略运行名称。')
    parser.add_argument('--mode', type=str, default='policy', choices=['policy', 'observation', 'action', 'replay'], help='仿真模式。')
    parser.add_argument('--log_dir', type=str, default=None, help='包含日志文件的目录。')
    parser.add_argument('--disable_gravity', action='store_true', help='在仿真中禁用重力。')
    # --- 新增的验证参数 ---
    parser.add_argument('--verify_replay_spin', action='store_true', help='生成自转日志并立即在replay模式下运行以进行验证。')
    
    args = parser.parse_args()
    cfg = Sim2simCfg()

    # --- 检查模型文件是否存在 ---
    model_path = cfg.sim_config.mujoco_model_path
    if not os.path.exists(model_path):
        print(f"错误: MuJoCo 模型文件未找到: {model_path}")
        print("请确保 'LEGGED_GYM_ROOT_DIR' 环境变量已正确设置, 或在脚本中手动修改路径。")
        exit()

    # --- 根据参数选择执行路径 ---
    if args.verify_replay_spin:
        # --- 路径1: 生成日志并立即验证 ---
        print("--- 开始执行“生成并验证”模式 ---")
        log_directory = "spin_verification_logs"
        
        # 步骤 1: 生成日志文件
        generate_logs_for_spin(cfg, output_dir=log_directory, duration=10.0, total_spin_revolutions=1.0)
        
        # 步骤 2: 在 replay 模式下运行仿真进行图形化验证
        print("\n--- 日志已生成，现在启动 MuJoCo 查看器进行回放验证 ---")
        # 在 replay 模式下不需要加载策略模型，所以传入 None
        run_mujoco(policy=None, cfg=cfg, mode='replay', log_dir=log_directory, disable_gravity=args.disable_gravity)
        print("\n--- 验证完成 ---")

    else:
        # --- 路径2: 原始的脚本执行逻辑 ---
        if not args.run_name:
            parser.error("当不使用 --verify_replay_spin 时, 必须提供 --run_name 参数。")

        policy = None
        if args.mode in ['policy', 'observation']:
            policy_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/MOSC/exported/policies/policy_" + args.run_name + ".pt")
            if not os.path.exists(policy_path):
                print(f"错误: 策略文件未找到: {policy_path}")
                exit()
            policy = torch.jit.load(policy_path)
            print(f"从 {policy_path} 加载策略")

        run_mujoco(policy, cfg, mode=args.mode, log_dir=args.log_dir, disable_gravity=args.disable_gravity)