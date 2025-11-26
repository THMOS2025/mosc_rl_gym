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
import cv2
import numpy as np
from isaacgym import gymapi
import matplotlib.pyplot as plt
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger, print_config_simple
from humanoid.utils.task_registry import recursive_override_class_cfg
from isaacgym.torch_utils import *

import hydra
import torch
from tqdm import tqdm
from datetime import datetime

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "cfg")
@hydra.main(config_path=config_path, config_name="play")
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    recursive_override_class_cfg(env_cfg, args)
    recursive_override_class_cfg(train_cfg, args)
    
    args.run_name = args.runner.run_name
    if args.runner.load_run == -1:
        # search the run name in the MOSC directory
        project_root = os.path.realpath(__file__).split('MOSC_RL_GYM')[0] + 'MOSC_RL_GYM' + os.sep
        root_dir = os.path.join(project_root, 'humanoid-gym/logs/MOSC')
        
        load_run = None
        latest_time = -1

        for d in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, d)
            if os.path.isdir(dir_path) and d.endswith(args.run_name):
                mtime = os.path.getmtime(dir_path)
                if mtime > latest_time:
                    latest_time = mtime
                    load_run = d
        args.runner.load_run = load_run
        print("find dir:", load_run)
        
    stop_state_log = 1000 # number of steps before plotting states
    start_plot = 0

    if RECDATA:
        log_dir_path = LEGGED_GYM_ROOT_DIR +'/logs/data_log/' + args.run_name + '/'
        if not os.path.exists(LEGGED_GYM_ROOT_DIR +'/logs/data_log/'):
            os.makedirs(LEGGED_GYM_ROOT_DIR +'/logs/data_log/')
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path) 
            
    if RECDATA or PLOT:            
        joint_name = [
                'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee_pitch', 'l_ankle_pitch', 'l_ankle_roll',
                'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee_pitch', 'r_ankle_pitch', 'r_ankle_roll']
        imu_name = ['roll', 'pitch', 'yaw', 'x', 'y', 'z']

        joint_actions_rec = np.empty((stop_state_log,12))
        joint_torques_rec = np.empty((stop_state_log,12))
        joint_dof_vel_rec = np.empty((stop_state_log,12))
        joint_dof_pos_rec = np.empty((stop_state_log,12))
        base_eu_angle_rec = np.empty((stop_state_log,3))
        base_anglevel_rec = np.empty((stop_state_log,3))

    # prepare environment
    print("env prepare")
    env, _ = task_registry.make_env_hydra(name=args.task, hydra_cfg=args)
    print("env is ready")
    print_config_simple(env_cfg, "env_cfg")
    
    if env.viewer:
        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner_hydra(env=env, name=args.task, hydra_cfg=args)
    print_config_simple(train_cfg, "train_cfg")
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, args.run_name)
        print('Exported policy as jit script to: ', path)

    robot_index = 0 

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1.5, -1, 1) 
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    # =========================================================================
    # [Setup] 矢量化绘图准备
    # =========================================================================
    cmd_color = np.array([[1.0, 0.0, 0.0]], dtype=np.float32) # Red
    ref_color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32) # Green
    
    thickness = 0.02 
    th = thickness
    offset_template = np.array([
        [0, 0, 0],
        [th, 0, 0], [-th, 0, 0],
        [0, th, 0], [0, -th, 0],
        [th, th, 0], [th, -th, 0],
        [-th, th, 0], [-th, -th, 0]
    ], dtype=np.float32) 
    
    num_thick_lines = len(offset_template)
    offsets_tensor = torch.tensor(offset_template, device=env.device) 

    # =========================================================================
    # [Setup] 生成多样化的固定指令 (Each Robot Different, but Constant)
    # =========================================================================
    if FIX_COMMAND:
        # 使用随机种子确保每次运行结果一致 (可选)
        torch.manual_seed(1234)
        
        num_envs = env.num_envs
        custom_commands = torch.zeros((num_envs, 3), device=env.device)
        
        # 1. 纵向速度 Vx: 范围 [-0.5, 1.5] m/s
        custom_commands[:, 0] = (torch.rand(num_envs, device=env.device) * 2.0) - 0.5
        
        # 2. 横向速度 Vy: 范围 [-0.5, 0.5] m/s
        custom_commands[:, 1] = (torch.rand(num_envs, device=env.device) * 1.0) - 0.5
        
        # 3. 转向角速度 Yaw: 范围 [-1.0, 1.0] rad/s
        custom_commands[:, 2] = (torch.rand(num_envs, device=env.device) * 2.0) - 1.0
        
        print(f"Generated {num_envs} random commands.")
        print("Example Command 0:", custom_commands[0].cpu().numpy())
    # =========================================================================

    for i in tqdm(range(stop_state_log)):

        actions = policy(obs.detach()) 
        env.debug_viz = True 
        
        # =================================================================
        # [Modify] 锁定指令
        # 强制将所有机器人的指令设置为我们预生成的 custom_commands
        # 这样即使环境内部想重置(Resample)，也会被我们覆盖回来
        # =================================================================
        if FIX_COMMAND:
            env.commands[:, 0:3] = custom_commands
        
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        # ==================== 可视化逻辑 (所有机器人 + 向量化 + 加粗) ====================
        if env.viewer:
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            
            all_base_pos = env.root_states[:, :3] 
            all_base_quat = env.root_states[:, 3:7]
            num_envs = env.num_envs
            
            # 计算局部指令向量 (直接使用 env.commands, 此时已经被我们锁定)
            local_cmds = torch.zeros((num_envs, 3), device=env.device)
            local_cmds[:, 0] = env.commands[:, 0]
            local_cmds[:, 1] = env.commands[:, 1]
            local_cmds[:, 2] = 0.0
            
            # 旋转到世界系
            global_cmds = quat_apply(all_base_quat, local_cmds)
            
            # 计算端点
            p_start_ref = all_base_pos 
            p_end_ref = all_base_pos + torch.tensor([0.0, 0.0, 1.0], device=env.device) 
            p_end_cmd = p_end_ref + global_cmds * 1.5 
            
            # 加粗 (Broadcasting)
            ref_starts_thick = p_start_ref.unsqueeze(1) + offsets_tensor.unsqueeze(0)
            ref_ends_thick = p_end_ref.unsqueeze(1) + offsets_tensor.unsqueeze(0)
            cmd_starts_thick = p_end_ref.unsqueeze(1) + offsets_tensor.unsqueeze(0)
            cmd_ends_thick = p_end_cmd.unsqueeze(1) + offsets_tensor.unsqueeze(0)
            
            # 展平
            verts_ref = torch.cat([ref_starts_thick, ref_ends_thick], dim=2).view(-1, 6).cpu().numpy()
            verts_cmd = torch.cat([cmd_starts_thick, cmd_ends_thick], dim=2).view(-1, 6).cpu().numpy()
            
            # 颜色
            total_lines = num_envs * num_thick_lines
            color_ref_all = np.tile(ref_color, (total_lines, 1))
            color_cmd_all = np.tile(cmd_color, (total_lines, 1))
            
            # 绘制
            env.gym.add_lines(env.viewer, None, total_lines, verts_ref, color_ref_all)
            env.gym.add_lines(env.viewer, None, total_lines, verts_cmd, color_cmd_all)
        # =========================================================================

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        if RECDATA or PLOT:            
            joint_actions_rec[i] = actions[robot_index, :].cpu().detach().numpy() * 0.25
            joint_torques_rec[i] = env.torques[robot_index,:].cpu().detach().numpy() 
            joint_dof_vel_rec[i] = env.dof_vel[robot_index,:].cpu().detach().numpy()
            joint_dof_pos_rec[i] = env.dof_pos[robot_index,:].cpu().detach().numpy()
            base_eu_angle_rec[i] = env.base_euler_xyz[robot_index,:].cpu().detach().numpy()
            base_anglevel_rec[i] = env.base_ang_vel[robot_index,:].cpu().detach().numpy()
    
    if RECDATA:
        np.savetxt(log_dir_path + "act_rec.txt",joint_actions_rec)
        np.savetxt(log_dir_path + "pos_rec.txt",joint_dof_pos_rec)
        np.savetxt(log_dir_path + "vel_rec.txt",joint_dof_vel_rec)
        np.savetxt(log_dir_path + "tor_rec.txt",joint_torques_rec)
        np.savetxt(log_dir_path + "base_ang_rec.txt",base_eu_angle_rec)
        np.savetxt(log_dir_path + "base_ang_rec.txt",base_anglevel_rec)

    if PLOT:
        time = np.linspace(start_plot * env.dt, stop_state_log * env.dt, stop_state_log - start_plot)
         # IMU
        fig, axs = plt.subplots(2, 3)
        for imu_index in range(3):
        
            a = axs[0, imu_index]
            a.plot(time, base_eu_angle_rec.T[imu_index][start_plot:])
            if (imu_index == 0):
                a.set(ylabel='Position [rad]', title= imu_name[imu_index] + ' Pos')
            else:
                a.set(title= imu_name[imu_index] + ' Eul')
            a.legend()
            
            a = axs[1, imu_index]
            a.plot(time, base_anglevel_rec.T[imu_index][start_plot:])
            if (imu_index == 0):
                a.set(ylabel='Vel [rad/s]', title=imu_name[imu_index] + ' Vel')
            else:
                a.set(title= imu_name[imu_index] + ' Vel')  
            a.legend()

        # joints
        fig, axs = plt.subplots(3, 6)
        for joint_index_l in range(6):
            joint_index = joint_index_l
            joint_index_r = joint_index + 6
            a = axs[0, joint_index]
            a.plot(time, joint_dof_pos_rec.T[joint_index][start_plot:] * 180 / 3.1415926, label= 'mea_l')
            a.plot(time, joint_actions_rec.T[joint_index][start_plot:] * 180 / 3.1415926, label= 'tar_l')
            a.plot(time, joint_dof_pos_rec.T[joint_index_r][start_plot:] * 180 / 3.1415926, label= 'mea_r')
            a.plot(time, joint_actions_rec.T[joint_index_r][start_plot:] * 180 / 3.1415926, label= 'tar_r')
            if (joint_index_l == 0):
                a.set(ylabel='Position [degree]', title= joint_name[joint_index] + ' Pos')
            else:
                a.set(title= joint_name[joint_index] + ' Pos')
            a.legend()
            
            a = axs[1, joint_index]
            a.plot(time, joint_torques_rec.T[joint_index][start_plot:], label= 'l')
            a.plot(time, joint_torques_rec.T[joint_index_r][start_plot:], label= 'r')
            if (joint_index_l == 0):
                a.set(ylabel='Torque [N*m]', title=joint_name[joint_index] + ' Tor')
            else:
                a.set(title= joint_name[joint_index] + ' Tor')  
            a.legend()
            
            a = axs[2, joint_index]
            a.plot(time, joint_dof_vel_rec.T[joint_index][start_plot:] * 180 / 3.1415926, label= 'l')
            a.plot(time, joint_dof_vel_rec.T[joint_index_r][start_plot:] * 180 / 3.1415926, label= 'r')
            if (joint_index_l == 0):
                a.set(xlabel='time [s]', ylabel='Velocity [degree/s]', title=joint_name[joint_index] + ' Vec')
            else:
                a.set(xlabel='time [s]', title=joint_name[joint_index] + ' Vec')  
            a.legend()
        plt.show()
        

    
    if RENDER:
        video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    RECDATA = False
    PLOT = False
    FIX_COMMAND = True
    play()