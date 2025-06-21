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
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    #env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5    
    env_cfg.terrain.max_init_terrain_level = 2
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.rewards.cycle_time = 0.6
    env_cfg.asset.test_ref_dof = False
    env_cfg.asset.disable_gravity = False
    env_cfg.asset.fix_base_link = False
    env_cfg.noise.noise_level = 0.5
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = "Jun20_21-01-00_v3_s2"
    train_cfg.runner.checkpoint = 3200 # model_.pt
    env_cfg.domain_rand.add_com_x = [-0.000,0.000]
    env_cfg.domain_rand.add_com_y = [-0.000,0.000]
    env_cfg.domain_rand.add_com_z = [-0.000,0.000]  
    
    env_cfg.domain_rand.kp_rand_ratio = 0.000
    env_cfg.domain_rand.kd_rand_ratio = 0.000
    env_cfg.domain_rand.torque_rand_ratio = 0.00
    stop_state_log = 1000 # number of steps before plotting states
    start_plot = 0
    env_cfg.viewer.debug_viz = True

    train_cfg.seed = 126
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)
    
    
    # search the run name in the MOSC directory
    root_dir = "./humanoid-gym/logs/MOSC/"
    load_run = None

    # find the first directory that matches the run name
    for d in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, d)
        if os.path.isdir(dir_path) and d.endswith(args.run_name):
            load_run = d
            break  # fine the first match and exit
    train_cfg.runner.load_run = load_run

    print("find dir:", load_run)


    if RECDATA:
        log_dir_path = LEGGED_GYM_ROOT_DIR +'/logs/data_log/' + args.run_name + '/'
        if not os.path.exists(LEGGED_GYM_ROOT_DIR +'/logs/data_log/'):
            os.makedirs(LEGGED_GYM_ROOT_DIR +'/logs/data_log/')
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path) 
            
    if RECDATA or PLOT:           
        joint_name = [
                'l_hip_yaw',
                'l_hip_roll',
                'l_hip_pitch',
                'l_knee_pitch',
                'l_ankle_pitch',
                'l_ankle_roll',
                'r_hip_roll',
                'r_hip_yaw',
                'r_hip_pitch',
                'r_knee_pitch',
                'r_ankle_pitch',
                'r_ankle_roll']
        imu_name = [
                'roll',
                'pitch',
                'yaw',
                'x',
                'y',
                'z',]

        joint_actions_rec = np.empty((stop_state_log,12))
        joint_torques_rec = np.empty((stop_state_log,12))
        joint_dof_vel_rec = np.empty((stop_state_log,12))
        joint_dof_pos_rec = np.empty((stop_state_log,12))
        base_eu_angle_rec = np.empty((stop_state_log,3))
        base_anglevel_rec = np.empty((stop_state_log,3))

    # prepare environment
    print("env prepare")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("env is ready")
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, args.run_name)
        print('Exported policy as jit script to: ', path)

    robot_index = 0 # which robot is used for logging

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1.5, -1, 1) #(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
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

    for i in tqdm(range(stop_state_log)):

        actions = policy(obs.detach()) 
        env.debug_viz = True
        
        if FIX_COMMAND:
            env.commands[:, 0] =  1.0
            env.commands[:, 1] =  0.0
            env.commands[:, 2] =  0.0
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

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
    RECDATA = True
    PLOT = False
    FIX_COMMAND = True
    args = get_args()
    play(args)
