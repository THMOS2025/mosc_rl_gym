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
import matplotlib.pyplot as plt
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
import torch
from datetime import datetime
import time
USD_JOINT_NAMES = ['b_Lh','Lh_Ll','Ll_Ll1','Ll1_Ll2','Ll2_La','La_Lf', 
                'b_Rh','Rh_Rl','Rl_Rl1','Rl1_Rl2','Rl2_Ra','Ra_Rf']


class Data_log:
    log_path = 'data_logs/' + datetime.now().strftime('%b%d_%H-%M-%S') + '/'
    if not os.path.exists('data_logs/'):
        os.makedirs('data_logs/')
    if not os.path.exists(log_path):
        os.makedirs(log_path) 
    joint_name = [
                'l_hip_pitch',
                'l_hip_roll',
                'l_hip_yaw',
                'l_knee_pitch',
                'l_ankle_pitch',
                'l_ankle_roll',
                'r_hip_pitch',
                'r_hip_roll',
                'r_hip_yaw',
                'r_knee_pitch',
                'r_ankle_pitch',
                'r_ankle_roll']
    imu_name = [
                'roll',
                'pitch',
                'yaw',
                'x',
                'y',
                'z']
    path_name = [
                log_path + 'joint_act.txt',
                log_path + 'joint_pos.txt',
                log_path + 'joint_vel.txt',
                log_path + 'base_ang_eul.txt',
                log_path + 'base_ang_vel.txt',
                log_path + 'cmd.txt']
    
    rpy = np.zeros((3), dtype=np.double)
    omega =np.zeros((3), dtype=np.double)
    act = np.zeros((12), dtype=np.double)
    q = np.zeros((12), dtype=np.double)
    dq = np.zeros((12), dtype=np.double)
    cmd = np.zeros((3), dtype=np.double) 

    def rec(self):
        with open(self.path_name[0], 'a') as f:
            str_arr = np.array2string(np.array(self.act[:14]), separator=',')
            str_arr = str_arr.replace('[', '').replace(']', '').replace('\n', '')
            f.writelines(str_arr)
            f.writelines("\n")
        with open(self.path_name[1], 'a') as f:
            str_arr = np.array2string(np.array(self.q[:14]), separator=',')
            str_arr = str_arr.replace('[', '').replace(']', '').replace('\n', '')
            f.writelines(str_arr)
            f.writelines("\n")
        with open(self.path_name[2], 'a') as f:
            str_arr = np.array2string(np.array(self.dq[:14]), separator=',')
            str_arr = str_arr.replace('[', '').replace(']', '').replace('\n', '')
            f.writelines(str_arr)
            f.writelines("\n")
        with open(self.path_name[3], 'a') as f:
            str_arr = np.array2string(np.array(self.rpy[:3]), separator=',')
            str_arr = str_arr.replace('[', '').replace(']', '').replace('\n', '')
            f.writelines(str_arr)
            f.writelines("\n")
        with open(self.path_name[4], 'a') as f:
            str_arr = np.array2string(np.array(self.omega[:3]), separator=',')
            str_arr = str_arr.replace('[', '').replace(']', '').replace('\n', '')
            f.writelines(str_arr)
            f.writelines("\n")
        with open(self.path_name[5], 'a') as f:
            str_arr = np.array2string(np.array(self.act[:3]), separator=',')
            str_arr = str_arr.replace('[', '').replace(']', '').replace('\n', '')
            f.writelines(str_arr)
            f.writelines("\n")

REC = True

if REC:
    data_rec = Data_log()



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
    '''Extracts an observation from the mujoco data structure
    '''
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
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    # TODO
    # model.body_mass[:] = model.body_mass * 0.5
    print(model.body_mass)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    print(actuator_names)

    # 设置 qpos 中的初始关节值
    for joint_name, value in cfg.robot_config.init_joint_pos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            qpos_index = model.jnt_qposadr[joint_id]
            data.qpos[qpos_index] = value
            
    action_offset = np.array([cfg.robot_config.init_joint_pos.get(joint, 0.0) for joint in USD_JOINT_NAMES])

    # 更新模型状态
    mujoco.mj_forward(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data)

    joint_names = [model.joint(i).name for i in range(model.njnt)]
    #print(joint_names)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    count_highlevel = 0


    for i in range(3):
        mujoco.mj_step(model, data)
        viewer.render()

    if REC:
        global data_rec
    for _step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, omega_base, quat, v_base = get_obs(data, cfg)
        q = q[-cfg.env.num_actions:] 
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            eu_ang[:3] *= 1
            
            # omega_base = np.zeros_like(omega_base)
            # eu_ang = np.zeros_like(eu_ang)

            # add noise
            # eu_ang = eu_ang + np.random.randn(*eu_ang.shape) * 0.12 * 0.6
            # omega_base = omega_base + np.random.randn(*omega_base.shape) * 0.12 * 0.6

            #print(eu_ang)
            obs_parts = []
            obs_parts.append(np.array([math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)])) 
            obs_parts.append(np.array([math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)]))
            obs_parts.append(np.array([cmd.vx]))
            obs_parts.append(np.array([cmd.vy]))
            obs_parts.append(np.array([cmd.az]))
            obs_parts.append((q - action_offset) * cfg.normalization.obs_scales.dof_pos)
            obs_parts.append(dq * cfg.normalization.obs_scales.dof_vel)
            obs_parts.append(action)
            obs_parts.append(omega_base)
            obs_parts.append(eu_ang) 
            obs = np.expand_dims(np.concatenate(obs_parts, axis=-1), axis=0).astype(np.float32)

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)  
            
            # print(policy_input)
            # print(action)
            # time.sleep(100.0)
            
            target_q = action * cfg.control.action_scale + action_offset 
            
            alpha = 0.8
            if _step != 0:
                target_q = alpha * target_q + (1 - alpha) * last_target_q
            last_target_q = target_q.copy()
            
            
            # target_q[4] = 0.3 * np.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time * 0.1)
            # target_q[10] = -0.3 * np.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time * 0.1)  
            # print([q[3]/3.141593*180,q[9]/3.141593*180])
            if REC:
                data_rec.rpy = eu_ang
                data_rec.omega = omega_base
                data_rec.q = q
                data_rec.dq = dq
                data_rec.act = target_q
                data_rec.rec()

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
                    
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit) # Clamp torques

        data.ctrl = tau
                        
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
    viewer.close()



class cmd:
    vx = 0.3
    vy = 0.0
    az = 0.0

class Sim2simCfg():

    class env():
        # change the observation dim
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
        mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/MOSC0516/MOSC_OL_up_hand.xml'
        sim_duration = 2000 * 0.01
        dt = 0.001
        decimation = 20

        
    class rewards:
        cycle_time = 0.8#0.60# sec
    
    class robot_config:
        name_list = USD_JOINT_NAMES
        
        kps = np.array([300.0, 300.0,45.0, 50.0, 50.0, 7.0,
                        300.0, 300.0,45.0, 50.0, 50.0, 7.0], dtype=np.double) 

        kds = np.array([20.0, 20.0, 3.0, 3.0,1.5,0.3,
                        20.0, 20.0, 3.0, 3.0,1.5,0.3], dtype=np.double) 
        
        # kps = np.array([100.0, 100.0,100.0, 100.0, 50.0, 24.0,
        #                 100.0, 100.0,100.0, 100.0, 50.0, 24.0], dtype=np.double) 

        # kds = np.array([2.0, 2.0, 2.0, 2.0,1.5,0.3,
        #                 2.0, 2.0, 2.0, 2.0,1.5,0.3], dtype=np.double) 


        # kps = np.array([10.0, 10.0,10.0, 10.0, 10, 10.0,
        #                 10.0, 10.0,10.0, 10.0, 10, 10.0], dtype=np.double) 
        # kds = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        #                 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.double)
        # kds = np.array([0.263, 0.263, 0.263, 0.263, 0.0495,0.0495,
        #                 0.263, 0.263, 0.263, 0.263, 0.0495,0.0495], dtype=np.double) 
        # kps = np.array([100, 100, 100, 100, 50, 50, 100, 100, 100, 100, 50, 50])
        # kds = np.array([0.6, 0.6, 0.6, 0.6, 0.05, 0.05, 0.6, 0.6, 0.6, 0.6, 0.05, 0.05]) 
        init_joint_pos = {
            "b_Lh": 0.3,
            "Ll1_Ll2": -0.6,
            "Ll2_La": 0.3,

            "b_Rh": -0.3,
            "Rl1_Rl2": 0.6,
            "Rl2_Ra": -0.3,
            }
        
        if_joint_command_offset = True
        
        tau_limit = np.array([
            60.0, 60.0, 
            60.0, 60.0,
            60.0, 60.0, 
            60.0, 60.0, 
            60.0, 60.0, 
            10.0, 10.0], dtype=np.double)
        
    class control:
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Run to load from.')
    args = parser.parse_args()
        
    policy = torch.jit.load(LEGGED_GYM_ROOT_DIR + "/logs/MOSC/exported/policies/policy_" + args.run_name + ".pt")
    print(f"loading policy from {LEGGED_GYM_ROOT_DIR}/logs/MOSC/exported/policies/policy_{args.run_name}.pt")

    
    run_mujoco(policy, Sim2simCfg())
