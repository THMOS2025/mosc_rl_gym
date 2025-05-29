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
import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# MOSC_DIR = "/home/thmos/MOSC_RL/"
MOSC_DIR = "/home/bravo/mosc-rl/"


USD_JOINT_NAMES = ['b_Lh', 'b_Ls', 'b_Rh', 'b_Rs', 'b_n', 
                  'Lh_Ll', 'Ls_La', 'Rh_Rl', 'Rs_Ra', 'n_h',  
                  'Ll_Ll1', 'La_Lh', 'Rl_Rl1', 'Ra_Rh', 'Ll1_Ll2', 
                  'Rl1_Rl2', 'Ll2_La', 'Rl2_Ra', 'La_Lf', 'Ra_Rf']





def plot_joint_time_series(data_dict: dict, dt: float, joint_names: list, plot_type: str, x_num: int = 4, y_num: int = 5):
    """
    绘制包含20个子图的时间序列图。
    
    参数:
    - data: dict，key 是数据标签（如 'motor pos'），value 是形状为 (n, 20) 的 ndarray。
    - dt: float，横坐标的时间间隔。
    - joint_names: list of str，长度为20，每个关节对应的名称。
    """

    n = next(iter(data_dict.values())).shape[0]
    time = np.arange(n) * dt

    fig, axs = plt.subplots(x_num, y_num, figsize=(20, 12))
    axs = axs.flatten()

    for i in range(x_num * y_num):
        for key, data in data_dict.items():
            axs[i].plot(time, data[:, i], label=key)
        axs[i].set_title(joint_names[i])
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Value")
        axs[i].grid(True)
        axs[i].legend()
    
    fig.text(0.52, 0.96, f"mujoco {plot_type}", ha='center', fontsize=24) 

    # Adjust layout and save the plot, [left, bottom, right, top]
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    # plt.show()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(f"{script_dir}/plots/mujoco_{plot_type.replace(' ', '_')}")




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
    name_list = ['b_Lh', 'b_Ls', 'b_Rh', 'b_Rs', 'b_n',
                 'Lh_Ll', 'Ls_La', 'Rh_Rl', 'Rs_Ra', 'n_h',
                 'Ll_Ll1', 'La_Lh', 'Rl_Rl1', 'Ra_Rh', 
                 'Ll1_Ll2', 'Rl1_Rl2', 'Ll2_La', 'Rl2_Ra', 'La_Lf', 'Ra_Rf']
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
    return (q, dq, omega_base, gvec, v_base)


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
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)


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
    print(joint_names)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    count_lowlevel = 0
    position_history = []
    tau_history = []
    action_history = []
    obs_history = []

    for i in range(3):
        mujoco.mj_step(model, data)
        viewer.render()

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, omega_base, gvec, v_base = get_obs(data, cfg)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs_parts = []
            obs_parts.append(v_base)
            obs_parts.append(omega_base)
            obs_parts.append(gvec)
            obs_parts.append(np.array([cmd.vx]))
            obs_parts.append(np.array([cmd.vy]))
            obs_parts.append(np.array([cmd.az]))
            obs_parts.append(q)
            obs_parts.append(dq)
            obs_parts.append(action)
            obs = np.expand_dims(np.concatenate(obs_parts, axis=-1), axis=0).astype(np.float32)

            # obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            action[:] = policy(torch.tensor(obs))[0].detach().numpy()
            # action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            
            
            if cfg.robot_config.if_joint_command_offset:
                target_q = action * cfg.control.action_scale + action_offset
            else:
                target_q = action * cfg.control.action_scale

            target_q[[4, 9]] = 0.0   # neck 和 head 的 action 设置 0
            
            
            obs_history.append(np.concatenate([v_base, omega_base, gvec, np.array([cmd.vx, cmd.vy, cmd.az])], axis=-1))
            action_history.append((action * cfg.control.action_scale).copy())
            position_history.append(q.copy())


        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques


        tau_history.append(tau.copy())

        data.ctrl = tau
        
        mujoco.mj_step(model, data)
        if count_lowlevel % (cfg.sim_config.decimation * 1.2) == 0:
            viewer.render()


        count_lowlevel += 1


    viewer.close()


    # 将数据整合，然后绘制
    tau_history = np.array(tau_history)
    position_history = np.array(position_history)
    action_history = np.array(action_history)
    obs_history = np.array(obs_history)

    tau_dict = {'apply torque': tau_history}
    pos_dict = {'motor pos': position_history, 'actions': action_history}
    obs_dict = {'obs': obs_history}

    plot_joint_time_series(tau_dict, cfg.sim_config.dt, cfg.robot_config.full_good_name_list, "apply torque", x_num=4, y_num=5)
    plot_joint_time_series(pos_dict, cfg.sim_config.dt * cfg.sim_config.decimation, cfg.robot_config.full_good_name_list, "actions", x_num=4, y_num=5)
    plot_joint_time_series(obs_dict, cfg.sim_config.dt * cfg.sim_config.decimation, cfg.robot_config.obs_name_list, "obs", x_num=4, y_num=3)


class cmd:
    vx = 0.3
    vy = 0.0
    az = 0.0

class Sim2simCfg():

    class env():
        num_actions = 20

    class normalization:
        clip_observations = 18.
        clip_actions = 18.

    class sim_config:
        mujoco_model_path = f'{MOSC_DIR}/source/MOSC_RL/MOSC_RL/tasks/manager_based/mosc_rl/config/MOSC/assets_0516/MOSC.xml'
        
        time_frames = 500
        dt = 0.001
        decimation = 20
        sim_duration = time_frames * decimation * dt

    class robot_config:
        name_list = ['b_Lh', 'b_Ls', 'b_Rh', 'b_Rs', 'b_n',
                    'Lh_Ll', 'Ls_La', 'Rh_Rl', 'Rs_Ra', 'n_h',
                    'Ll_Ll1', 'La_Lh', 'Rl_Rl1', 'Ra_Rh', 
                    'Ll1_Ll2', 'Rl1_Rl2', 'Ll2_La', 'Rl2_Ra', 
                    'La_Lf', 'Ra_Rf']
        
        full_name_list = ['body_left_hip', 'body_left_shoulder', 'body_right_hip', 'body_right_shoulder', 'body_neck',
                        'left_hip_left_lap', 'left_shoulder_left_arm', 'right_hip_right_lap', 'right_shoulder_right_arm', 'neck_head',
                        'left_lap_left_leg1', 'left_arm_left_hand', 'right_lap_right_leg1', 'right_arm_right_hand', 
                        'left_leg1_left_leg2', 'right_leg1_right_leg2', 'left_leg2_left_ankle', 'right_leg2_right_ankle', 
                        'left_ankle_left_foot', 'right_ankle_right_foot']
        
        full_good_name_list = [
            'left_hip_pitch', 'left_shoulder_pitch', 'right_hip_pitch', 'right_shoulder_pitch', 'neck_yaw',
            'left_hip_roll', 'left_shoulder_roll', 'right_hip_roll', 'right_shoulder_roll', 'neck_pitch',
            'left_hip_yaw', 'left_elbow', 'right_hip_yaw', 'right_elbow',
            'left_knee', 'right_knee', 'left_ankle_pitch', 'right_ankle_pitch',
            'left_ankle_roll', 'right_ankle_roll'
            ]

        obs_name_list =  ['vel_base_x', 'vel_base_y', 'vel_base_z', 'omega_base_x', 'omega_base_y', 'omega_base_z', 'gvec_x', 'gvec_y', 'gvec_z', 'cmd_vx', 'cmd_vy', 'cmd_az',]
        
        
        
        kps = np.array([100.0, 50.0, 100.0, 50.0, 10.0, 
                        100.0, 50.0, 100.0, 50.0, 10.0,
                        100.0, 50.0, 100.0, 50.0,
                        100.0, 100.0, 50.0, 50.0, 
                        50.0, 50.0], dtype=np.double) 

        kds = np.array([2.0, 1.5, 2.0, 1.5, 0.1,
                        2.0, 1.5, 2.0, 1.5, 0.1,
                        2.0, 1.5, 2.0, 1.5,
                        2.0, 2.0, 1.5, 1.5, 
                        1.5, 1.5], dtype=np.double) 
        

        init_joint_pos = {
            "La_Lh": -1.57,
            "Ra_Rh": -1.57,
            "Ls_La": 1.57,
            "Rs_Ra": -1.57,
            }
        
        if_joint_command_offset = True
        
        
        tau_limit = np.array([
            60.0, 50.0, 60.0, 50.0, 10.0, 
            60.0, 50.0, 60.0, 50.0, 10.0,
            60.0, 50.0, 60.0, 50.0,
            60.0, 60.0, 25.0, 25.0, 
            25.0, 25.0], dtype=np.double)

    class control:
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
            

if __name__ == '__main__':
    policy = torch.jit.load(MOSC_DIR + "logs/rsl_rl/mosc_flat/2025-05-28_17-30-56/exported/policy.pt")
    run_mujoco(policy, Sim2simCfg())
