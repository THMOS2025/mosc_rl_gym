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
import csv
from datetime import datetime
import time
import argparse

# --- New Imports for Logging and Plotting ---
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    print("Warning: 'pandas' or 'matplotlib' not found. Plotting will be disabled.")
    print("Please install them using: pip install pandas matplotlib")
    PLOTTING_ENABLED = False


try:
    from humanoid import LEGGED_GYM_ROOT_DIR
except ImportError:
    # If you don't have a module named humanoid, or LEGGED_GYM_ROOT_DIR is not defined in it
    # you need to set the path to LEGGED_GYM_ROOT_DIR manually here
    # For example: LEGGED_GYM_ROOT_DIR = "/path/to/your/legged_gym"
    # For the code to run smoothly, we temporarily set it to the current directory
    print("Warning: 'humanoid' module not found. Setting 'LEGGED_GYM_ROOT_DIR' to the current working directory.")
    LEGGED_GYM_ROOT_DIR = os.getcwd()

import torch


USD_JOINT_NAMES = ['b_Lh','Lh_Ll','Ll_Ll1','Ll1_Ll2','Ll2_La','La_Lf', 
                   'b_Rh','Rh_Rl','Rl_Rl1','Rl1_Rl2','Rl2_Ra','Ra_Rf']

def euler_to_quaternion(rpy):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion (w, x, y, z).
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
    '''Extract observations from the MuJoCo data structure'''
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
    '''Calculate torque from position commands'''
    return (target_q - q) * kp + (target_dq - dq) * kd

def load_data_from_log(log_dir):
    """Load data from log files"""
    joint_pos_path = os.path.join(log_dir, 'joint_pos.txt')
    joint_vel_path = os.path.join(log_dir, 'joint_vel.txt')
    imu_path = os.path.join(log_dir, 'base_ang_eul.txt')
    joint_act_path = os.path.join(log_dir, 'joint_act.txt')
    # New: Define the path for input_imu.txt
    imu_ang_vel_path = os.path.join(log_dir, 'input_imu.txt')

    joint_pos = np.loadtxt(joint_pos_path, delimiter=',') if os.path.exists(joint_pos_path) else None
    joint_vel = np.loadtxt(joint_vel_path, delimiter=',') if os.path.exists(joint_vel_path) else None
    imu_rpy = np.loadtxt(imu_path, delimiter=',') if os.path.exists(imu_path) else None
    joint_act = np.loadtxt(joint_act_path, delimiter=',') if os.path.exists(joint_act_path) else None
    # New: Load angular velocity data
    imu_ang_vel = np.loadtxt(imu_ang_vel_path, delimiter=',') if os.path.exists(imu_ang_vel_path) else None
    
    # New: Return angular velocity data
    return joint_pos, joint_vel, imu_rpy, joint_act, imu_ang_vel

def setup_logging(log_dir_base, run_name):
    """Create a unique subdirectory for the run and return the file path for the CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a unique subdirectory for this run, e.g., "run_my_policy_20251011_203945"
    run_dir_name = f"run_{run_name}_{timestamp}"
    run_log_dir = os.path.join(log_dir_base, run_dir_name)
    os.makedirs(run_log_dir, exist_ok=True)

    # The log file will be inside the new subdirectory
    log_filename = "log_data.csv"
    log_filepath = os.path.join(run_log_dir, log_filename)
    
    headers = ['time']
    for name in USD_JOINT_NAMES:
        headers.extend([f'q_{name}', f'tau_{name}', f'action_{name}', f'target_q_{name}'])
    headers.extend(['omega_x', 'omega_y', 'omega_z', 'roll', 'pitch', 'yaw'])
    
    with open(log_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
    return log_filepath

def append_log_data(log_filepath, data_row):
    """Append a row of data to the CSV log file."""
    with open(log_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_row)

def plot_log_data(log_filepath):
    """Read the log file and generate plots for each joint and IMU data."""
    if not PLOTTING_ENABLED:
        print("Plotting is disabled because required libraries are missing.")
        return

    print(f"Generating plots from {log_filepath}...")
    df = pd.read_csv(log_filepath)
    # The log directory is the directory containing the csv file
    log_dir = os.path.dirname(log_filepath)

    # --- Plot data for each joint individually ---
    for i, name in enumerate(tqdm(USD_JOINT_NAMES, desc="Generating joint plots")):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        # UPDATED: Added joint index 'i' to the plot title
        fig.suptitle(f'Joint Data: [{i}] {name}', fontsize=16)

        # Subplot 1: Position (q) and Target Position (target_q)
        ax1.plot(df['time'], df[f'q_{name}'], label='Measured Position (q)')
        ax1.plot(df['time'], df[f'target_q_{name}'], label='Target Position (target_q)', linestyle='--', alpha=0.8)
        ax1.set_ylabel('Angle (rad)')
        ax1.set_title('Joint Position')
        ax1.legend()
        ax1.grid(True)

        # Subplot 2: Torque (tau)
        ax2.plot(df['time'], df[f'tau_{name}'], label='Applied Torque (tau)', color='orange')
        ax2.set_ylabel('Torque (Nm)')
        ax2.set_title('Joint Torque')
        ax2.legend()
        ax2.grid(True)

        # Subplot 3: Raw Action
        ax3.plot(df['time'], df[f'action_{name}'], label='Raw Policy Action', color='green')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Action Value')
        ax3.set_title('Raw Policy Action')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plot_filename = os.path.join(log_dir, f"joint_{name}.png")
        plt.savefig(plot_filename)
        plt.close(fig) # Close the figure to free up memory

    # --- Plot IMU Data (combined) ---
    fig_imu, (ax_imu1, ax_imu2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax_imu1.plot(df['time'], df['roll'], label='Roll')
    ax_imu1.plot(df['time'], df['pitch'], label='Pitch')
    ax_imu1.plot(df['time'], df['yaw'], label='Yaw')
    ax_imu1.set_title('IMU Euler Angles')
    ax_imu1.set_ylabel('Angle (rad)')
    ax_imu1.legend()
    ax_imu1.grid(True)

    ax_imu2.plot(df['time'], df['omega_x'], label='Omega X')
    ax_imu2.plot(df['time'], df['omega_y'], label='Omega Y')
    ax_imu2.plot(df['time'], df['omega_z'], label='Omega Z')
    ax_imu2.set_title('IMU Angular Velocities')
    ax_imu2.set_xlabel('Time (s)')
    ax_imu2.set_ylabel('Angular Velocity (rad/s)')
    ax_imu2.legend()
    ax_imu2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "imu_data.png"))
    plt.close(fig_imu)

    print(f"Plots and log file saved in: {log_dir}")


def run_mujoco(policy, cfg, run_name, mode='policy', log_dir=None, disable_gravity=False):
    """
    Run the Mujoco simulation with the provided policy and configuration.

    Args:
        policy: The policy to control the simulation.
        cfg: The configuration object containing simulation settings.
        run_name: The name of the policy run, used for logging directory.
        mode: 'policy', 'observation', 'action', or 'replay'
        log_dir: Directory containing log files for certain modes.
        disable_gravity (bool): If True, disables gravity in the simulation.
    """
    # --- Key Time Parameters ---
    # Physics simulation step: dt = 0.001s
    # High-level control/logging decimation: decimation = 20
    # Therefore, high-level control and logging interval:
    # high_level_interval = dt * decimation = 0.001s * 20 = 0.02s (50Hz)
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt

    if disable_gravity:
        model.opt.gravity[:] = 0
        print("Gravity has been DISABLED for this simulation.")
    else:
        print(f"Gravity is ENABLED. Default value: {model.opt.gravity}")
    
    data = mujoco.MjData(model)

    joint_pos_log, joint_vel_log, imu_rpy_log, joint_act_log, imu_ang_vel_log = None, None, None, None, None
    if log_dir:
        joint_pos_log, joint_vel_log, imu_rpy_log, joint_act_log, imu_ang_vel_log = load_data_from_log(log_dir)

    if mode == 'observation' and (joint_pos_log is None or joint_vel_log is None or imu_rpy_log is None or imu_ang_vel_log is None):
        raise ValueError("In 'observation' mode, 'joint_pos.txt', 'joint_vel.txt', 'base_ang_eul.txt', and 'input_imu.txt' must exist.")
    if mode == 'action' and joint_act_log is None:
        raise ValueError("In 'action' mode, 'joint_act.txt' must exist.")
    if mode == 'replay' and (joint_pos_log is None or imu_rpy_log is None):
        raise ValueError("In 'replay' mode, 'joint_pos.txt' and 'base_ang_eul.txt' must exist.")

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

    # --- Setup Logging ---
    log_dir_path = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, '..', 'data_logs'))
    log_filepath = setup_logging(log_dir_path, run_name)
    print(f"Logging data to a new folder within: {os.path.dirname(log_filepath)}")
    
    if mode in ['observation', 'action', 'replay']:
        valid_logs = [log for log in [joint_pos_log, joint_vel_log, imu_rpy_log, joint_act_log, imu_ang_vel_log] if log is not None]
        if not valid_logs:
            print(f"Warning: No log files provided in '{mode}' mode. Simulation will not run.")
            num_steps = 0
        else:
            num_steps = min(len(log) for log in valid_logs)
        sim_duration_steps = num_steps * cfg.sim_config.decimation
    else:
        num_steps = int(cfg.sim_config.sim_duration / (cfg.sim_config.dt * cfg.sim_config.decimation))
        sim_duration_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)

    for _step in tqdm(range(sim_duration_steps), desc=f"Simulating in {mode} mode..."):
        # --- Replay Mode Logic ---
        if mode == 'replay':
            if _step % cfg.sim_config.decimation == 0:
                if log_idx < len(joint_pos_log):
                    for i, joint_name in enumerate(cfg.robot_config.name_list):
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        if joint_id != -1:
                            qpos_adr = model.jnt_qposadr[joint_id]
                            data.qpos[qpos_adr] = joint_pos_log[log_idx, i]
                    
                    if log_idx < len(imu_rpy_log):
                        rpy = imu_rpy_log[log_idx]
                        quat = euler_to_quaternion(rpy)
                        data.qpos[0:4] = quat

                    mujoco.mj_forward(model, data)
                    log_idx += 1
            
            viewer.render()
            time.sleep(cfg.sim_config.dt) 
            continue

        # --- Policy, Observation, Action Mode Logic ---
        if mode == 'observation' and log_idx < len(joint_pos_log):
            q = joint_pos_log[log_idx]
            dq = joint_vel_log[log_idx]
            eu_ang = imu_rpy_log[log_idx]
            omega_base = imu_ang_vel_log[log_idx]
            quat = euler_to_quaternion(eu_ang)
        else:
            q, dq, omega_base, quat, v_base = get_obs(data, cfg)
        
        q = q[-cfg.env.num_actions:] 
        dq = dq[-cfg.env.num_actions:]

        # High-level control loop at 50Hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            if mode != 'observation':
                 eu_ang = quaternion_to_euler_array(quat)
                 eu_ang[eu_ang > math.pi] -= 2 * math.pi

            # --- Build Policy Input ---
            obs_parts = []
            current_time = count_lowlevel * cfg.sim_config.dt
            obs_parts.append(np.array([math.sin(2 * math.pi * current_time / cfg.rewards.cycle_time)])) 
            obs_parts.append(np.array([math.cos(2 * math.pi * current_time / cfg.rewards.cycle_time)]))
            obs_parts.append(np.array([cmd.vx]))
            obs_parts.append(np.array([cmd.vy]))
            obs_parts.append(np.array([cmd.az]))
            obs_parts.append((q - action_offset) * cfg.normalization.obs_scales.dof_pos)
            obs_parts.append(dq * cfg.normalization.obs_scales.dof_vel)
            obs_parts.append(action) # Previous action
            obs_parts.append(omega_base)
            obs_parts.append(eu_ang) 
            obs = np.expand_dims(np.concatenate(obs_parts, axis=-1), axis=0).astype(np.float32)

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            
            # --- Calculate Action ---
            if mode == 'action':
                if log_idx < len(joint_act_log):
                    target_q = - joint_act_log[log_idx]
                else:
                    target_q = action_offset
            else: # policy and observation modes
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)  
                target_q = action * cfg.control.action_scale + action_offset
            
            alpha = 0.8
            if _step > 0:
                target_q = alpha * target_q + (1 - alpha) * last_target_q
            last_target_q = target_q.copy()
            
            if log_dir and log_idx < num_steps:
                log_idx += 1
            
            # --- Data Logging ---
            current_log_data = [current_time]
            for i in range(cfg.env.num_actions):
                current_log_data.extend([q[i], data.ctrl[i], action[i], target_q[i]])
            current_log_data.extend(omega_base)
            current_log_data.extend(eu_ang)
            append_log_data(log_filepath, current_log_data)


        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
                    
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)

        data.ctrl = tau
                        
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()
    return log_filepath

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
        # mujoco_model_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/robots/MOSC0516/MOSC_rotaion_test.xml')
        mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/MOSC0516/MOSC_OL_up_hand.xml'
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
            "b_Lh": 0.3,
            "Ll1_Ll2": -0.6,
            "Ll2_La": 0.3,

            "b_Rh": -0.3,
            "Rl1_Rl2": 0.6,
            "Rl2_Ra": -0.3,
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
        print(f"Error: MuJoCo model file not found: {model_path}")
        print("Please ensure 'LEGGED_GYM_ROOT_DIR' environment variable is set correctly, or modify the path manually in the script.")
        exit()

    policy = None
    if args.mode in ['policy', 'observation']:
        policy_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/MOSC/exported/policies/policy_" + args.run_name + ".pt")
        if not os.path.exists(policy_path):
            print(f"Error: Policy file not found: {policy_path}")
            print("Please ensure the policy file exists at the correct path.")
            exit()
            
        policy = torch.jit.load(policy_path)
        print(f"Loaded policy from {policy_path}")

    # Run the simulation and get the path to the log file
    log_file_path = run_mujoco(policy, Sim2simCfg(), args.run_name, mode=args.mode, log_dir=args.log_dir, disable_gravity=args.disable_gravity)

    # Generate plots from the log file
    if log_file_path and os.path.exists(log_file_path):
        plot_log_data(log_file_path)