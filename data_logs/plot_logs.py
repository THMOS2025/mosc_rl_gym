"""
Copyright (C) [2024] [Fourier Intelligence Ltd.]

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
"""

import os
import sys
import time
import numpy
import matplotlib.pyplot as plt


def find_dir():
    root_dir = "./data_logs"

    # 收集所有以 run_name 结尾的目录及其创建时间
    matched_dirs = []

    for d in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, d)
        if os.path.isdir(dir_path):
            # 获取目录的创建时间
            ctime = os.path.getctime(dir_path)
            matched_dirs.append((ctime, d))

    # 选择创建时间最新的那个目录
    if matched_dirs:
        # 按创建时间倒序排序，取第一个
        latest_dir = max(matched_dirs, key=lambda x: x[0])[1]
        print(f"Latest directory found: {latest_dir}")
        return latest_dir
    else:
        return None
    
    

#====record====
PLOT = True
PLOT_SHOW = False
fig_dpi = 800
fig_width = 15
fig_height = 12
your_date = find_dir()
log_path = f"{os.path.abspath(os.path.dirname(__file__))}/{your_date}/"
print(f"Log path: {log_path}")
now_state_log = 0
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
            'z',]

path_name = [
            log_path + 'joint_act.txt',
            log_path + 'joint_pos.txt',
            log_path + 'joint_vel.txt',
            log_path + 'base_ang_eul.txt',
            log_path + 'base_ang_vel.txt']
joint_actions_rec = numpy.loadtxt(path_name[0], delimiter=',').T / 3.1415926 * 180
joint_dof_vec_rec = numpy.loadtxt(path_name[2], delimiter=',').T / 3.1415926 * 180
joint_dof_pos_rec = numpy.loadtxt(path_name[1], delimiter=',').T / 3.1415926 * 180
base_eu_angle_rec = numpy.loadtxt(path_name[3], delimiter=',').T / 3.1415926 * 180
base_anglevel_rec = numpy.loadtxt(path_name[4], delimiter=',').T / 3.1415926 * 180

stop_state_log = min(600, joint_actions_rec.shape[1])
    
    
    
    

def plot_fig(target_control_period_in_s):
    global PLOT, stop_state_log ,joint_actions_rec ,joint_torques_rec ,joint_dof_vec_rec ,joint_dof_pos_rec, joint_name, imu_name, base_eu_angle_rec, base_anglevel_rec
    
    if PLOT:
        time = numpy.linspace(0, stop_state_log * target_control_period_in_s, stop_state_log)
        
        # IMU
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        for imu_index in range(3):
        
            a = axs[0, imu_index]
            a.plot(time, base_eu_angle_rec[imu_index,:stop_state_log])
            if (imu_index == 0):
                a.set(ylabel='Position [degree]', title= imu_name[imu_index])
            else:
                a.set(title= imu_name[imu_index])
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
            a = axs[1, imu_index]
            a.plot(time, base_anglevel_rec[imu_index,:stop_state_log])
            if (imu_index == 0):
                a.set(ylabel='Vel [degree/s]')
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
        if PLOT_SHOW:
            plt.show()
        else:
            plt.savefig(log_path + 'imu.png', bbox_inches='tight', dpi=fig_dpi)

        # left 1-3
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        for joint_index_l in range(3):
            joint_index = joint_index_l
            
            a = axs[0, joint_index]
            a.plot(time, joint_dof_pos_rec[joint_index,:stop_state_log], label= 'measured')
            a.plot(time, joint_actions_rec[joint_index,:stop_state_log], label= 'target')
            if (joint_index_l == 0):
                a.set(ylabel='Position [degree]', title= joint_name[joint_index])
            else:
                a.set(title= joint_name[joint_index])
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
            # a = axs[1, joint_index]
            # a.plot(time, joint_torques_rec[joint_index,:stop_state_log])
            # if (joint_index_l == 0):
            #     a.set(ylabel='Torque [N*m]') 
            # handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
            a = axs[1, joint_index]
            a.plot(time, joint_dof_vec_rec[joint_index,:stop_state_log])
            if (joint_index_l == 0):
                a.set(xlabel='time [s]', ylabel='Velocity [degree/s]')
            else:
                a.set(xlabel='time [s]')  
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
        if PLOT_SHOW:
            plt.show()
        else:
            plt.savefig(log_path + 'left_leg_1_3.png', bbox_inches='tight', dpi=fig_dpi)
        
        # left 4-6
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        for joint_index_l in range(3):
            joint_index = joint_index_l + 3
            
            a = axs[0, joint_index_l]
            a.plot(time, joint_dof_pos_rec[joint_index,:stop_state_log], label= 'measured')
            a.plot(time, joint_actions_rec[joint_index,:stop_state_log], label= 'target')
            if (joint_index_l == 0):
                a.set(ylabel='Position [degree]', title= joint_name[joint_index])
            else:
                a.set(title= joint_name[joint_index])
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
            # a = axs[1, joint_index_l]
            # a.plot(time, joint_torques_rec[joint_index,:stop_state_log])
            # if (joint_index_l == 0):
            #     a.set(ylabel='Torque [N*m]')
            # handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
            a = axs[1, joint_index_l]
            a.plot(time, joint_dof_vec_rec[joint_index,:stop_state_log])
            if (joint_index_l == 0):
                a.set(xlabel='time [s]', ylabel='Velocity [degree/s]')
            else:
                a.set(xlabel='time [s]')  
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
        if PLOT_SHOW:
            plt.show()
        else:
            plt.savefig(log_path + 'left_leg_4_6.png', bbox_inches='tight', dpi=fig_dpi)
        
        # right 1-3
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        for joint_index_r in range(3):
            joint_index = joint_index_r + 6
            
            a = axs[0, joint_index_r]
            a.plot(time, joint_dof_pos_rec[joint_index,:stop_state_log], label= 'measured')
            a.plot(time, joint_actions_rec[joint_index,:stop_state_log], label= 'target')
            if (joint_index_r == 0):
                a.set(ylabel='Position [degree]', title= joint_name[joint_index])
            else:
                a.set(title= joint_name[joint_index])
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
            # a = axs[1, joint_index_r]
            # a.plot(time, joint_torques_rec[joint_index,:stop_state_log])
            # if (joint_index_r == 0):
            #     a.set(ylabel='Torque [N*m]')    
            # handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
            a = axs[1, joint_index_r]
            a.plot(time, joint_dof_vec_rec[joint_index,:stop_state_log])
            if (joint_index_r == 0):
                a.set(xlabel='time [s]', ylabel='Velocity [degree/s]')
            else:
                a.set(xlabel='time [s]')             
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
        if PLOT_SHOW:
            plt.show()
        else:
            plt.savefig(log_path + 'right_leg_1_3', bbox_inches='tight', dpi=fig_dpi)

        # right 4-6
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        for joint_index_r in range(3):
            joint_index = joint_index_r + 9
            
            a = axs[0, joint_index_r]
            a.plot(time, joint_dof_pos_rec[joint_index,:stop_state_log], label= 'measured')
            a.plot(time, joint_actions_rec[joint_index,:stop_state_log], label= 'target')
            if (joint_index_r == 0):
                a.set(ylabel='Position [degree]', title= joint_name[joint_index])
            else:
                a.set(title= joint_name[joint_index])
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
            # a = axs[1, joint_index_r]
            # a.plot(time, joint_torques_rec[joint_index,:stop_state_log])
            # if (joint_index_r == 0):
            #     a.set(ylabel='Torque [N*m]')        
            # handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
            
            a = axs[1, joint_index_r]
            a.plot(time, joint_dof_vec_rec[joint_index,:stop_state_log])
            if (joint_index_r == 0):
                a.set(xlabel='time [s]', ylabel='Velocity [degree/s]')
            else:
                a.set(xlabel='time [s]')             
            handles, labels = a.get_legend_handles_labels()
            if len(handles) > 0:
                a.legend()
        if PLOT_SHOW:
            plt.show()
        else:
            plt.savefig(log_path + 'right_leg_4_6', bbox_inches='tight', dpi=fig_dpi)
        





plot_fig(0.02)