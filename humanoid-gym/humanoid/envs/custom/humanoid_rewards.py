# SPDX-License-Identifier: BSD-3-Clause
# (版权声明与之前相同)
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from humanoid.envs import LeggedRobotCfg
from humanoid.envs import LeggedRobot

class HumanoidRewards(LeggedRobot):
    '''
    HumanoidRewards 类，用于根据“通用、惩罚导向”的思路计算奖励。
    (注释与之前相同)
    '''

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 准备用于碰撞检测的身体部位索引
        self.penalised_contact_indices = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name) for name in self.cfg.asset.penalize_contacts_on]
        self.penalised_contact_indices = torch.tensor(self.penalised_contact_indices, device=self.device, dtype=torch.long)

# ========================== 核心目标与稳定性奖励 (保留与新增) ==========================

    def _reward_tracking_lin_vel(self):
        # 追踪线速度：使用高斯核将误差映射到 (0, 1]
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # 追踪角速度：使用高斯核
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_orientation(self):
        # 姿态保持：惩罚非水平的重力投影
        # projected_gravity 是重力向量在机身坐标系下的投影，理想情况是 [0, 0, -1]
        # 使用高斯核奖励正直姿态
        gravity_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-gravity_error / self.cfg.rewards.orientation_sigma)

    def _reward_lin_vel_z(self):
        """惩罚Z轴线速度 (防止跳跃)"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """惩罚XY轴角速度 (防止身体摇晃)"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)


# ========================== 步态与接触相关奖励 (保留与修改) ==========================

    def _reward_feet_air_time(self):
        """奖励合理的空中时间以鼓励动态步态"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        rew_air_time = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1)
        rew_air_time *= (torch.norm(self.commands[:, :2], dim=1) > 0.1) # 只在移动时奖励
        self.feet_air_time *= ~contact
        return rew_air_time

    def _reward_foot_slip(self):
        """惩罚脚部滑动"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        foot_vel_xy = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        slip_rew = torch.sum(foot_vel_xy * contact, dim=1)
        return slip_rew
        
    def _reward_collision(self):
        """惩罚除脚部以外的身体部位发生碰撞"""
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    

    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos[:, :12] - pos_target[:, :12]
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r
    

# ========================== 能量与消耗相关奖励 (保留) ==========================
 
    def _reward_torques(self):
        """惩罚过大的力矩"""
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_action_rate(self):
        """惩罚动作变化率过大，鼓励平滑动作"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_vel(self):
        """惩罚过大的关节速度"""
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """惩罚过大的关节加速度"""
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        
# ========================== 极限惩罚 (新增) ==========================

    def _reward_dof_pos_limits(self):
        """惩罚关节角度接近极限"""
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
        
    def _reward_termination(self):
        """终止惩罚"""
        return self.reset_buf * ~self.time_out_buf