import os
import math
import numpy as np
import mujoco
import mujoco_viewer
import time

# ================= 配置区域 =================
XML_PATH = '/home/thmos/mos9_rl_codes/mosc_rl_gym/humanoid-gym/resources/robots/MOSC0516/MOSC_OL_up_hand.xml' 

USD_JOINT_NAMES = [
    'b_Lh', 'Lh_Ll', 'Ll_Ll1', 'Ll1_Ll2', 'Ll2_La', 'La_Lf', 
    'b_Rh', 'Rh_Rl', 'Rl_Rl1', 'Rl1_Rl2', 'Rl2_Ra', 'Ra_Rf'
]

DEFAULT_JOINT_ANGLES = {
    'b_Lh': 0.0, 'b_Rh': 0.0,
    'Lh_Ll': 0.0, 'Rh_Rl': 0.0,
    'Ll_Ll1': 0.0, 'Rl_Rl1': 0.0, 
    'Ll1_Ll2': 0.0, 'Rl1_Rl2': 0.0, 
    'Ll2_La': 0.0, 'Rl2_Ra': 0.0, 
    'La_Lf': 0.0, 'Ra_Rf': 0.0
}
DEFAULT_JOINT_ANGLES.update({
    "b_Lh": -0.3, "Ll1_Ll2": 0.6, "Ll2_La": -0.3,
    "b_Rh": 0.3, "Rl1_Rl2": -0.6, "Rl2_Ra": 0.3,
})

class RefParams:
    cycle_time = 0.55
    double_stand_phase = 0.07 # 对应 cfg.rewards.double_stand_phase
    target_joint_pos_scale = 0.25
    ref_pos_dir = [-1, 1, -1, -1, 1, -1]
    dt = 0.002

class Command:
    vx = 0.0 
    vy = 0.0

def get_default_dof_pos_array(model):
    default_pos = np.zeros(model.nu)
    for i, name in enumerate(USD_JOINT_NAMES):
        default_pos[i] = DEFAULT_JOINT_ANGLES.get(name, 0.0)
    return default_pos

# =========================================================================
# 1. 旧逻辑 (User Provided Code)
# =========================================================================
def compute_old_ref_state(current_time, cmd, default_pos, cfg):
    """
    你提供的原始逻辑 (移植到 Numpy)
    """
    phase = (current_time % cfg.cycle_time) / cfg.cycle_time
    
    # torch.sin -> np.sin
    sin_pos = np.sin(2 * np.pi * phase)
    
    # clone() - 0.2 / + 0.2
    sin_pos_l = sin_pos - 0.5
    sin_pos_r = sin_pos + 0.5
    
    ref_dof_pos = default_pos.copy()
    
    scale_1 = cfg.target_joint_pos_scale
    scale_2 = 2 * cfg.target_joint_pos_scale
    
    # Left foot stance phase set to default joint pos
    # Logic: sin_pos_l[sin_pos_l > 0] = 0
    val_l = sin_pos_l if sin_pos_l <= 0 else 0.0
    
    # Apply to Left (indices 0, 3, 4)
    ref_dof_pos[0] += cfg.ref_pos_dir[0] * (val_l * scale_1)
    ref_dof_pos[3] += cfg.ref_pos_dir[1] * (val_l * scale_2)
    ref_dof_pos[4] += cfg.ref_pos_dir[2] * (val_l * scale_1)
    
    # Right foot stance phase set to default joint pos
    # Logic: sin_pos_r[sin_pos_r < 0] = 0
    val_r = sin_pos_r if sin_pos_r >= 0 else 0.0
    
    # Apply to Right (indices 6, 9, 10)
    ref_dof_pos[6] += cfg.ref_pos_dir[3] * (val_r * scale_1)
    ref_dof_pos[9] += cfg.ref_pos_dir[4] * (val_r * scale_2)
    ref_dof_pos[10]+= cfg.ref_pos_dir[5] * (val_r * scale_1)
    
    # Double support phase
    # if np.abs(sin_pos) < cfg.double_stand_phase:
    #     ref_dof_pos = default_pos.copy()
        
    return ref_dof_pos

# =========================================================================
# 2. 新逻辑 (Proposed Logic with Hip Extension)
# =========================================================================
def compute_new_ref_state(current_time, cmd, default_pos, cfg):
    """
    新逻辑：支撑相髋关节后摆，全程连续运动
    """
    phase = (current_time % cfg.cycle_time) / cfg.cycle_time
    phase_l = phase
    phase_r = phase + 0.5
    sin_l = np.sin(2 * np.pi * phase_l)
    sin_r = np.sin(2 * np.pi * phase_r)
    
    # 简单的动态缩放模拟 (假设有速度)
    cmd_norm = np.sqrt(cmd.vx**2 + cmd.vy**2)
    dynamic_scale = 0.3 + 1.5 * np.clip(cmd_norm / 0.5, 0.0, 1.0)
    is_moving = 1.0 if cmd_norm > 0.01 else 0.0
    dynamic_scale *= is_moving
    
    scale_1 = cfg.target_joint_pos_scale
    scale_2 = 2 * scale_1
    
    ref_dof_pos = default_pos.copy()
    
    # --- Left Leg ---
    # Hip: 全程跟随正弦波 (摆动向前，支撑向后)
    ref_dof_pos[0] += cfg.ref_pos_dir[0] * (-sin_l) * dynamic_scale * scale_1
    # Knee/Ankle: 只在摆动相动
    if sin_l < 0:
        amp_l = np.abs(sin_l) * dynamic_scale
        ref_dof_pos[3] += cfg.ref_pos_dir[1] * amp_l * scale_2
        ref_dof_pos[4] += cfg.ref_pos_dir[2] * amp_l * scale_1

    # --- Right Leg ---
    # Hip: 全程跟随正弦波
    ref_dof_pos[6] += cfg.ref_pos_dir[3] * (-sin_r) * dynamic_scale * scale_1
    # Knee/Ankle: 只在摆动相动
    if sin_r < 0:
        amp_r = np.abs(sin_r) * dynamic_scale
        ref_dof_pos[9] += cfg.ref_pos_dir[4] * amp_r * scale_2
        ref_dof_pos[10]+= cfg.ref_pos_dir[5] * amp_r * scale_1

    # 去掉 Double Support 的强制归零，保持 Hip 连续性
    
    return ref_dof_pos

# =========================================================================
# Main Loop
# =========================================================================
def main():
    if not os.path.exists(XML_PATH):
        print(f"Error: XML file not found at {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    cfg = RefParams()
    cmd = Command()
    default_dof_pos = get_default_dof_pos_array(model)
    
    print("\n" + "="*50)
    print("开始对比可视化 (Comparison Visualization)")
    print("每 10 秒切换一次算法。")
    print("机器人将悬浮在空中。")
    print("="*50 + "\n")

    sim_time = 0.0
    
    # 给一点初始速度让动作幅度显现出来
    cmd.vx = 1.0 

    while True:
        # 1. 切换逻辑
        # mode 0: Old (User)
        # mode 1: New (Continuous Hip)
        mode = 0
        
        if mode == 0:
            target_q = compute_old_ref_state(sim_time, cmd, default_dof_pos, cfg)
            algo_name = ">>> OLD ALGO (Static Stance) <<<"
        else:
            target_q = compute_new_ref_state(sim_time, cmd, default_dof_pos, cfg)
            algo_name = ">>> NEW ALGO (Active Hip) <<<"

        # 2. 纯运动学赋值 (Kinematic Update)
        data.qpos[0] = 1.0 
        data.qpos[1] = 0.0 
        data.qpos[2] = 0.0  # 抬高
        data.qpos[3:7] = [0.0, 0.0, 0.0, 0.5] # 竖直姿态
        
        data.qpos[-model.nu:] = target_q
        data.qvel[:] = 0.0

        # 3. 刷新位置
        mujoco.mj_forward(model, data)

        # 4. 渲染与时间步进
        sim_time += cfg.dt
        
        if hasattr(viewer, 'is_alive') and not viewer.is_alive:
            break
        
        if int(sim_time / cfg.dt) % 20 == 0:
            viewer.render()
            # 在控制台打印当前状态
            print(f"\rTime: {sim_time:.1f}s | Mode: {algo_name}", end="")
            
        time.sleep(cfg.dt)

    viewer.close()

if __name__ == "__main__":
    main()