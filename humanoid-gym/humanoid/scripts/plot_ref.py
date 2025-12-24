import numpy as np
import matplotlib.pyplot as plt

def visualize_complete_gait_logic():
    # 模拟参数
    num_points = 1000
    phase = np.linspace(0, 2, num_points) # 2个周期
    double_stand_threshold = 0.1 # 假设阈值，对应 cfg.rewards.double_stand_phase
    
    # ================= 逻辑复现 (Numpy版) =================
    
    # 1. 基础波形
    sin_pos = np.sin(2 * np.pi * phase)
    
    # 2. 原始左右腿波形 (Step 1)
    sin_pos_l = sin_pos - 0.5
    sin_pos_r = sin_pos + 0.5
    
    # 3. 运动逻辑计算 (Step 3)
    # Left: clamp max=0
    val_l = np.clip(sin_pos_l, -np.inf, 0.0)
    # Right: clamp min=0
    val_r = np.clip(sin_pos_r, 0.0, np.inf)
    
    # 4. 双支撑逻辑 (Step 4 & Mask Logic)
    # 判断是否处于双支撑相位 (基于原始 sin_pos 的绝对值)
    is_double_support = np.abs(sin_pos) < double_stand_threshold
    
    # === 关键点：Step 4 强制归零 ===
    # 代码原话: self.ref_dof_pos[double_support] = self.default_dof_pos...
    # 这意味着在双支撑期间，偏移量 val_l 和 val_r 被强制设为 0
    val_l_final = val_l.copy()
    val_l_final[is_double_support] = 0.0
    
    val_r_final = val_r.copy()
    val_r_final[is_double_support] = 0.0
    
    # 5. Mask 逻辑 (_get_gait_phase)
    # Left Mask
    mask_l = (sin_pos >= 0).astype(float)
    mask_l[is_double_support] = 1.0 # 强制设为支撑
    
    # Right Mask
    mask_r = (sin_pos < 0).astype(float)
    mask_r[is_double_support] = 1.0 # 强制设为支撑

    # ================= 绘图 =================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 辅助函数：绘制背景区域
    def plot_background(ax, mask, is_ds):
        # 1. 单腿支撑 (绿色) = Mask是1 且 不是双支撑
        single_stance = (mask == 1) & (~is_ds)
        ax.fill_between(phase, -1.5, 1.5, where=single_stance, color='#d9f7be', alpha=0.8, label='Single Stance (Load)')
        
        # 2. 双支撑 (金色) = explicitly defined
        ax.fill_between(phase, -1.5, 1.5, where=is_ds, color='#fffb8f', alpha=1.0, label='Double Support (Reset)')
        
        # 3. 摆动相 (粉色) = Mask是0
        swing = (mask == 0)
        ax.fill_between(phase, -1.5, 1.5, where=swing, color='#ffccc7', alpha=0.5, label='Swing (Air)')

    # --- 左腿绘图 ---
    ax1.set_title("Left Leg Logic", fontsize=14, fontweight='bold')
    plot_background(ax1, mask_l, is_double_support)
    
    # 绘制参考线
    ax1.plot(phase, sin_pos, '--', color='gray', alpha=0.3, label='Base Sine')
    ax1.plot(phase, sin_pos_l, ':', color='blue', alpha=0.3, label='Raw Offset (-0.5)')
    
    # 绘制最终运动曲线 (加粗)
    # 注意：这里画的是 val_l_final，体现了 Step 4 的归零效果
    ax1.plot(phase, val_l_final, color='blue', linewidth=2.5, label='Final Motion Ref')
    
    ax1.set_ylim(-1.2, 0.5)
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # --- 右腿绘图 ---
    ax2.set_title("Right Leg Logic", fontsize=14, fontweight='bold')
    plot_background(ax2, mask_r, is_double_support)
    
    # 绘制参考线
    ax2.plot(phase, sin_pos, '--', color='gray', alpha=0.3, label='Base Sine')
    ax2.plot(phase, sin_pos_r, ':', color='orange', alpha=0.3, label='Raw Offset (+0.5)')
    
    # 绘制最终运动曲线
    ax2.plot(phase, val_r_final, color='#d46b08', linewidth=2.5, label='Final Motion Ref')
    
    ax2.set_ylim(-0.5, 1.2)
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Phase (Cycles)")
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Gait Logic Visualization with Double Support Reset (Thresh={double_stand_threshold})", fontsize=16)
    plt.show()

if __name__ == "__main__":
    visualize_complete_gait_logic()