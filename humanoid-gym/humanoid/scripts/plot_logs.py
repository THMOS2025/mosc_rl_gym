# joint_act: act （缩放后target_q）
# joint_pos: obs_q （未经缩放）
# joint_vel: obs_dq （未经缩放）
# base_ang_eul: rpy 
# base_ang_vel: omega
# cmd: ???前三个act
# imput_omega: omega(指数平滑)

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_data_from_folder(folder_path):
    """
    扫描指定文件夹中的所有.txt文件，将它们作为CSV格式读取，
    并为每个文件生成一个折线图。文件中的每一列数据都会被绘制在一个独立的子图中。
    生成的图表（.png）保存在同一文件夹下。

    Args:
        folder_path (str): 包含.txt文件的数据文件夹路径。
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return

    # 获取文件夹下所有文件名
    try:
        files = os.listdir(folder_path)
    except OSError as e:
        print(f"错误: 无法访问文件夹 '{folder_path}': {e}")
        return

    # 筛选出.txt文件
    txt_files = [f for f in files if f.endswith('.txt')]

    if not txt_files:
        print(f"在 '{folder_path}' 文件夹中未找到.txt文件。")
        return

    print(f"找到 {len(txt_files)} 个.txt文件，开始绘图...")

    # 遍历并处理每个.txt文件
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        # 从文件名中提取图表标题 (去掉.txt后缀)
        plot_title = os.path.splitext(file_name)[0]
        output_path = os.path.join(folder_path, f"{plot_title}.png")

        print(f"正在处理 '{file_name}'...")

        try:
            # 使用pandas读取数据，将文件视为无表头的CSV
            # - header=None: 文件没有表头行
            # - comment='#': 忽略以'#'开头的行
            # - sep=',': 使用逗号作为分隔符
            data = pd.read_csv(file_path, header=None, comment='#', sep=',')
            
            # 删除因行尾多余逗号而产生的全为NaN的列
            data.dropna(axis=1, how='all', inplace=True)

            # 如果读取后数据为空，则跳过此文件
            if data.empty:
                print(f"  - 警告: '{file_name}' 为空或只包含注释。已跳过。")
                continue

            num_columns = data.shape[1]

            # 创建一个图形(figure)和N个子图(subplots)，N为数据列数
            # 垂直排列子图，共享X轴
            fig, axes = plt.subplots(num_columns, 1, figsize=(12, 2.5 * num_columns), sharex=True)
            
            # 如果只有一列数据，plt.subplots返回的不是数组，我们将其转换为列表以便统一处理
            if num_columns == 1:
                axes = [axes]

            # 设置整个图表的总标题
            fig.suptitle(plot_title, fontsize=16)

            # 遍历每一列数据并在对应的子图上绘图
            for i, ax in enumerate(axes):
                ax.plot(data.index, data.iloc[:, i])
                ax.set_title(f'Column {i+1}')
                ax.set_ylabel('Value')
                ax.grid(True)

            # 为最下方的子图设置X轴标签
            axes[-1].set_xlabel('Sample Index')

            # 调整布局，防止标题和标签重叠
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # 保存图表到文件
            plt.savefig(output_path)
            print(f"  - 图表已保存至 '{output_path}'")

            # 关闭当前图表，释放内存
            plt.close(fig)

        except Exception as e:
            print(f"  - 处理 '{file_name}' 时发生错误: {e}")
            # 如果在绘图过程中出错，也确保关闭图表
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)

    print("\n所有文件处理完毕。")

# --- 脚本主执行区 ---
if __name__ == '__main__':
    # ====================================================================
    # 使用说明:
    # 1. 将下面的 'target_folder' 变量的值修改为您存放.txt文件的文件夹路径。
    #    - 您可以使用相对路径 (例如 'data') 或绝对路径
    #      (例如 'C:/Users/YourUser/Documents/MyData')。
    # 2. 运行此脚本。
    # ====================================================================

    # 【【【 请修改这里 】】】
    # 指定包含.txt文件的文件夹路径
    target_folder = '/home/thmos/MOSC_RL_GYM/data_logs/observation_intheair_vel0_video' # <--- 重要：请务必设置此路径

    # --- 使用示例 ---
    # 示例1: 如果您的.txt文件在一个名为"data"的子文件夹中
    # target_folder = 'data'
    
    # 示例2: 如果您的.txt文件与此脚本在同一个文件夹中
    # target_folder = '.'


    # 检查用户是否已修改路径，如果未修改则打印提示信息
    if target_folder == 'path/to/your/txt/files':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 请先编辑此脚本，设置 'target_folder' 变量的正确路径 !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # 调用主函数开始执行绘图任务
        plot_data_from_folder(target_folder)