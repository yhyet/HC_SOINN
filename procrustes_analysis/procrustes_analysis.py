import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 对每个文件，按Task分组计算平均Procrustes距离
def calculate_task_averages(df):
    """计算每个task的平均procrustes距离"""
    task_averages = df.groupby('Task')['Procrustes_Dist'].mean().reset_index()
    return task_averages

def plot_procrustes_comparison(dataset_name, cllora_file, codaprompt_file, dualprompt_file, 
                                task_range, output_filename, title_suffix="", show_legend=False):
    """绘制Procrustes距离比较图"""
    # 读取三个CSV文件
    cllora_df = pd.read_csv(cllora_file)
    codaprompt_df = pd.read_csv(codaprompt_file)
    dualprompt_df = pd.read_csv(dualprompt_file)
    
    # 计算每个task的平均值
    cllora_avg = calculate_task_averages(cllora_df)
    codaprompt_avg = calculate_task_averages(codaprompt_df)
    dualprompt_avg = calculate_task_averages(dualprompt_df)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制三条折线（加粗线条和标记）
    ax.plot(cllora_avg['Task'], cllora_avg['Procrustes_Dist'], 
             marker='o', color='orange', label='CL-LoRA', linewidth=4, markersize=12)
    ax.plot(codaprompt_avg['Task'], codaprompt_avg['Procrustes_Dist'], 
             marker='s', color='blue', label='CODA-Prompt', linewidth=4, markersize=12)
    ax.plot(dualprompt_avg['Task'], dualprompt_avg['Procrustes_Dist'], 
             marker='^', color='green', label='DualPrompt', linewidth=4, markersize=12)
    
    # 在y=0.1处绘制横虚线作为参考线（加粗）
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=3, alpha=0.7, label='Reference (y=0.1)')
    
    # 设置图形属性（加大字体，不加粗）
    ax.set_xlabel('Task', fontsize=24)
    ax.set_ylabel('Procrustes Distance', fontsize=24)
    
    # 只在CUB数据集显示图例，并加大图例标记
    if show_legend:
        # 使用prop参数设置字体属性（不加粗，加大字体）
        font_prop = fm.FontProperties(size=22)
        legend = ax.legend(loc='best', prop=font_prop,
                          frameon=True, fancybox=True, shadow=True,
                          markerscale=3.5, handlelength=3.5, handletextpad=1.2)
        # 加大图例中的标记和线条
        for handle in legend.legend_handles:
            if hasattr(handle, 'set_markersize'):
                handle.set_markersize(25)  # 增大标记大小
            if hasattr(handle, 'set_linewidth'):
                handle.set_linewidth(5)  # 增大线条宽度
    
    # 加粗坐标轴
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    
    # 设置刻度标签（加大字体）
    ax.tick_params(axis='both', which='major', labelsize=20, width=3, length=8)
    ax.tick_params(axis='both', which='minor', width=2, length=5)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=2)
    ax.set_xticks(range(task_range[0], task_range[1] + 1))  # 设置x轴刻度
    ax.set_ylim(0.05, 0.45)  # 统一y轴范围
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图形已保存为 {output_filename}")
    
    # 显示图形
    plt.show()

# 处理CIFAR数据集（不显示图例）
print("处理CIFAR数据集...")
plot_procrustes_comparison(
    dataset_name='CIFAR',
    cllora_file='cllora_cifar_procrustes_distances.csv',
    codaprompt_file='codaprompt_cifar_procrustes_distances.csv',
    dualprompt_file='dualprompt_cifar_procrustes_distances.csv',
    task_range=(2, 10),
    output_filename='procrustes_distance_comparison_cifar.png',
    show_legend=False
)

# 处理CUB数据集（显示图例）
print("\n处理CUB数据集...")
plot_procrustes_comparison(
    dataset_name='CUB',
    cllora_file='cllora_cub_procrustes_distances.csv',
    codaprompt_file='codaprompt_cub_procrustes_distances.csv',
    dualprompt_file='dualprompt_cub_procrustes_distances.csv',
    task_range=(2, 20),
    output_filename='procrustes_distance_comparison_cub.png',
    show_legend=True
)

