import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def plot_node_budget_sensitivity():
    """绘制hcsoinn_max_proto_per_class (Node Budget $K_{max}$) 参数敏感性分析折线图"""
    
    # 数据
    kmax_values = [20, 40, 60, 80, 100, 500]
    avg_acc = [90.92, 91.06, 91.62, 91.39, 91.49, 92.28]
    last_acc = [86.66, 87.06, 87.56, 87.49, 87.66, 89.03]
    
    # 创建x轴位置映射：实际值 -> 显示位置
    # 20, 40, 60, 80, 100保持线性，500映射到130
    x_positions = [20, 40, 60, 80, 100, 130]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制两条折线（使用映射后的x轴位置）
    ax.plot(x_positions, avg_acc, 
             marker='o', color='blue', label='$A_{Avg}$', linewidth=4, markersize=12)
    ax.plot(x_positions, last_acc, 
             marker='s', color='red', label='$A_{Last}$', linewidth=4, markersize=12)
    
    # 设置图形属性（加大字体）
    ax.set_xlabel('$K_{max}$', fontsize=32)
    ax.set_ylabel('Accuracy (%)', fontsize=24)
    # ax.set_title('Sensitivity Analysis of Node Budget $K_{max}$', fontsize=26, pad=20)
    
    # 显示图例（加大字体）
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
    
    # 设置x轴刻度：使用映射后的位置，但标签显示实际值
    ax.set_xticks(x_positions)
    ax.set_xticklabels(kmax_values)
    # 设置x轴范围
    ax.set_xlim(0, 150)
    
    # 设置y轴范围（留出一些边距）
    y_min = min(min(avg_acc), min(last_acc)) - 1
    y_max = max(max(avg_acc), max(last_acc)) + 1
    ax.set_ylim(y_min, y_max)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_filename = 'node_budget_sensitivity_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图形已保存为 {output_filename}")
    
    # 显示图形
    plt.show()

if __name__ == '__main__':
    plot_node_budget_sensitivity()

