import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def plot_alpha_sensitivity():
    """Handle plot alpha sensitivity."""
    dataset_name = "CIFAR-100"  # e.g. "ImageNet-R", "CUB-200", etc.

    alpha_values = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    avg_acc = [64.31, 68.17, 69.69, 69.97, 70.43, 69.40, 68.05]
    last_acc = [58.28, 62.02, 63.63, 63.93, 63.98, 63.00, 61.35]
    
    # avg_acc = [64.31, 68.17, 69.69, 69.97, 70.43, 69.40, 68.05]
    # last_acc = [58.28, 62.02, 63.63, 63.93, 63.98, 63.00, 61.35]
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.plot(alpha_values, avg_acc, 
             marker='o', color='blue', label='$A_{Avg}$', linewidth=4, markersize=12)
    ax.plot(alpha_values, last_acc, 
             marker='s', color='red', label='$A_{Last}$', linewidth=4, markersize=12)
    
    ax.set_xlabel('$\\alpha$', fontsize=32)
    ax.set_ylabel('Accuracy (%)', fontsize=24)
    # ax.set_title('Sensitivity Analysis of hcsoinn_alpha', fontsize=26, pad=20)
    
    font_prop = fm.FontProperties(size=22)
    legend = ax.legend(loc='best', prop=font_prop,
                      frameon=True, fancybox=True, shadow=True,
                      markerscale=3.5, handlelength=3.5, handletextpad=1.2)
    
    for handle in legend.legend_handles:
        if hasattr(handle, 'set_markersize'):
            handle.set_markersize(25)
        if hasattr(handle, 'set_linewidth'):
            handle.set_linewidth(5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    
    ax.tick_params(axis='both', which='major', labelsize=20, width=3, length=8)
    ax.tick_params(axis='both', which='minor', width=2, length=5)
    
    ax.set_xticks(alpha_values)
    ax.set_xlim(-0.1, 1.1)
    
    y_min = min(min(avg_acc), min(last_acc)) - 1
    y_max = max(max(avg_acc), max(last_acc)) + 1
    ax.set_ylim(y_min, y_max)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=2)
    
    caption = (
        r"Fig. Sensitivity of $A_{\mathrm{Avg}}$ and $A_{\mathrm{Last}}$ to HC-SOINN "
        r"fusion weight $\alpha$ (distance mix of NCM and sub-cluster prototypes)."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=16)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    output_filename = 'alpha_sensitivity_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图形已保存为 {output_filename}")
    
    plt.show()

if __name__ == '__main__':
    plot_alpha_sensitivity()

