import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def plot_node_budget_sensitivity():
    """Handle plot node budget sensitivity."""
    
    kmax_values = [20, 40, 60, 80, 100, 500]
    avg_acc = [90.92, 91.06, 91.62, 91.39, 91.49, 92.28]
    last_acc = [86.66, 87.06, 87.56, 87.49, 87.66, 89.03]
    
    x_positions = [20, 40, 60, 80, 100, 130]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.plot(x_positions, avg_acc, 
             marker='o', color='blue', label='$A_{Avg}$', linewidth=4, markersize=12)
    ax.plot(x_positions, last_acc, 
             marker='s', color='red', label='$A_{Last}$', linewidth=4, markersize=12)
    
    ax.set_xlabel('$K_{init}$', fontsize=32)
    ax.set_ylabel('Accuracy (%)', fontsize=24)
    # ax.set_title('Sensitivity Analysis of Node Budget $K_{max}$', fontsize=26, pad=20)
    
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
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(kmax_values)
    ax.set_xlim(0, 150)
    
    y_min = min(min(avg_acc), min(last_acc)) - 1
    y_max = max(max(avg_acc), max(last_acc)) + 1
    ax.set_ylim(y_min, y_max)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=2)

    plt.tight_layout(rect=[0, 0.11, 1, 1])

    # caption = (
    #     r"Sensitivity of $A_{\mathrm{Avg}}$ and $A_{\mathrm{Last}}$ to HC-SOINN "
    #     r"per-class prototype budget $K_{\max}$ "
    #     r"(maximum sub-prototypes after hierarchical clustering)."
    # )
    footnote = "Setup: CODA-Prompt + HC-SOINN on CIFAR-100"
    # fig.text(0.5, 0.065, caption, ha="center", va="bottom", fontsize=20)
    fig.text(0.5, 0.025, footnote, ha="center", va="bottom", fontsize=24, style="italic")

    output_filename = 'node_budget_sensitivity_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图形已保存为 {output_filename}")
    
    plt.show()

if __name__ == '__main__':
    plot_node_budget_sensitivity()

