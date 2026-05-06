import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def calculate_task_averages(df):
    """Handle calculate task averages."""
    task_averages = df.groupby('Task')['Procrustes_Dist'].mean().reset_index()
    return task_averages

def plot_procrustes_comparison(dataset_name, cllora_file, codaprompt_file, dualprompt_file, 
                                task_range, output_filename, title_suffix="", show_legend=False):
    """Handle plot procrustes comparison."""
    cllora_df = pd.read_csv(cllora_file)
    codaprompt_df = pd.read_csv(codaprompt_file)
    dualprompt_df = pd.read_csv(dualprompt_file)
    
    cllora_avg = calculate_task_averages(cllora_df)
    codaprompt_avg = calculate_task_averages(codaprompt_df)
    dualprompt_avg = calculate_task_averages(dualprompt_df)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.plot(cllora_avg['Task'], cllora_avg['Procrustes_Dist'], 
             marker='o', color='orange', label='CL-LoRA', linewidth=4, markersize=12)
    ax.plot(codaprompt_avg['Task'], codaprompt_avg['Procrustes_Dist'], 
             marker='s', color='blue', label='CODA-Prompt', linewidth=4, markersize=12)
    ax.plot(dualprompt_avg['Task'], dualprompt_avg['Procrustes_Dist'], 
             marker='^', color='green', label='DualPrompt', linewidth=4, markersize=12)
    
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=3, alpha=0.7, label='Reference (y=0.1)')
    
    ax.set_xlabel('Task', fontsize=24)
    ax.set_ylabel('Procrustes Distance', fontsize=24)
    
    if show_legend:
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
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=2)
    ax.set_xticks(range(task_range[0], task_range[1] + 1))
    ax.set_ylim(0.05, 0.45)
    
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"{output_filename}")
    
    plt.show()


plot_procrustes_comparison(
    dataset_name='CIFAR',
    cllora_file='cllora_cifar_procrustes_distances.csv',
    codaprompt_file='codaprompt_cifar_procrustes_distances.csv',
    dualprompt_file='dualprompt_cifar_procrustes_distances.csv',
    task_range=(2, 10),
    output_filename='procrustes_distance_comparison_cifar.png',
    show_legend=False
)


plot_procrustes_comparison(
    dataset_name='CUB',
    cllora_file='cllora_cub_procrustes_distances.csv',
    codaprompt_file='codaprompt_cub_procrustes_distances.csv',
    dualprompt_file='dualprompt_cub_procrustes_distances.csv',
    task_range=(2, 20),
    output_filename='procrustes_distance_comparison_cub.png',
    show_legend=True
)

