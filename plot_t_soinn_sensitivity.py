"""
HC-SOINN SOINN 精炼迭代次数 T_soinn（hcsoinn_soinn_max_iter）敏感性：
三个独立子图分别绘制 A_Avg、A_Last、每类平均原型数。
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


def _style_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis="both", which="major", labelsize=20, width=3, length=8)
    ax.tick_params(axis="both", which="minor", width=2, length=5)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=2)


def plot_t_soinn_sensitivity():
    t_soinn = [0, 1, 2, 3, 4, 5]
    mean_prototypes_per_class = [58.95, 19.29, 18.84, 17.91, 16.59, 15.13]
    avg_acc = [71.08, 69.97, 70.01, 69.72, 69.50, 69.31]
    last_acc = [65.00, 63.93, 63.80, 63.62, 63.38, 63.12]

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    color_acc_avg = "blue"
    color_acc_last = "red"
    color_proto = "#2ca02c"

    font_prop = fm.FontProperties(size=20)

    # --- (a) A_Avg ---
    ax0 = axes[0]
    ax0.plot(
        t_soinn,
        avg_acc,
        marker="o",
        color=color_acc_avg,
        linewidth=4,
        markersize=12,
        label=r"$A_{\mathrm{Avg}}$",
    )
    ax0.set_ylabel("Accuracy (%)", fontsize=24)
    leg0 = ax0.legend(loc="best", prop=font_prop, frameon=True, fancybox=True, shadow=True)
    for h in leg0.legend_handles:
        if hasattr(h, "set_markersize"):
            h.set_markersize(22)
        if hasattr(h, "set_linewidth"):
            h.set_linewidth(5)
    y0_min, y0_max = min(avg_acc) - 1, max(avg_acc) + 1
    ax0.set_ylim(y0_min, y0_max)
    ax0.set_xticks(t_soinn)
    ax0.set_xlim(-0.3, 5.3)
    _style_axis(ax0)

    # --- (b) A_Last ---
    ax1 = axes[1]
    ax1.plot(
        t_soinn,
        last_acc,
        marker="s",
        color=color_acc_last,
        linewidth=4,
        markersize=12,
        label=r"$A_{\mathrm{Last}}$",
    )
    ax1.set_ylabel("Accuracy (%)", fontsize=24)
    leg1 = ax1.legend(loc="best", prop=font_prop, frameon=True, fancybox=True, shadow=True)
    for h in leg1.legend_handles:
        if hasattr(h, "set_markersize"):
            h.set_markersize(22)
        if hasattr(h, "set_linewidth"):
            h.set_linewidth(5)
    y1_min, y1_max = min(last_acc) - 1, max(last_acc) + 1
    ax1.set_ylim(y1_min, y1_max)
    _style_axis(ax1)

    # --- (c) Mean prototypes per class ---
    ax2 = axes[2]
    ax2.plot(
        t_soinn,
        mean_prototypes_per_class,
        marker="^",
        color=color_proto,
        linewidth=4,
        markersize=12,
        label="Mean prototypes / class",
    )
    ax2.set_ylabel("Mean prototypes / class", fontsize=22)
    leg2 = ax2.legend(loc="best", prop=font_prop, frameon=True, fancybox=True, shadow=True)
    for h in leg2.legend_handles:
        if hasattr(h, "set_markersize"):
            h.set_markersize(22)
        if hasattr(h, "set_linewidth"):
            h.set_linewidth(5)
    proto_pad = (max(mean_prototypes_per_class) - min(mean_prototypes_per_class)) * 0.08 + 1
    ax2.set_ylim(
        min(mean_prototypes_per_class) - proto_pad,
        max(mean_prototypes_per_class) + proto_pad,
    )
    ax2.set_xlabel(r"$T_{\mathrm{soinn}}$", fontsize=32)
    _style_axis(ax2)
    # ax2.spines["left"].set_edgecolor(color_proto)
    # ax2.tick_params(axis="y", labelcolor=color_proto)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    # caption = (
    #     r"Setup: SimpleCIL + HC-SOINN on Imagenet-R"

    # )
    footnote =  "Setup: SimpleCIL + HC-SOINN on Imagenet-R"
    # fig.text(0.5, 0.048, caption, ha="center", va="bottom", fontsize=18)
    fig.text(0.5, 0.018, footnote, ha="center", va="bottom", fontsize=24, style="italic")

    output_filename = "t_soinn_sensitivity_analysis.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"图形已保存为 {output_filename}")
    plt.show()


if __name__ == "__main__":
    plot_t_soinn_sensitivity()
