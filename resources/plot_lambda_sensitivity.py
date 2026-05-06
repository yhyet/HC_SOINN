"""Core component."""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


def plot_lambda_sensitivity():
    lambda_values = [1.0, 0.999, 0.99, 0.95, 0.9]
    avg_acc = [92.60, 92.60, 92.60, 92.62, 92.63]
    last_acc = [89.15, 89.15, 89.16, 89.22, 89.23]
    x_display = [0.0, 1.5, 4.0, 7.5, 12.0]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(
        x_display,
        avg_acc,
        marker="o",
        color="blue",
        label=r"$A_{\mathrm{Avg}}$",
        linewidth=4,
        markersize=12,
    )
    ax.plot(
        x_display,
        last_acc,
        marker="s",
        color="red",
        label=r"$A_{\mathrm{Last}}$",
        linewidth=4,
        markersize=12,
    )

    ax.set_xlabel(r"$\lambda$", fontsize=32)
    ax.set_ylabel("Accuracy (%)", fontsize=24)

    font_prop = fm.FontProperties(size=22)
    legend = ax.legend(
        loc="best",
        prop=font_prop,
        frameon=True,
        fancybox=True,
        shadow=True,
        markerscale=3.5,
        handlelength=3.5,
        handletextpad=1.2,
    )
    for handle in legend.legend_handles:
        if hasattr(handle, "set_markersize"):
            handle.set_markersize(25)
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(5)

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    ax.tick_params(axis="both", which="major", labelsize=20, width=3, length=8)
    ax.tick_params(axis="both", which="minor", width=2, length=5)

    ax.set_xticks(x_display)
    ax.set_xticklabels(["1", "0.999", "0.99", "0.95", "0.9"])
    ax.set_xlim(x_display[0] - 0.8, x_display[-1] + 0.8)

    pad = 0.12
    y_min = min(min(avg_acc), min(last_acc)) - pad
    y_max = max(max(avg_acc), max(last_acc)) + pad
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=2)

    plt.tight_layout(rect=[0, 0.11, 1, 1])

    # caption = (
    #     r"Sensitivity of $A_{\mathrm{Avg}}$ and $A_{\mathrm{Last}}$ to STAR trajectory "
    #     r"mixing weight $\lambda$."
    # )
    footnote = "Setup: CODA-Prompt + HC-SOINN + STAR on CIFAR-100"
    # fig.text(0.5, 0.065, caption, ha="center", va="bottom", fontsize=20)
    fig.text(0.5, 0.025, footnote, ha="center", va="bottom", fontsize=24, style="italic")

    output_filename = "lambda_sensitivity_analysis.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"图形已保存为 {output_filename}")
    plt.show()


if __name__ == "__main__":
    plot_lambda_sensitivity()
