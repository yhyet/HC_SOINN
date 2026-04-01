"""
HC-SOINN SOINN 边老化上限 age_max（代码中 hcsoinn_soinn_ad）敏感性；
绘图风格对齐 plot_alpha_sensitivity.py。生成两张图：SimpleCIL+INR、CODA-Prompt+CIFAR。
"""
from __future__ import annotations

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def _plot_age_max_core(
    avg_acc: list[float],
    last_acc: list[float],
    footnote: str,
    output_filename: str,
) -> None:
    x = [1, 3, 5, 10, 20, 30]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(
        x,
        avg_acc,
        marker="o",
        color="blue",
        label=r"$A_{\mathrm{Avg}}$",
        linewidth=4,
        markersize=12,
    )
    ax.plot(
        x,
        last_acc,
        marker="s",
        color="red",
        label=r"$A_{\mathrm{Last}}$",
        linewidth=4,
        markersize=12,
    )

    ax.set_xlabel(r"$\mathrm{age}_{\max}$", fontsize=32)
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

    ax.set_xticks(x)
    ax.set_xlim(x[0] - 1.5, x[-1] + 1.5)

    y_min = min(min(avg_acc), min(last_acc)) - 0.6
    y_max = max(max(avg_acc), max(last_acc)) + 0.6
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=2)

    plt.tight_layout(rect=[0, 0.11, 1, 1])

    # caption = (
    #     r"Sensitivity of $A_{\mathrm{Avg}}$ and $A_{\mathrm{Last}}$ to SOINN edge aging "
    #     r"budget $\mathrm{age}_{\max}$ ($T_{\mathrm{ad}}$ in implementation)."
    # )
    # fig.text(0.5, 0.065, caption, ha="center", va="bottom", fontsize=18)
    fig.text(0.5, 0.025, footnote, ha="center", va="bottom", fontsize=22, style="italic")

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图形已保存为 {output_filename}")


def plot_age_max_simplecil_inr() -> None:
    avg_acc = [70.97, 69.96, 69.96, 69.96, 69.96, 69.96]
    last_acc = [64.60, 63.93, 63.93, 63.93, 63.93, 63.93]
    _plot_age_max_core(
        avg_acc,
        last_acc,
        footnote="Setup: SimpleCIL + HC-SOINN on ImageNet-R",
        output_filename="age_max_sensitivity_simplecil_imagenetr.png",
    )


def plot_age_max_coda_cifar() -> None:
    avg_acc = [90.73, 91.48, 91.48, 91.48, 91.48, 91.48]
    last_acc = [87.01, 87.58, 87.58, 87.58, 87.58, 87.58]
    _plot_age_max_core(
        avg_acc,
        last_acc,
        footnote="Setup: CODA-Prompt + HC-SOINN on CIFAR-100",
        output_filename="age_max_sensitivity_coda_cifar100.png",
    )


if __name__ == "__main__":
    plot_age_max_simplecil_inr()
    plot_age_max_coda_cifar()
