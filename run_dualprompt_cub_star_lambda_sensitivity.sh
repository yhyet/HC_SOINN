#!/bin/bash
# DualPrompt + HC-SOINN + STAR 在 CUB 上的 star_lambda 参数敏感性实验脚本
# 依次运行 star_lambda = 0.999, 0.99, 0.95, 0.9 的实验

echo "开始 DualPrompt + HC-SOINN + STAR (CUB) 的 star_lambda 参数敏感性实验"
echo "实验参数: star_lambda = 0.999, 0.99, 0.95, 0.9"
if [ $# -gt 0 ]; then
    echo "附加运行参数: $*"
fi
echo ""

lambda_values=(0p999 0p99 0p95 0p9)

for lambda_tag in "${lambda_values[@]}"; do
    config_file="exps/dualprompt_hc_soinn_cub_star_lambda_${lambda_tag}.json"

    echo "========================================"
    echo "运行实验: star_lambda = ${lambda_tag//p/.}"
    echo "配置文件: $config_file"
    echo "========================================"
    echo ""

    python main.py --config="$config_file" "$@"

    if [ $? -ne 0 ]; then
        echo "实验失败: star_lambda = ${lambda_tag//p/.}"
        read -p "是否继续运行下一个实验? (Y/N): " continue_run
        if [ "$continue_run" != "Y" ] && [ "$continue_run" != "y" ]; then
            echo "实验已停止"
            exit 1
        fi
    else
        echo "实验完成: star_lambda = ${lambda_tag//p/.}"
    fi

    echo ""
done

echo "========================================"
echo "所有实验已完成!"
echo "========================================"
