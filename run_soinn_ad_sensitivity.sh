#!/bin/bash
# HC-SOINN soinn_ad 参数敏感性分析实验脚本
# 依次运行 hcsoinn_soinn_ad = 5, 10, 15, 25, 30 的实验

echo "开始 HC-SOINN soinn_ad 参数敏感性分析实验"
echo "实验参数: hcsoinn_soinn_ad = 5, 10, 15, 25, 30"
if [ $# -gt 0 ]; then
    echo "附加运行参数: $*"
fi
echo ""

ad_values=(5 10 15 25 30)

for ad in "${ad_values[@]}"; do
    config_file="exps/simplecil_hc_soinn_cub_ad_${ad}.json"

    echo "========================================"
    echo "运行实验: hcsoinn_soinn_ad = $ad"
    echo "配置文件: $config_file"
    echo "========================================"
    echo ""

    python main.py --config="$config_file" "$@"

    if [ $? -ne 0 ]; then
        echo "实验失败: hcsoinn_soinn_ad = $ad"
        read -p "是否继续运行下一个实验? (Y/N): " continue_run
        if [ "$continue_run" != "Y" ] && [ "$continue_run" != "y" ]; then
            echo "实验已停止"
            exit 1
        fi
    else
        echo "实验完成: hcsoinn_soinn_ad = $ad"
    fi

    echo ""
done

echo "========================================"
echo "所有实验已完成!"
echo "========================================"
