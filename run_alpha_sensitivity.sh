#!/bin/bash
# HC-SOINN Alpha 参数敏感性分析实验脚本
# 依次运行 hcsoinn_alpha = 0, 0.2, 0.4, 0.6, 0.8 的实验

echo "开始 HC-SOINN Alpha 参数敏感性分析实验"
echo "实验参数: hcsoinn_alpha = 0, 0.2, 0.4, 0.6, 0.8"
echo ""

alpha_values=(0 0.2 0.4 0.6 0.8)

for alpha in "${alpha_values[@]}"; do
    config_file="exps/coda_prompt_hc_soinn_alpha_${alpha}.json"
    
    echo "========================================"
    echo "运行实验: hcsoinn_alpha = $alpha"
    echo "配置文件: $config_file"
    echo "========================================"
    echo ""
    
    python main.py --config=$config_file
    
    if [ $? -ne 0 ]; then
        echo "实验失败: hcsoinn_alpha = $alpha"
        read -p "是否继续运行下一个实验? (Y/N): " continue
        if [ "$continue" != "Y" ] && [ "$continue" != "y" ]; then
            echo "实验已停止"
            exit 1
        fi
    else
        echo "实验完成: hcsoinn_alpha = $alpha"
    fi
    
    echo ""
done

echo "========================================"
echo "所有实验已完成!"
echo "========================================"

