# HC-SOINN Alpha 参数敏感性分析实验脚本
# 依次运行 hcsoinn_alpha = 0, 0.2, 0.4, 0.6, 0.8 的实验

$alpha_values = @(0, 0.2, 0.4, 0.6, 0.8)

Write-Host "开始 HC-SOINN Alpha 参数敏感性分析实验" -ForegroundColor Green
Write-Host "实验参数: hcsoinn_alpha = $($alpha_values -join ', ')" -ForegroundColor Cyan
Write-Host ""

foreach ($alpha in $alpha_values) {
    $config_file = "exps/coda_prompt_hc_soinn_alpha_$alpha.json"
    
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "运行实验: hcsoinn_alpha = $alpha" -ForegroundColor Yellow
    Write-Host "配置文件: $config_file" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    
    & python main.py --config=$config_file
    $success = $?
    
    if (-not $success) {
        Write-Host "实验失败: hcsoinn_alpha = $alpha" -ForegroundColor Red
        Write-Host "是否继续运行下一个实验? (Y/N)" -ForegroundColor Yellow
        $continue = Read-Host
        if ($continue -ne "Y" -and $continue -ne "y") {
            Write-Host "实验已停止" -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host "实验完成: hcsoinn_alpha = $alpha" -ForegroundColor Green
    }
    
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "所有实验已完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
