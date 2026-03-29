# HC-SOINN soinn_max_iter sensitivity runner
# Runs hcsoinn_soinn_max_iter = 0, 1, 2, 3, 4, 5 sequentially

$experiments = @(
    @{ max_iter = 1; config = "exps/coda_prompt_hc_soinn.json" },
    @{ max_iter = 0; config = "exps/coda_prompt_hc_soinn_maxiter_0.json" },
    @{ max_iter = 2; config = "exps/coda_prompt_hc_soinn_maxiter_2.json" },
    @{ max_iter = 3; config = "exps/coda_prompt_hc_soinn_maxiter_3.json" },
    @{ max_iter = 4; config = "exps/coda_prompt_hc_soinn_maxiter_4.json" },
    @{ max_iter = 5; config = "exps/coda_prompt_hc_soinn_maxiter_5.json" }
)

Write-Host "Starting HC-SOINN soinn_max_iter sensitivity runs" -ForegroundColor Green
Write-Host "Values: hcsoinn_soinn_max_iter = 0, 1, 2, 3, 4, 5" -ForegroundColor Cyan
Write-Host ""

$failedExperiments = @()
$succeededExperiments = @()

foreach ($exp in $experiments) {
    $max_iter = $exp.max_iter
    $config_file = $exp.config

    if (-not (Test-Path $config_file)) {
        Write-Host "Missing config file: $config_file" -ForegroundColor Red
        Write-Host "Stopping script." -ForegroundColor Red
        exit 1
    }

    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "Running experiment: hcsoinn_soinn_max_iter = $max_iter" -ForegroundColor Yellow
    Write-Host "Config file: $config_file" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""

    & python main.py --config=$config_file
    $success = $?

    if (-not $success) {
        Write-Host "Experiment failed: hcsoinn_soinn_max_iter = $max_iter" -ForegroundColor Red
        $failedExperiments += $max_iter
        Write-Host "Continuing to the next experiment..." -ForegroundColor Yellow
    }
    else {
        Write-Host "Experiment finished: hcsoinn_soinn_max_iter = $max_iter" -ForegroundColor Green
        $succeededExperiments += $max_iter
    }

    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "All experiments finished." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "Succeeded: $($succeededExperiments -join ', ')" -ForegroundColor Green
if ($failedExperiments.Count -gt 0) {
    Write-Host "Failed: $($failedExperiments -join ', ')" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "Failed: none" -ForegroundColor Green
    exit 0
}
