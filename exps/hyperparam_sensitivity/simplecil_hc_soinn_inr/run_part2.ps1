# SimpleCIL + HC-SOINN INR sensitivity runner - part 2
# Runs 11 experiments sequentially.

$configDir = "exps/hyperparam_sensitivity/simplecil_hc_soinn_inr"
$configs = @(
    "maxproto_500.json",
    "maxiter_0.json",
    "maxiter_2.json",
    "maxiter_3.json",
    "maxiter_4.json",
    "maxiter_5.json",
    "soinn_ad_1.json",
    "soinn_ad_2.json",
    "soinn_ad_3.json",
    "soinn_ad_4.json",
    "soinn_ad_5.json"
)

$failed = @()
$succeeded = @()

Write-Host "Starting INR sensitivity runs - part 2" -ForegroundColor Green
Write-Host "Config directory: $configDir" -ForegroundColor Cyan
Write-Host "Total experiments: $($configs.Count)" -ForegroundColor Cyan
Write-Host ""

foreach ($file in $configs) {
    $configPath = Join-Path $configDir $file

    if (-not (Test-Path $configPath)) {
        Write-Host "Missing config file: $configPath" -ForegroundColor Red
        $failed += $file
        continue
    }

    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "Running: $configPath" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""

    & python main.py --config=$configPath
    if ($?) {
        $succeeded += $file
        Write-Host "Finished: $file" -ForegroundColor Green
    }
    else {
        $failed += $file
        Write-Host "Failed: $file" -ForegroundColor Red
    }

    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "Part 2 finished." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Succeeded count: $($succeeded.Count)" -ForegroundColor Green

if ($failed.Count -gt 0) {
    Write-Host "Failed count: $($failed.Count)" -ForegroundColor Red
    Write-Host "Failed files: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "Failed count: 0" -ForegroundColor Green
    exit 0
}
