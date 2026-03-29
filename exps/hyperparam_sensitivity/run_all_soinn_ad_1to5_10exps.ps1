# Sequential runner: 10 hcsoinn_soinn_ad sensitivity experiments (ad = 1..5 each).
# 1-5: SimpleCIL + HC-SOINN, ImageNet-R (imagenetr)
# 6-10: CodaPrompt + HC-SOINN, CIFAR-224 (cifar224)
#
# Run from repo root: D:\LAMDA-PILOT_2

# This script lives in exps/hyperparam_sensitivity/ -> repo root is two levels up.
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
if (-not (Test-Path (Join-Path $repoRoot "main.py"))) {
    Write-Host "Could not find main.py under $repoRoot. Run this script from repo root or fix path." -ForegroundColor Red
    exit 1
}

$experiments = @(
    @{ label = "simplecil_hc_soinn_inr soinn_ad_1"; path = "exps/hyperparam_sensitivity/simplecil_hc_soinn_inr/soinn_ad_1.json" },
    @{ label = "simplecil_hc_soinn_inr soinn_ad_2"; path = "exps/hyperparam_sensitivity/simplecil_hc_soinn_inr/soinn_ad_2.json" },
    @{ label = "simplecil_hc_soinn_inr soinn_ad_3"; path = "exps/hyperparam_sensitivity/simplecil_hc_soinn_inr/soinn_ad_3.json" },
    @{ label = "simplecil_hc_soinn_inr soinn_ad_4"; path = "exps/hyperparam_sensitivity/simplecil_hc_soinn_inr/soinn_ad_4.json" },
    @{ label = "simplecil_hc_soinn_inr soinn_ad_5"; path = "exps/hyperparam_sensitivity/simplecil_hc_soinn_inr/soinn_ad_5.json" },
    @{ label = "coda_prompt_hc_soinn_cifar soinn_ad_1"; path = "exps/hyperparam_sensitivity/coda_prompt_hc_soinn_cifar/soinn_ad_1.json" },
    @{ label = "coda_prompt_hc_soinn_cifar soinn_ad_2"; path = "exps/hyperparam_sensitivity/coda_prompt_hc_soinn_cifar/soinn_ad_2.json" },
    @{ label = "coda_prompt_hc_soinn_cifar soinn_ad_3"; path = "exps/hyperparam_sensitivity/coda_prompt_hc_soinn_cifar/soinn_ad_3.json" },
    @{ label = "coda_prompt_hc_soinn_cifar soinn_ad_4"; path = "exps/hyperparam_sensitivity/coda_prompt_hc_soinn_cifar/soinn_ad_4.json" },
    @{ label = "coda_prompt_hc_soinn_cifar soinn_ad_5"; path = "exps/hyperparam_sensitivity/coda_prompt_hc_soinn_cifar/soinn_ad_5.json" }
)

$failed = @()
$succeeded = @()

Write-Host "Running 10 hcsoinn_soinn_ad experiments (1-5 INR + 1-5 CIFAR coda)" -ForegroundColor Green
Write-Host "Repo root: $repoRoot" -ForegroundColor Cyan
Write-Host ""

Push-Location $repoRoot
try {
    foreach ($exp in $experiments) {
        $configPath = Join-Path $repoRoot $exp.path
        if (-not (Test-Path $configPath)) {
            Write-Host "MISSING: $($exp.path)" -ForegroundColor Red
            $failed += $exp.label
            continue
        }
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "$($exp.label)" -ForegroundColor Yellow
        Write-Host $configPath -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow
        & python main.py --config=$configPath
        if ($?) {
            $succeeded += $exp.label
            Write-Host "OK: $($exp.label)" -ForegroundColor Green
        }
        else {
            $failed += $exp.label
            Write-Host "FAIL: $($exp.label)" -ForegroundColor Red
        }
        Write-Host ""
    }
}
finally {
    Pop-Location
}

Write-Host "Done. Succeeded: $($succeeded.Count) / $($experiments.Count)" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "Failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}
exit 0
