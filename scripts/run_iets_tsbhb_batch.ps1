$ErrorActionPreference = "Stop"
$R = "C:\Program Files\R\R-4.5.3\bin\Rscript.exe"
$Repo = Split-Path -Parent $PSScriptRoot
$Src = Join-Path $Repo "src"
$Out = Join-Path $Repo "outputs"

# Nested shells often lack conda on PATH; prepend common install locations.
$condaRoots = @(
    "$env:USERPROFILE\anaconda3",
    "$env:USERPROFILE\miniconda3",
    "$env:LOCALAPPDATA\anaconda3",
    "$env:LOCALAPPDATA\miniconda3"
)
foreach ($root in $condaRoots) {
    if (Test-Path $root) {
        $env:Path = (Join-Path $root "Scripts") + ";" + (Join-Path $root "Library\bin") + ";" + $env:Path
    }
}

Set-Location $Src

function Run-Step {
    param([string]$Name, [string[]]$PyArgs)
    Write-Host "===== $Name =====" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    conda run -n tsbhb310 python @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "Step failed ($Name): exit $LASTEXITCODE" }
    $sw.Stop()
    Write-Host "Done $Name in $($sw.Elapsed)" -ForegroundColor Green
}

# 1–2: Point forecasts, 1000 series each (TSB-HB + iETS)
Run-Step "point_online_1000" @(
    "-m", "experiments.run_point",
    "--dataset", "online_retail",
    "--max-series", "1000",
    "--baseline-mode", "hb_only",
    "--with-iets",
    "--iets-rscript", $R,
    "--out", (Join-Path $Out "point_online_iets_tsbhb_1000"),
    "--seed", "42"
)

Run-Step "point_m5_1000" @(
    "-m", "experiments.run_point",
    "--dataset", "m5",
    "--m5-sample-size", "1000",
    "--baseline-mode", "hb_only",
    "--with-iets",
    "--iets-rscript", $R,
    "--out", (Join-Path $Out "point_m5_iets_tsbhb_1000"),
    "--seed", "42"
)

# 3–4: Probabilistic, 500 series each
Run-Step "prob_online_500" @(
    "-m", "experiments.run_prob",
    "--dataset", "online",
    "--max-series", "500",
    "--baseline-mode", "hb_only",
    "--with-iets",
    "--iets-rscript", $R,
    "--out", (Join-Path $Out "prob_online_iets_tsbhb_500"),
    "--seed", "42"
)

Run-Step "prob_m5_500" @(
    "-m", "experiments.run_prob",
    "--dataset", "m5",
    "--m5-sample-size", "500",
    "--baseline-mode", "hb_only",
    "--with-iets",
    "--iets-rscript", $R,
    "--out", (Join-Path $Out "prob_m5_iets_tsbhb_500"),
    "--seed", "42"
)

Write-Host "All batch steps completed." -ForegroundColor Magenta
