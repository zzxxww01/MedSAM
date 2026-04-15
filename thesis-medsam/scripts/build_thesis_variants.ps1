$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Exe,
    [Parameter(Mandatory = $true)]
    [string[]]$Args
  )

  & $Exe @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE"
  }
}

$variants = @(
  @{ Source = "build_formal.tex"; Job = "formal_build"; Target = "基于平衡损失与局部适配的医学图像分割_正式版.pdf" },
  @{ Source = "build_blind.tex"; Job = "blind_build"; Target = "基于平衡损失与局部适配的医学图像分割.pdf" }
)

foreach ($variant in $variants) {
  Invoke-Checked -Exe "xelatex" -Args @("-interaction=nonstopmode", "-halt-on-error", "-jobname=$($variant.Job)", $variant.Source)
  Invoke-Checked -Exe "bibtex" -Args @($variant.Job)
  Invoke-Checked -Exe "xelatex" -Args @("-interaction=nonstopmode", "-halt-on-error", "-jobname=$($variant.Job)", $variant.Source)
  Invoke-Checked -Exe "xelatex" -Args @("-interaction=nonstopmode", "-halt-on-error", "-jobname=$($variant.Job)", $variant.Source)
  Move-Item -Force "$($variant.Job).pdf" $variant.Target
}

python scripts\verify_thesis_variants.py
