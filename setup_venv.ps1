#Requires -Version 5.1
<#
.SYNOPSIS
    Create and populate a virtual environment for PyTaRGET.
.PARAMETER DeepSeek
    Also clone and install HuggingFace/transformers from source (required for DeepSeek support).
#>
param([switch]$DeepSeek)

$ErrorActionPreference = 'Stop'
$ScriptDir  = $PSScriptRoot
$VenvDir    = Join-Path $ScriptDir 'venv'

# ── find Python 3.12+ ────────────────────────────────────────────────────────
function Find-Python312 {
    # Try the Windows Python Launcher with explicit version flags first
    foreach ($ver in '3.13', '3.12') {
        try {
            $null = & py "-$ver" --version 2>&1
            if ($LASTEXITCODE -eq 0) { return "py -$ver" }
        } catch {}
    }
    # Fall back to bare executables
    foreach ($cmd in 'python3.13', 'python3.12', 'python3', 'python') {
        try {
            $null = & $cmd --version 2>&1
            if ($LASTEXITCODE -ne 0) { continue }
            $ok = & $cmd -c "import sys; exit(0 if sys.version_info>=(3,12) else 1)" 2>&1
            if ($LASTEXITCODE -eq 0) { return $cmd }
        } catch {}
    }
    return $null
}

$PythonCmd = Find-Python312
if (-not $PythonCmd) {
    Write-Error "Python 3.12+ not found. Install it from https://python.org and ensure it is on your PATH."
    exit 1
}

# Split into executable + args so we can call it properly
$PythonParts = $PythonCmd -split ' '
$PyExe  = $PythonParts[0]
$PyArgs = $PythonParts[1..($PythonParts.Length - 1)]

$version = & $PyExe @PyArgs --version
Write-Host "Python : $version  ($PythonCmd)"

# ── create venv ──────────────────────────────────────────────────────────────
if (Test-Path $VenvDir) {
    Write-Host "Venv   : already exists at $VenvDir (skipping creation)"
    Write-Host "         Delete it first if you want a clean rebuild:  Remove-Item -Recurse -Force $VenvDir"
} else {
    Write-Host "Venv   : creating at $VenvDir"
    & $PyExe @PyArgs -m venv $VenvDir
}

$Pip    = Join-Path $VenvDir 'Scripts\pip.exe'
$Python = Join-Path $VenvDir 'Scripts\python.exe'

# ── install dependencies ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "Installing requirements.txt ..."
& $Pip install --upgrade pip --quiet
& $Pip install -r (Join-Path $ScriptDir 'requirements.txt')

Write-Host ""
Write-Host "Pinning tree-sitter==0.24.0 (conflict warnings below are expected) ..."
& $Pip install tree-sitter==0.24.0

# ── optional: DeepSeek transformers ──────────────────────────────────────────
if ($DeepSeek) {
    Write-Host ""
    Write-Host "DeepSeek: cloning HuggingFace/transformers from source ..."
    $TransformersDir = Join-Path $ScriptDir 'transformers'
    if (Test-Path $TransformersDir) {
        Write-Host "  Directory already exists, pulling latest ..."
        git -C $TransformersDir pull --ff-only
    } else {
        git clone --depth 1 https://github.com/huggingface/transformers.git $TransformersDir
    }
    Write-Host "  Installing in editable mode ..."
    & $Pip install -e $TransformersDir
}

# ── done ─────────────────────────────────────────────────────────────────────
$ParentDir = Split-Path $ScriptDir -Parent
$ActivatePath = Join-Path $VenvDir 'Scripts\Activate.ps1'

Write-Host ""
Write-Host ("=" * 57)
Write-Host "Setup complete."
Write-Host ""
Write-Host "Activate:"
Write-Host "  & '$ActivatePath'"
Write-Host ""
Write-Host "Run code from the PARENT directory of PyTaRGET:"
Write-Host "  cd $ParentDir"
Write-Host "  python -c 'from PyTaRGET.data_processing.encode_tune_test import Eftt'"
Write-Host ""
if (-not $DeepSeek) {
    Write-Host "For DeepSeek support, re-run with:  .\setup_venv.ps1 -DeepSeek"
    Write-Host ""
}
Write-Host ("=" * 57)
