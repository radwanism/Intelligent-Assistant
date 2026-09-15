<#
.SYNOPSIS
    Start the Telecom Egypt Assistant locally (Windows / PowerShell).

.DESCRIPTION
    Starts speech (:8001), core (:8000) and the Gradio UI (:7860) in separate
    windows, after checking the things that actually go wrong: missing venv,
    missing ffmpeg, an empty knowledge base.

    Services start in dependency order, but core does NOT require speech — if
    the speech service fails to start, text chat still works and the UI hides
    the microphone. That is intentional.

.EXAMPLE
    .\scripts\run_local.ps1
    .\scripts\run_local.ps1 -SkipSpeech      # text-only, fastest startup
    .\scripts\run_local.ps1 -Profile cpu-lite
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu-lite', 'gpu-colab', 'auto')]
    [string]$Profile = 'auto',
    [switch]$SkipSpeech,
    [switch]$NoWait
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root 'smart_assistant\Scripts\python.exe'

Write-Host "`n=== Telecom Egypt Assistant ===" -ForegroundColor Cyan

# --- preflight ------------------------------------------------------------
if (-not (Test-Path $python)) {
    Write-Host "ERROR: venv not found at smart_assistant\" -ForegroundColor Red
    Write-Host "  Create it:  uv venv smart_assistant --python 3.11"
    Write-Host "              uv pip install -e `".[dev]`""
    exit 1
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "WARNING: ffmpeg not on PATH - voice input will fail." -ForegroundColor Yellow
    Write-Host "         Text chat is unaffected.`n"
}

# An empty index is the single most common cause of "it answers nothing".
$chroma = Join-Path $root 'data\chroma'
if (-not (Test-Path $chroma) -or -not (Get-ChildItem $chroma -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: no knowledge base at data\chroma" -ForegroundColor Red
    Write-Host "  Build it:  $python -m te_assistant.ingest.build_index"
    exit 1
}

if ($Profile -ne 'auto') {
    $env:TE_PROFILE = $Profile
    Write-Host "profile : $Profile (forced)"
} else {
    Write-Host "profile : auto-detected"
}

function Start-Service-Window {
    param([string]$Title, [string]$Module, [string[]]$Arguments = @())
    $argList = @('-m', $Module) + $Arguments
    Start-Process -FilePath $python -ArgumentList $argList -WindowStyle Normal `
        -WorkingDirectory $root
    Write-Host "started : $Title" -ForegroundColor Green
}

function Wait-For-Health {
    param([string]$Url, [int]$TimeoutSeconds = 180, [string]$Name)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-RestMethod -Uri $Url -TimeoutSec 3
            Write-Host "healthy : $Name" -ForegroundColor Green
            return $true
        } catch { Start-Sleep -Seconds 2 }
    }
    Write-Host "timeout : $Name did not become healthy" -ForegroundColor Yellow
    return $false
}

# --- start ----------------------------------------------------------------
if (-not $SkipSpeech) {
    Start-Service-Window -Title 'speech  :8001' -Module 'te_assistant.speech.service'
} else {
    Write-Host "skipped : speech (text-only mode)" -ForegroundColor Yellow
}

Start-Service-Window -Title 'core    :8000' -Module 'te_assistant.api'

if (-not $NoWait) {
    # Core loads the embedder and builds the BM25 index at startup; on a slow
    # CPU that is a minute or more, and launching the UI first shows an error.
    Write-Host "`nwaiting for core to load models..."
    Wait-For-Health -Url 'http://127.0.0.1:8000/health' -Name 'core' -TimeoutSeconds 300 | Out-Null
}

Start-Service-Window -Title 'ui      :7860' -Module 'te_assistant.ui.gradio_app'

Write-Host "`n  UI      http://127.0.0.1:7860"
Write-Host "  core    http://127.0.0.1:8000/health"
Write-Host "  metrics http://127.0.0.1:8000/metrics"
if (-not $SkipSpeech) { Write-Host "  speech  http://127.0.0.1:8001/health" }
Write-Host "`nClose the service windows to stop.`n"
