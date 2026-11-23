# run_pipeline.ps1 — atualiza cotações e abre o painel
#   .\run_pipeline.ps1              atualiza ativos acompanhados + operações abertas e abre o painel
#   .\run_pipeline.ps1 -Universo    antes, reavalia o ranking de liquidez dos ativos com opções (~6 min)
#   .\run_pipeline.ps1 -SemApp      só atualiza (útil no Agendador de Tarefas)

param([switch]$Universo, [switch]$SemApp)

$PYTHON = if ($env:PAINEL_PYTHON) { $env:PAINEL_PYTHON } else { "$env:USERPROFILE\miniconda3\envs\myenv\python.exe" }
Set-Location $PSScriptRoot

$argsAtualizar = @("atualizar.py")
if ($Universo) { $argsAtualizar += "--universo" }

Write-Host "▶ Atualizando cotações (Yahoo Finance + opcoes.net.br)"
& $PYTHON @argsAtualizar
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Erro na atualização (exitcode=$LASTEXITCODE)"; exit 1 }

Write-Host "▶ Exportando MetaStock ASCII"
& $PYTHON export_ms7_incremental.py
if ($LASTEXITCODE -ne 0) { Write-Host "⚠️ export_ms7_incremental.py falhou (exitcode=$LASTEXITCODE)" }

if (-not $SemApp) {
    Write-Host "🚀 Abrindo o painel (Ctrl+C para encerrar)"
    & $PYTHON -m streamlit run app.py
}
