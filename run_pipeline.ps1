# run_pipeline.ps1 — atualiza cotações e abre o painel (ambiente gerenciado pelo uv: pyproject.toml + uv.lock)
#   .\run_pipeline.ps1              atualiza ativos acompanhados + operações abertas e abre o painel
#   .\run_pipeline.ps1 -Universo    antes, reavalia o ranking de liquidez dos ativos com opções
#   .\run_pipeline.ps1 -Operacoes   atualiza só os ativos com operações abertas (rápido)
#   .\run_pipeline.ps1 -SemApp      só atualiza (útil no Agendador de Tarefas)

param([switch]$Universo, [switch]$Operacoes, [switch]$SemApp)

Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "❌ uv não encontrado. Instale com:  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    exit 1
}

$argsAtualizar = @("run", "atualizar.py")
if ($Universo) { $argsAtualizar += "--universo" }
if ($Operacoes) { $argsAtualizar += "--operacoes" }

Write-Host "▶ Atualizando cotações (Yahoo Finance + opcoes.net.br)"
& uv @argsAtualizar
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Erro na atualização (exitcode=$LASTEXITCODE)"; exit 1 }

if (Test-Path "export_ms7_incremental.py") {
    Write-Host "▶ Exportando MetaStock ASCII"
    & uv run --with tqdm export_ms7_incremental.py
    if ($LASTEXITCODE -ne 0) { Write-Host "⚠️ export_ms7_incremental.py falhou (exitcode=$LASTEXITCODE)" }
}

if (-not $SemApp) {
    Write-Host "🚀 Abrindo o painel (Ctrl+C para encerrar)"
    & uv run streamlit run app.py
}
