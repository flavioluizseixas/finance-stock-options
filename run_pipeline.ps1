# run_pipeline.ps1

$PYTHON    = "C:\Users\Flavio\anaconda3\envs\myenv\python.exe"
$STREAMLIT = "C:\Users\Flavio\anaconda3\envs\myenv\Scripts\streamlit.exe"

$SCRIPTS = @(
    "finance_options_pipeline.py",
    "export_ms7_incremental.py",
    "export_to_parquet.py"
)

foreach ($script in $SCRIPTS) {
    Write-Host "▶ Executando $script"
    & $PYTHON $script

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Erro em $script (exitcode=$LASTEXITCODE)"
        exit 1
    }
}

Write-Host "✅ Pipeline finalizado com sucesso"
Write-Host "🚀 Iniciando Streamlit..."

# Sobe o app (fica "preso" no terminal até você parar com Ctrl+C)
& $STREAMLIT run ".\app_streamlit_options.py"