# Always uses the project venv (Python 3.10) regardless of what is active in the terminal.
$ROOT = $PSScriptRoot
$PYTHON = "$ROOT\.venv\Scripts\python.exe"
$CHATBOT = "$ROOT\chatbot.py"

if (-not (Test-Path $PYTHON)) {
    Write-Error "Venv not found at $ROOT\.venv — run: uv sync"
    exit 1
}

Write-Host "Using Python: $(& $PYTHON --version)"
Write-Host "Starting Streamlit..."
& $PYTHON -m streamlit run $CHATBOT
