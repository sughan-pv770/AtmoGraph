Write-Host "Starting AtmoGraph Backend (http://127.0.0.1:8000) and Frontend (http://localhost:5173)..." -ForegroundColor Cyan

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location 'f:\atomgraph'; .\.venv\Scripts\uvicorn.exe backend.main:app --reload --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location 'f:\atomgraph\frontend'; npm run dev"

Write-Host "Open http://localhost:5173 in your browser!" -ForegroundColor Green

