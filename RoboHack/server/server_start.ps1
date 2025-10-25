try {
  $VenvName = "venv"

  if (-not (Test-Path -Path $VenvName)) {
      Write-Host "Creating virtual environment: $VenvName"
      python -m venv $VenvName
  }

  Write-Host "Activating virtual environment..."
  try {
      . "./$VenvName/Scripts/Activate.ps1"
  } catch {
      Write-Error "Failed to activate venv."
      Write.Host "Press Enter to exit... " -ForegroundColor Red
      Read-Host | Out-Null
      exit 1
  }

  Write-Host "Installing requirements"
  pip install fastapi asyncio pydantic speechrecognition pyttsx3 uvicorn python-multipart

  Write-Host "Starting server on http://0.0.0.0:8000"
  Write-Host "Press Ctrl+C to stop the server."
  uvicorn server_TTS:app --host 0.0.0.0 --port 8000

} catch {
  Write-Error "Script fail"
  Write-Hose "Press Enter to exit... " -ForegroundColor Red
  Read-Host | Out-Null
  exit 1
}