@echo off
echo Starting Video Subtitle Remover...

echo.
echo Starting API service...
start "API" /min cmd /c "python app.py ^|^| python start_api.py"

timeout /t 5 /nobreak >nul

echo.
echo Starting GUI service...
start "GUI" cmd /c "python app_gui.py"

echo.
echo Services started:
echo API:  http://localhost:8000
echo GUI:  http://localhost:8002
echo.
echo Press any key to exit...
pause >nul