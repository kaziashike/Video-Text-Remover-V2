@echo off
:: Update script for Video Subtitle Remover on RunPod (Windows)

echo Updating Video Subtitle Remover code...

:: Check if git is available
where git >nul 2>&1
if %errorlevel% == 0 (
    echo Git detected. Pulling latest changes...
    git pull
) else (
    echo Git not available. Please update code manually or install git.
    echo You can also rebuild and push the Docker image with new code.
)

:: Install/upgrade any new dependencies if requirements.txt was updated
if exist requirements.txt (
    echo Installing/upgrading dependencies...
    pip install --no-cache-dir -r requirements.txt
)

echo Code update completed. Please restart the application for changes to take effect.
pause