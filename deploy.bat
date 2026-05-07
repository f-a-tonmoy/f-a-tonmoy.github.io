@echo off
REM Deploy the portfolio to GitHub Pages (Windows version).
REM
REM Usage:
REM   deploy.bat                    -- commits with a default message
REM   deploy.bat "your message"     -- commits with a custom message

cd /d "%~dp0"

git status --porcelain >nul
if errorlevel 1 goto :err

for /f %%i in ('git status --porcelain') do goto :proceed
echo Nothing to commit -- working tree is clean.
exit /b 0

:proceed
set "MESSAGE=%~1"
if "%MESSAGE%"=="" set "MESSAGE=Update portfolio site"

echo ==^> Staging all changes...
git add -A || goto :err

echo ==^> Committing...
git commit -m "%MESSAGE%" || goto :err

echo ==^> Pushing to origin/main...
git push origin main || goto :err

echo.
echo Done. Live in ~30-60s at: https://f-a-tonmoy.github.io
exit /b 0

:err
echo.
echo Deploy failed. See messages above.
exit /b 1
