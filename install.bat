@echo off
chcp 65001 >nul
title TinyH3 Video: установка
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\bootstrap.ps1" %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo.
  echo  Установка не завершена ^(код %RC%^). Запустите install.bat снова: она продолжится с того же места.
)
pause
exit /b %RC%
