@echo off
title PromptEval AI - Public Online Tunnel
echo ======================================================================
echo  Starting PromptEval AI with Public Global Access
echo ======================================================================
cd /d "%~dp0"

echo [1/2] Starting local Python NLP Backend...
start /B "" ".venv\Scripts\python.exe" app.py

timeout /t 3 /nobreak >nul

echo [2/2] Starting Cloudflare Public Tunnel...
echo.
echo Your public link will be shown below (e.g. https://xxxx.trycloudflare.com)
echo Anyone on any phone, laptop, or computer can access this link!
echo.
cloudflared.exe tunnel --url http://localhost:5000
pause
