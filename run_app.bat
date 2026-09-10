@echo off
title PromptEval AI - NLP Prompt Engineering Assessment System
echo ======================================================================
echo  Starting PromptEval AI Web Application (PyTorch + SentenceTransformers)
echo ======================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please ensure .venv is installed.
    pause
    exit /b 1
)

echo Activating Python environment and launching server...
".venv\Scripts\python.exe" app.py
pause
