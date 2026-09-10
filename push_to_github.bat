@echo off
title Push PromptEval AI to GitHub
echo ======================================================================
echo  Pushing PromptEval AI to https://github.com/Soumya23698/PromptEval-AI
echo ======================================================================
cd /d "%~dp0"

set GIT="C:\Users\SOUMYADEEP\AppData\Local\MinGit\cmd\git.exe"

echo Step 1: Configuring remote origin...
%GIT% remote remove origin >nul 2>&1
%GIT% remote add origin https://github.com/Soumya23698/PromptEval-AI.git

echo Step 2: Pushing to main branch...
echo (If prompted, log in with your GitHub browser window or Personal Access Token)
%GIT% push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ======================================================================
    echo  SUCCESS! Repository pushed to:
    echo  https://github.com/Soumya23698/PromptEval-AI
    echo ======================================================================
) else (
    echo.
    echo [NOTE] If the repository 'PromptEval-AI' does not exist yet on GitHub:
    echo 1. Go to: https://github.com/new
    echo 2. Name: PromptEval-AI (Leave "Add README" unchecked)
    echo 3. Click "Create repository"
    echo 4. Run this script again!
)

pause
