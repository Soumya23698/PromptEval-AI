@echo off
title PromptEval AI - CLI Test Suite
echo ======================================================================
echo  Running PromptEval AI CLI Test across Benchmarks
echo ======================================================================
cd /d "%~dp0"

echo [1/4] Testing Python Code Refactoring Scenario...
".venv\Scripts\python.exe" nlp_engine.py --scenario code --prompt "Act as a Senior Python Engineer. Refactor the code to O(n) runtime complexity, add PEP 8 docstrings and type hints."
echo.

echo [2/4] Testing Socratic AI Tutor Scenario...
".venv\Scripts\python.exe" nlp_engine.py --scenario tutor --prompt "You are a Socratic high school math tutor. Guide me to understand derivatives using the speedometer analogy without giving the formula directly."
echo.

echo [3/4] Testing JSON Data Extraction Scenario...
".venv\Scripts\python.exe" nlp_engine.py --scenario data --prompt "Extract customer ticket info into strict RFC 8259 JSON format with fields: customer_id, sentiment, urgency. Do not include markdown text."
echo.

echo [4/4] Testing Executive Briefing Scenario...
".venv\Scripts\python.exe" nlp_engine.py --scenario creative --prompt "You are a Chief Strategy Officer. Summarize the quarterly report into top 3 milestones and KPI table under 400 words."
echo.

echo ======================================================================
echo  All CLI Tests Complete!
echo ======================================================================
pause
