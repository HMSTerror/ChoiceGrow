@echo off
setlocal
rem Windows entry point: prefer explicit Python, project venv, known local environment, then PATH.
rem Forward all arguments unchanged; the Python entry point merges and validates configuration.
if defined CHOICE_PYTHON (
  "%CHOICE_PYTHON%" -B "%~dp0run.py" %*
) else if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -B "%~dp0run.py" %*
) else if exist "E:\anaco\python.exe" (
  "E:\anaco\python.exe" -B "%~dp0run.py" %*
) else (
  python -B "%~dp0run.py" %*
)
exit /b %errorlevel%
