@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Transcriptor - Offline Whisper Pipeline
cd /d "%~dp0"

echo.
echo ========================================
echo   Transcriptor - Offline Transcription
echo ========================================
echo.

set "PYTHON_EXE="

REM --- Prefer a real python.exe (skip Windows Store stub) ---

REM 1) python on PATH
for /f "delims=" %%I in ('where python 2^>nul') do (
    echo %%I | findstr /i "WindowsApps\\python.exe" >nul
    if errorlevel 1 (
        if exist "%%I" (
            set "PYTHON_EXE=%%I"
            goto :py_found
        )
    )
)

REM 2) Explicit known install (this machine)
if exist "%LocalAppData%\Programs\Python\Python314\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python314\python.exe"
    goto :py_found
)

REM 3) Any LocalAppData Python3x install (newest first)
for /f "delims=" %%D in ('dir /b /ad /o-n "%LocalAppData%\Programs\Python\Python*" 2^>nul') do (
    if exist "%LocalAppData%\Programs\Python\%%D\python.exe" (
        set "PYTHON_EXE=%LocalAppData%\Programs\Python\%%D\python.exe"
        goto :py_found
    )
)

REM 4) Program Files
for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles%\Python*" 2^>nul') do (
    if exist "%ProgramFiles%\%%D\python.exe" (
        set "PYTHON_EXE=%ProgramFiles%\%%D\python.exe"
        goto :py_found
    )
)

REM 5) py launcher -> ask it for the executable path
where py >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        if exist "%%I" (
            set "PYTHON_EXE=%%I"
            goto :py_found
        )
    )
)

echo [ERROR] Python 3 was not found.
echo Looked on PATH and under:
echo   %LocalAppData%\Programs\Python\
echo.
echo Your install appears to be:
echo   %LocalAppData%\Programs\Python\Python314\python.exe
echo If that path exists, reopen this .bat after fixing PATH, or run:
echo   "%LocalAppData%\Programs\Python\Python314\python.exe" -u "%~dp0transcribe.py" --gui
echo.
pause
exit /b 1

:py_found
echo Using: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -c "import faster_whisper, tkinterdnd2" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing / updating dependencies from requirements.txt...
    "%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo.
)

echo Launching preview interface...
echo.
REM Already cd'd to script dir; use relative path to avoid quote/space issues.
"%PYTHON_EXE%" -u transcribe.py --gui %*

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] Exit code %EXITCODE%
    pause
)
exit /b %EXITCODE%
