@echo off
setlocal enabledelayedexpansion

set REPO=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Interim_Migration\Landing Page
set LOG=%REPO%\Automator\run_log.txt
set SCRIPT=%REPO%\Code\ingest.py
set MAILER=%REPO%\Automator\send_mail.py

:: Prevent Git Credential Manager from showing an interactive dialog in unattended runs.
set GCM_INTERACTIVE=never
set GIT_TERMINAL_PROMPT=0
echo. >> "%LOG%"
echo ======================================== >> "%LOG%"
echo %DATE% %TIME% -- Landing Page ingest START >> "%LOG%"

python "%SCRIPT%" >> "%LOG%" 2>&1
set ERR=!ERRORLEVEL!
if !ERR! NEQ 0 (
    echo %DATE% %TIME% -- INGEST FAILED >> "%LOG%"
    python "%MAILER%" FAIL "Landing Page sync failed — source parquets may be missing. Check run_log.txt."
    exit /b 1
)
echo %DATE% %TIME% -- Ingest complete >> "%LOG%"

:: Git commit and push, so the deployed Streamlit Cloud app picks up the refresh
cd /d "%REPO%"
git add "Database/" >> "%LOG%" 2>&1
git diff --cached --quiet
if !ERRORLEVEL! NEQ 0 (
    git commit -m "auto: Landing Page database sync %DATE%" >> "%LOG%" 2>&1
    git push >> "%LOG%" 2>&1
    set PUSH_ERR=!ERRORLEVEL!
    if !PUSH_ERR! NEQ 0 (
        echo %DATE% %TIME% -- GIT PUSH FAILED >> "%LOG%"
        python "%MAILER%" FAIL "Sync OK but git push failed. Check run_log.txt."
        exit /b 1
    )
    echo %DATE% %TIME% -- Git push OK >> "%LOG%"
    python "%MAILER%" SUCCESS "Landing Page database synced and pushed to GitHub."
) else (
    echo %DATE% %TIME% -- No changes, skipping commit >> "%LOG%"
    python "%MAILER%" SUCCESS "Landing Page sync ran — no parquet changes detected."
)

echo %DATE% %TIME% -- DONE >> "%LOG%"
endlocal
