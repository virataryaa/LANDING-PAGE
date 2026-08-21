@echo off
setlocal enabledelayedexpansion

set REPO=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Interim_Migration\Landing Page
set LOG=%REPO%\Automator\run_log.txt
set SCRIPT=%REPO%\Code\check_sources.py
set MAILER=%REPO%\Automator\send_mail.py

echo. >> "%LOG%"
echo ======================================== >> "%LOG%"
echo %DATE% %TIME% -- Landing Page source check START >> "%LOG%"

python "%SCRIPT%" >> "%LOG%" 2>&1
set ERR=!ERRORLEVEL!
if !ERR! NEQ 0 (
    echo %DATE% %TIME% -- SOURCE CHECK FAILED >> "%LOG%"
    python "%MAILER%" FAIL "One or more source parquets are missing or stale. Check run_log.txt."
    exit /b 1
)

echo %DATE% %TIME% -- Source check OK >> "%LOG%"
python "%MAILER%" SUCCESS "All source parquets present and fresh."
echo %DATE% %TIME% -- DONE >> "%LOG%"
endlocal
