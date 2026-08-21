import sys
import datetime
import win32com.client

TO     = "virat.arya@etgworld.com"
STATUS = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
MSG    = sys.argv[2] if len(sys.argv) > 2 else ""
NOW    = datetime.datetime.now().strftime("%d %b %Y  %H:%M")

if STATUS == "SUCCESS":
    subject = f"Landing Page — Source Check OK — {NOW}"
    body    = (
        f"Landing Page source freshness check passed.\n\n"
        f"Time:    {NOW}\n"
        f"Detail:  {MSG}\n\n"
        f"No action needed — this project holds no data of its own, it just "
        f"reads the other Interim_Migration projects' databases live."
    )
else:
    subject = f"Landing Page — Source Check FAILED — {NOW}"
    body    = (
        f"Landing Page source freshness check found a problem.\n\n"
        f"Time:    {NOW}\n"
        f"Status:  FAILED\n"
        f"Detail:  {MSG}\n\n"
        f"Check Automator\\run_log.txt for the missing/stale file list."
    )

outlook = win32com.client.Dispatch("Outlook.Application")
mail    = outlook.CreateItem(0)
mail.To      = TO
mail.Subject = subject
mail.Body    = body
mail.Send()
print(f"Mail sent: {subject}")
