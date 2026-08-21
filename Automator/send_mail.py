import sys
import datetime
import win32com.client

TO     = "virat.arya@etgworld.com"
STATUS = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
MSG    = sys.argv[2] if len(sys.argv) > 2 else ""
NOW    = datetime.datetime.now().strftime("%d %b %Y  %H:%M")

if STATUS == "SUCCESS":
    subject = f"Landing Page Update — {NOW}"
    body    = (
        f"Landing Page database sync completed successfully.\n\n"
        f"Time:    {NOW}\n"
        f"Status:  Parquets synced & pushed to GitHub\n"
        f"Detail:  {MSG}\n\n"
        f"Streamlit dashboard will auto-redeploy shortly."
    )
else:
    subject = f"Landing Page FAILED — {NOW}"
    body    = (
        f"Landing Page database sync encountered an error.\n\n"
        f"Time:    {NOW}\n"
        f"Status:  FAILED\n"
        f"Detail:  {MSG}\n\n"
        f"Check Automator\\run_log.txt for full output."
    )

outlook = win32com.client.Dispatch("Outlook.Application")
mail    = outlook.CreateItem(0)
mail.To      = TO
mail.Subject = subject
mail.Body    = body
mail.Send()
print(f"Mail sent: {subject}")
