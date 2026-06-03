# Daily Investment Briefing — VPS Setup Guide

**Server:** Hetzner CX22, Ubuntu 24.04  
**IP:** 5.78.237.242  
**Sends at:** 7:35 AM Pacific Time daily

---

## 1. SSH Into the Server

```bash
ssh root@5.78.237.242
```

---

## 2. Create a Dedicated User

Running as root is risky. Create a `briefing` user instead:

```bash
adduser briefing
# Set a password when prompted, or just press Enter to skip
usermod -aG sudo briefing
```

Switch to the new user:

```bash
su - briefing
```

---

## 3. Install Python

Ubuntu 24.04 ships Python 3.12. Verify it's present:

```bash
python3 --version
# Should print Python 3.12.x or later
```

If missing:

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv
```

---

## 4. Upload the Project Files

From your **local machine**, copy the files to the server:

```bash
scp scan.py requirements.txt .env.example briefing@5.78.237.242:/home/briefing/
```

Or, if you prefer to clone from git on the server:

```bash
# On the server
sudo apt install -y git
git clone https://github.com/milesdolson/work.git /home/briefing/work
cd /home/briefing/work
```

---

## 5. Set Up a Virtual Environment

```bash
cd /home/briefing
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Configure Environment Variables

Copy the example file and fill in real values:

```bash
cp .env.example .env
nano .env
```

Edit the file to look like this (replace all placeholders):

```
ANTHROPIC_API_KEY=sk-ant-YOUR_REAL_KEY_HERE
GMAIL_USER=Blorgo.Olson@gmail.com
GMAIL_APP_PASSWORD=YOUR_APP_PASSWORD_HERE
TO_EMAIL=milesdavidolson@gmail.com
```

Secure the file so only your user can read it:

```bash
chmod 600 .env
```

> **Note:** Never commit `.env` to git. The `.env.example` file (with placeholder values) is safe to commit; `.env` is not.

---

## 7. Get a Gmail App Password

Gmail requires an App Password when 2-Step Verification is enabled (required for SMTP access).

1. Go to [myaccount.google.com/security](https://myaccount.google.com/security)
2. Ensure **2-Step Verification** is turned on for `Blorgo.Olson@gmail.com`
3. Search for **"App Passwords"** at the top of the page (or go directly to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords))
4. Click **Create**, give it a name like `briefing-vps`, click **Create**
5. Copy the 16-character password shown — paste it as `GMAIL_APP_PASSWORD` in your `.env`

---

## 8. Test the Script Manually

With the venv active:

```bash
cd /home/briefing
source venv/bin/activate
python scan.py
```

You should see:
```
Briefing sent for Tuesday, June 3, 2026
```

And an email should arrive in your inbox within a few seconds. If there's an error, you'll receive a red "FAILED" email with the traceback, and the error will also print to the terminal.

---

## 9. Schedule with Cron

Open the crontab editor:

```bash
crontab -e
```

Add these two lines at the top (the `TZ` line handles Daylight Saving Time automatically):

```
TZ=America/Los_Angeles
35 7 * * * /home/briefing/venv/bin/python /home/briefing/scan.py >> /home/briefing/briefing.log 2>&1
```

Save and exit. Verify the cron job was saved:

```bash
crontab -l
```

Log output goes to `/home/briefing/briefing.log`. Check it after the first run:

```bash
tail -f /home/briefing/briefing.log
```

---

## 10. (Optional) Keep Dependencies Updated

To update the Anthropic SDK when new versions ship:

```bash
source /home/briefing/venv/bin/activate
pip install --upgrade anthropic
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: anthropic` | Run `source venv/bin/activate` first, or use the full venv Python path in cron |
| `SMTPAuthenticationError` | Double-check the App Password; make sure 2FA is enabled on the Gmail account |
| `KeyError: 'ANTHROPIC_API_KEY'` | The `.env` file is missing or in the wrong directory; run from `/home/briefing/` |
| Email received but no opportunities | Check `briefing.log`; Claude may have returned unexpected JSON — the error fallback email will contain the traceback |
| Cron job doesn't run at 7:35 AM PT | Verify `TZ=America/Los_Angeles` is the first line of your crontab (not inside a comment) |
