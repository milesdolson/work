#!/usr/bin/env python3
"""Daily government signal investment briefing."""

import html
import json
import os
import smtplib
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
TO_EMAIL = os.environ["TO_EMAIL"]
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SCAN_PROMPT = """You are a government signal investing analyst. Using your web search tool, scan the following public sources for congressional trading activity from the past 45 days:

Search these sources:
- Quiver Quant congressional trading (quiverquant.com/congresstrading)
- Capitol Trades (capitoltrades.com)
- Unusual Whales political trades (unusualwhales.com/political_trades)
- SEC EDGAR Form 4 filings for recent insider transactions

For each trade candidate, analyze:
1. Congressional trade details: member name, party, chamber, committee memberships
2. Related persons activity: spouse or dependent family member trades
3. Committee relevance: does this legislator oversee sectors they're trading?
4. Insider proximity: High/Medium/Low
5. Fundamentals: P/E, revenue growth, debt, sector tailwinds
6. Upcoming catalysts: earnings, FDA decisions, government contracts, regulatory votes
7. News alignment: supporting/contradicting news from past 2 weeks

Score each 0–100 across 5 components:
- Congressional Signal Quality (max 25 pts)
- Related Persons Activity (max 15 pts)
- Fundamentals (max 25 pts)
- Upcoming Catalysts (max 20 pts)
- News Alignment (max 15 pts)

Return ONLY a JSON array of the top 5 opportunities — no other text, no markdown fences:
[
  {
    "rank": 1,
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "total_score": 82,
    "component_scores": {
      "congressional_signal": 21,
      "related_persons": 11,
      "fundamentals": 22,
      "upcoming_catalysts": 16,
      "news_alignment": 12
    },
    "legislator_name": "Rep. Jane Smith (D-CA)",
    "purchase_date": "2026-05-20",
    "trade_size": "$50,001 – $100,000",
    "committee": "House Energy & Commerce",
    "insider_proximity_rating": "High",
    "explanation": "2-3 sentence explanation."
  }
]"""


def call_claude(client: anthropic.Anthropic) -> str:
    """Call Claude with web search, handling pause_turn continuations."""
    messages = [{"role": "user", "content": SCAN_PROMPT}]
    tools = [{"type": "web_search_20260209", "name": "web_search"}]

    for _ in range(6):
        response = client.messages.create(
            model=MODEL,
            max_tokens=8096,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason == "pause_turn":
            messages = [
                {"role": "user", "content": SCAN_PROMPT},
                {"role": "assistant", "content": response.content},
            ]
            continue
        text_blocks = [
            b.text for b in response.content
            if hasattr(b, "text") and b.type == "text"
        ]
        return "\n".join(text_blocks).strip()

    raise RuntimeError("Scan did not complete after maximum continuation attempts")


def extract_json(text: str) -> list:
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"No JSON array found in response: {text[:300]}")


def score_color(score: int) -> tuple:
    """Returns (text_color, bg_color, border_color) for the given score tier."""
    if score >= 75:
        return "#166534", "#dcfce7", "#16a34a"
    elif score >= 50:
        return "#92400e", "#fef3c7", "#d97706"
    else:
        return "#991b1b", "#fee2e2", "#dc2626"


def score_label(score: int) -> str:
    if score >= 75:
        return "STRONG"
    elif score >= 50:
        return "MODERATE"
    else:
        return "WEAK"


def component_bar(label: str, value: int, max_val: int) -> str:
    pct = int((value / max_val) * 100) if max_val > 0 else 0
    bar_color = "#16a34a" if pct >= 75 else "#d97706" if pct >= 50 else "#dc2626"
    return f"""
      <tr>
        <td style="font-size:11px;color:#6b7280;padding:2px 8px 2px 0;white-space:nowrap;">{html.escape(label)}</td>
        <td style="padding:2px 0;">
          <div style="background:#e5e7eb;border-radius:4px;height:8px;width:120px;">
            <div style="background:{bar_color};border-radius:4px;height:8px;width:{pct}%;"></div>
          </div>
        </td>
        <td style="font-size:11px;color:#374151;padding:2px 0 2px 6px;">{value}/{max_val}</td>
      </tr>"""


def build_opportunity_card(opp: dict) -> str:
    score = opp.get("total_score", 0)
    text_col, bg_col, border_col = score_color(score)
    label = score_label(score)
    cs = opp.get("component_scores", {})

    bars = (
        component_bar("Congressional Signal", cs.get("congressional_signal", 0), 25)
        + component_bar("Related Persons", cs.get("related_persons", 0), 15)
        + component_bar("Fundamentals", cs.get("fundamentals", 0), 25)
        + component_bar("Upcoming Catalysts", cs.get("upcoming_catalysts", 0), 20)
        + component_bar("News Alignment", cs.get("news_alignment", 0), 15)
    )

    return f"""
  <div style="margin:0 0 20px 0;border-radius:10px;overflow:hidden;border:2px solid {border_col};font-family:Arial,sans-serif;">
    <!-- Header -->
    <div style="background:{border_col};padding:12px 16px;display:flex;justify-content:space-between;align-items:center;">
      <div>
        <span style="font-size:22px;font-weight:700;color:#ffffff;letter-spacing:1px;">#{opp.get('rank','?')} {html.escape(opp.get('ticker',''))}</span>
        <span style="font-size:13px;color:rgba(255,255,255,0.85);margin-left:8px;">{html.escape(opp.get('company_name',''))}</span>
      </div>
      <div style="text-align:right;">
        <div style="background:#ffffff;border-radius:6px;padding:4px 10px;display:inline-block;">
          <span style="font-size:20px;font-weight:700;color:{border_col};">{score}</span>
          <span style="font-size:11px;color:{border_col};font-weight:600;margin-left:2px;">/100</span>
        </div>
        <div style="font-size:10px;color:rgba(255,255,255,0.9);font-weight:600;margin-top:2px;">{label}</div>
      </div>
    </div>
    <!-- Body -->
    <div style="background:{bg_col};padding:14px 16px;">
      <!-- Component scores -->
      <table style="margin-bottom:12px;border-collapse:collapse;">
        <tbody>{bars}
        </tbody>
      </table>
      <!-- Detail table -->
      <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:12px;">
        <tbody>
          <tr>
            <td style="color:#6b7280;padding:3px 0;width:38%;">Legislator</td>
            <td style="color:#111827;font-weight:500;">{html.escape(opp.get('legislator_name',''))}</td>
          </tr>
          <tr>
            <td style="color:#6b7280;padding:3px 0;">Purchase Date</td>
            <td style="color:#111827;font-weight:500;">{html.escape(opp.get('purchase_date',''))}</td>
          </tr>
          <tr>
            <td style="color:#6b7280;padding:3px 0;">Trade Size</td>
            <td style="color:#111827;font-weight:500;">{html.escape(opp.get('trade_size',''))}</td>
          </tr>
          <tr>
            <td style="color:#6b7280;padding:3px 0;">Committee</td>
            <td style="color:#111827;font-weight:500;">{html.escape(opp.get('committee',''))}</td>
          </tr>
          <tr>
            <td style="color:#6b7280;padding:3px 0;">Insider Proximity</td>
            <td style="color:{text_col};font-weight:600;">{html.escape(opp.get('insider_proximity_rating',''))}</td>
          </tr>
        </tbody>
      </table>
      <!-- Explanation -->
      <div style="font-size:13px;color:#374151;line-height:1.5;border-top:1px solid {border_col}33;padding-top:10px;">
        {html.escape(opp.get('explanation',''))}
      </div>
    </div>
  </div>"""


def build_email_html(opportunities: list, scan_date: str) -> str:
    cards = "".join(build_opportunity_card(opp) for opp in opportunities)
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f3f4f6;">
  <div style="max-width:600px;margin:0 auto;padding:16px;">
    <!-- Title bar -->
    <div style="background:#1e3a5f;border-radius:10px 10px 0 0;padding:18px 20px;margin-bottom:0;">
      <div style="color:#ffffff;font-size:18px;font-weight:700;font-family:Arial,sans-serif;">
        Government Signal Investment Briefing
      </div>
      <div style="color:#93c5fd;font-size:13px;font-family:Arial,sans-serif;margin-top:4px;">
        {html.escape(scan_date)} &nbsp;·&nbsp; Top 5 Congressional Trade Signals
      </div>
    </div>
    <!-- Legend -->
    <div style="background:#ffffff;padding:10px 20px;display:flex;gap:16px;font-size:11px;font-family:Arial,sans-serif;margin-bottom:16px;border-radius:0 0 6px 6px;">
      <span style="color:#16a34a;font-weight:600;">&#9646; STRONG ≥75</span>
      <span style="color:#d97706;font-weight:600;">&#9646; MODERATE 50–74</span>
      <span style="color:#dc2626;font-weight:600;">&#9646; WEAK &lt;50</span>
    </div>
    {cards}
    <div style="text-align:center;font-size:10px;color:#9ca3af;font-family:Arial,sans-serif;margin-top:8px;padding-bottom:8px;">
      Not financial advice. For informational purposes only.
    </div>
  </div>
</body>
</html>"""


def build_error_html(error_msg: str, scan_date: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f3f4f6;">
  <div style="max-width:600px;margin:0 auto;padding:16px;">
    <div style="background:#991b1b;border-radius:10px 10px 0 0;padding:18px 20px;">
      <div style="color:#ffffff;font-size:18px;font-weight:700;font-family:Arial,sans-serif;">
        Investment Briefing — Scan Failed
      </div>
      <div style="color:#fca5a5;font-size:13px;font-family:Arial,sans-serif;margin-top:4px;">
        {html.escape(scan_date)}
      </div>
    </div>
    <div style="background:#fee2e2;border:2px solid #dc2626;border-top:none;border-radius:0 0 10px 10px;padding:16px;">
      <p style="font-family:Arial,sans-serif;font-size:13px;color:#7f1d1d;margin:0 0 12px 0;">
        The daily scan encountered an error and could not complete. Details below:
      </p>
      <pre style="background:#ffffff;border:1px solid #fca5a5;border-radius:6px;padding:12px;font-size:11px;color:#374151;overflow-x:auto;white-space:pre-wrap;word-break:break-all;">{html.escape(error_msg)}</pre>
    </div>
  </div>
</body>
</html>"""


def send_email(subject: str, html_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = TO_EMAIL
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())


def run_scan() -> None:
    scan_date = datetime.now().strftime("%A, %B %-d, %Y")
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        raw = call_claude(client)
        opportunities = extract_json(raw)
        html_body = build_email_html(opportunities, scan_date)
        send_email(f"Investment Briefing — {scan_date}", html_body)
        print(f"Briefing sent for {scan_date}")
    except Exception:
        error_detail = traceback.format_exc()
        print(error_detail)
        try:
            html_body = build_error_html(error_detail[:3000], scan_date)
            send_email(f"Investment Briefing FAILED — {scan_date}", html_body)
            print("Error notification email sent.")
        except Exception as mail_err:
            print(f"Could not send error email: {mail_err}")


if __name__ == "__main__":
    run_scan()
