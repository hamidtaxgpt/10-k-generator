import os
import time
import json
import logging
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

from openai import OpenAI
import requests
import trafilatura
from bs4 import BeautifulSoup
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

__all__ = [
    "save_job_status",
    "load_job_status",
    "cleanup_old_jobs",
    "get_google_credentials",
    "_apply_inline_styles",
    "convert_markdown_to_docs_format",
    "generate_title",
    "create_google_doc",
    "extract_text_from_url",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job-status persistence helpers
# ---------------------------------------------------------------------------

JOBS_DIR = "/tmp/sec_analysis_jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

def save_job_status(job_id: str, status_data: Dict[str, Any]) -> None:
    """Persist job status across restarts so the front-end can poll."""
    try:
        with open(f"{JOBS_DIR}/{job_id}.json", "w") as fh:
            json.dump(status_data, fh)
        logger.debug("Saved job status for %s", job_id)
    except Exception as exc:
        logger.error("Failed to save job status for %s: %s", job_id, exc)


def load_job_status(job_id: str) -> Dict[str, Any] | None:
    """Return stored job status or None if the job ID is unknown."""
    try:
        path = f"{JOBS_DIR}/{job_id}.json"
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    except Exception as exc:
        logger.error("Failed to load job status for %s: %s", job_id, exc)
    return None


def cleanup_old_jobs(max_age_hours: int = 24) -> None:
    """Delete job files older than *max_age_hours* to avoid /tmp clutter."""
    now = time.time()
    horizon = max_age_hours * 3600
    for fn in os.listdir(JOBS_DIR):
        if fn.endswith(".json"):
            full = os.path.join(JOBS_DIR, fn)
            if now - os.path.getmtime(full) > horizon:
                try:
                    os.remove(full)
                    logger.debug("Removed old job file %s", fn)
                except Exception as exc:
                    logger.warning("Could not delete %s: %s", full, exc)


# ---------------------------------------------------------------------------
# Google-API helpers
# ---------------------------------------------------------------------------

def get_google_credentials():
    """Return fresh OAuth2 credentials using the env-var refresh-token trio."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")
    if not all([client_id, client_secret, refresh_token]):
        logger.error("Missing Google OAuth credentials; Docs export disabled")
        return None

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri="https://oauth2.googleapis.com/token",
    )
    if not creds.valid:
        try:
            creds.refresh(Request())
            logger.debug("Refreshed Google creds")
        except Exception as exc:
            logger.error("Failed refreshing Google creds: %s", exc)
            return None
    return creds


# ---------------------------------------------------------------------------
# Markdown-to-Google-Docs converter
# ---------------------------------------------------------------------------

def _apply_inline_styles(text_with_markers: str, base_index: int, requests: List[Dict[str, Any]]):
    """Apply **bold** and *italic* styling for Google-Docs batchUpdate payload."""
    orig_to_clean: dict[int, int] = {}
    clean_chars: list[str] = []
    for i, ch in enumerate(text_with_markers):
        if ch != "*":
            orig_to_clean[i] = len(clean_chars)
            clean_chars.append(ch)
    def translate_span(start_orig: int, end_orig: int) -> tuple[int, int]:
        while start_orig < end_orig and start_orig not in orig_to_clean:
            start_orig += 1
        last_orig_char = end_orig - 1
        while last_orig_char >= start_orig and last_orig_char not in orig_to_clean:
            last_orig_char -= 1
        if start_orig > last_orig_char:
            return (0, 0)
        start_clean = orig_to_clean[start_orig]
        end_clean = orig_to_clean[last_orig_char] + 1
        return (start_clean, end_clean)
    # Bold spans
    for m in re.finditer(r"\*\*(.+?)\*\*", text_with_markers):
        span_start, span_end = translate_span(m.start() + 2, m.end() - 2)
        if span_start != span_end:
            requests.append({
                "updateTextStyle": {
                    "range": {"startIndex": base_index + span_start, "endIndex": base_index + span_end},
                    "textStyle": {"bold": True},
                    "fields": "bold",
                }
            })
    # Italic spans (avoid **bold**)
    italics_pattern = r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"
    for m in re.finditer(italics_pattern, text_with_markers):
        span_start, span_end = translate_span(m.start() + 1, m.end() - 1)
        if span_start != span_end:
            requests.append({
                "updateTextStyle": {
                    "range": {"startIndex": base_index + span_start, "endIndex": base_index + span_end},
                    "textStyle": {"italic": True},
                    "fields": "italic",
                }
            })


def convert_markdown_to_docs_format(text: str) -> List[Dict[str, Any]]:
    """Convert a subset of Markdown into Docs API batchUpdate requests."""
    requests_batch: List[Dict[str, Any]] = []
    lines = text.split("\n")
    current_index = 1

    for line in lines:
        line = line.strip()
        if not line:
            # Handle empty lines
            requests_batch.append({
                "insertText": {
                    "location": {"index": current_index},
                    "text": "\n"
                }
            })
            current_index += 1
            continue

        # Handle headings
        handled_heading = False
        for prefix, style in [("####", "HEADING_4"), ("###", "HEADING_3"), ("##", "HEADING_2"), ("#", "HEADING_1")]:
            if line.startswith(prefix):
                heading_text = line[len(prefix):].strip() + "\n"
                requests_batch.append({
                    "insertText": {
                        "location": {"index": current_index},
                        "text": heading_text
                    }
                })
                requests_batch.append({
                    "updateParagraphStyle": {
                        "range": {
                            "startIndex": current_index,
                            "endIndex": current_index + len(heading_text)
                        },
                        "paragraphStyle": {"namedStyleType": style},
                        "fields": "namedStyleType"
                    }
                })
                current_index += len(heading_text)
                handled_heading = True
                break
        
        if not handled_heading:
            # Handle regular text with inline styling
            line_text = line + "\n"
            requests_batch.append({
                "insertText": {
                    "location": {"index": current_index},
                    "text": line_text
                }
            })
            _apply_inline_styles(line_text, current_index, requests_batch)
            current_index += len(line_text)

    return requests_batch


# ---------------------------------------------------------------------------
# Google Docs integration helpers
# ---------------------------------------------------------------------------

def generate_title(report_text: str) -> str:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert copywriter generating short, professional report titles. Return only the title—no quotes, no punctuation before or after. Limit to 12 words."},
                {"role": "user", "content": f"Draft a title for the following tax-planning report:\n\n{report_text}\n\nTitle:"},
            ],
            max_completion_tokens=32,
            temperature=0.4,
        )
        raw = resp.choices[0].message.content.strip()
        return raw.strip('"\n').strip()
    except Exception as exc:
        logger.warning("Title generation failed: %s", exc)
        return "Tax Planning Report"


def create_google_doc(report_text: str, job_id: str, title: str, format_markdown: bool = True) -> str | None:
    creds = get_google_credentials()
    if not creds:
        return None
    try:
        docs = build("docs", "v1", credentials=creds)
        drive = build("drive", "v3", credentials=creds)
        doc = docs.documents().create(body={"title": title}).execute()
        doc_id = doc["documentId"]
        if format_markdown:
            batch = convert_markdown_to_docs_format(report_text)
        else:
            # Insert raw text without styling
            batch = [
                {
                    "insertText": {
                        "location": {"index": 1},
                        "text": report_text
                    }
                }
            ]
        if batch:
            docs.documents().batchUpdate(documentId=doc_id, body={"requests": batch}).execute()
        try:
            drive.permissions().create(fileId=doc_id, body={"role": "reader", "type": "anyone"}).execute()
        except Exception as exc:
            logger.warning("Couldn't set public permissions: %s", exc)
        return f"https://docs.google.com/document/d/{doc_id}/edit?usp=sharing"
    except Exception as exc:
        logger.error("Error creating Google Doc: %s", exc)
        return None


# ---------------------------------------------------------------------------
# SEC filing text extraction helper
# ---------------------------------------------------------------------------

def extract_text_from_url(url: str) -> str:
    ua_email = os.getenv("USER_AGENT_EMAIL", "support@taxgpt.com")
    headers = {
        # SEC guidelines: UA should include contact info. We mimic a browser + mailto.
        "User-Agent": (
            f"Mozilla/5.0 (compatible; TaxGPTSecAnalyzer/1.0; +https://app.taxgpt.com; {ua_email})"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sec.gov/",
        "Connection": "keep-alive",
    }
    try:
        logger.debug("Downloading %s", url)
        resp = requests.get(url, headers=headers, timeout=30)
        # 403? try alternative URLs once each
        if resp.status_code == 403:
            alt_attempts = []
            # 1) append ?download=1 (SEC allows manual download links)
            alt_attempts.append(url + ("&download=1" if "?" in url else "?download=1"))
            # 2) switch to plain-text version if original looked like HTML
            if url.lower().endswith((".htm", ".html")):
                alt_attempts.append(re.sub(r"\.html?$", ".txt", url, flags=re.IGNORECASE))
            for alt_url in alt_attempts:
                logger.debug("403 – retrying with alt URL: %s", alt_url)
                time.sleep(1)
                alt_resp = requests.get(alt_url, headers=headers, timeout=30)
                if alt_resp.status_code < 400:
                    resp = alt_resp
                    break
        resp.raise_for_status()
        text = trafilatura.extract(resp.text)
        if text and len(text) > 100:
            return text
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()
        lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
        return "\n".join(lines)
    except Exception as exc:
        logger.error("Text extraction error for %s: %s", url, exc)
        raise

# Run cleanup when module imported
cleanup_old_jobs() 