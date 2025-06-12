import os
import time
import logging
import re
import threading
import uuid
import json
from flask import Flask, request, jsonify
import requests
import trafilatura
from bs4 import BeautifulSoup
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Job persistence directory
JOBS_DIR = "/tmp/sec_analysis_jobs"
os.makedirs(JOBS_DIR, exist_ok=True)


def save_job_status(job_id, status_data):
    """Save job status to file for persistence across restarts."""
    try:
        with open(f"{JOBS_DIR}/{job_id}.json", "w") as f:
            json.dump(status_data, f)
        logger.debug(f"Saved job status for {job_id}")
    except Exception as e:
        logger.error(f"Failed to save job status for {job_id}: {e}")


def load_job_status(job_id):
    """Load job status from file."""
    try:
        path = f"{JOBS_DIR}/{job_id}.json"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load job status for {job_id}: {e}")
    return None


def cleanup_old_jobs():
    """Remove job files older than 24 hours."""
    now = time.time()
    for fn in os.listdir(JOBS_DIR):
        if fn.endswith(".json"):
            path = os.path.join(JOBS_DIR, fn)
            if now - os.path.getmtime(path) > 24 * 3600:
                os.remove(path)
                logger.debug(f"Removed old job file {fn}")


cleanup_old_jobs()


def get_google_credentials():
    """Return OAuth2 credentials, refreshing if needed."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    if not all([client_id, client_secret, refresh_token]):
        logger.error("Missing Google OAuth credentials")
        return None

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri="https://oauth2.googleapis.com/token",
    )
    if not creds.valid:
        creds.refresh(Request())
        logger.debug("Refreshed Google credentials")
    return creds


def _apply_inline_styles(text_with_markers, base_index, requests):
    """Emit updateTextStyle calls for **bold** and *italic* taking into account
    that the text which will be/has been inserted into the document has all
    Markdown asterisk markers stripped out.

    The function therefore builds a mapping from the character positions in the
    *original* markdown string (containing asterisks) to the character
    positions of the *clean* string (with all * removed). Using this mapping we
    can calculate the correct Google-Docs index ranges after insertion.
    """
    # Build forward mapping: original_index -> cleaned_index (or None for markers)
    orig_to_clean: dict[int, int] = {}
    clean_chars = []
    for i, ch in enumerate(text_with_markers):
        if ch != "*":
            orig_to_clean[i] = len(clean_chars)
            clean_chars.append(ch)
        else:
            # asterisk markers are removed; they do not appear in clean text
            pass

    # Helper to translate a span (start, end) in the original text into the
    # equivalent (start, end) span in the cleaned text, assuming * markers have
    # been removed. `end` is exclusive.
    def translate_span(start_orig: int, end_orig: int) -> tuple[int, int]:
        """Return (start_clean, end_clean) for the provided original span."""
        # Advance forward from start_orig until we hit a non-marker character to
        # guard against unexpected multiple consecutive asterisks.
        while start_orig < end_orig and start_orig not in orig_to_clean:
            start_orig += 1
        # Likewise move the end index backwards so that end_orig-1 maps to the
        # last character *inside* the style span. Google Docs endIndex is
        # exclusive so we subsequently add +1.
        last_orig_char = end_orig - 1
        while last_orig_char >= start_orig and last_orig_char not in orig_to_clean:
            last_orig_char -= 1
        if start_orig > last_orig_char:
            # Nothing left after stripping (should not happen, but guard anyway)
            return (0, 0)
        start_clean = orig_to_clean[start_orig]
        end_clean = orig_to_clean[last_orig_char] + 1  # exclusive
        return (start_clean, end_clean)

    # Bold (**text**)
    for m in re.finditer(r"\*\*(.+?)\*\*", text_with_markers):
        # Exclude the leading/trailing ** when translating
        span_start, span_end = translate_span(m.start() + 2, m.end() - 2)
        if span_start == span_end:
            continue  # empty span after stripping, skip
        requests.append({
            "updateTextStyle": {
                "range": {
                    "startIndex": base_index + span_start,
                    "endIndex": base_index + span_end,
                },
                "textStyle": {"bold": True},
                "fields": "bold",
            }
        })

    # Italic (*text*) – use look-arounds to make sure we do not match **bold**
    italics_pattern = r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"
    for m in re.finditer(italics_pattern, text_with_markers):
        span_start, span_end = translate_span(m.start() + 1, m.end() - 1)
        if span_start == span_end:
            continue
        requests.append({
            "updateTextStyle": {
                "range": {
                    "startIndex": base_index + span_start,
                    "endIndex": base_index + span_end,
                },
                "textStyle": {"italic": True},
                "fields": "italic",
            }
        })


def convert_markdown_to_docs_format(text):
    r"""
    Convert Markdown into Google Docs batchUpdate requests. The converter now
    makes three key improvements:
      1. **All asterisk markers (\*) are stripped** from the text before it is
         inserted into the document – this guarantees that no stray markup
         remains visible.
      2. **Bold/italic offsets are corrected** via the improved
         `_apply_inline_styles` helper which calculates style ranges relative to
         the cleaned text.
      3. **Markdown tables are converted to tab-delimited text** which renders
         as columns in Google Docs and avoids invalid `tableCellLocation`
         payloads.
    """
    requests_batch = []
    lines = text.split("\n")
    idx = 0
    current_index = 1  # Google Docs body content starts at index 1

    while idx < len(lines):
        line = lines[idx]

        # 1) Table detection – convert to tab-delimited plain text
        if line.strip().startswith("|") and "|" in line:
            tbl_lines = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                tbl_lines.append(lines[idx].strip())
                idx += 1

            # Parse and render each row as TAB-delimited text
            for row in tbl_lines:
                cells = [cell.strip() for cell in row.strip("| ").split("|")]
                line_text = "\t".join(cells) + "\n"
                requests_batch.append({
                    "insertText": {
                        "location": {"index": current_index},
                        "text": line_text,
                    }
                })
                current_index += len(line_text)
            continue  # proceed to next outer while iteration

        # 2) Blank line
        if not line.strip():
            requests_batch.append({
                "insertText": {
                    "location": {"index": current_index},
                    "text": "\n",
                }
            })
            current_index += 1
            idx += 1
            continue

        # Helper to insert cleaned text and apply inline styles
        def _insert_paragraph(raw_text: str, paragraph_style: str | None = None, bullet: bool = False):
            nonlocal current_index, requests_batch
            clean_text = re.sub(r"\*+", "", raw_text) + "\n"
            # Insert the clean text first
            requests_batch.append({
                "insertText": {
                    "location": {"index": current_index},
                    "text": clean_text,
                }
            })
            # Apply heading style if requested
            if paragraph_style:
                requests_batch.append({
                    "updateParagraphStyle": {
                        "range": {
                            "startIndex": current_index,
                            "endIndex": current_index + len(clean_text) - 1,
                        },
                        "paragraphStyle": {"namedStyleType": paragraph_style},
                        "fields": "namedStyleType",
                    }
                })
            # Apply bullet preset if requested
            if bullet:
                requests_batch.append({
                    "createParagraphBullets": {
                        "range": {
                            "startIndex": current_index,
                            "endIndex": current_index + len(clean_text) - 1,
                        },
                        "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE",
                    }
                })
            # Apply bold / italic styling based on the *original* raw text.
            _apply_inline_styles(raw_text + "\n", current_index, requests_batch)
            current_index += len(clean_text)

        # 3) Headings
        handled_heading = False
        for prefix, style in [("###", "HEADING_3"), ("##", "HEADING_2"), ("#", "HEADING_1")]:
            if line.startswith(prefix):
                heading_text = line[len(prefix):].strip()
                _insert_paragraph(heading_text, paragraph_style=style)
                idx += 1
                handled_heading = True
                break
        if handled_heading:
            continue

        # 4) Bullet list item
        if line.strip().startswith("- "):
            bullet_text = line.strip()[2:]
            _insert_paragraph(bullet_text, bullet=True)
            idx += 1
            continue

        # 5) Plain paragraph text
        _insert_paragraph(line)
        idx += 1

    return requests_batch


def create_google_doc(report_text, job_id):
    """Create & share a Google Doc, return its public URL."""
    creds = get_google_credentials()
    if not creds:
        return None

    docs = build("docs", "v1", credentials=creds)
    drive = build("drive", "v3", credentials=creds)

    # Create the document
    doc = docs.documents().create(body={
        "title": f"Tax Report - {job_id}"
    }).execute()
    doc_id = doc["documentId"]

    # Batch-update with our formatting requests
    requests_batch = convert_markdown_to_docs_format(report_text)
    if requests_batch:
        docs.documents().batchUpdate(documentId=doc_id,
                                     body={
                                         "requests": requests_batch
                                     }).execute()

    # Make it publicly viewable
    try:
        drive.permissions().create(fileId=doc_id,
                                   body={
                                       "role": "reader",
                                       "type": "anyone"
                                   }).execute()
    except Exception as e:
        logger.warning(f"Couldn't set public permissions: {e}")

    return f"https://docs.google.com/document/d/{doc_id}/edit?usp=sharing"


def extract_text_from_url(url):
    """Try trafilatura first, then BeautifulSoup as fallback."""
    headers = {"User-Agent": "your-email@domain.com"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    text = trafilatura.extract(resp.text)
    if text and len(text) > 100:
        return text

    # BeautifulSoup fallback
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()
    body = soup.get_text(separator="\n")
    return "\n".join(line.strip() for line in body.splitlines()
                     if line.strip())


def analyze_with_openai_async(job_id, filing_text):
    """Run the OAI model in a background thread."""
    save_job_status(job_id, {
        "status": "processing",
        "answer": "",
        "sources": [],
        "doc_url": None
    })

    # Truncate if too long
    if len(filing_text) > 100_000:
        filing_text = filing_text[:100_000] + "\n\n[Truncated]"

    prompt = (
        "You are a senior corporate tax advisor... [rest of your prompt here]\n\n"
        f"SEC Filing Content:\n{filing_text}")
    try:
        res = openai_client.chat.completions.create(model="o1-mini",
                                                    messages=[{
                                                        "role": "user",
                                                        "content": prompt
                                                    }],
                                                    max_completion_tokens=4000)
        answer = res.choices[0].message.content
        sources = re.findall(r"https?://\S+", answer)

        doc_url = create_google_doc(answer, job_id)
        save_job_status(
            job_id, {
                "status": "done",
                "answer": answer,
                "sources": sources,
                "doc_url": doc_url
            })
        logger.info(f"Job {job_id} done, doc: {doc_url}")

    except Exception as e:
        save_job_status(job_id, {
            "status": "error",
            "answer": str(e),
            "sources": [],
            "doc_url": None
        })
        logger.error(f"Analysis error for {job_id}: {e}")


def create_analysis_job(text):
    job_id = str(uuid.uuid4())
    save_job_status(
        job_id, {
            "status": "processing",
            "answer": "In progress...",
            "sources": [],
            "doc_url": None
        })
    thread = threading.Thread(target=analyze_with_openai_async,
                              args=(job_id, text))
    thread.daemon = True
    thread.start()
    return job_id


def get_analysis_status(job_id):
    data = load_job_status(job_id)
    if not data:
        return {
            "status": "not_found",
            "answer": "No such job",
            "sources": [],
            "doc_url": None
        }
    return data


@app.route("/")
def index():
    return jsonify({
        "message": "SEC Filing Analysis API",
        "endpoints": {
            "start": "/analyze/start (POST)",
            "status": "/analyze/status/<job_id> (GET)",
            "health": "/health (GET)"
        }
    })


@app.route("/analyze/start", methods=["POST"])
def start_analysis():
    payload = request.get_json() or {}
    url = payload.get("url")
    if not url:
        return jsonify({"error": "Missing url"}), 400
    try:
        text = extract_text_from_url(url)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    job_id = create_analysis_job(text)
    return jsonify({"job_id": job_id})


@app.route("/analyze/status/<job_id>")
def status(job_id):
    return jsonify(get_analysis_status(job_id))


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "o1-mini",
        "time": time.time()
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
