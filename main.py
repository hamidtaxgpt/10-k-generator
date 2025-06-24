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
from datetime import datetime, timezone

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # Fallback minimal loader so missing python-dotenv won't crash production.
    from pathlib import Path
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

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

        # 1) Table detection – fallback to tab-delimited monospace so columns align
        if line.strip().startswith("|") and "|" in line:
            tbl_lines = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                tbl_lines.append(lines[idx].strip())
                idx += 1

            # Build cleaned table cells
            parsed_rows = []
            for r in tbl_lines:
                cells = [re.sub(r"\*+", "", c.strip()) for c in r.strip("| ").split("|")]
                parsed_rows.append(cells)

            # Remove separator row of dashes if present
            if parsed_rows and all(re.fullmatch(r"-+", c) for c in parsed_rows[1]):
                parsed_rows.pop(1)

            # Compute maximum width of each column
            col_count = max(len(r) for r in parsed_rows)
            col_widths = [0] * col_count
            for r in parsed_rows:
                for i, cell in enumerate(r):
                    col_widths[i] = max(col_widths[i], len(cell))

            # Emit rows with space-padded columns
            for row_cells in parsed_rows:
                padded_cells = [row_cells[i].ljust(col_widths[i]) if i < len(row_cells) else ''.ljust(col_widths[i])
                                for i in range(col_count)]
                line_text = "  ".join(padded_cells) + "\n"  # two spaces between columns

                requests_batch.append({
                    "insertText": {
                        "location": {"index": current_index},
                        "text": line_text,
                    }
                })

                requests_batch.append({
                    "updateTextStyle": {
                        "range": {
                            "startIndex": current_index,
                            "endIndex": current_index + len(line_text) - 1,
                        },
                        "textStyle": {
                            "weightedFontFamily": {"fontFamily": "Courier New"}
                        },
                        "fields": "weightedFontFamily",
                    }
                })

                current_index += len(line_text)

                # Add a blank line after each table row for visual spacing
                requests_batch.append({
                    "insertText": {
                        "location": {"index": current_index},
                        "text": "\n",
                    }
                })
                current_index += 1

            continue  # table handled

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
        def _insert_paragraph(raw_text: str, paragraph_style: str | None = None, bullet: bool = False, nesting_level: int = 0):
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


def generate_title(report_text: str) -> str:
    """Use a higher-capacity model to generate a concise document title."""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # falls back to o1-mini if unavailable
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert copywriter generating short, professional report titles. Return only the title—no quotes, no punctuation before or after. Limit to 12 words."
                },
                {
                    "role": "user",
                    "content": f"Draft a title for the following tax-planning report:\n\n{report_text}\n\nTitle:"
                }
            ],
            max_completion_tokens=32,
            temperature=0.4,
        )
        title_raw = resp.choices[0].message.content.strip()
        # Remove leading/trailing quotes or punctuation
        return title_raw.strip('"\n').strip()
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        return "Tax Planning Report"


def create_google_doc(report_text, job_id, title):
    """Create & share a Google Doc, return its public URL."""
    creds = get_google_credentials()
    if not creds:
        return None

    docs = build("docs", "v1", credentials=creds)
    drive = build("drive", "v3", credentials=creds)

    # Create the document with the generated title
    doc = docs.documents().create(body={
        "title": title
    }).execute()
    doc_id = doc["documentId"]

    # Use the report text directly; the LLM already includes a title/heading
    requests_batch = convert_markdown_to_docs_format(report_text)
    if requests_batch:
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests_batch}).execute()

    # Make it publicly viewable
    try:
        drive.permissions().create(fileId=doc_id, body={"role": "reader", "type": "anyone"}).execute()
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

    from datetime import datetime, timezone
    today_str = datetime.now(timezone.utc).strftime("%B %d, %Y")

    prompt = (
        f"Report date: {today_str}\n\n"  # give the model the current date
        """You are a senior corporate tax advisor analyzing the full 10-K filing for a public U.S. company. Your task is to produce a structured tax planning report for the company's executive tax team.

Use the language of the filing when appropriate. Anchor your insights in specific phrases, footnotes, or financial disclosures from the document.

Your report must include the following sections with headings:

1. Tax Savings Opportunities
    - Highlight deductions or credits mentioned in the filing.
    - Identify areas where tax treatments (e.g., depreciation, capitalization, NOLs) are significant.
    - Use estimates where appropriate and include numerical tax impact where you can.

2. Underutilized Tax Credits
    - Identify whether credits like R&D, 179D, WOTC, AMT, or foreign tax credits are used.
    - Call out credits that are *not mentioned* in the filing but could apply based on business model.
    - Provide implementation recommendations and potential tax value ranges.

3. Strategies to Reduce Effective Tax Rate (ETR)
    - Recommend legal structuring, timing, or planning opportunities.
    - Show how changes might impact the ETR with rough % estimates.
    - Address IRC limitations like 163(j) and 382 if relevant.

4. Peer Comparison & Benchmarking
    - Compare the company's tax position with 2–3 peers (if peer names not available, simulate).
    - Note practices others use (e.g., IP migration, aggressive credit use, entity structuring).
    - Include a table if possible for revenue/ETR comparison.

Important Style Instructions:
- Use bullet points for clarity.
- Quantify tax opportunities and ETR impact wherever possible.
- Use professional, data-driven language.
- Cite the 10-K using section names, table titles, or footnote numbers when making claims.

Close the report with a summary and 3 high-priority next steps for the tax team.

SEC Filing Content:
""" + filing_text)
    try:
        res = openai_client.chat.completions.create(model="o1-mini",
                                                    messages=[{
                                                        "role": "user",
                                                        "content": prompt
                                                    }],
                                                    max_completion_tokens=4000)
        answer = res.choices[0].message.content
        sources = re.findall(r"https?://\S+", answer)

        title = generate_title(answer)
        doc_url = create_google_doc(answer, job_id, title)
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
    # Default to port 8080 for Replit
    port = int(os.environ.get("PORT", 8080))
    # In development, use Flask's built-in server
    app.run(host='0.0.0.0', port=port)
else:
    # In production (gunicorn), app is imported directly
    port = int(os.environ.get("PORT", 8080))
    # Configure any production-specific settings here
    pass
