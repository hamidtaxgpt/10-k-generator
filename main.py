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


def _apply_inline_styles(text, base_index, requests):
    """Emit updateTextStyle calls for **bold** and *italic*."""
    # Bold formatting - skip the leading and trailing '**'
    for m in re.finditer(r'\*\*(.+?)\*\*', text):
        start = base_index + m.start() + 2
        end = base_index + m.end() - 2
        requests.append({
            'updateTextStyle': {
                'range': {
                    'startIndex': start,
                    'endIndex': end
                },
                'textStyle': {
                    'bold': True
                },
                'fields': 'bold'
            }
        })
    
    # Italic formatting - tighter regex to avoid catching bold markers
    italics_pattern = r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)'
    for m in re.finditer(italics_pattern, text):
        start = base_index + m.start() + 1
        end = base_index + m.end() - 1
        requests.append({
            'updateTextStyle': {
                'range': {
                    'startIndex': start,
                    'endIndex': end
                },
                'textStyle': {
                    'italic': True
                },
                'fields': 'italic'
            }
        })


def convert_markdown_to_docs_format(text):
    """
    Convert Markdown into Google Docs batchUpdate requests,
    handling headings, bullets, tables, bold and italics.
    """
    requests_batch = []
    lines = text.split("\n")
    idx = 0
    current_index = 1

    while idx < len(lines):
        line = lines[idx]

        # Table detection - fallback to tab-delimited plain text
        if line.strip().startswith("|") and "|" in line:
            # Collect table block
            tbl = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                tbl.append(lines[idx].strip())
                idx += 1

            # Parse rows and render as tab-delimited text
            rows = [row.strip("| ").split("|") for row in tbl]
            
            # Render each row as a single line with tabs
            for row in rows:
                line_text = "\t".join(cell.strip() for cell in row) + "\n"
                requests_batch.append({
                    "insertText": {
                        "location": {"index": current_index},
                        "text": line_text
                    }
                })
                current_index += len(line_text)

            continue  # skip the normal processing

        # Blank line
        if not line.strip():
            requests_batch.append({
                'insertText': {
                    'location': {
                        'index': current_index
                    },
                    'text': "\n"
                }
            })
            current_index += 1
            idx += 1
            continue

        # Headings
        for prefix, style in [("###", "HEADING_3"), ("##", "HEADING_2"),
                              ("#", "HEADING_1")]:
            if line.startswith(prefix):
                content = line[len(prefix):].strip() + "\n"
                requests_batch.append({
                    'insertText': {
                        'location': {
                            'index': current_index
                        },
                        'text': content
                    }
                })
                requests_batch.append({
                    'updateParagraphStyle': {
                        'range': {
                            'startIndex': current_index,
                            'endIndex': current_index + len(content) - 1
                        },
                        'paragraphStyle': {
                            'namedStyleType': style
                        },
                        'fields': 'namedStyleType'
                    }
                })
                current_index += len(content)
                idx += 1
                break
        else:
            # Bullet list
            if line.strip().startswith("- "):
                content = line.strip()[2:] + "\n"
                requests_batch.append({
                    'insertText': {
                        'location': {
                            'index': current_index
                        },
                        'text': content
                    }
                })
                requests_batch.append({
                    'createParagraphBullets': {
                        'range': {
                            'startIndex': current_index,
                            'endIndex': current_index + len(content) - 1
                        },
                        'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
                    }
                })
                _apply_inline_styles(content, current_index, requests_batch)
                current_index += len(content)
                idx += 1
            else:
                # Plain text (with inline styling)
                content = line + "\n"
                clean = re.sub(r"\*+", "", content)
                requests_batch.append({
                    'insertText': {
                        'location': {
                            'index': current_index
                        },
                        'text': clean
                    }
                })
                _apply_inline_styles(content, current_index, requests_batch)
                current_index += len(clean)
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
        logger.warning(f"Couldn’t set public permissions: {e}")

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
    app.run(host="0.0.0.0", port=5000, debug=True)
