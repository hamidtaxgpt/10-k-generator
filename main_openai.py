import os
import time
import logging
import re
import threading
import uuid
from flask import Flask, request, jsonify
import requests
import trafilatura
from urllib.parse import urlparse
from openai import OpenAI
import json
from textwrap import wrap
from utils import (
    save_job_status,
    load_job_status,
    generate_title,
    create_google_doc,
    extract_text_from_url,
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Global dictionary to store analysis results
analysis_results = {}

# Split the filing into ~30 000-character chunks (~7 500 tokens). This keeps each
# compression request safely inside the o1-mini context window and prevents
# "maximum context length" errors.
COMPRESS_CHUNK_SIZE = 30_000   # chars (~7.5k tokens)
# Hard ceiling on how many chunks we process, regardless of filing length.
MAX_COMPRESS_CALLS = 6

COMPRESS_SCHEMA = """
Return ONLY JSON with this exact schema:
{
  "company":                string,
  "fiscalYearEnd":          string | null,
  "keyNumbers":             object,      // any numeric figure you find, keys lower-snake
  "creditsMentioned":       string[],    // e.g., ["R&D", "ITC"]
  "segmentBreakdown": [
      { "name": string, "revenue": number | null, "etr": number | null }
  ],
  "verbatimExtracts": [
      { "label": string, "text": string }
  ]
}
If a value is missing write null. Do NOT use placeholders like $XX.

------------------------------------------------------------------
Extraction guidelines (added 2025-06-19)
------------------------------------------------------------------
- For every monetary figure or percentage that appears in the excerpt
  (revenue, margins, tax expense, effective tax rate, assets, liabilities,
  share count, etc.) add an entry under `keyNumbers`.
- Name keys in lower_snake_case and suffix with a unit code:
    _usd_m  → millions USD (e.g., 150 for $150m)
    _usd_bn → billions USD (e.g., 1.5 for $1.5bn)
    _pct    → percentage values (e.g., 25 for 25%).
- If the source says "2.3 billion," convert to billions (2.3) and use _usd_bn.
  "250 million" → 250 with _usd_m.
- Include every numeric figure ≥ $ 1 million or any percentage, even if it
  doesn't look directly tax-related.
- Pay special attention to any figure that contains the word "tax" (e.g.,
  income tax expense, cash taxes paid, effective tax rate, tax credit value,
  net operating loss carryforwards, deferred tax asset/liability).
- Avoid duplicate keys—keep the first occurrence.
"""

def _compress_chunk(chunk: str) -> dict:
    prompt = (
        "You are an expert SEC-filing parser.\n"
        f"{COMPRESS_SCHEMA}\n\nSEC excerpt:\n```\n{chunk}\n```"
    )
    resp = openai_client.chat.completions.create(
        model="o1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4096,
    )
    raw = resp.choices[0].message.content or ""
    # Remove ```json ... ``` or ``` fences if present
    if raw.lstrip().startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip(), flags=re.S)
    return json.loads(raw)

def compress_with_openai(full_text: str) -> dict:
    merged = {
        "company": None,
        "fiscalYearEnd": None,
        "keyNumbers": {},
        "creditsMentioned": set(),
        "segmentBreakdown": [],
        "verbatimExtracts": [],
    }
    for i, chunk in enumerate(wrap(full_text, COMPRESS_CHUNK_SIZE)):
        if i >= MAX_COMPRESS_CALLS:
            logger.info("compress_with_openai: stopped after %d chunks (limit)", MAX_COMPRESS_CALLS)
            break
        try:
            part = _compress_chunk(chunk)
            merged["company"]          = merged["company"] or part.get("company")
            merged["fiscalYearEnd"]    = merged["fiscalYearEnd"] or part.get("fiscalYearEnd")
            merged["keyNumbers"].update(part.get("keyNumbers", {}))
            merged["creditsMentioned"].update(part.get("creditsMentioned", []))
            merged["segmentBreakdown"].extend(part.get("segmentBreakdown", []))
            merged["verbatimExtracts"].extend(part.get("verbatimExtracts", []))
        except Exception as e:
            logger.warning(f"Compression chunk failed: {e}")
    merged["creditsMentioned"] = list(merged["creditsMentioned"])
    return merged

def analyze_with_openai_async(analysis_id, filing_text):
    """
    Analyze SEC filing using OpenAI o3-mini model asynchronously.
    This function stores results in the global analysis_results dictionary.
    """
    global analysis_results
    
    logger.debug(f"Starting OpenAI o3-mini analysis for job {analysis_id}")
    
    try:
        # Mark as processing
        analysis_results[analysis_id] = {
            "status": "processing",
            "answer": "",
            "sources": []
        }
        
        # Truncate very large filings to manageable size for OpenAI
        max_content_length = 100000  # 100K characters for o3-mini
        if len(filing_text) > max_content_length:
            logger.debug(f"Large filing detected ({len(filing_text)} chars), truncating for OpenAI processing")
            filing_text = filing_text[:max_content_length] + "\n\n[Content truncated for analysis]"
        
        # Construct comprehensive tax analysis prompt
        prompt = f"""You are a professional tax analyst specializing in corporate tax strategy and SEC filing analysis.

STRICT CONTENT RULES (read carefully):
1. You may use ONLY the financial metrics, text extracts, and other facts that appear in the JSON payload below.  
2. DO NOT introduce generic IRS publications, statutes, or web links that are not present in the JSON.  
3. If a figure is null, write "N/A (not disclosed)"—never guess.

Start immediately with the first heading—no introductory sentences.

Structure your analysis with the following sections exactly as titled:

## 1. Tax Savings Opportunities
- Available tax credits and incentives not fully utilized
- Potential deductions that could be maximized
- Strategic timing opportunities for tax benefits

## 2. Underutilized Tax Credits
- Research & Development credits
- Foreign tax credits
- Alternative minimum tax credits
- Other applicable credits mentioned or implied in the filing

## 3. Effective Tax Rate (ETR) Reduction Strategies
- Geographic tax optimization
- Transfer pricing opportunities
- Corporate structure improvements
- Timing strategies for recognition

## 4. Peer Comparison and Benchmarking
- Industry average effective tax rates
- Similar-sized companies in the same sector
- Best practices observed in comparable filings

Provide actionable recommendations with estimated tax impact where possible, citing the JSON keys/labels you are using (e.g., "keyNumbers.research_and_development") so the reader can trace every claim to the provided data.

Do NOT include:
• Introductory paragraphs or summaries before Section 1.
• A concluding paragraph or disclaimer after the Actionable Recommendations.
• Generic boilerplate (e.g., "Based on the indexed information").

For the final section titled "Actionable Recommendations":
• Begin with a concise bulleted list – one bullet per action, 1–2 sentences each, highest-impact item first.  
• Immediately after the bullets, insert a single Markdown table **without** code-fences, using these exact columns:  
Priority | Action | Estimated Savings |  
List the highest-impact item first.  Do not include any additional columns beyond these three.

End the report after the last recommendation.

SEC Filing Content:
{filing_text}

Provide a detailed, professional analysis with specific recommendations and quantified benefits where applicable.

Do NOT include any disclaimer or note about excerpted data or model limitations.

# ------------------------------------------------------------------
# Quantification and data referencing rules
# ------------------------------------------------------------------
Quantification and data referencing rules
• For every number or fact, reference the exact JSON key (e.g., `keyNumbers.backlog_usd_million_dec_31_2024`).
• If a value is missing or null, write "N/A" instead of a number.
• Each of Sections 1–3 must cite **at least two numeric figures** taken from the JSON (e.g., keyNumbers.revenue_2024_total_millions_usd).
• Show units – use **$ m** for millions USD, **$ bn** for billions USD, and **%** for percentages.
• Never invent numbers; if a value is null write "N/A".
• Avoid ranges; give a single figure or midpoint.
• Bold the figures when they appear inside prose for quick scanning.
• When referencing qualitative statements, cite the `verbatimExtracts.label` (e.g., "see verbatimExtracts.Backlog").
• When discussing business segments, use the `segmentBreakdown` array and show a table with columns: Segment | Revenue (USD m) | ETR (if available). Use the JSON keys for each value.

Linking rules
• When citing a source, embed the URL once using Markdown format with concise descriptive text (e.g., [IRS R&D credit guide]).  
• Do **not** repeat the bare URL in parentheses or elsewhere; include each link only once.

# ------------------------------------------------------------------
# Key metrics table
# ------------------------------------------------------------------
Before Section 1, insert a Markdown table titled **"Key Metrics from Filing"** that lists every entry found in the `keyNumbers` object plus any other numeric figure you choose to reference.  Use this format (no code-fence):

| Metric | Value | Source Label |

---
"""

        logger.debug(f"Sending analysis request to OpenAI o3-mini")
        
        # Call OpenAI o3-mini
        response = openai_client.chat.completions.create(
            model="o3-mini",  # Using o3-mini as requested
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional tax analyst with expertise in corporate tax strategy, SEC filings, and tax optimization. Provide detailed, actionable analysis with specific recommendations."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=4000,
            temperature=0.3
        )
        
        analysis_result = response.choices[0].message.content
        
        # Strip intro and conclusion for cleanliness
        analysis_result = _sanitize_answer(analysis_result)
        
        # Extract any URLs mentioned in the analysis as sources
        sources = re.findall(r"https?://\S+", analysis_result)
        
        # Store successful result
        analysis_results[analysis_id] = {
            "status": "done",
            "answer": analysis_result,
            "sources": sources
        }
        
        logger.info(f"OpenAI o3-mini analysis complete for job {analysis_id}. Response length: {len(analysis_result)}")
        
    except Exception as e:
        logger.error(f"OpenAI analysis error for job {analysis_id}: {str(e)}")
        analysis_results[analysis_id] = {
            "status": "error",
            "answer": f"Analysis failed: {str(e)}",
            "sources": []
        }

TAXGPT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://app.taxgpt.com",
    "Referer": "https://app.taxgpt.com/",
    "Authorization": f"Bearer {os.environ.get('TAXGPT_API_KEY', '')}"
}
TAXGPT_CHAT_URL   = "https://api.taxgpt.com/api/chats/"
TAXGPT_PROMPT_URL = "https://api.taxgpt.com/api/chats/{chat_id}/prompts/"

TAXGPT_MAIN_PROMPT = """
You are a senior tax strategy executive preparing a high-level analysis for C-suite executives.

STRICT CONTENT RULES:
1. Use ONLY facts and figures from the provided JSON data
2. NEVER mention JSON, source labels, or technical references
3. NEVER include external links or citations
4. Present all numbers in a clean format:
   - Use "$ XXm" for millions (e.g., "$ 150m")
   - Use "$ XXbn" for billions (e.g., "$ 1.5bn")
   - Use "XX%" for percentages (e.g., "25%")
5. NEVER use ranges - provide specific numbers
6. If data is unavailable, use "Not disclosed" (never use N/A, null, or TBD)

FORMAT AND STRUCTURE:
Start with the title "Tax Strategy Analysis for [Company Name] - FY [Year]"
Then immediately begin with Section 1 - no introduction needed.

## 1. Tax Savings Opportunities
- Cite at least two specific financial figures where available; if fewer than two exist, write "Not disclosed" for the missing items
- Focus on immediate actionable opportunities
- Quantify potential savings where possible

## 2. Underutilized Tax Credits
- Cite at least two specific financial figures where available; if fewer than two exist, write "Not disclosed" for the missing items
- Focus on credits mentioned in or implied by financial data
- Quantify credit values where possible

## 3. ETR Reduction Strategies
- Cite at least two specific financial figures where available; if fewer than two exist, write "Not disclosed" for the missing items
- Focus on structural and operational opportunities
- Quantify ETR impact where possible

## 4. Peer Comparison
- Compare key metrics to industry standards
- Focus on tax efficiency metrics
- Identify competitive advantages/disadvantages

## 5. Tax Savings Summary
Present ONE table at the end with this EXACT format:

| Strategy | Estimated Savings |
|----------|-------------------|
| [Clear strategy name] | [Projected saving in $] |

Table Rules:
- Include 4–6 highest-impact strategies
- Every figure must be drawn directly from the data
- Use consistent formatting ($ XXm or $ XXbn)
- No placeholder values (Not disclosed, TBD, etc.)

EXECUTIVE COMMUNICATION RULES:
1. Write in clear, executive-level language
2. Focus on material impacts (>$ 1m)
3. Be specific with numbers but concise with explanations
4. Avoid technical jargon
5. No disclaimers or hedging language
6. No mentions of JSON, keys, or technical terms

END the report after the summary table. No conclusion needed.
"""

def _sanitize_answer(raw: str) -> str:
    """Return content up to (but excluding) any 'Conclusion' heading. Intro lines are preserved."""
    import re
    lines = raw.splitlines()
    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if re.match(r"\s*#?\s*Conclusion", lines[i], re.I):
            end = i
            break
    trimmed = "\n".join(lines[:end]).strip()
    return trimmed

def analyze_with_taxgpt_async(job_id: str, compressed_json: dict):
    save_job_status(job_id, {"status": "processing", "answer": "", "sources": [], "doc_url": None})
    try:
        # 1 create chat
        chat_id = requests.post(TAXGPT_CHAT_URL, headers=TAXGPT_HEADERS,
                                json={"type": "professional"}, timeout=30).json()["id"]

        # 2 send prompt
        full_prompt = TAXGPT_MAIN_PROMPT + "\n\nJSON DATA:\n" + json.dumps(compressed_json, indent=2)
        requests.post(
            TAXGPT_PROMPT_URL.format(chat_id=chat_id),
            headers=TAXGPT_HEADERS,
            json={"prompt": full_prompt},
            timeout=120,
        )

        # 3 poll every 15 s, up to 5 min total
        answer = ""
        for _ in range(20):
            time.sleep(15)
            try:
                resp = requests.get(
                    TAXGPT_PROMPT_URL.format(chat_id=chat_id),
                    headers=TAXGPT_HEADERS,
                    timeout=60,
                )
                resp.raise_for_status()
                hist = resp.json()
                if hist and hist[-1].get("prompt_answer"):
                    answer = hist[-1]["prompt_answer"]
                    break
            except requests.RequestException as poll_exc:
                logger.warning("TaxGPT poll error: %s", poll_exc)
                continue  # keep polling
        if not answer:
            raise RuntimeError("TaxGPT did not return an answer within 5 minutes")

        # 4 google doc
        title = generate_title(answer)
        # Remove any leading/trailing ``` fences and strip intro/conclusion
        cleaned_answer = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", answer.strip(), flags=re.M)
        cleaned_answer = _sanitize_answer(cleaned_answer)
        doc_url = create_google_doc(cleaned_answer, job_id, title, format_markdown=True)
        save_job_status(job_id, {"status": "done", "answer": cleaned_answer,
                                 "sources": [], "doc_url": doc_url})
    except Exception as e:
        save_job_status(job_id, {"status": "error", "answer": str(e),
                                 "sources": [], "doc_url": None})
        logger.error(f"TaxGPT error for {job_id}: {e}")

# ---------------------------------------------------------------------------
# Wrapper to run compression in the worker thread instead of the request
# ---------------------------------------------------------------------------

def _compress_and_taxgpt_async(job_id: str, raw_text: str):
    """Background helper: compress SEC filing with o1-mini then send to TaxGPT."""
    try:
        logger.debug("Job %s – starting compression for TaxGPT", job_id)
        compressed = compress_with_openai(raw_text)
        logger.debug("Job %s – compression done (%d keys)", job_id, len(compressed))
        # --- Dump compressed JSON for debugging/inspection
        try:
            with open(f"/tmp/{job_id}_compressed.json", "w") as fh:
                json.dump(compressed, fh, indent=2)
            logger.debug("Job %s – compressed JSON written to /tmp/%s_compressed.json", job_id, job_id)
        except Exception as dump_exc:
            logger.warning("Job %s – could not dump compressed JSON: %s", job_id, dump_exc)
        analyze_with_taxgpt_async(job_id, compressed)
    except Exception as exc:
        logger.error("Job %s – compression/TaxGPT pipeline failed: %s", job_id, exc)
        save_job_status(job_id, {"status": "error", "answer": str(exc), "sources": [], "doc_url": None})

def create_analysis_job(text: str):
    job_id = str(uuid.uuid4())
    # Log which pipeline will run (OpenAI vs TaxGPT) for easier debugging
    use_taxgpt = os.getenv("USE_TAXGPT", "false").lower() == "true"
    logger.info("USE_TAXGPT=%s – %s pipeline selected", os.getenv("USE_TAXGPT"), "TaxGPT" if use_taxgpt else "OpenAI")
    # compress filing first
    save_job_status(job_id, {"status": "processing", "answer": "In progress…",
                             "sources": [], "doc_url": None})
    if use_taxgpt:
        thread = threading.Thread(target=_compress_and_taxgpt_async,
                                  args=(job_id, text))
    else:
        thread = threading.Thread(target=analyze_with_openai_async,
                                  args=(job_id, text))
    thread.daemon = True
    thread.start()
    return job_id

def get_analysis_status(job_id):
    """
    Check analysis status and return result if complete.
    """
    global analysis_results
    
    # Check in-memory results (OpenAI pipeline)
    if job_id in analysis_results:
        result = analysis_results[job_id]
        logger.debug("Found in-memory result for job %s: %s", job_id, result["status"])
        return result

    # Fallback to disk (TaxGPT pipeline)
    file_result = load_job_status(job_id)
    if file_result:
        logger.debug("Loaded result for job %s from disk: %s", job_id, file_result["status"])
        return file_result

    # Not found anywhere
    logger.warning("Job %s not found in memory or on disk", job_id)
    return {
        "status": "not_found",
        "answer": "Job ID not found. Please check the job ID or start a new analysis.",
        "sources": []
    }

# Flask routes
@app.route('/')
def index():
    """Root endpoint with API information"""
    return jsonify({
        "message": "SEC Filing Analysis API using OpenAI o3-mini",
        "endpoints": {
            "start_analysis": "POST /analyze/start - Start SEC filing analysis",
            "check_status": "GET /analyze/status/<job_id> - Check analysis status",
            "health": "GET /health - Health check"
        },
        "model": "OpenAI o3-mini"
    })

@app.route('/analyze/start', methods=['POST'])
def start_analysis():
    """
    Start SEC filing analysis endpoint.
    Accepts JSON: {"url": "https://www.sec.gov/Archives/edgar/data/..."}
    Returns: {"job_id": "analysis_job_id"}
    """
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "URL is required in JSON body"}), 400
        
        url = data['url']
        logger.info(f"Starting analysis for URL: {url}")
        
        # Extract text from URL
        text = extract_text_from_url(url)
        if not text or len(text.strip()) < 100:
            return jsonify({"error": "Unable to extract sufficient text content from URL"}), 400
        
        # Create analysis job
        job_id = create_analysis_job(text)
        
        logger.info(f"Analysis started with job ID: {job_id}")
        return jsonify({"job_id": job_id})
        
    except Exception as e:
        logger.error(f"Error in start_analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/status/<job_id>')
def check_analysis_status(job_id):
    """
    Check analysis status endpoint.
    URL param: job_id (Analysis job ID)
    Returns: {"status": "processing"} or {"status": "done", "answer": "...", "sources": [...]}
    """
    try:
        logger.info(f"Checking status for job ID: {job_id}")
        
        result = get_analysis_status(job_id)
        
        logger.info(f"Status check complete for job ID: {job_id}, status: {result['status']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error checking status for job {job_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "OpenAI o3-mini",
        "timestamp": time.time()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)