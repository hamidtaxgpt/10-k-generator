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

COMPRESS_CHUNK_SIZE = 80_000          # chars; safe for o1-mini
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
        temperature=0.1,
    )
    return json.loads(resp.choices[0].message.content)

def compress_with_openai(full_text: str) -> dict:
    merged = {
        "company": None,
        "fiscalYearEnd": None,
        "keyNumbers": {},
        "creditsMentioned": set(),
        "segmentBreakdown": [],
        "verbatimExtracts": [],
    }
    for chunk in wrap(full_text, COMPRESS_CHUNK_SIZE):
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
        prompt = f"""You are a professional tax analyst specializing in corporate tax strategy and SEC filing analysis. Analyze the following SEC filing excerpt and provide a comprehensive tax analysis report.

Please structure your analysis with the following sections:

## 1. Tax Savings Opportunities
Identify specific tax optimization strategies based on the filing data, including:
- Available tax credits and incentives not fully utilized
- Potential deductions that could be maximized
- Strategic timing opportunities for tax benefits

## 2. Underutilized Tax Credits
Analyze and highlight:
- Research & Development credits
- Foreign tax credits
- Alternative minimum tax credits
- Other applicable credits mentioned or implied in the filing

## 3. Effective Tax Rate (ETR) Reduction Strategies
Provide recommendations for:
- Geographic tax optimization
- Transfer pricing opportunities
- Corporate structure improvements
- Timing strategies for recognition

## 4. Peer Comparison and Benchmarking
Compare the company's tax position with:
- Industry average effective tax rates
- Similar-sized companies in the same sector
- Best practices observed in comparable filings

Please cite specific data points from the filing and provide actionable recommendations with estimated tax impact where possible.

SEC Filing Content:
{filing_text}

Provide a detailed, professional analysis with specific recommendations and quantified benefits where applicable.

Do NOT include any disclaimer or note about excerpted data or model limitations.
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
You are a professional tax analyst specializing in corporate tax strategy and SEC filing analysis.

Please structure your analysis with the following sections:

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

Please cite specific data points from the filing and provide actionable recommendations with estimated tax impact where possible.

Use ONLY the JSON data provided below. If a number is null, write "N/A (not disclosed)" and do not invent numbers.

Do NOT include any disclaimer or note about excerpted data or model limitations.
"""

def analyze_with_taxgpt_async(job_id: str, compressed_json: dict):
    save_job_status(job_id, {"status": "processing", "answer": "", "sources": [], "doc_url": None})
    try:
        # 1 create chat
        chat_id = requests.post(TAXGPT_CHAT_URL, headers=TAXGPT_HEADERS,
                                json={"type": "professional"}, timeout=30).json()["id"]

        # 2 send prompt
        full_prompt = TAXGPT_MAIN_PROMPT + "\n\nJSON DATA:\n" + json.dumps(compressed_json, indent=2)
        requests.post(TAXGPT_PROMPT_URL.format(chat_id=chat_id),
                      headers=TAXGPT_HEADERS, json={"prompt": full_prompt}, timeout=30)

        # 3 wait & poll
        time.sleep(45)
        hist = requests.get(TAXGPT_PROMPT_URL.format(chat_id=chat_id),
                            headers=TAXGPT_HEADERS, timeout=30).json()
        answer = hist[-1].get("prompt_answer", "") if hist else ""
        links = re.findall(r"https?://\\S+", answer)

        # 4 google doc
        title = generate_title(answer)
        doc_url = create_google_doc(answer, job_id, title)
        save_job_status(job_id, {"status": "done", "answer": answer,
                                 "sources": links, "doc_url": doc_url})
    except Exception as e:
        save_job_status(job_id, {"status": "error", "answer": str(e),
                                 "sources": [], "doc_url": None})
        logger.error(f"TaxGPT error for {job_id}: {e}")

def create_analysis_job(text: str):
    job_id = str(uuid.uuid4())
    # compress filing first
    compressed = compress_with_openai(text) if os.getenv("USE_TAXGPT", "false").lower() == "true" else None
    save_job_status(job_id, {"status": "processing", "answer": "In progress…",
                             "sources": [], "doc_url": None})
    if os.getenv("USE_TAXGPT", "false").lower() == "true":
        thread = threading.Thread(target=analyze_with_taxgpt_async,
                                  args=(job_id, compressed))
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
    
    # Check if we have results
    if job_id in analysis_results:
        result = analysis_results[job_id]
        logger.debug(f"Found result for job {job_id}: status={result['status']}")
        return result
    
    # Job not found
    logger.warning(f"Job {job_id} not found in results")
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