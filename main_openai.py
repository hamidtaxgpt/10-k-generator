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

def extract_text_from_url(url):
    """
    Extract clean text content from a URL using trafilatura.
    This provides better text extraction than manual HTML parsing.
    """
    try:
        logger.debug(f"Fetching content from URL: {url}")
        
        # Set proper headers for SEC.gov requests
        headers = {
            'User-Agent': 'hamid@taxgpt.com'
        }
        
        downloaded = trafilatura.fetch_url(url, headers=headers)
        if not downloaded:
            raise Exception("Failed to download content from URL")
        
        text = trafilatura.extract(downloaded)
        if not text:
            raise Exception("Failed to extract text from downloaded content")
        
        logger.debug(f"Successfully extracted {len(text)} characters of text")
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from URL {url}: {str(e)}")
        raise Exception(f"Text extraction failed: {str(e)}")

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

Provide a detailed, professional analysis with specific recommendations and quantified benefits where applicable."""

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

def create_analysis_job(text):
    """
    Create analysis job and start OpenAI processing, return job_id for async processing.
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        logger.debug(f"Creating analysis job with ID: {job_id}")
        
        # Start background analysis
        logger.debug(f"Starting background OpenAI analysis")
        analysis_thread = threading.Thread(
            target=analyze_with_openai_async,
            args=(job_id, text)
        )
        analysis_thread.daemon = True
        analysis_thread.start()
        logger.debug(f"Background thread started - returning job ID")
        
        return job_id
        
    except Exception as e:
        logger.error(f"Error creating analysis job: {str(e)}")
        raise Exception(f"Failed to create analysis job: {str(e)}")

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