# SEC Filing Analysis API with Google Docs Integration

A robust Flask-based API that analyzes SEC filings using OpenAI's o1-mini model and automatically creates formatted Google Docs reports with professional tax planning recommendations.

## Features

- **Asynchronous SEC Filing Analysis**: Processes large 10-K filings using OpenAI o1-mini
- **Robust Text Extraction**: Multi-method extraction with trafilatura and BeautifulSoup fallbacks
- **Google Docs Integration**: Automatically creates formatted tax reports in Google Docs
- **Persistent Job Tracking**: File-based job persistence survives server restarts
- **Professional Tax Analysis**: Senior corporate tax advisor prompts with quantified recommendations

## API Endpoints

### Start Analysis
```
POST /analyze/start
Content-Type: application/json

{
  "url": "https://www.sec.gov/Archives/edgar/data/[company]/[filing].htm"
}
```

**Response:**
```json
{
  "job_id": "uuid-string"
}
```

### Check Analysis Status
```
GET /analyze/status/<job_id>
```

**Response (Processing):**
```json
{
  "status": "processing",
  "answer": "Analysis in progress...",
  "sources": [],
  "doc_url": null
}
```

**Response (Complete):**
```json
{
  "status": "done",
  "answer": "Full tax analysis report text...",
  "sources": ["https://example.com/..."],
  "doc_url": "https://docs.google.com/document/d/[doc_id]/edit?usp=sharing"
}
```

## Google OAuth2 Setup

### Required Replit Secrets

Set these three secrets in your Replit environment:

1. **GOOGLE_CLIENT_ID**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create or select a project
   - Enable these APIs:
     - Google Docs API
     - Google Drive API (required for public sharing)
   - Create OAuth 2.0 credentials (Desktop application)
   - Copy the Client ID

2. **GOOGLE_CLIENT_SECRET**
   - Copy the Client Secret from the same OAuth 2.0 credentials

3. **GOOGLE_REFRESH_TOKEN**
   - Use Google OAuth Playground or run a one-time authorization flow
   - Get the refresh token for the scopes:
     - `https://www.googleapis.com/auth/documents`
     - `https://www.googleapis.com/auth/drive.file`

### Setting Secrets in Replit

1. Open your Replit project
2. Click the "Secrets" tab (lock icon) in the left sidebar
3. Add each secret:
   ```
   Key: GOOGLE_CLIENT_ID
   Value: your-client-id.apps.googleusercontent.com
   
   Key: GOOGLE_CLIENT_SECRET
   Value: your-client-secret
   
   Key: GOOGLE_REFRESH_TOKEN
   Value: your-refresh-token
   ```

## Usage Examples

### Starting an Analysis

```bash
curl -X POST https://your-replit-app.replit.app/analyze/start \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.sec.gov/Archives/edgar/data/861459/000086145925000007/gva-20241231.htm"}'
```

**Response:**
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```

### Checking Status

```bash
curl https://your-replit-app.replit.app/analyze/status/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

### Opening the Google Doc

Once analysis is complete and `doc_url` is provided:

1. **Copy the URL** from the `doc_url` field in the API response
2. **Open in browser** - paste the URL into your browser
3. **View the report** - the document will open with full formatting:
   - Professional headings and structure
   - Bullet point recommendations
   - Bold text for emphasis
   - Peer comparison tables
   - Executive action items

Example `doc_url`:
```
https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlWkMg4n-mw6kfHA/edit?usp=sharing
```

## Report Features

The generated Google Docs contain:

- **Executive Summary** with company context
- **Tax Savings Opportunities** with quantified impact estimates
- **Underutilized Tax Credits** (R&D, Energy, WOTC, etc.)
- **ETR Reduction Strategies** with percentage impact projections
- **Peer Benchmarking Tables** comparing tax positions
- **High-Priority Action Items** with timelines and ownership

## Architecture

- **Flask API** with async job processing
- **OpenAI o1-mini** for complex tax analysis reasoning
- **Persistent Storage** using file-based job tracking
- **Google APIs** for document creation and sharing
- **Multi-method Text Extraction** for robust SEC filing processing

## Error Handling

The system handles:
- Complex SEC HTML structures
- Google API authentication refresh
- OpenAI rate limiting and timeouts
- Server restarts with job persistence
- Network connectivity issues

## Development

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_CLIENT_ID="your-client-id"
export GOOGLE_CLIENT_SECRET="your-client-secret"
export GOOGLE_REFRESH_TOKEN="your-refresh-token"

# Run the server
python main.py
```

### Production Deployment

The application is optimized for Replit deployment with:
- Gunicorn WSGI server
- Automatic worker reloading
- Environment-based configuration
- Persistent job storage in `/tmp`

## Dependencies

- Flask
- OpenAI Python SDK
- Google API Client Libraries
- Trafilatura (text extraction)
- BeautifulSoup4 (HTML parsing)
- Requests (HTTP client)

## Security

- OAuth2 with refresh token management
- Environment variable configuration
- Public document sharing with view-only permissions
- No persistent storage of sensitive credentials