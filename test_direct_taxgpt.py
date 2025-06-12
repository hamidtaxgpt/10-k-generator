#!/usr/bin/env python3
"""
Direct TaxGPT test using the exact working pattern to verify authentic responses.
This demonstrates that TaxGPT returns legitimate tax analysis when the API works correctly.
"""

import requests
import time
import re
import os

def test_direct_taxgpt():
    """Test TaxGPT using the exact working pattern that we know succeeds."""
    
    api_key = os.getenv('TAXGPT_API_KEY')
    if not api_key:
        print("ERROR: TAXGPT_API_KEY not found")
        return
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": "https://app.taxgpt.com",
        "Referer": "https://app.taxgpt.com/",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        # 1. Create chat (exact working pattern)
        print("Creating TaxGPT chat...")
        create_chat_resp = requests.post(
            "https://api.taxgpt.com/api/chats/",
            headers=headers,
            json={"type": "professional"},
            timeout=30
        )
        create_chat_resp.raise_for_status()
        chat_id = create_chat_resp.json()["id"]
        print(f"Chat created: {chat_id}")

        # 2. Send prompt (exact working pattern)
        question = "Analyze the following SEC filing excerpt for tax optimization opportunities: 'The Company recorded an effective tax rate of 28% for fiscal year 2024, with significant deferred tax assets related to research and development activities. International operations contributed 40% of total revenue with varying tax rates across jurisdictions.'"
        
        print("Sending prompt to TaxGPT...")
        prompt_resp = requests.post(
            f"https://api.taxgpt.com/api/chats/{chat_id}/prompts/",
            headers=headers,
            json={"prompt": question},
            timeout=30
        )
        prompt_resp.raise_for_status()
        print("Prompt sent successfully")

        # 3. Wait (exact working pattern)
        print("Waiting 45 seconds for TaxGPT processing...")
        time.sleep(45)

        # 4. Get response (exact working pattern)
        print("Retrieving TaxGPT response...")
        history_resp = requests.get(
            f"https://api.taxgpt.com/api/chats/{chat_id}/prompts/",
            headers=headers,
            timeout=30
        )
        history_resp.raise_for_status()
        history = history_resp.json()
        
        if history and len(history) > 0:
            latest = history[-1].get("prompt_answer", "")
            if latest and len(latest.strip()) > 50:
                sources = re.findall(r"https?://\S+", latest)
                
                print(f"\n=== AUTHENTIC TAXGPT RESPONSE ===")
                print(f"Response length: {len(latest)} characters")
                print(f"Sources found: {len(sources)}")
                print(f"First 500 characters:")
                print(latest[:500] + "..." if len(latest) > 500 else latest)
                print(f"\nSources: {sources}")
                print("=== END RESPONSE ===\n")
                
                return {
                    "success": True,
                    "answer": latest,
                    "sources": sources,
                    "chat_id": chat_id
                }
            else:
                print("TaxGPT response was empty or too short")
                return {"success": False, "error": "Empty response"}
        else:
            print("No response history found")
            return {"success": False, "error": "No history"}
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = test_direct_taxgpt()
    if result["success"]:
        print("SUCCESS: TaxGPT returned authentic tax analysis")
    else:
        print(f"FAILED: {result['error']}")