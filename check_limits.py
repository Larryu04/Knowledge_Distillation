import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Error] OPENAI_API_KEY is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("Sending a minimal 1-token request to check OpenAI rate limits...\n")
    try:
        response = client.chat.completions.with_raw_response.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1
        )
        
        headers = response.headers
        
        print("=== Rate Limit Status ===")
        print(f"Requests Limit (Per Day):  {headers.get('x-ratelimit-limit-requests', 'N/A')}")
        print(f"Requests Remaining:        {headers.get('x-ratelimit-remaining-requests', 'N/A')}")
        print(f"Requests Reset Time:       {headers.get('x-ratelimit-reset-requests', 'N/A')}")
        print("-" * 25)
        print(f"Tokens Limit (Per Min):    {headers.get('x-ratelimit-limit-tokens', 'N/A')}")
        print(f"Tokens Remaining:          {headers.get('x-ratelimit-remaining-tokens', 'N/A')}")
        print(f"Tokens Reset Time:         {headers.get('x-ratelimit-reset-tokens', 'N/A')}")
        print("=========================")
        
    except Exception as e:
        print(f"[API Error] {e}")
        if hasattr(e, 'response') and e.response is not None:
            headers = e.response.headers
            print("\nExtracted headers from the error response:")
            print(f"Requests Reset Time: {headers.get('x-ratelimit-reset-requests', 'N/A')}")
            print(f"Requests Remaining:  {headers.get('x-ratelimit-remaining-requests', 'N/A')}")

if __name__ == "__main__":
    main()