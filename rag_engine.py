import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize Google GenAI client if API key is present
API_KEY = os.getenv("GEMINI_API_KEY")
client = None

if API_KEY:
    try:
        from google import genai
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(f"Warning: Could not initialize Google GenAI client: {e}")

def generate_rag_response(prompt: str, context: str) -> str:
    """
    Generates an evidence-grounded response using Google Gemini.
    Falls back to returning raw context if the API is unavailable or unconfigured.
    """
    full_prompt = (
        "You are an expert SOC (Security Operations Center) Copilot.\n"
        "Analyze the security alert using ONLY the provided context and retrieved MITRE ATT&CK techniques.\n\n"
        f"--- CONTEXT & RETRIEVED EVIDENCE ---\n{context}\n\n"
        f"--- ANALYST PROMPT ---\n{prompt}"
    )

    # Try generating response using Gemini Cloud API
    if client:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"Gemini API generation error: {e}")

    # Fallback mode if API key is missing or call fails
    fallback_output = (
        "**[System Notice: Cloud LLM API Inactive / Offline]**\n\n"
        "Displaying direct hybrid retrieval and grounding context:\n\n"
        f"{context}"
    )
    return fallback_output