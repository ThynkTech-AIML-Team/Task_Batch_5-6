import os
from dotenv import load_dotenv

# Load .env from the same folder as this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# Read API key from environment
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

MIN_QUESTIONS = 5

if not API_KEY:
    raise ValueError("API key not found. Add GOOGLE_API_KEY=... in AI_BOT/.env")