import os
from dotenv import load_dotenv

load_dotenv()

# Together AI Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", " ")  #add your own api key
TOGETHER_MODEL_NAME = os.getenv("TOGETHER_MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
TOGETHER_BASE_URL = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")

# Ensure Together AI configuration is correct
print(f"Using Together AI - Model: {TOGETHER_MODEL_NAME}")
print(f"Base URL: {TOGETHER_BASE_URL}")
print(f"API Key: {TOGETHER_API_KEY[:10]}...")

# FastAPI Configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
