import os
from dotenv import load_dotenv

load_dotenv()

# Together AI Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "0440017c7e550247f094a703e2f4b5cf804bfaa44c3c6b2c74c725c98de61fe4")
TOGETHER_MODEL_NAME = os.getenv("TOGETHER_MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
TOGETHER_BASE_URL = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")

# 确保Together AI的配置正确
print(f"Using Together AI - Model: {TOGETHER_MODEL_NAME}")
print(f"Base URL: {TOGETHER_BASE_URL}")
print(f"API Key: {TOGETHER_API_KEY[:10]}...")

# FastAPI Configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
