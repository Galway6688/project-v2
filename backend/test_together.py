#!/usr/bin/env python3
"""
Test script to verify Together AI configuration
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Together AI Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "0440017c7e550247f094a703e2f4b5cf804bfaa44c3c6b2c74c725c98de61fe4")
TOGETHER_MODEL_NAME = os.getenv("TOGETHER_MODEL_NAME", "meta-llama/Llama-Vision-Free")
TOGETHER_BASE_URL = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")

print(f"Testing Together AI Configuration:")
print(f"Model: {TOGETHER_MODEL_NAME}")
print(f"Base URL: {TOGETHER_BASE_URL}")
print(f"API Key: {TOGETHER_API_KEY[:10]}...")

try:
    # Initialize the LLM
    llm = ChatOpenAI(
        api_key=TOGETHER_API_KEY,
        base_url=TOGETHER_BASE_URL,
        model=TOGETHER_MODEL_NAME,
        temperature=0.7
    )
    
    print("\n✅ LLM initialized successfully")
    
    # Test a simple message
    message = HumanMessage(content="Hello! Please respond with 'Test successful' if you can see this message.")
    
    print("\n🔄 Testing LLM call...")
    response = llm.invoke([message])
    
    print(f"✅ LLM response: {response.content}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

