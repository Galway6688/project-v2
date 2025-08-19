"""
FastAPI Backend for Interactive Multimodal GPT Application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from config import CORS_ORIGINS
from agent import multimodal_agent

# Initialize FastAPI app
app = FastAPI(
    title="Interactive Multimodal GPT API",
    description="Backend API for multimodal AI reasoning with LangGraph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Request/Response Models
class ReasoningRequest(BaseModel):
    question: str
    mode: str  # 'tactile', 'vision', or 'combined'
    tactile_image: Optional[str] = None  # Base64 encoded image
    vision_image: Optional[str] = None   # Base64 encoded image

class ReasoningResponse(BaseModel):
    response: str
    mode: str
    optimized_question: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Interactive Multimodal GPT API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Main reasoning endpoint
@app.post("/api/reasoning", response_model=ReasoningResponse)
async def generate_reasoning(request: ReasoningRequest):
    """
    Main endpoint for multimodal reasoning
    """
    try:
        # Validate inputs
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if request.mode not in ["tactile", "vision", "combined"]:
            raise HTTPException(status_code=400, detail="Mode must be 'tactile', 'vision', or 'combined'")
        
        # Validate that required images are provided for each mode
        if request.mode == "tactile" and not request.tactile_image:
            raise HTTPException(status_code=400, detail="Tactile image is required for tactile mode")
        
        if request.mode == "vision" and not request.vision_image:
            raise HTTPException(status_code=400, detail="Visual image is required for vision mode")
        
        if request.mode == "combined":
            if not request.tactile_image and not request.vision_image:
                raise HTTPException(status_code=400, detail="At least one image is required for combined mode")
        
        # Log the request for debugging
        print(f"Processing request - Mode: {request.mode}, Question: {request.question[:50]}...")
        print(f"Tactile image: {'Present' if request.tactile_image else 'None'}")
        print(f"Vision image: {'Present' if request.vision_image else 'None'}")
        
        # Process the request through the LangGraph agent
        result = multimodal_agent.process_request(
            question=request.question,
            mode=request.mode,
            tactile_image=request.tactile_image,
            vision_image=request.vision_image
        )
        
        return ReasoningResponse(
            response=result["response"],
            mode=request.mode,
            optimized_question=result["optimized_question"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in reasoning endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Additional utility endpoints
@app.get("/api/modes")
async def get_available_modes():
    """Get available reasoning modes"""
    return {
        "modes": [
            {
                "value": "tactile",
                "label": "Tactile-Text",
                "description": "Analyze tactile data with text",
                "requires": ["tactile_image", "question"]
            },
            {
                "value": "vision", 
                "label": "Vision-Text",
                "description": "Analyze visual data with text",
                "requires": ["vision_image", "question"]
            },
            {
                "value": "combined",
                "label": "Tactile-Vision-Text Combined", 
                "description": "Combined multimodal analysis",
                "requires": ["question"],
                "optional": ["tactile_image", "vision_image"]
            }
        ]
    }

@app.get("/api/model-info")
async def get_model_info():
    """Get information about the AI model being used"""
    from config import TOGETHER_MODEL_NAME, TOGETHER_BASE_URL
    return {
        "model_name": TOGETHER_MODEL_NAME,
        "base_url": TOGETHER_BASE_URL,
        "provider": "Together AI",
        "capabilities": ["vision", "text", "multimodal"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
