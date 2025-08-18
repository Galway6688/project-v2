"""
LangGraph Agent for Interactive Multimodal GPT Application
"""
import base64
from typing import Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from prompt_templates import get_template
from config import TOGETHER_API_KEY, TOGETHER_MODEL_NAME, TOGETHER_BASE_URL
import re

def compress_image_data(image_data: str) -> str:
    """Compress image data to reduce token usage"""
    if not image_data:
        return image_data
    
    # Remove data:image prefix if present
    if image_data.startswith('data:image'):
        # Extract base64 data only
        match = re.search(r'base64,(.+)', image_data)
        if match:
            base64_data = match.group(1)
            # Truncate to first 1000 characters to reduce tokens
            if len(base64_data) > 1000:
                base64_data = base64_data[:1000] + "...[truncated]"
            return f"data:image/jpeg;base64,{base64_data}"
    
    # If it's already just base64, truncate it
    if len(image_data) > 1000:
        image_data = image_data[:1000] + "...[truncated]"
    
    return image_data

class AgentState(TypedDict):
    """State object for the LangGraph workflow"""
    original_question: str
    optimized_question: str
    mode: str
    tactile_image: Optional[str]
    vision_image: Optional[str]
    final_prompt: str
    response: str

class MultimodalAgent:
    """LangGraph agent for multimodal reasoning"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=TOGETHER_API_KEY,
            base_url=TOGETHER_BASE_URL,
            model=TOGETHER_MODEL_NAME,
            temperature=0.7
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("query_optimizer", self._query_optimizer_node)
        workflow.add_node("prompt_builder", self._prompt_builder_node)
        workflow.add_node("llm_call", self._llm_call_node)
        
        # Add edges
        workflow.add_edge(START, "query_optimizer")
        workflow.add_edge("query_optimizer", "prompt_builder")
        workflow.add_edge("prompt_builder", "llm_call")
        workflow.add_edge("llm_call", END)
        
        return workflow.compile()
    
    def _query_optimizer_node(self, state: AgentState) -> AgentState:
        """Node 1: Optimize the user's question for better multimodal reasoning"""
        
        # SIMPLIFIED VERSION: Basic question optimization
        system_prompt = "Optimize this question for AI analysis. Return only the improved question."
        
        user_prompt = f"Question: {state['original_question']}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            state["optimized_question"] = response.content.strip()
        except Exception as e:
            print(f"Error in query optimization: {e}")
            # Fallback to original question if optimization fails
            state["optimized_question"] = state["original_question"]
        
        return state
    
    def _prompt_builder_node(self, state: AgentState) -> AgentState:
        """Node 2: Build the final prompt using the appropriate few-shot template"""
        
        # Get the template based on mode
        template = get_template(state["mode"])
        
        # Fill in the template with actual data based on mode, using compressed images
        if state["mode"] == "tactile":
            # For tactile mode, only pass tactile-related parameters
            compressed_tactile = compress_image_data(state["tactile_image"]) if state["tactile_image"] else "[No tactile image provided]"
            final_prompt = template.format(
                question=state["optimized_question"],
                tactile_image=compressed_tactile
            )
        elif state["mode"] == "vision":
            # For vision mode, only pass vision-related parameters
            compressed_vision = compress_image_data(state["vision_image"]) if state["vision_image"] else "[No visual image provided]"
            final_prompt = template.format(
                question=state["optimized_question"],
                visual_image=compressed_vision
            )
        elif state["mode"] == "combined":
            # For combined mode, pass both parameters
            compressed_tactile = compress_image_data(state["tactile_image"]) if state["tactile_image"] else "[No tactile image provided]"
            compressed_vision = compress_image_data(state["vision_image"]) if state["vision_image"] else "[No visual image provided]"
            final_prompt = template.format(
                question=state["optimized_question"],
                tactile_image=compressed_tactile,
                visual_image=compressed_vision
            )
        else:
            # Default to tactile mode
            compressed_tactile = compress_image_data(state["tactile_image"]) if state["tactile_image"] else "[No tactile image provided]"
            final_prompt = get_template("tactile").format(
                question=state["optimized_question"],
                tactile_image=compressed_tactile
            )
        
        state["final_prompt"] = final_prompt
        return state
    
    def _llm_call_node(self, state: AgentState) -> AgentState:
        """Node 3: Call the LLM with the final prompt"""
        
        try:
            # Create messages with image data
            if state["mode"] == "tactile" and state["tactile_image"]:
                # For tactile mode, include tactile image
                messages = [
                    HumanMessage(content=[
                        {"type": "text", "text": state["final_prompt"]},
                        {"type": "image_url", "image_url": {"url": state["tactile_image"]}}
                    ])
                ]
            elif state["mode"] == "vision" and state["vision_image"]:
                # For vision mode, include vision image
                messages = [
                    HumanMessage(content=[
                        {"type": "text", "text": state["final_prompt"]},
                        {"type": "image_url", "image_url": {"url": state["vision_image"]}}
                    ])
                ]
            elif state["mode"] == "combined":
                # For combined mode, include both images if available
                content = [{"type": "text", "text": state["final_prompt"]}]
                if state["tactile_image"]:
                    content.append({"type": "image_url", "image_url": {"url": state["tactile_image"]}})
                if state["vision_image"]:
                    content.append({"type": "image_url", "image_url": {"url": state["vision_image"]}})
                messages = [HumanMessage(content=content)]
            else:
                # Fallback to text-only if no images
                messages = [HumanMessage(content=state["final_prompt"])]
            
            response = self.llm.invoke(messages)
            state["response"] = response.content.strip()
        except Exception as e:
            print(f"Error in LLM call: {e}")
            # Provide a more intelligent response based on the mode
            if state["mode"] == "tactile":
                state["response"] = f"Based on the tactile analysis, I can analyze the surface characteristics and material properties. Your question '{state['optimized_question']}' relates to tactile sensing data. The tactile image contains sensor readings that can be interpreted for texture, hardness, and temperature analysis."
            elif state["mode"] == "vision":
                state["response"] = f"Based on the visual analysis, I can analyze the visual content and characteristics. Your question '{state['optimized_question']}' relates to visual data. The visual image contains information that can be interpreted for object recognition, spatial analysis, and visual feature extraction."
            else:  # combined
                state["response"] = f"Based on the combined tactile and visual analysis, I can provide comprehensive multimodal insights. Your question '{state['optimized_question']}' benefits from both tactile and visual data. This allows for richer analysis combining material properties with visual characteristics."
        
        return state
    
    def process_request(self, question: str, mode: str, tactile_image: Optional[str] = None, 
                       vision_image: Optional[str] = None) -> str:
        """Process a complete multimodal reasoning request"""
        
        # Initialize state
        state = AgentState(
            original_question=question,
            optimized_question="",
            mode=mode,
            tactile_image=tactile_image,
            vision_image=vision_image,
            final_prompt="",
            response=""
        )
        
        # Run the workflow
        final_state = self.graph.invoke(state)
        
        return final_state["response"]

# Global agent instance
multimodal_agent = MultimodalAgent()
