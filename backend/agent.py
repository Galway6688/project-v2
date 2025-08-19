"""
LangGraph Agent for Interactive Multimodal GPT Application
"""
import base64
import os
from typing import Dict, Any, Optional, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
# Import the new template variables
from prompt_templates import (
    FEW_SHOT_EXAMPLE_TACTILE_TEXT, TASK_PROMPT_TACTILE,
    FEW_SHOT_EXAMPLE_VISION_TEXT, TASK_PROMPT_VISION,
    FEW_SHOT_EXAMPLE_COMBINED_TEXT, TASK_PROMPT_COMBINED
)
from config import TOGETHER_API_KEY, TOGETHER_MODEL_NAME, TOGETHER_BASE_URL
import re

# --- NEW HELPER FUNCTION ---
def image_to_base64(image_path: str) -> Optional[str]:
    """Converts an image file to a Base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Example image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def normalize_image_data(image_data: str) -> str:
    """Normalize image data format for consistency"""
    if not image_data:
        return image_data
    
    # Ensure consistent format
    if image_data.startswith('data:image'):
        return image_data  # Already in correct format
    else:
        return f"data:image/jpeg;base64,{image_data}"  # Add prefix if missing

def truncate_image_in_text(text: str, max_image_chars: int = 100) -> str:
    """Safely truncate base64 image data in text for printing"""
    # Pattern to match data:image URLs
    pattern = r'data:image/[^;]*;base64,[A-Za-z0-9+/=]+'
    
    def replace_image(match):
        image_data = match.group(0)
        if len(image_data) > max_image_chars:
            prefix = image_data[:50]  # Keep the data:image prefix
            return f"{prefix}...[truncated {len(image_data)} chars total]"
        return image_data
    
    return re.sub(pattern, replace_image, text)

# --- UPDATED CLASS ---
class AgentState(TypedDict):
    """State object for the LangGraph workflow."""
    original_question: str
    optimized_question: str
    mode: str
    tactile_image: Optional[str]
    vision_image: Optional[str]
    messages: List[Any]  # List of LangChain Message objects
    response: str

class MultimodalAgent:
    """LangGraph agent for multimodal reasoning."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=TOGETHER_API_KEY,
            base_url=TOGETHER_BASE_URL,
            model=TOGETHER_MODEL_NAME,
            temperature=0.7
        )
        # --- NEW DICTIONARY FOR EXAMPLE ASSETS ---
        # Assumes an 'assets' folder exists in the same directory as this script.
        self.example_assets = {
            "tactile_sandpaper": os.path.join("assets", "example_sandpaper_tactile.jpg"),
            "vision_marble": os.path.join("assets", "example_marble_vision.jpg"),
            "vision_corduroy": os.path.join("assets", "example_corduroy_vision.jpg"),
            "tactile_corduroy": os.path.join("assets", "example_corduroy_tactile.jpg"),
        }
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the two-stage conditional routing workflow"""
        workflow = StateGraph(AgentState)
        
        # ================================
        # ADD ALL NODES
        # ================================
        
        # Stage 1: Query Optimizers
        workflow.add_node("tactile_optimizer", self._tactile_query_optimizer)
        workflow.add_node("vision_optimizer", self._vision_query_optimizer)
        workflow.add_node("combined_optimizer", self._combined_query_optimizer)
        
        # Stage 2: Prompt Builders
        workflow.add_node("tactile_prompt_builder", self._tactile_prompt_builder)
        workflow.add_node("vision_prompt_builder", self._vision_prompt_builder)
        workflow.add_node("combined_prompt_builder", self._combined_prompt_builder)
        
        # Final LLM Call
        workflow.add_node("llm_call", self._llm_call_node)
        
        # ================================
        # WORKFLOW ROUTING STRUCTURE
        # ================================
        
        # Entry point: Start -> First conditional routing (Query Optimization)
        workflow.add_conditional_edges(
            START,
            self._query_router,
            {
                "tactile_optimizer": "tactile_optimizer",
                "vision_optimizer": "vision_optimizer", 
                "combined_optimizer": "combined_optimizer"
            }
        )
        
        # First convergence: All optimizers -> Second conditional routing (Prompt Building)
        workflow.add_conditional_edges(
            "tactile_optimizer",
            self._prompt_router,
            {
                "tactile_prompt_builder": "tactile_prompt_builder",
                "vision_prompt_builder": "vision_prompt_builder",
                "combined_prompt_builder": "combined_prompt_builder"
            }
        )
        workflow.add_conditional_edges(
            "vision_optimizer",
            self._prompt_router,
            {
                "tactile_prompt_builder": "tactile_prompt_builder",
                "vision_prompt_builder": "vision_prompt_builder",
                "combined_prompt_builder": "combined_prompt_builder"
            }
        )
        workflow.add_conditional_edges(
            "combined_optimizer",
            self._prompt_router,
            {
                "tactile_prompt_builder": "tactile_prompt_builder",
                "vision_prompt_builder": "vision_prompt_builder",
                "combined_prompt_builder": "combined_prompt_builder"
            }
        )
        
        # Second convergence: All builders -> LLM Call
        workflow.add_edge("tactile_prompt_builder", "llm_call")
        workflow.add_edge("vision_prompt_builder", "llm_call")
        workflow.add_edge("combined_prompt_builder", "llm_call")
        
        # Exit point: LLM Call -> End
        workflow.add_edge("llm_call", END)
        
        return workflow.compile()
    
    # ================================
    # QUERY OPTIMIZATION NODES (Stage 1)
    # ================================
    
    def _query_router(self, state: AgentState) -> str:
        """Route to appropriate query optimizer based on mode"""
        mode_to_optimizer = {
            "tactile": "tactile_optimizer",
            "vision": "vision_optimizer", 
            "combined": "combined_optimizer"
        }
        return mode_to_optimizer.get(state["mode"], "tactile_optimizer")
    
    def _tactile_query_optimizer(self, state: AgentState) -> AgentState:
        """Optimize questions for tactile analysis"""
        system_prompt = "Act as a haptics specialist. Refine the user's question to probe tactile properties (texture, hardness, friction, thermal) of a surface from the SSVTP database. Return ONLY the refined question."
        
        user_prompt = f"Question: {state['original_question']}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            state["optimized_question"] = response.content.strip()
        except Exception as e:
            print(f"Error in tactile query optimization: {e}")
            state["optimized_question"] = state["original_question"]
        
        return state
    
    def _vision_query_optimizer(self, state: AgentState) -> AgentState:
        """Optimize questions for vision analysis"""
        system_prompt = "Act as a computer vision expert. Refine the user's question to analyze visual characteristics (color, pattern, sheen, inferred texture) of a surface from the SSVTP database. Return ONLY the refined question."
        
        user_prompt = f"Question: {state['original_question']}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            state["optimized_question"] = response.content.strip()
        except Exception as e:
            print(f"Error in vision query optimization: {e}")
            state["optimized_question"] = state["original_question"]
        
        return state
    
    def _combined_query_optimizer(self, state: AgentState) -> AgentState:
        """Optimize questions for combined multimodal analysis"""
        system_prompt = "Act as a multimodal reasoning expert. Refine the user's question to explore the synergy between visual and tactile data of a surface from the SSVTP database, connecting appearance with physical sensations. Return ONLY the refined question."
        
        user_prompt = f"Question: {state['original_question']}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            state["optimized_question"] = response.content.strip()
        except Exception as e:
            print(f"Error in combined query optimization: {e}")
            state["optimized_question"] = state["original_question"]
        
        return state
    
    # ================================
    # PROMPT BUILDING NODES (Stage 2) - COMPLETELY REFACTORED
    # ================================
    
    def _prompt_router(self, state: AgentState) -> str:
        """Route to appropriate prompt builder based on mode"""
        mode_to_builder = {
            "tactile": "tactile_prompt_builder",
            "vision": "vision_prompt_builder",
            "combined": "combined_prompt_builder"
        }
        return mode_to_builder.get(state["mode"], "tactile_prompt_builder")
    
    def _tactile_prompt_builder(self, state: AgentState) -> AgentState:
        """Builds the tactile analysis message with separated messages for each image."""
        print("🔨 BUILDING TACTILE MESSAGE WITH FEW-SHOT IMAGE:")
        example_image_b64 = image_to_base64(self.example_assets["tactile_sandpaper"])
        user_image_b64 = state["tactile_image"]
        task_prompt = TASK_PROMPT_TACTILE.format(question=state["optimized_question"])

        messages_list = []
        
        # Message 1: Example User Turn (Question + Image)
        example_content = [{"type": "text", "text": "Based on the tactile data, what is this material and what are its key properties?"}]
        if example_image_b64:
            example_content.append({"type": "image_url", "image_url": {"url": example_image_b64}})
        messages_list.append(HumanMessage(content=example_content))
        
        # Message 2: Example AI Turn (Answer)
        messages_list.append(SystemMessage(content=FEW_SHOT_EXAMPLE_TACTILE_TEXT))

        # Message 3: Real User Turn (Instructions + Image)
        task_content = [{"type": "text", "text": task_prompt}]
        if user_image_b64:
            task_content.append({"type": "image_url", "image_url": {"url": user_image_b64}})
        messages_list.append(HumanMessage(content=task_content))

        state["messages"] = messages_list
        print(f"Built {len(messages_list)} separate messages")
        return state

    def _vision_prompt_builder(self, state: AgentState) -> AgentState:
        """Builds the vision analysis message with separated messages for each image."""
        print("🔨 BUILDING VISION MESSAGE WITH FEW-SHOT IMAGE:")
        example_image_b64 = image_to_base64(self.example_assets["vision_marble"])
        user_image_b64 = state["vision_image"]
        task_prompt = TASK_PROMPT_VISION.format(question=state["optimized_question"])

        messages_list = []
        
        # Message 1: Example User Turn (Question + Image)
        example_content = [{"type": "text", "text": "From this image, what would the surface feel like?"}]
        if example_image_b64:
            example_content.append({"type": "image_url", "image_url": {"url": example_image_b64}})
        messages_list.append(HumanMessage(content=example_content))
        
        # Message 2: Example AI Turn (Answer)
        messages_list.append(SystemMessage(content=FEW_SHOT_EXAMPLE_VISION_TEXT))

        # Message 3: Real User Turn (Instructions + Image)
        task_content = [{"type": "text", "text": task_prompt}]
        if user_image_b64:
            task_content.append({"type": "image_url", "image_url": {"url": user_image_b64}})
        messages_list.append(HumanMessage(content=task_content))

        state["messages"] = messages_list
        print(f"Built {len(messages_list)} separate messages")
        return state

    def _combined_prompt_builder(self, state: AgentState) -> AgentState:
        """Builds the combined analysis message with separated messages for multiple images."""
        print("🔨 BUILDING COMBINED MESSAGE WITH FEW-SHOT IMAGES:")
        example_vision_b64 = image_to_base64(self.example_assets["vision_corduroy"])
        example_tactile_b64 = image_to_base64(self.example_assets["tactile_corduroy"])
        user_vision_b64 = state["vision_image"]
        user_tactile_b64 = state["tactile_image"]
        task_prompt = TASK_PROMPT_COMBINED.format(question=state["optimized_question"])
        
        messages_list = []
        
        # Message 1: Example User Turn with Vision Image
        if example_vision_b64:
            example_vision_content = [
                {"type": "text", "text": "Identify this material and describe its characteristics. Here is the visual image:"},
                {"type": "image_url", "image_url": {"url": example_vision_b64}}
            ]
            messages_list.append(HumanMessage(content=example_vision_content))
        
        # Message 2: Example User Turn with Tactile Image
        if example_tactile_b64:
            example_tactile_content = [
                {"type": "text", "text": "And here is the tactile data for the same material:"},
                {"type": "image_url", "image_url": {"url": example_tactile_b64}}
            ]
            messages_list.append(HumanMessage(content=example_tactile_content))
        
        # Message 3: Example AI Turn (Answer)
        messages_list.append(SystemMessage(content=FEW_SHOT_EXAMPLE_COMBINED_TEXT))

        # Message 4: Real User Turn with Task Instructions
        messages_list.append(HumanMessage(content=[{"type": "text", "text": task_prompt}]))
        
        # Message 5: User Vision Image (if available)
        if user_vision_b64:
            user_vision_content = [
                {"type": "text", "text": "Here is the visual image to analyze:"},
                {"type": "image_url", "image_url": {"url": user_vision_b64}}
            ]
            messages_list.append(HumanMessage(content=user_vision_content))
        
        # Message 6: User Tactile Image (if available)
        if user_tactile_b64:
            user_tactile_content = [
                {"type": "text", "text": "And here is the tactile data:"},
                {"type": "image_url", "image_url": {"url": user_tactile_b64}}
            ]
            messages_list.append(HumanMessage(content=user_tactile_content))
            
        state["messages"] = messages_list
        print(f"Built {len(messages_list)} separate messages")
        return state
    
    # --- SIMPLIFIED LLM CALL NODE ---
    def _llm_call_node(self, state: AgentState) -> AgentState:
        """Node 3: Invokes the LLM with the fully constructed message list."""
        print("\n🚀 SENDING FULLY CONSTRUCTED MESSAGE TO LLM:")
        
        messages_list = state.get("messages", [])
        if not messages_list:
            print("Error: Message list is empty.")
            state["response"] = "Error: Message list was not built correctly."
            return state

        # Use the message list directly (already contains LangChain Message objects)
        print(f"Total messages to send: {len(messages_list)}")

        try:
            response = self.llm.invoke(messages_list)
            state["response"] = response.content.strip()
            print("✅ LLM call successful")
        except Exception as e:
            print(f"❌ Error in LLM call: {e}")
            state["response"] = "An error occurred while communicating with the AI model."
        
        return state
    
    # --- UPDATED PROCESS REQUEST METHOD ---
    def process_request(self, question: str, mode: str, tactile_image: Optional[str] = None, 
                       vision_image: Optional[str] = None) -> str:
        """Processes a complete multimodal reasoning request."""
        state = AgentState(
            original_question=question,
            optimized_question="",
            mode=mode,
            tactile_image=tactile_image,
            vision_image=vision_image,
            messages=[],  # Initialize with empty list
            response=""
        )
        final_state = self.graph.invoke(state)
        return final_state["response"]

# Global agent instance
multimodal_agent = MultimodalAgent()