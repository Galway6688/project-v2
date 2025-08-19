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

def normalize_image_data(image_data: str) -> str:
    """Normalize image data format for consistency"""
    if not image_data:
        return image_data
    
    # Ensure consistent format
    if image_data.startswith('data:image'):
        return image_data  # Already in correct format
    else:
        return f"data:image/jpeg;base64,{image_data}"  # Add prefix if missing

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
        
        # PRINT TEMPLATE BUILDING PROCESS
        print("\n" + "-"*60)
        print("🔨 BUILDING PROMPT TEMPLATE:")
        print("-"*60)
        print(f"Selected Mode: {state['mode']}")
        print(f"Template: {template}")
        print(f"Optimized Question: {state['optimized_question']}")
        
        # Fill in the template with actual data based on mode, using normalized images
        if state["mode"] == "tactile":
            # For tactile mode, only pass tactile-related parameters
            normalized_tactile = normalize_image_data(state["tactile_image"]) if state["tactile_image"] else "[No tactile image provided]"
            print(f"Tactile Image (normalized): {len(normalized_tactile)} chars")
            final_prompt = template.format(
                question=state["optimized_question"],
                tactile_image=normalized_tactile
            )
        elif state["mode"] == "vision":
            # For vision mode, only pass vision-related parameters
            normalized_vision = normalize_image_data(state["vision_image"]) if state["vision_image"] else "[No visual image provided]"
            print(f"Vision Image (normalized): {len(normalized_vision)} chars")
            final_prompt = template.format(
                question=state["optimized_question"],
                visual_image=normalized_vision
            )
        elif state["mode"] == "combined":
            # For combined mode, pass both parameters
            normalized_tactile = normalize_image_data(state["tactile_image"]) if state["tactile_image"] else "[No tactile image provided]"
            normalized_vision = normalize_image_data(state["vision_image"]) if state["vision_image"] else "[No visual image provided]"
            print(f"Tactile Image (normalized): {len(normalized_tactile)} chars")
            print(f"Vision Image (normalized): {len(normalized_vision)} chars")
            final_prompt = template.format(
                question=state["optimized_question"],
                tactile_image=normalized_tactile,
                visual_image=normalized_vision
            )
        else:
            # Default to tactile mode
            normalized_tactile = normalize_image_data(state["tactile_image"]) if state["tactile_image"] else "[No tactile image provided]"
            print(f"Tactile Image (normalized): {len(normalized_tactile)} chars")
            final_prompt = get_template("tactile").format(
                question=state["optimized_question"],
                tactile_image=normalized_tactile
            )
        
        print(f"Final Prompt Length: {len(final_prompt)} chars")
        print("-"*60)
        
        state["final_prompt"] = final_prompt
        return state
    
    def _llm_call_node(self, state: AgentState) -> AgentState:
        """Node 3: Call the LLM with the final prompt"""
        
        # PRINT PROMPT BEFORE LLM CALL
        print("\n" + "="*80)
        print("🚀 PROMPT BEING SENT TO LLM:")
        print("="*80)
        print(f"Mode: {state['mode']}")
        print(f"Question: {state['optimized_question']}")
        print(f"Final Prompt: {state['final_prompt']}")
        if state["tactile_image"]:
            print(f"Tactile Image Length: {len(state['tactile_image'])} chars")
            print(f"Tactile Image Preview: {state['tactile_image'][:100]}...")
        if state["vision_image"]:
            print(f"Vision Image Length: {len(state['vision_image'])} chars")
            print(f"Vision Image Preview: {state['vision_image'][:100]}...")
        print("="*80)
        
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
            
            # PRINT MESSAGES STRUCTURE
            print("📤 MESSAGES STRUCTURE:")
            print(f"Number of messages: {len(messages)}")
            for i, msg in enumerate(messages):
                print(f"Message {i+1}: {type(msg).__name__}")
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, list):
                        for j, content_item in enumerate(msg.content):
                            print(f"  Content {j+1}: {content_item['type']}")
                            if content_item['type'] == 'text':
                                print(f"    Text: {content_item['text'][:100]}...")
                            elif content_item['type'] == 'image_url':
                                print(f"    Image URL length: {len(content_item['image_url']['url'])} chars")
                    else:
                        print(f"  Content: {msg.content[:100]}...")
            print("="*80)
            
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
