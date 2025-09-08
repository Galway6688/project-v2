"""
LangGraph Agent for Interactive Multimodal GPT Application
"""
import base64
import os
from typing import Dict, Any, Optional, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
# Import the template variables
from prompt_templates import (
    TASK_PROMPT_TACTILE,
    TASK_PROMPT_VISION,
    TASK_PROMPT_COMBINED
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
    
    def __init__(self, num_shots: int = 4, base_image_path: str = "."):
        self.llm = ChatOpenAI(
            api_key=TOGETHER_API_KEY,
            base_url=TOGETHER_BASE_URL,
            model=TOGETHER_MODEL_NAME,
            temperature=0.7
        )
        self.num_shots = num_shots
        self.graph = self._build_graph()

        if self.num_shots > 0:
            # --- HARDCODED 5-SHOT EXAMPLES ---
            # Based on the user-provided file list and train.csv
            # Assumes 'images_rgb' and 'images_tac' folders are in the same directory as the script.
            self.few_shot_examples = [
                {
                    "vision_path": "images_rgb/image_2101_rgb.jpg",
                    "tactile_path": "images_tac/image_2101_tac.jpg",
                    "caption": "lined, woven, patterned"
                },
                {
                    "vision_path": "images_rgb/image_143_rgb.jpg",
                    "tactile_path": "images_tac/image_143_tac.jpg",
                    "caption": "hard, splotchy, glossy"
                },
                {
                    "vision_path": "images_rgb/image_856_rgb.jpg",
                    "tactile_path": "images_tac/image_856_tac.jpg",
                    "caption": "absorbent, fluffy"
                },
                {
                    "vision_path": "images_rgb/image_871_rgb.jpg",
                    "tactile_path": "images_tac/image_871_tac.jpg",
                    "caption": "metallic, rigid"
                },
                {
                    "vision_path": "images_rgb/image_2492_rgb.jpg",
                    "tactile_path": "images_tac/image_2492_tac.jpg",
                    "caption": "mesh, flexible, coarse"
                }
            ]
    
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
        system_prompt = """**Role:** You are a Query Standardization Bot.
        **Task:** Your job is to convert any user question into a standard format for analyzing a tactile sensor data visualization. The user's original intent does not matter.

        **Output Format:**
        - You must always respond with one of the following canonical questions, choosing the one that seems most appropriate, or defaulting to the first one.
        - Example Formats: "This tactile data visualization suggests what attributes?", "This tactile data implies a surface quality of?", "What tactile properties does this sensor data represent?"
        - Return ONLY the chosen standardized question, with no preamble.
        """

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
        system_prompt = """**Role:** You are a Query Standardization Bot.
        **Task:** Your job is to convert any user question into a standard format for analyzing a visual photograph to infer tactile qualities. The user's original intent does not matter.

        **Output Format:**
        - You must always respond with one of the following canonical questions, choosing the one that seems most appropriate, or defaulting to the first one.
        - Example Formats: "This photograph conveys a touchable quality of?", "This image hints at what tactile textures?", "What tactile feelings does this picture resonate with?"
        - Return ONLY the chosen standardized question, with no preamble.
        """

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
        system_prompt = """**Role:** You are a Query Standardization Bot.
        **Task:** Your job is to convert any user question into a standard format for a multimodal analysis of both a photograph and its corresponding tactile sensor data. The user's original intent does not matter.

        **Output Format:**
        - You must always respond with one of the following canonical questions, choosing the one that seems most appropriate, or defaulting to the first one.
        - Example Formats: "This visual work and its tactile data hint at what textures?", "What tactile attributes do this image and its sensor data convey?", "This depiction and its haptic feedback resonate with what tactile feelings?"
        - Return ONLY the chosen standardized question, with no preamble.
        """

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
        """Builds the tactile analysis message with CONSISTENT few-shot examples."""
        print(f"🔨 BUILDING TACTILE MESSAGE WITH {self.num_shots}-SHOT EXAMPLES (Corrected Logic):")
        
        messages_list = []

        # 1. Loop through all fixed few-shot examples (only if num_shots > 0)
        if self.num_shots > 0:
            examples_to_use = self.few_shot_examples[:self.num_shots]
            for example in examples_to_use:  # <-- Changed to iterate over sliced list
                # For each example, use the standard, strengthened TASK_PROMPT as the instruction
                # The 'question' field can be a generic placeholder since it's just an example
                example_prompt_text = TASK_PROMPT_TACTILE.format(question="Analyze the following example.")
                
                example_image_b64 = image_to_base64(example["tactile_path"])
                if not example_image_b64:
                    print(f"Warning: Skipping missing example image at {example['tactile_path']}")
                    continue

                # Example User Turn (Instruction + Image)
                example_content = [{"type": "text", "text": example_prompt_text}]
                if example_image_b64:
                    example_content.append({"type": "image_url", "image_url": {"url": example_image_b64}})
                messages_list.append(HumanMessage(content=example_content))

                # Example AI Turn (The correct, perfectly formatted answer)
                messages_list.append(SystemMessage(content=example["caption"]))
        
        # 2. Add the actual user query at the end
        user_image_b64 = state["tactile_image"]
        # The final task prompt uses the optimized question from the previous step
        task_prompt = TASK_PROMPT_TACTILE.format(question=state["optimized_question"])

        task_content = [{"type": "text", "text": task_prompt}]
        if user_image_b64:
            task_content.append({"type": "image_url", "image_url": {"url": user_image_b64}})
        messages_list.append(HumanMessage(content=task_content))

        state["messages"] = messages_list
        print(f"Built {len(messages_list)} separate messages for the final prompt.")
        return state

    def _vision_prompt_builder(self, state: AgentState) -> AgentState:
        """Builds the vision analysis message with CONSISTENT few-shot examples."""
        print(f"🔨 BUILDING VISION MESSAGE WITH {self.num_shots}-SHOT EXAMPLES (Corrected Logic):")
        
        messages_list = []

        # 1. Loop through all fixed few-shot examples (only if num_shots > 0)
        if self.num_shots > 0:
            examples_to_use = self.few_shot_examples[:self.num_shots]
            for example in examples_to_use:  # <-- Changed to iterate over sliced list
                # For each example, use the standard, strengthened TASK_PROMPT as the instruction
                # The 'question' field can be a generic placeholder since it's just an example
                example_prompt_text = TASK_PROMPT_VISION.format(question="Analyze the following example.")
                
                example_image_b64 = image_to_base64(example["vision_path"])
                if not example_image_b64:
                    print(f"Warning: Skipping missing example image at {example['vision_path']}")
                    continue

                # Example User Turn (Instruction + Image)
                example_content = [{"type": "text", "text": example_prompt_text}]
                if example_image_b64:
                    example_content.append({"type": "image_url", "image_url": {"url": example_image_b64}})
                messages_list.append(HumanMessage(content=example_content))

                # Example AI Turn (The correct, perfectly formatted answer)
                messages_list.append(SystemMessage(content=example["caption"]))
        
        # 2. Add the actual user query at the end
        user_image_b64 = state["vision_image"]
        # The final task prompt uses the optimized question from the previous step
        task_prompt = TASK_PROMPT_VISION.format(question=state["optimized_question"])

        task_content = [{"type": "text", "text": task_prompt}]
        if user_image_b64:
            task_content.append({"type": "image_url", "image_url": {"url": user_image_b64}})
        messages_list.append(HumanMessage(content=task_content))

        state["messages"] = messages_list
        print(f"Built {len(messages_list)} separate messages for the final prompt.")
        return state

    # Please use this corrected version of the function to replace the existing _combined_prompt_builder in your agent.py

    def _combined_prompt_builder(self, state: AgentState) -> AgentState:
        """
        Builds the combined analysis message.
        CORRECTED to send each image in a separate message to comply with API limits.
        """
        print("🔨 BUILDING COMBINED MESSAGE WITH 5-SHOT EXAMPLES (API-Compliant Logic):")

        messages_list = []

        # 1. Loop through all fixed few-shot examples
        if self.num_shots > 0:
            examples_to_use = self.few_shot_examples[:self.num_shots]
            for example in examples_to_use:  # <-- Changed to iterate over sliced list
                example_prompt_text = TASK_PROMPT_COMBINED.format(question="Analyze the following example.")

                example_vision_b64 = image_to_base64(example["vision_path"])
                example_tactile_b64 = image_to_base64(example["tactile_path"])

                # --- This is the core change: send images separately ---

                # Message 1: Instructions and first image (vision)
                content_part1 = [
                    {"type": "text", "text": example_prompt_text},
                    {"type": "image_url", "image_url": {"url": example_vision_b64}}
                ]
                messages_list.append(HumanMessage(content=content_part1))

                # Message 2: Prompt and second image (tactile)
                content_part2 = [
                    {"type": "text", "text": "Here is the corresponding tactile data:"},
                    {"type": "image_url", "image_url": {"url": example_tactile_b64}}
                ]
                messages_list.append(HumanMessage(content=content_part2))

                # Example AI Turn (Correct, perfectly formatted answer)
                messages_list.append(SystemMessage(content=example["caption"]))

        # 2. Add the actual user query at the end
        user_vision_b64 = state["vision_image"]
        user_tactile_b64 = state["tactile_image"]
        task_prompt = TASK_PROMPT_COMBINED.format(question=state["optimized_question"])

        # --- Similarly, send user's two images separately ---

        # Message with instruction and user's vision image
        task_content_vision = [
            {"type": "text", "text": task_prompt},
            {"type": "image_url", "image_url": {"url": user_vision_b64}}
        ]
        messages_list.append(HumanMessage(content=task_content_vision))

        # Message with user's tactile image
        task_content_tactile = [
            {"type": "text", "text": "And here is the corresponding tactile data:"},
            {"type": "image_url", "image_url": {"url": user_tactile_b64}}
        ]
        messages_list.append(HumanMessage(content=task_content_tactile))

        state["messages"] = messages_list
        print(f"Built {len(messages_list)} separate messages for the final prompt.")
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
        
        # Print detailed information for each message
        print("\n📋 MESSAGE DETAILS:")
        for i, message in enumerate(messages_list, 1):
            message_type = type(message).__name__
            print(f"--- Message {i} ({message_type}) ---")
            
            # Handle different content formats
            if hasattr(message, 'content'):
                content = message.content
                
                # If content is a list (multimodal content)
                if isinstance(content, list):
                    for j, item in enumerate(content):
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_content = item.get('text', '')
                                # Truncate long text for readability
                                if len(text_content) > 200:
                                    print(f"  Content[{j}] (text): {text_content[:200]}...")
                                else:
                                    print(f"  Content[{j}] (text): {text_content}")
                            elif item.get('type') == 'image_url':
                                image_url = item.get('image_url', {}).get('url', '')
                                if image_url.startswith('data:image'):
                                    print(f"  Content[{j}] (image): [IMAGE DATA - {len(image_url)} chars]")
                                else:
                                    print(f"  Content[{j}] (image): {image_url}")
                            else:
                                print(f"  Content[{j}] (unknown): {str(item)[:100]}...")
                        else:
                            print(f"  Content[{j}]: {str(item)[:100]}...")
                else:
                    # Content is a string
                    if len(str(content)) > 200:
                        print(f"  Content (text): {str(content)[:200]}...")
                    else:
                        print(f"  Content (text): {content}")
            else:
                print(f"  No content attribute found")
            print()

        try:
            response = self.llm.invoke(messages_list)
            state["response"] = response.content.strip()
            print("✅ LLM call successful")
            print(f"📤 LLM Response: {state['response']}")
        except Exception as e:
            print(f"❌ Error in LLM call: {e}")
            state["response"] = "An error occurred while communicating with the AI model."
        
        return state
    
    # --- UPDATED PROCESS REQUEST METHOD ---
    def process_request(self, question: str, mode: str, tactile_image: Optional[str] = None, 
                      vision_image: Optional[str] = None) -> Dict[str, str]:
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
        return {
            "response": final_state["response"],
            "optimized_question": final_state["optimized_question"]
        }

# Global agent instance
multimodal_agent = MultimodalAgent()