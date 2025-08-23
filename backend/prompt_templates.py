#
# File: prompt_templates.py
#
"""
Final Optimized Prompt Templates (English), with strengthened instructions.
"""

# --- TACTILE MODE ---
TASK_PROMPT_TACTILE = """**Role:** You are an expert haptic data interpreter.
**Task:** Analyze the provided tactile sensor data visualization. Based on the patterns, colors, and gradients, list the key tactile attributes of the surface.
**Candidate Words:**
You **must** choose your answer ONLY from the following comma-separated list of candidate words:
`{word_list}`  # <-- 在这里预留“槽位”
**Output Format Requirements:**
- You **must** only provide your answer as a comma-separated list of descriptive adjectives.
- **Focus only on the 5 most essential tactile attributes.
- **DO NOT** write full sentences, explanations, or any introductory text.
- Your output **must** strictly match the format shown in the examples.

**User's Question:**
{question}
"""


# --- VISION MODE ---
TASK_PROMPT_VISION = """**Role:** You are an expert visual observer who infers tactile properties from photographs.
**Task:** Analyze the provided photograph, focusing on the main material. Based on what you see, list the tactile attributes you can infer.
**Candidate Words:**
You **must** choose your answer ONLY from the following comma-separated list of candidate words:
`{word_list}`  # <-- 在这里预留“槽位”

**Output Format Requirements:**
- You **must** only provide your answer as a comma-separated list of descriptive adjectives.
- **Focus only on the 5 most essential tactile attributes.
- **DO NOT** write full sentences, explanations, or any introductory text.
- Your output **must** strictly match the format shown in the examples.

**User's Question:**
{question}
"""


# --- COMBINED MODE ---
TASK_PROMPT_COMBINED = """**Role:** You are an expert multimodal analyst, skilled at correlating visual photographs with abstract sensor data.
**Task:** Synthesize the RGB photograph and the tactile sensor data visualization to list the most relevant tactile attributes of the material shown.
**Candidate Words:**
You **must** choose your answer ONLY from the following comma-separated list of candidate words:
`{word_list}`  # <-- 在这里预留“槽位”
**Output Format Requirements:**
- You **must** only provide your answer as a comma-separated list of descriptive adjectives.
- **Focus only on the 5 most essential tactile attributes.
- **DO NOT** write full sentences, explanations, or any introductory text.
- Your output **must** strictly match the format shown in the examples.

**User's Question:**
{question}
"""