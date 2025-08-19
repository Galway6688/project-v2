"""
Optimized, project-specific prompt templates in English for a few-shot system with real images.

Note: These templates only contain the text portions. The example images will be dynamically loaded and injected by the agent code.
"""

# --- TACTILE MODE ---
# The text of the AI's example ANSWER
FEW_SHOT_EXAMPLE_TACTILE_TEXT = """
**Analysis:**
1.  **Describe Texture:** The tactile data indicates a highly irregular and coarse surface with sharp, high-frequency peaks. The texture is abrasive and non-uniform.
2.  **Infer Properties:** The material feels very hard with high friction. It does not seem to deform under pressure. Thermal properties are neutral.
3.  **Justify:** The sharp peaks in the data directly correlate to a rough, abrasive feel, typical of materials designed for sanding or gripping.
**Conclusion:** The material is likely a form of sandpaper or a similar abrasive surface.
"""
# The text of the INSTRUCTIONS for the actual task
TASK_PROMPT_TACTILE = """**Role:** You are a material scientist specializing in haptics.
**Task:** Analyze the provided tactile image data to answer the user's question. Your response should be detailed and strictly grounded in the data, following the format of the example provided in the previous turn.

**User's Question:**
{question}
"""


# --- VISION MODE ---
# The text of the AI's example ANSWER
FEW_SHOT_EXAMPLE_VISION_TEXT = """
**Analysis:**
1.  **Visual Description:** The image shows a surface with a very high gloss, reflecting light sharply. There are smooth, vein-like patterns but no discernible surface texture.
2.  **Infer Tactile Sensation:** The high gloss and lack of texture strongly suggest the surface is extremely smooth, hard, and has low friction. It would likely feel cool to the touch due to its density.
3.  **Material Identification:** The visual characteristics are typical of polished stone, such as marble or granite.
**Conclusion:** The surface would feel smooth, hard, and cool.
"""
# The text of the INSTRUCTIONS for the actual task
TASK_PROMPT_VISION = """**Role:** You are a computer vision analyst with expertise in material science.
**Task:** Analyze the provided image of a surface to answer the user's question, inferring physical properties from visual information. Follow the format of the example provided in the previous turn.

**User's Question:**
{question}
"""


# --- COMBINED MODE ---
# The text of the AI's example ANSWER
FEW_SHOT_EXAMPLE_COMBINED_TEXT = """
**Analysis:**
1.  **Synthesize, Don't Just List:** The analysis integrates both data sources into a unified description.
2.  **Establish Correlation:** The visual image clearly shows parallel lines or 'wales', which is characteristic of corduroy fabric. This visual pattern directly corresponds to the tactile data, which registers a series of uniform, soft, and raised ridges.
3.  **Provide a Unified Conclusion:** Based on the visual evidence of a textile and the tactile confirmation of its distinct soft, ridged pattern, the material is identified as corduroy fabric. It would feel soft, warm, and textured.
"""
# The text of the INSTRUCTIONS for the actual task
TASK_PROMPT_COMBINED = """**Role:** You are an advanced multimodal AI, an expert in fusing visual and tactile data for comprehensive material analysis.
**Task:** Synthesize information from BOTH the visual image and the tactile data to provide a holistic answer, following the format of the example provided in the previous turn.

**User's Question:**
{question}
"""