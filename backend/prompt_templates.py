# """
# Optimized, project-specific prompt templates in English for a few-shot system with real images.
#
# Note: These templates only contain the text portions. The example images will be dynamically loaded and injected by the agent code.
# """
#
# # --- TACTILE MODE ---
# # The text of the AI's example ANSWER
# # OLD VERSION - COMMENTED OUT:
# # FEW_SHOT_EXAMPLE_TACTILE_TEXT = """
# # **Analysis:**
# # 1.  **Describe Texture:** The tactile data indicates a highly irregular and coarse surface with sharp, high-frequency peaks. The texture is abrasive and non-uniform.
# # 2.  **Infer Properties:** The material feels very hard with high friction. It does not seem to deform under pressure. Thermal properties are neutral.
# # 3.  **Justify:** The sharp peaks in the data directly correlate to a rough, abrasive feel, typical of materials designed for sanding or gripping.
# # **Conclusion:** The material is likely a form of sandpaper or a similar abrasive surface.
# # """
#
# # NEW VERSION - Updated for image_2355:
# FEW_SHOT_EXAMPLE_TACTILE_TEXT = """
# **Analysis:**
# 1.  **Describe Texture:** The tactile data reveals a structured surface with moderate variations in height and texture patterns. The surface shows controlled roughness with defined tactile features.
# 2.  **Infer Properties:** The material exhibits moderate friction and firmness. The texture pattern suggests engineered properties rather than natural variation.
# 3.  **Justify:** The tactile signature corresponds to a material with deliberate surface texture design, potentially for functional or aesthetic purposes.
# **Conclusion:** The material appears to be an engineered surface with controlled tactile properties, possibly a textured synthetic material or processed natural surface.
# """
# # The text of the INSTRUCTIONS for the actual task
# TASK_PROMPT_TACTILE = """**Role:** You are a material scientist specializing in haptics.
# **Task:** Analyze the provided tactile image data to answer the user's question. Your response should be detailed and strictly grounded in the data, following the format of the example provided in the previous turn.
#
# **User's Question:**
# {question}
# """
#
#
# # --- VISION MODE ---
# # The text of the AI's example ANSWER
# # OLD VERSION - COMMENTED OUT:
# # FEW_SHOT_EXAMPLE_VISION_TEXT = """
# # **Analysis:**
# # 1.  **Visual Description:** The image shows a surface with a very high gloss, reflecting light sharply. There are smooth, vein-like patterns but no discernible surface texture.
# # 2.  **Infer Tactile Sensation:** The high gloss and lack of texture strongly suggest the surface is extremely smooth, hard, and has low friction. It would likely feel cool to the touch due to its density.
# # 3.  **Material Identification:** The visual characteristics are typical of polished stone, such as marble or granite.
# # **Conclusion:** The surface would feel smooth, hard, and cool.
# # """
#
# # NEW VERSION - Updated for image_2355:
# FEW_SHOT_EXAMPLE_VISION_TEXT = """
# **Analysis:**
# 1.  **Visual Description:** The image displays a surface with visible texture patterns and structured appearance. The material shows consistent coloration with distinguishable surface features and moderate light reflection.
# 2.  **Infer Tactile Sensation:** Based on the visual texture patterns, the surface would likely feel moderately textured with defined tactile feedback. The material appears to have a controlled surface structure that would provide grip and tactile recognition.
# 3.  **Material Identification:** The visual characteristics suggest an engineered or processed material with deliberate surface structuring.
# **Conclusion:** The surface would feel textured, structured, and provide tactile feedback with moderate friction properties.
# """
# # The text of the INSTRUCTIONS for the actual task
# TASK_PROMPT_VISION = """**Role:** You are a computer vision analyst with expertise in material science.
# **Task:** Analyze the provided image of a surface to answer the user's question, inferring physical properties from visual information. Follow the format of the example provided in the previous turn.
#
# **User's Question:**
# {question}
# """
#
#
# # --- COMBINED MODE ---
# # The text of the AI's example ANSWER
# # OLD VERSION - COMMENTED OUT:
# # FEW_SHOT_EXAMPLE_COMBINED_TEXT = """
# # **Analysis:**
# # 1.  **Synthesize, Don't Just List:** The analysis integrates both data sources into a unified description.
# # 2.  **Establish Correlation:** The visual image clearly shows parallel lines or 'wales', which is characteristic of corduroy fabric. This visual pattern directly corresponds to the tactile data, which registers a series of uniform, soft, and raised ridges.
# # 3.  **Provide a Unified Conclusion:** Based on the visual evidence of a textile and the tactile confirmation of its distinct soft, ridged pattern, the material is identified as corduroy fabric. It would feel soft, warm, and textured.
# # """
#
# # NEW VERSION - Updated for image_2355:
# FEW_SHOT_EXAMPLE_COMBINED_TEXT = """
# **Analysis:**
# 1.  **Synthesize, Don't Just List:** The analysis integrates both visual and tactile data sources into a unified material description.
# 2.  **Establish Correlation:** The visual image shows a structured surface with defined patterns and consistent texture features. This visual evidence directly correlates with the tactile data, which reveals controlled surface variations and engineered texture properties.
# 3.  **Provide a Unified Conclusion:** The combination of visual structure and tactile feedback indicates an engineered material with deliberate surface design. The material exhibits controlled texture properties that provide both visual appeal and functional tactile characteristics, suggesting an application-specific surface treatment or synthetic material design.
# """
# # The text of the INSTRUCTIONS for the actual task
# TASK_PROMPT_COMBINED = """**Role:** You are an advanced multimodal AI, an expert in fusing visual and tactile data for comprehensive material analysis.
# **Task:** Synthesize information from BOTH the visual image and the tactile data to provide a holistic answer, following the format of the example provided in the previous turn.
#
# **User's Question:**
# {question}
# """

"""
Final Optimized Prompt Templates (English), specifically tailored to the provided dataset.

These prompts are engineered to align with the unique characteristics of the visual (RGB)
and tactile (TAC sensor visualization) data, and to produce the target output format
of a comma-separated list of adjectives as seen in finetune.json.
"""

# --- TACTILE MODE ---

# The instructions now explicitly state that the input is a SENSOR DATA VISUALIZATION.
TASK_PROMPT_TACTILE = """**Role:** You are an expert haptic data interpreter.
**Task:** Analyze the provided tactile sensor data visualization. Based on the patterns, colors, and gradients in the visualization, list the key tactile attributes of the surface.

**Output Format:**
- Provide your answer as a comma-separated list of descriptive adjectives.
- Do not use full sentences or explanations.
- The format should match the example provided in the previous turn.

**User's Question:**
{question}
"""


# --- VISION MODE ---

# The instructions guide the AI to focus on the main subject in potentially complex images.
TASK_PROMPT_VISION = """**Role:** You are an expert visual observer who infers tactile properties from photographs.
**Task:** Analyze the provided photograph, focusing on the main material. Based on what you see, list the tactile attributes you can infer.

**Output Format:**
- Provide your answer as a comma-separated list of descriptive adjectives.
- Do not use full sentences or explanations.
- The format should match the example provided in the previous turn.

**User's Question:**
{question}
"""


# --- COMBINED MODE ---

# The instructions are highly specific, demanding the AI to link the RGB photo to the TAC data visualization.
TASK_PROMPT_COMBINED = """**Role:** You are an expert multimodal analyst, skilled at correlating visual photographs with abstract sensor data.
**Task:** Synthesize the RGB photograph and the tactile sensor data visualization to list the most relevant tactile attributes of the material shown.

**Instructions:**
- Use the RGB image to identify the material (e.g., towel, denim, metal).
- Use the tactile data visualization to confirm its properties (e.g., the soft, loopy texture of a towel would create a specific pattern in the tactile data).
- Combine your findings into a single list of descriptors.

**Output Format:**
- Provide your answer as a comma-separated list of descriptive adjectives.
- Do not use full sentences or explanations.
- The format should match the example provided in the previous turn.

**User's Question:**
{question}
"""