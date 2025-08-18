"""
Few-shot prompt templates for different reasoning modes
"""

# BALANCED VERSION: Include image data but keep prompts concise
TACTILE_ONLY_TEMPLATE = """Analyze the tactile data and answer: {question}

Tactile Data: {tactile_image}"""

VISION_ONLY_TEMPLATE = """Analyze the image and answer: {question}

Image: {visual_image}"""

COMBINED_TEMPLATE = """Analyze both data sources and answer: {question}

Tactile: {tactile_image}
Visual: {visual_image}"""

def get_template(mode: str) -> str:
    """Get the appropriate template based on the reasoning mode"""
    templates = {
        "tactile": TACTILE_ONLY_TEMPLATE,
        "vision": VISION_ONLY_TEMPLATE,
        "combined": COMBINED_TEMPLATE
    }
    return templates.get(mode, TACTILE_ONLY_TEMPLATE)
