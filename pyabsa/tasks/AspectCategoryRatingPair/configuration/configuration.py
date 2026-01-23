# configuration/configuration.py
import copy
from pyabsa.framework.configuration_class.configuration_template import ConfigManager

PROMPT_TEMPLATE = """Task: Predict aspect ratings for a restaurant review.

Aspects:
- food: taste, freshness, portion, menu quality
- service: staff attitude, speed, accuracy, helpfulness
- atmosphere: cleanliness, comfort, ambience, noise, decor

Rating scale:
1 = very negative, 2 = negative, 3 = neutral/mixed/unclear, 4 = positive, 5 = very positive

Rules:
- Output MUST contain exactly 3 pairs in this exact order: food, service, atmosphere.
- Output format MUST be exactly: (food, X), (service, Y), (atmosphere, Z) where X,Y,Z are integers only in rating scale 1 to 5
- Use ONLY integers 1 to 5. No extra text. No explanations.
- If an aspect is not mentioned, infer from the overall sentiment then match it back to the rating scale; if truly unclear, use 3.

Review:
{REVIEW_TEXT}

Output: 
"""

_t5_config_template = {
    "model": None,
    "model_name_or_path": "t5-base",
    "optimizer": "adamw",
    "learning_rate": 3e-4,
    "max_source_length": 512,
    "max_target_length": 64,
    "batch_size": 2,
    "num_epoch": 10,
    "seed": 42,
    "output_dir": "checkpoints",
    "overwrite_output_dir": True,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 100,
    "prompt_template": PROMPT_TEMPLATE,
}

class T5ConfigManager(ConfigManager):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    @staticmethod
    def get_t5_config_template():
        return T5ConfigManager(copy.deepcopy(_t5_config_template))