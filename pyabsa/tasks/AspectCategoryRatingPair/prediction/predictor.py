# prediction/predictor.py
import os
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pyabsa.tasks.AspectCategoryRatingPair.models.model import T5GenModel

_PAIR_RE = re.compile(
    r"<\s*(food|service|atmosphere)\s*,\s*([1-5])\s*>",
    re.IGNORECASE
)

class T5Predictor:
    def __init__(self, config, model_path):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name_or_path)
        self.model = T5GenModel(config)

        if os.path.isdir(model_path):
            print(f"Loading from folder: {model_path}")
            self.model.model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            print(f"Loading from file: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

        self.model.to(config.device)
        self.model.eval()

        if not getattr(self.config, "prompt_template", None):
            raise ValueError("config.prompt_template is missing. Please set it in configuration.py")

    def _repair_to_strict_schema(self, text: str) -> str:
        # Extract all pairs found
        found = {m.group(1).lower(): int(m.group(2)) for m in _PAIR_RE.finditer(text or "")}
        if all(k in found for k in ("food", "service", "atmosphere")):
            return f"<food, {found['food']}>, <service, {found['service']}>, <atmosphere, {found['atmosphere']}>"
        return (text or "").strip()

    def predict(self, review_text: str) -> str:
        prompt = self.config.prompt_template.format(REVIEW_TEXT=(review_text or "").strip())

        inputs = self.tokenizer(
            prompt,
            max_length=self.config.max_source_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.to(self.config.device)
        attention_mask = inputs.attention_mask.to(self.config.device)

        generated_ids = self.model.generate(input_ids, attention_mask)
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return self._repair_to_strict_schema(output_text)
