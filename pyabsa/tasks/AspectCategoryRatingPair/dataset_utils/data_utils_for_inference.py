# dataset_utils/data_utils_for_inference.py
from torch.utils.data import Dataset

class T5InferenceDataset(Dataset):
    def __init__(self, config, tokenizer, texts):
        self.config = config
        self.tokenizer = tokenizer
        self.texts = texts

        if not getattr(self.config, "prompt_template", None):
            raise ValueError("config.prompt_template is missing. Please set it in configuration.py")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        review = (self.texts[index] or "").strip()
        prompt = self.config.prompt_template.format(REVIEW_TEXT=review)

        inputs = self.tokenizer(
            prompt,
            max_length=self.config.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
        }
