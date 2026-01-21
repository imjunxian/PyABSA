# dataset_utils/data_utils_for_training.py
import json
from torch.utils.data import Dataset

class T5ABSADataset(Dataset):
    def __init__(self, config, tokenizer, dataset_type="train"):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.data = []
        self.load_data()

    def load_data(self):
        file_path_list = self.config.dataset_file[self.dataset_type]
        file_path = file_path_list[0] if isinstance(file_path_list, list) else file_path_list

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find dataset file at: {file_path}")

        prompt_template = getattr(self.config, "prompt_template", None)
        if not prompt_template:
            raise ValueError("config.prompt_template is missing. Please set it in configuration.py")

        for entry in raw_data:
            review_text = (entry.get("review_text") or "").strip()
            if not review_text:
                continue

            # ✅ Require all 3 scores (food/service/atmosphere)
            food = entry.get("food_score")
            service = entry.get("service_score")
            atmosphere = entry.get("atmosphere_score")

            if food is None or service is None or atmosphere is None:
                continue

            try:
                food = int(food)
                service = int(service)
                atmosphere = int(atmosphere)
            except Exception:
                continue

            # Optional: enforce 1–5
            if not (1 <= food <= 5 and 1 <= service <= 5 and 1 <= atmosphere <= 5):
                continue

            source_text = prompt_template.format(REVIEW_TEXT=review_text)
            target_text = f"<food, {food}>, <service, {service}>, <atmosphere, {atmosphere}>"

            self.data.append({"source_text": source_text, "target_text": target_text})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        source_text = row["source_text"]
        target_text = row["target_text"]

        source = self.tokenizer(
            source_text,
            max_length=self.config.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = self.tokenizer(
            target_text,
            max_length=self.config.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source["input_ids"].flatten(),
            "attention_mask": source["attention_mask"].flatten(),
            "labels": target["input_ids"].flatten(),
        }
