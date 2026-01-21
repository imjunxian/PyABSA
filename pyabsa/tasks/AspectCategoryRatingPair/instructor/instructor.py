# pyabsa/tasks/AspectCategoryRatingPair/instructor/instructor.py
import os
import json
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tqdm
from ..models.model import T5GenModel
from ..dataset_utils.data_utils_for_training import T5ABSADataset

# Relaxed Regex for extraction (we enforce strict format in prompt/predictor)
_PAIR_RE = re.compile(r"<?\s*(food|service|atmosphere)\s*,\s*([1-5])\s*>", re.IGNORECASE)
REQUIRED = ("food", "service", "atmosphere")

class T5TrainingInstructor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name_or_path)
        self.model = T5GenModel(config)
        self.model.to(self.config.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_dataset(self):
        train_set = T5ABSADataset(self.config, self.tokenizer, dataset_type="train")
        valid_set = T5ABSADataset(self.config, self.tokenizer, dataset_type="valid")
        test_set = T5ABSADataset(self.config, self.tokenizer, dataset_type="test")

        # ✅ CRITICAL FIX: num_workers=0 and pin_memory=False to prevent Bus Error
        self.train_dataloader = DataLoader(
            train_set, batch_size=self.config.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True  # Changed from 0 and False
        )
        self.valid_dataloader = DataLoader(
            valid_set, batch_size=self.config.batch_size, shuffle=False, 
            num_workers=0, pin_memory=False
        )
        self.test_dataloader = DataLoader(
            test_set, batch_size=self.config.batch_size, shuffle=False, 
            num_workers=0, pin_memory=False
        )

        print(f"✅ Loaded: {len(train_set)} Train, {len(valid_set)} Valid, {len(test_set)} Test")

    def save_model(self, folder_name: str):
        save_path = os.path.join(self.config.output_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)
        self.model.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ Saved to: {save_path}")

    def load_best_model(self, folder_name: str):
        load_path = os.path.join(self.config.output_dir, folder_name)
        if not os.path.exists(load_path):
            return
        self.model.model = T5ForConditionalGeneration.from_pretrained(load_path)
        self.model.to(self.config.device)
        print(f"🔄 Loaded best model: {load_path}")

    def extract_scores(self, text: str):
        found = {}
        for m in _PAIR_RE.finditer(text or ""):
            asp = m.group(1).lower().strip()
            val = int(m.group(2))
            if asp not in found:
                found[asp] = val
        return found

    def evaluate(self, dataloader):
        self.model.eval()
        metric_data = {a: {"preds": [], "labels": []} for a in REQUIRED}
        
        # Track strict format success
        total = 0
        strict_format_match = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                generated_ids = self.model.generate(input_ids, attention_mask)
                preds_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = self.tokenizer.pad_token_id
                labels_text = self.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

                for p_str, l_str in zip(preds_text, labels_text):
                    total += 1
                    p_scores = self.extract_scores(p_str)
                    l_scores = self.extract_scores(l_str)

                    # Check if output strictly contains all 3 required aspects
                    if len(p_scores) == 3 and all(k in p_scores for k in REQUIRED):
                        strict_format_match += 1

                    if all(a in p_scores for a in REQUIRED) and all(a in l_scores for a in REQUIRED):
                        for a in REQUIRED:
                            metric_data[a]["preds"].append(p_scores[a])
                            metric_data[a]["labels"].append(l_scores[a])

        final = {}
        final["strict_accuracy"] = (strict_format_match / total) if total else 0.0

        maes, rmses = [], []
        for a in REQUIRED:
            preds = np.array(metric_data[a]["preds"], dtype=float)
            labs = np.array(metric_data[a]["labels"], dtype=float)
            if len(preds) == 0: 
                final[f"{a}_RMSE"] = 0.0
                continue
            rmse = float(np.sqrt(np.mean((preds - labs) ** 2)))
            final[f"{a}_RMSE"] = rmse
            rmses.append(rmse)

        final["avg_RMSE"] = float(np.mean(rmses)) if rmses else 0.0

        print(f"\n📊 Metrics - RMSE: {final['avg_RMSE']:.4f} | Strict Format Acc: {final['strict_accuracy']:.2%}")
        return final

    def run(self):
        self.load_dataset()
        best_valid_rmse = float("inf")

        for epoch in range(self.config.num_epoch):
            self.model.train()
            total_loss = 0.0
            
            pbar = tqdm.tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                
                labels = labels.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            print(f"--- Validating Epoch {epoch+1} ---")
            val_metrics = self.evaluate(self.valid_dataloader)

            if val_metrics["avg_RMSE"] < best_valid_rmse:
                print(f"🔥 Best Model (RMSE {val_metrics['avg_RMSE']:.4f}) -> Saving...")
                best_valid_rmse = val_metrics["avg_RMSE"]
                self.save_model("best_model")

        self.load_best_model("best_model")
        print("📌 Test Results:")
        print(json.dumps(self.evaluate(self.test_dataloader), indent=2))