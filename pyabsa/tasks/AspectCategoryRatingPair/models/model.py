# pyabsa/tasks/AspectCategoryRatingPair/models/model.py
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class T5GenModel(nn.Module):
    def __init__(self, config):
        super(T5GenModel, self).__init__()
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_name_or_path)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.config.max_target_length
        )