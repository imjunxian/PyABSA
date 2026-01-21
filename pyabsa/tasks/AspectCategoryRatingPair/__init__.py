# -*- coding: utf-8 -*-
# file: __init__.py
# description: Package initialization for Generative ABSA Subtask

# Expose the Trainer
from .trainer.trainer import T5Trainer

# Expose the Configuration Manager
from .configuration.configuration import T5ConfigManager

# Expose the Dataset List (so you can refer to 'GoogleReviews' by name)
from .dataset_utils.dataset_list import T5DatasetList

# Expose the Predictor (for inference after training)
from .prediction.predictor import T5Predictor

# (Optional) If you have a model list file, you can expose it here, 
# but usually ConfigManager handles model selection for T5.