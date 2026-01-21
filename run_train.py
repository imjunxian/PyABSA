# run_train.py
import os
import sys

# --- REMOVED CPU OPTIMIZATIONS ---
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pyabsa.tasks.AspectCategoryRatingPair.configuration.configuration import T5ConfigManager
from pyabsa.tasks.AspectCategoryRatingPair.trainer.trainer import T5Trainer

def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def main():
    config = T5ConfigManager.get_t5_config_template()

    # ✅ CHANGE 1: Set device to cuda
    config.device = "cuda"

    base_path = os.getcwd()

    # Locate datasets
    train_file = _first_existing(
        os.path.join(base_path, "datasets", "train.json"),
        os.path.join(base_path, "pyabsa", "datasets", "train.json"),
    )
    valid_file = _first_existing(
        os.path.join(base_path, "datasets", "valid.json"),
        os.path.join(base_path, "pyabsa", "datasets", "valid.json"),
    )
    test_file = _first_existing(
        os.path.join(base_path, "datasets", "test.json"),
        os.path.join(base_path, "pyabsa", "datasets", "test.json"),
    )

    if not train_file:
        raise FileNotFoundError("Cannot find train.json in datasets/ or pyabsa/datasets/")
    if not valid_file:
        raise FileNotFoundError("Cannot find valid.json in datasets/ or pyabsa/datasets/")
    if not test_file:
        raise FileNotFoundError("Cannot find test.json in datasets/ or pyabsa/datasets/")

    config.dataset_file = {
        "train": [train_file],
        "valid": [valid_file],
        "test": [test_file],
    }

    # Configuration
    config.model_name_or_path = "google/flan-t5-base"
    config.max_source_length = 512
    config.max_target_length = 64
    
    # ✅ CHANGE 2: Increase batch size for GPU efficiency (try 8, 16, or 32)
    config.batch_size = 16 
    
    config.num_epoch = 10
    config.learning_rate = 3e-4
    config.output_dir = "checkpoints"

    # Start Trainer
    _ = T5Trainer(config=config, dataset=None, auto_device=False)

if __name__ == "__main__":
    main()