# run_demo.py
import os
from pyabsa.tasks.AspectCategoryRatingPair.prediction.predictor import T5Predictor
from pyabsa.tasks.AspectCategoryRatingPair.configuration.configuration import T5ConfigManager

def main():
    base_path = os.getcwd()
    model_path = os.path.join(base_path, "checkpoints", "best_model")

    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find model folder at: {model_path}")
        print("Please run 'run_train.py' first.")
        return

    config = T5ConfigManager.get_t5_config_template()
    config.model_name_or_path = "google/flan-t5-base"
    config.device = "cpu"

    predictor = T5Predictor(config, model_path)

    examples = [
        "The food was delicious and the service was amazing!",
        "Bad food.",
        "Terrible atmosphere but the location is good.",
        "I hated the steak.",
        "This place is a GEM! We tried lunch set menu."
    ]

    for text in examples:
        print(f"\nReview:  {text}")
        print(f"Output:  {predictor.predict(text)}")

if __name__ == "__main__":
    main()
