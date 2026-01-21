# run_zero_shot_safe.py
import os

# ---------------------------------------------------------
# 🛡️ CRASH PREVENTION HEADER (MUST BE AT THE TOP)
# These flags force single-threaded execution to stop Mac Bus Errors
# ---------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ---------------------------------------------------------
# 📝 CONFIGURATION
# ---------------------------------------------------------
MODEL_NAME = "google/flan-t5-base"

PROMPT_TEMPLATE = """Task: Extract ratings (1-5) for food, service, and atmosphere.

Input Review:
"{REVIEW_TEXT}"

Constraints:
1. Output format MUST be: <food, X>, <service, Y>, <atmosphere, Z>
2. If an aspect is missing, infer it or use 3.
3. Output NOTHING else.

Response:
"""

def clean_output(text):
    """
    Ensures the output strictly follows the <aspect, score> format.
    """
    # Find all pairs like "food, 5" or "Service: 4"
    matches = re.findall(r"(food|service|atmosphere)\D+([1-5])", text, re.IGNORECASE)
    
    scores = {"food": 3, "service": 3, "atmosphere": 3}
    for aspect, score in matches:
        scores[aspect.lower()] = int(score)
        
    return f"<food, {scores['food']}>, <service, {scores['service']}>, <atmosphere, {scores['atmosphere']}>"

def main():
    print(f"🚀 Initializing on CPU (Safe Mode)...")
    
    # ✅ Force CPU explicitly
    device = torch.device("cpu")
    
    try:
        print("⏳ Loading Tokenizer...")
        # legacy=False fixes the protobuf warning
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
        
        print("⏳ Loading Model...")
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR loading model: {e}")
        print("👉 Try running: pip install sentencepiece protobuf")
        return

    # ---------------------------------------------------------
    # 🧪 TEST DATA
    # ---------------------------------------------------------
    reviews = [
        "The steak was amazing but the server was so rude!",
        "Terrible place, dirty and loud.",
        "I loved the vibe, but the food was cold.",
        "Average experience.",
    ]

    print("\nStarting Inference...\n")

    for review in reviews:
        # 1. Format Prompt
        input_text = PROMPT_TEMPLATE.format(REVIEW_TEXT=review)
        
        # 2. Tokenize
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(device)

        # 3. Generate (Greedy Search for stability)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=64,
                num_beams=1, 
                do_sample=False
            )

        # 4. Decode
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_output = clean_output(raw_output)

        print(f"📝 Review: {review}")
        print(f"🤖 Raw:    {raw_output}")
        print(f"✨ Final:  {final_output}")
        print("-" * 30)

if __name__ == "__main__":
    main()