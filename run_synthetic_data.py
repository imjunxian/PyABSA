import json
import random
import os

def generate_synthetic_data():
    food_adj_pos = ["delicious", "yummy", "great", "tasty", "fresh"]
    food_adj_neg = ["disgusting", "gross", "cold", "stale", "salty"]
    
    serv_adj_pos = ["friendly", "fast", "attentive", "kind"]
    serv_adj_neg = ["rude", "slow", "ignored us", "terrible"]

    data = []

    # Generate 100 examples
    for _ in range(100):
        is_pos = random.choice([True, False])
        
        if is_pos:
            f_adj = random.choice(food_adj_pos)
            s_adj = random.choice(serv_adj_pos)
            text = f"The food was {f_adj} and the service was {s_adj}."
            food_score = 5
            service_score = 5
        else:
            f_adj = random.choice(food_adj_neg)
            s_adj = random.choice(serv_adj_neg)
            text = f"The food was {f_adj} and the waiter was {s_adj}."
            food_score = 1
            service_score = 1

        data.append({
            "review_text": text,
            "food_score": food_score,
            "service_score": service_score,
            # We leave atmosphere blank sometimes to teach it flexibility
            "atmosphere_score": 5 if is_pos else 1
        })

    # Save
    os.makedirs("datasets", exist_ok=True)
    with open("datasets/train.json", "w") as f:
        json.dump(data[:90], f, indent=2)
    with open("datasets/test.json", "w") as f:
        json.dump(data[90:], f, indent=2)
        
    print("✅ Generated 100 synthetic reviews in datasets/train.json")

if __name__ == "__main__":
    generate_synthetic_data()