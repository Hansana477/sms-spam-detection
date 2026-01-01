import pickle
import os

# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")

# Load model
with open(MODEL_PATH, "rb") as f:
    model, vectorizer = pickle.load(f)

messages = [
    "Congratulations! You won a free iPhone",
    "Can we meet tomorrow at 10?"
]

X = vectorizer.transform(messages)
predictions = model.predict(X)

for msg, pred in zip(messages, predictions):
    print(f"{msg} → {'SPAM 🚫' if pred == 1 else 'NOT SPAM ✅'}")
