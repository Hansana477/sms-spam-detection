import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from load_data import load_data
from preprocess import preprocess
from sklearn.metrics import accuracy_score

# Load dataset
df = load_data()

# Preprocess text
X, vectorizer = preprocess(df["message"])
y = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"✅ Model trained successfully")
print(f"🎯 Accuracy: {accuracy:.2f}")

# Save model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "spam_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump((model, vectorizer), f)

print("💾 Model saved successfully")