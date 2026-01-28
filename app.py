from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model, vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    explanation = []
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        X = vectorizer.transform([message])

        
        result = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        confidence = round(max(probs) * 100, 2)
        prediction = "SPAM 🚫" if result == 1 else "NOT SPAM ✅"

        
        feature_names = vectorizer.get_feature_names_out()
        word_scores = X.toarray()[0]

        important_words = [
            feature_names[i]
            for i, score in enumerate(word_scores)
            if score > 0
        ]

        explanation = important_words[:5]  

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
