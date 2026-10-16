
from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/")
def home():
    return render_template("test.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    news_text = data.get("text", "")

    if not isinstance(news_text, str) or not news_text.strip():
        return jsonify({"error": "Please enter news text."}), 400

    cleaned_text = clean_text(news_text)

    if not cleaned_text:
        return jsonify({"error": "Please enter valid news text."}), 400

    features = vectorizer.transform([cleaned_text])
    prediction = int(model.predict(features)[0])

    result = "Fake News" if prediction == 0 else "Real News"

    print("Model prediction:", prediction)
    print("Result:", result)

    return jsonify({
        "prediction": result,
        "model_label": prediction
    })


if __name__ == "__main__":
    app.run(debug=True)
