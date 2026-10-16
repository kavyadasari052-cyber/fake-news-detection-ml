
import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Step 2: Assign labels
fake["label"] = 0
true["label"] = 1

# Step 3: Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Step 4: Clean and validate data
data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str)
data = data.drop_duplicates(subset=["text"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["clean_text"] = data["text"].apply(clean_text)
data = data[data["clean_text"] != ""]

# Step 5: Create TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

print("TF-IDF Shape:", X.shape)
print("Labels Shape:", y.shape)
print("Fake News:", (y == 0).sum())
print("Real News:", (y == 1).sum())

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 7: Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Step 8: Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake News", "Real News"]
))

# Step 9: Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and Vectorizer Saved Successfully!")
print("Label mapping: 0 = Fake News, 1 = Real News")

# Step 10: Test sample news
test_news = [
    "NASA's Perseverance rover landed on Mars on February 18, 2021.",
    "Scientists confirm that drinking hot water makes humans invisible.",
    "India successfully landed Chandrayaan 3 near the Moon's south pole on August 23, 2023.",
    "The World Health Organization was established on April 7, 1948."
]

print("\nSample Predictions:")

for news in test_news:
    cleaned = clean_text(news)
    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    result = "Fake News" if prediction == 0 else "Real News"

    print("\nNews:", news)
    print("Prediction:", result)
    print("Model label:", prediction)
    print("Confidence:", round(float(max(probabilities)) * 100, 2), "%")

print("\nAll tests completed!")
