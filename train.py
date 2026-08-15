from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

texts = [
    "win a free prize now", "claim your cash reward", "limited offer click now", "meeting at 10 tomorrow",
    "please send the project report", "your appointment is confirmed", "free gift waiting for you", "can we discuss the assignment"
]
labels = [1, 1, 1, 0, 0, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=1000).fit(X_train_vec, y_train)
predictions = model.predict(X_test_vec)
print(classification_report(y_test, predictions, zero_division=0))
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(model, "artifacts/model.joblib")
joblib.dump(vectorizer, "artifacts/vectorizer.joblib")
print("Saved model artifacts to artifacts/")
