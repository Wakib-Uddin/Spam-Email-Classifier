import sys
import joblib

model = joblib.load("artifacts/model.joblib")
vectorizer = joblib.load("artifacts/vectorizer.joblib")
text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Message: ")
prediction = model.predict(vectorizer.transform([text]))[0]
print("SPAM" if prediction == 1 else "HAM")
