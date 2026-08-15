# 🤖 Spam Email Classifier

An end-to-end NLP machine-learning project that classifies messages as **spam** or **ham** using TF-IDF features and supervised learning.

## 🧠 Pipeline
`Raw text → cleaning → TF-IDF → model training → evaluation → prediction`

## Models
- Logistic Regression
- Multinomial Naive Bayes

## Metrics
Accuracy, precision, recall, F1-score and confusion matrix.

## 🚀 Run
```bash
pip install -r requirements.txt
python train.py
python predict.py "Congratulations! You won a free prize"
```

## 📁 Structure
```text
spam-email-classifier/
├── data/
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 👨‍💻 Author
A.M. Wakib Uddin — CSE Engineer | Python | Data Science | Machine Learning
