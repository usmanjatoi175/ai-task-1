# sentiment_analysis.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. Dataset Preparation
# ---------------------------
data = {
    "text": [
        # Positive
        "I love this product",
        "The movie was amazing",
        "This is the best phone I ever used",
        "I am so happy with the service",
        "What a fantastic experience",
        "Absolutely wonderful",
        "The food was great and tasty",
        "The game is really fun",
        "I enjoy using this app",
        "This is awesome",
        
        # Negative
        "I hate this service",
        "The food was terrible",
        "This is the worst day",
        "I am very disappointed",
        "The movie was boring",
        "Absolutely horrible",
        "The phone is useless",
        "I regret buying this",
        "The app keeps crashing",
        "This is bad"
    ],
    "label": [
        "Positive","Positive","Positive","Positive","Positive",
        "Positive","Positive","Positive","Positive","Positive",
        "Negative","Negative","Negative","Negative","Negative",
        "Negative","Negative","Negative","Negative","Negative"
    ]
}

df = pd.DataFrame(data)

# ---------------------------
# 2. Preprocessing
# ---------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# ---------------------------
# 3. Model Training
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------------------
# 4. Model Testing