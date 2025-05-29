import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import joblib
import os


df = pd.read_csv("intent_data.csv")
X = df["message"]
y = df["intent"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)


clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
clf.fit(X_vec, y)


os.makedirs("intent_model", exist_ok=True)
joblib.dump(clf, "intent_model/model.pkl")
joblib.dump(vectorizer, "intent_model/vectorizer.pkl")
