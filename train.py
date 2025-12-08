# train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("car_complaints_10000.csv")  # or your dataset
df.dropna(subset=["Complaint", "Problem"], inplace=True)

X = df["Complaint"]
y = df["Problem"]

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words='english')
X_vec = vec.fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(Xtr, ytr)

print("Train done. Example accuracy:", model.score(Xte, yte))

joblib.dump(vec, "vectorizer.pkl")
joblib.dump(model, "model.pkl")
print("Saved vectorizer.pkl and model.pkl")
