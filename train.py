import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv('car_complaints_10000.csv')

# Rename columns
df.rename(columns={"Complaint": "Complaint", "Problem": "Category"}, inplace=True)

# Drop empty
df.dropna(subset=["Complaint", "Category"], inplace=True)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Transform data
X = vectorizer.fit_transform(df["Complaint"])
y = df["Category"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save vectorizer + model
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print("\nTraining complete!")
print("Saved vectorizer.pkl and model.pkl")
