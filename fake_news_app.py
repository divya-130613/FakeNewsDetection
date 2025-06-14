
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("sample_fake_news_dataset.csv")

# Split into features and labels
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Model training
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article or headline below to check if it's likely **REAL** or **FAKE**.")

user_input = st.text_area("🖊️ News Content", height=150)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        st.success(f"🔍 The news is predicted to be: **{prediction}**")
