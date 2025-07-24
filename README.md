# 🧠 Fake News Detector

This project uses **machine learning** to detect fake news articles based on their content.

## 🔍 Project Overview

A Logistic Regression model is trained using **TF-IDF vectorization** on a combined dataset of fake and true news headlines. The goal is to classify whether a news article is **fake (1)** or **true (0)**.

## 📊 Dataset

- Combined from `Fake.csv` and `True.csv`
- Shape: 44,898 rows × 3 columns
- Labels: `1` → Fake, `0` → True

## 🔧 Technologies Used

- Python
- Pandas
- scikit-learn
- TF-IDF Vectorizer
  

## ⚙️ ML Workflow

1. Load and combine datasets
2. Clean and preprocess data
3. Apply TF-IDF vectorization
4. Train Logistic Regression model
5. Evaluate accuracy, precision, recall

## 🚀 Result

Achieved **~98% accuracy** on test set.

## 💡 Future Work

- Build a **Streamlit web app**
- Add **NLP enhancements**
- Deploy on **Heroku / Hugging Face Spaces**


> Feel free to clone the repo and test it yourself!

## 👨‍💻 Author

**Shivam Panwar**  
[LinkedIn](https://www.linkedin.com/in/shivam-panwar-76a1a5211/)
# Fake--News--Detector
A machine learning project to detect fake news using TF--IDF and Logistic Regression.
