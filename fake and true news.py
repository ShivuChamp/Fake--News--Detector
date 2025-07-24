import pandas as pd

# Step 1: Load the datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Step 2: Add label column (1 for fake, 0 for real)
fake_df['label'] = 1
true_df['label'] = 0

# Step 3: Combine both datasets
df = pd.concat([fake_df, true_df], axis=0)

# Step 4: Shuffle the combined data
df = df.sample(frac=1).reset_index(drop=True)

# Step 5: Drop unnecessary columns (optional but cleaner)
df = df.drop(columns=['subject', 'date'])

# Step 6: View the data
print(df.head())
print("Shape:", df.shape)
print("Label distribution:\n", df['label'].value_counts())


#-------------------------scikit-learn part------------------#
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Use only the 'text' and 'label' columns
X = df['text']
y = df['label']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data, transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Confirm shapes
print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)

 #-----------------------------prediction ----------------------#
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Initialize model
model = LogisticRegression(max_iter=1000)  # Increased iterations to ensure convergence

# Step 2: Train model
model.fit(X_train_tfidf, y_train)

# Step 3: Predict on test data
y_pred = model.predict(X_test_tfidf)

# Step 4: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
