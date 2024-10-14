import os
import pandas as pd
import numpy as np
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

# File paths for positive and negative review zip files
pos_zip_path = r"C:\Users\cyase\OneDrive\Documents\Syracuse Masters\Text Mining\pos-20240820T011722Z-001.zip"
neg_zip_path = r"C:\Users\cyase\OneDrive\Documents\Syracuse Masters\Text Mining\neg-20240820T011834Z-001.zip"

# Function to load reviews from a zip file
def load_reviews_from_zip(zip_path, label):
    reviews = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for idx, filename in enumerate(zip_ref.namelist()):
            with zip_ref.open(filename) as file:
                reviews.append({'review': file.read().decode('utf-8'), 'sentiment': label})
            # Print progress
            if idx % 100 == 0:
                print(f"Processed {idx+1} files from {zip_path}")
    return reviews

# Load positive and negative reviews from zip files
print("Loading positive reviews...")
pos_reviews = load_reviews_from_zip(pos_zip_path, 'pos')
print("Loading negative reviews...")
neg_reviews = load_reviews_from_zip(neg_zip_path, 'neg')

# Combine the datasets into a single DataFrame
df = pd.DataFrame(pos_reviews + neg_reviews)

# Text Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

print("Preprocessing text...")
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Generate Word Clouds for positive and negative reviews
print("Generating word clouds...")

# Positive reviews word cloud
pos_text = ' '.join(df[df['sentiment'] == 'pos']['cleaned_review'])
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Reviews')
plt.show()

# Negative reviews word cloud
neg_text = ' '.join(df[df['sentiment'] == 'neg']['cleaned_review'])
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(neg_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Negative Reviews')
plt.show()

# Feature Extraction using CountVectorizer
print("Extracting features...")
vectorizer = CountVectorizer(max_features=10000)  # Limit the number of features for SVM efficiency
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment'].map({'pos': 1, 'neg': 0})

# Split the Data into Training and Testing Sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# List to hold models, their names, and their predicted results
models = []
model_names = []
model_predictions = []

# Train a Multinomial Naive Bayes model
print("Training Multinomial Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

models.append(nb_model)
model_names.append("Multinomial Naive Bayes")
model_predictions.append(nb_preds)

# Train Linear SVM model using LinearSVC
print(f"Training Linear SVM...")
linear_svm_model = LinearSVC(max_iter=1000, verbose=1)  # Added max_iter and verbose
linear_svm_model.fit(X_train, y_train)
linear_svm_preds = linear_svm_model.predict(X_test)

models.append(linear_svm_model)
model_names.append("Linear SVM")
model_predictions.append(linear_svm_preds)

# Standard Scaler for SVM models
scaler = StandardScaler(with_mean=False)  # with_mean=False because we are working with sparse matrices

# Pipeline for Sigmoid kernel
print("Optimizing SVM with Sigmoid kernel...")
pipe_sigmoid = Pipeline([
    ('scaler', scaler),
    ('svc', SVC(kernel='sigmoid', max_iter=1000))
])
param_grid_sigmoid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto']
}
grid_sigmoid = GridSearchCV(pipe_sigmoid, param_grid_sigmoid, cv=3, verbose=1, n_jobs=-1)
grid_sigmoid.fit(X_train, y_train)
best_sigmoid_model = grid_sigmoid.best_estimator_
sigmoid_preds = best_sigmoid_model.predict(X_test)

models.append(best_sigmoid_model)
model_names.append("Optimized SVM with Sigmoid Kernel (Scaled)")
model_predictions.append(sigmoid_preds)

# Pipeline for RBF kernel
print("Optimizing SVM with RBF kernel...")
pipe_rbf = Pipeline([
    ('scaler', scaler),
    ('svc', SVC(kernel='rbf', max_iter=1000))
])
param_grid_rbf = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto', 0.1, 0.01]
}
grid_rbf = GridSearchCV(pipe_rbf, param_grid_rbf, cv=3, verbose=1, n_jobs=-1)
grid_rbf.fit(X_train, y_train)
best_rbf_model = grid_rbf.best_estimator_
rbf_preds = best_rbf_model.predict(X_test)

models.append(best_rbf_model)
model_names.append("Optimized SVM with RBF Kernel (Scaled)")
model_predictions.append(rbf_preds)

# Generate Confusion Matrices and Classification Reports
for model_name, preds in zip(model_names, model_predictions):
    cm = confusion_matrix(y_test, preds)
    print(f"{model_name} Confusion Matrix:")
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, preds))

# Compare Accuracies
accuracies = [accuracy_score(y_test, preds) for preds in model_predictions]

# Accuracy Comparison Visualization
plt.figure(figsize=(12, 6))  # Increase figure size for better label visibility
sns.barplot(x=model_names, y=accuracies)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45, ha='right')  # Rotate labels and align them to the right
plt.tight_layout()  # Adjust layout to ensure labels are not cut off
plt.show()





