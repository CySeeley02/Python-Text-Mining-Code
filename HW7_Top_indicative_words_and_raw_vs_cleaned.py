import os
import pandas as pd
import numpy as np
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string

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

# Function to truncate text to fit within table cells
def truncate_text(text, max_length=100):
    if len(text) > max_length:
        return text[:max_length] + '...'  # Truncate and add ellipsis
    return text

df['truncated_review'] = df['review'].apply(lambda x: truncate_text(x, max_length=100))
df['truncated_cleaned_review'] = df['cleaned_review'].apply(lambda x: truncate_text(x, max_length=100))

# Display sample raw data with truncated text
print("Displaying sample raw data...")
sample_raw_df = df[['truncated_review', 'sentiment']].head(5)

# Configure figure and axis for displaying raw data
fig, ax = plt.subplots(figsize=(12, 2))  # Increase figure size for better visibility
ax.axis('off')
ax.axis('tight')

# Create table for raw data sample
table_raw = ax.table(
    cellText=sample_raw_df.values,
    colLabels=sample_raw_df.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.8, 0.2]  # Adjust column widths for better readability
)

# Adjust font size for raw data table
table_raw.auto_set_font_size(False)
table_raw.set_fontsize(10)
plt.title('Sample Raw Reviews Data', fontsize=14)
plt.show()

# Display sample cleaned data with truncated text
print("Displaying sample cleaned data...")
sample_cleaned_df = df[['truncated_cleaned_review', 'sentiment']].head(5)

# Configure figure and axis for displaying cleaned data
fig, ax = plt.subplots(figsize=(12, 2))  # Increase figure size for better visibility
ax.axis('off')
ax.axis('tight')

# Create table for cleaned data sample
table_cleaned = ax.table(
    cellText=sample_cleaned_df.values,
    colLabels=sample_cleaned_df.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.8, 0.2]  # Adjust column widths for better readability
)

# Adjust font size for cleaned data table
table_cleaned.auto_set_font_size(False)
table_cleaned.set_fontsize(10)
plt.title('Sample Cleaned Reviews Data', fontsize=14)
plt.show()

# Feature Extraction using CountVectorizer
print("Extracting features...")
vectorizer = CountVectorizer(max_features=10000)  # Limit the number of features for SVM efficiency
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment'].map({'pos': 1, 'neg': 0})

# Split the Data into Training and Testing Sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Multinomial Naive Bayes model
print("Training Multinomial Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Get log probabilities for positive and negative classes
log_prob_pos = nb_model.feature_log_prob_[1]  # Positive class
log_prob_neg = nb_model.feature_log_prob_[0]  # Negative class

# Find the top 10 indicative words for each class
top_pos_indices = np.argsort(log_prob_pos)[-10:]  # Top 10 for positive
top_neg_indices = np.argsort(log_prob_neg)[-10:]  # Top 10 for negative

top_pos_words = feature_names[top_pos_indices]
top_neg_words = feature_names[top_neg_indices]

top_pos_log_probs = log_prob_pos[top_pos_indices]
top_neg_log_probs = log_prob_neg[top_neg_indices]

# Plot top 10 indicative words for positive reviews
plt.figure(figsize=(12, 6))
sns.barplot(x=top_pos_log_probs, y=top_pos_words, palette="Blues_d")
plt.title('Top 10 Indicative Words for Positive Reviews')
plt.xlabel('Log Probability')
plt.ylabel('Word')
plt.show()

# Plot top 10 indicative words for negative reviews
plt.figure(figsize=(12, 6))
sns.barplot(x=top_neg_log_probs, y=top_neg_words, palette="Reds_d")
plt.title('Top 10 Indicative Words for Negative Reviews')
plt.xlabel('Log Probability')
plt.ylabel('Word')
plt.show()

# Raw vs. Cleaned Data Comparison

# Create word frequency dataframes for raw and cleaned text
def get_word_freq(text_series):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_series)
    word_freq = np.asarray(X.sum(axis=0)).flatten()
    word_freq_df = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 'frequency': word_freq})
    return word_freq_df.sort_values(by='frequency', ascending=False).head(10)

raw_word_freq_df = get_word_freq(df['review'])
cleaned_word_freq_df = get_word_freq(df['cleaned_review'])

# Plot word frequencies for raw reviews
plt.figure(figsize=(12, 6))
sns.barplot(x='frequency', y='word', data=raw_word_freq_df, palette="Greens_d")
plt.title('Top 10 Words in Raw Reviews')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()

# Plot word frequencies for cleaned reviews
plt.figure(figsize=(12, 6))
sns.barplot(x='frequency', y='word', data=cleaned_word_freq_df, palette="Purples_d")
plt.title('Top 10 Words in Cleaned Reviews')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()




