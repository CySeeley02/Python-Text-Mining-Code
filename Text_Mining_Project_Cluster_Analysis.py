import re  # regular expressions
import pandas as pd  # for dataframes
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Use the provided paths
input_filepath = r"C:\Users\cyase\OneDrive\Documents\Syracuse Masters\Text Mining\emotions_full.csv"
output_basepath = r"C:\Users\cyase\OneDrive\Documents\Syracuse Masters\Text Mining"

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(input_filepath, usecols=[1, 2])
labels = df['label']
texts = df['text']

# Data cleaning
print("Cleaning data...")
texts = texts.str.replace(r'http\S+' or r'http[s]?://\S+', '', regex=True)  # Remove URLs
texts = texts.str.replace(r'\b\w*\d+\w*\b', '', regex=True)  # Remove numerals and words with numerals
texts = texts.str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove punctuation and non-alphabetical characters

# Exploratory Analysis: Label Distribution
print("Visualizing label distribution...")
plt.figure(figsize=(8, 6))
sns.countplot(x=labels)
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Exploratory Analysis: Text Length Distribution
print("Visualizing text length distribution...")
text_lengths = texts.apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(text_lengths, bins=50, kde=True)
plt.title("Text Length Distribution")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Vectorization for further EDA
print("Vectorizing text data...")
MyCountV = CountVectorizer(input="content", lowercase=True, stop_words="english", max_features=5000)  # Reduced max_features
tokens = MyCountV.fit_transform(texts)
column_names = MyCountV.get_feature_names_out()

# Exploratory Analysis: Most Frequent Words
print("Visualizing most frequent words...")
word_freq = pd.DataFrame(tokens.toarray(), columns=column_names).sum().sort_values(ascending=False).head(20)
plt.figure(figsize=(10, 6))
sns.barplot(x=word_freq.values, y=word_freq.index)
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.tight_layout()
plt.show()

# Exploratory Analysis: Word Cloud
print("Generating word cloud...")
wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(texts))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Word Cloud of Text Data")
plt.axis('off')
plt.tight_layout()
plt.show()

# Prepare data for cluster analysis and PCA visualization (reusing the optimized code)
print("Applying TF-IDF transformation...")
tfidf_transformer = TfidfTransformer()
df_tfidf_sparse = tfidf_transformer.fit_transform(tokens)

# Split data for training and testing
print("Splitting data into train and test sets...")
labels_array = labels.values
df_tfidf_train, df_tfidf_test, labels_train, labels_test, texts_train, texts_test = train_test_split(
    df_tfidf_sparse, labels_array, texts, test_size=0.2, random_state=42, stratify=labels_array
)

# Perform cluster analysis
print("Performing cluster analysis...")
kmeans = MiniBatchKMeans(n_clusters=5, n_init='auto', random_state=42)
kmeans.fit(df_tfidf_train)

# Add cluster labels to a DataFrame for visualization
print("Preparing data for visualization...")
df_clusters = pd.DataFrame({
    'text': texts_train.values,
    'label': labels_train,
    'cluster': kmeans.labels_
})

# Perform TruncatedSVD for visualization
print("Performing TruncatedSVD for visualization...")
svd = TruncatedSVD(n_components=2, random_state=42)
svd_result = svd.fit_transform(df_tfidf_train)

df_clusters['svd1'] = svd_result[:, 0]
df_clusters['svd2'] = svd_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='svd1', y='svd2', hue='cluster', data=df_clusters, palette='Set1', alpha=0.6)
plt.title("Truncated SVD Visualization of Clusters")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.legend()
plt.show()

print("Task completed.")








