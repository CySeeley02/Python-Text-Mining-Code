import requests
import pandas as pd
import os

API_KEY = '5321a6ec84ac49a09ffe07a2919accc5'

def fetch_articles(topic):
    url = f'https://newsapi.org/v2/everything?q={topic}&apiKey={API_KEY}&language=en'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

def articles_to_dataframe(articles):
    df = pd.DataFrame(articles)
    return df[['source', 'author', 'title', 'description', 'url', 'publishedAt', 'content']]

def save_to_csv(df, filename):
    output_dir = r'C:\Users\cyase\OneDrive\Documents\Syracuse Masters\Text Mining'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)

topics = ['baseball', 'football', 'olympics']
for topic in topics:
    articles = fetch_articles(topic)
    raw_df = pd.DataFrame(articles)
    cleaned_df = articles_to_dataframe(articles)
    save_to_csv(raw_df, f'{topic}_articles_raw.csv')
    save_to_csv(cleaned_df, f'{topic}_articles_cleaned.csv')

print("CSV files for raw and cleaned data have been created in the specified directory.")





