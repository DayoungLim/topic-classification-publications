import pandas as pd
import sqlite3
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException

import nltk
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = text.lower().strip()
    return text

def remove_stopwords(text, lang='en'):
    tokens = word_tokenize(text)
    try:
        lang_stopwords = stopwords.words(lang)
    except OSError:
        lang_stopwords = stopwords.words('english')
    tokens = [token for token in tokens if token not in lang_stopwords]
    return ' '.join(tokens)

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'en'  

conn = sqlite3.connect('test-works-abstracts.db')
df = pd.read_sql_query("SELECT * FROM limited_abstract", conn)
conn.close()

for index, row in df.iterrows():
    cleaned_text = clean_text(row['work_abstract'])

    lang = detect_language(cleaned_text)

    cleaned_text_no_stopwords = remove_stopwords(cleaned_text, lang)

    df.at[index, 'work_abstract'] = cleaned_text_no_stopwords
    df.at[index, 'language'] = lang

conn_write = sqlite3.connect('test-cleaned-works-with-abstracts-topics-bm25.db')

df.to_sql('cleaned_limited_abstract_bm25', conn_write, if_exists='replace', index=False)

print("Data cleaning complete.")
