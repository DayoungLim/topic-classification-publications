import tiktoken
import sqlite3
import pandas as pd
from openai import OpenAI

client = OpenAI()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

conn = sqlite3.connect('cleaned-stratified-samples.db')
cursor = conn.cursor()
cursor.execute("SELECT work_title, cleaned_abstract, work_names FROM cleaned_stratified_sample")  
rows = cursor.fetchall()
conn.close()

column_names = ['work_title', 'cleaned_abstract', 'work_names']
df = pd.DataFrame(rows, columns=column_names)

df['combined_text'] = 'title: ' + df['work_title'].astype(str) + \
                      ' abstract: ' + df['cleaned_abstract'].astype(str)

df['num_tokens'] = df['combined_text'].apply(lambda x: num_tokens_from_string(x, "cl100k_base"))

df.to_csv('token_counts.csv', index=False)