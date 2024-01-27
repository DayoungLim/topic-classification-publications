import pandas as pd
import sqlite3
import re
import html


def clean_abstract(abstract):
    # Decode HTML entities
    abstract = html.unescape(abstract)
    
    # Remove instances of the word "abstract"
    abstract = re.sub(r'\babstract[.:]?', '', abstract, flags=re.IGNORECASE)
    
    # Remove HTML tags
    abstract = re.sub(r'<.*?>', '', abstract)
    
    # Remove URLs
    abstract = re.sub(r'https?://\S+|www\.\S+', '', abstract)
    
    # Remove LaTeX artefacts
    abstract = re.sub(r'\{\\.*?\}', '', abstract)
    abstract = re.sub(r'\\[a-zA-Z]+', '', abstract)

    # Handle spaces and newlines
    abstract = abstract.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    abstract = re.sub(r'\s+', ' ', abstract)
    abstract = abstract.strip()

    return abstract


conn_read = sqlite3.connect('test-works-abstracts.db')
df = pd.read_sql_query("SELECT * FROM limited_abstract", conn_read)
conn_read.close()

df['cleaned_abstract'] = df['work_abstract'].apply(clean_abstract)

df = df[df['cleaned_abstract'].str.strip().apply(lambda x: len(x.split()) > 1)]

conn_write = sqlite3.connect('test-cleaned-abstracts.db')

df.to_sql('cleaned_limited_abstract', conn_write, if_exists='replace', index=False)

conn_write.close()

print("Data cleaning complete.")
