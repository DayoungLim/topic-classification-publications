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

conn_read = sqlite3.connect('stratified-samples.db')

chunk_size = 10000  

conn_write = sqlite3.connect('cleaned-stratified-samples.db')

for chunk in pd.read_sql_query("SELECT * FROM stratified_sample_agg", conn_read, chunksize=chunk_size):
    chunk['cleaned_abstract'] = chunk['work_abstract'].apply(clean_abstract)

    chunk = chunk[chunk['cleaned_abstract'].str.strip().apply(lambda x: len(x.split()) > 1)]

    chunk.to_sql('cleaned_stratified_sample', conn_write, if_exists='append', index=False)

# Close both connections
conn_read.close()
conn_write.close()
print("Data cleaning complete.")
