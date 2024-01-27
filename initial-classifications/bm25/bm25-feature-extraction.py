import sqlite3
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

conn = sqlite3.connect('test-cleaned-works-with-abstracts-topics-bm25.db')

df = pd.read_sql_query("SELECT work_abstract, work_names FROM cleaned_limited_abstract_bm25", conn)

conn.close()
df['labels'] = df['work_names'].apply(lambda x: x.split(','))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])
joblib.dump(mlb, 'mlb.joblib')

pd.DataFrame(y).to_csv('bm25_encoded_labels.csv', index=False)

tokenized_corpus = [doc.split(" ") for doc in df['work_abstract'].tolist()]

bm25 = BM25Okapi(tokenized_corpus)

bm25_scores = [bm25.get_scores(doc) for doc in tokenized_corpus]

bm25_df = pd.DataFrame(bm25_scores)

bm25_df.to_csv('bm25_features.csv', index=False)
print("done bm25")