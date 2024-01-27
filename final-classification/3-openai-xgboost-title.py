import sqlite3
import pandas as pd
from openai import OpenAI
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import ast

client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

conn = sqlite3.connect('cleaned-stratified-samples.db')
cursor = conn.cursor()
cursor.execute("SELECT work_title, cleaned_abstract, work_names FROM cleaned_stratified_sample")  
rows = cursor.fetchall()
conn.close()

column_names = ['work_title', 'cleaned_abstract', 'work_names']
df = pd.DataFrame(rows, columns=column_names)

df['combined_text'] = 'title: ' + df['work_title'].astype(str) + \
                      ' abstract: ' + df['cleaned_abstract'].astype(str)

df['ada_002_embeddings'] = df['combined_text'].apply(lambda x: get_embedding(x))

df['labels'] = df['work_names'].apply(lambda x: x.split(','))

df.to_csv('embedded-data-stratified-samples.csv', index=False)

# df = pd.read_csv('embedded-data-stratified-samples.csv')
# df['ada_002_embeddings'] = df['ada_002_embeddings'].apply(ast.literal_eval)
# df['labels'] = df['work_names'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])

X = pd.DataFrame(df['ada_002_embeddings'].tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

model = OneVsRestClassifier(xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', eta=0.8, n_estimators=1000, max_depth=20))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
hamming = hamming_loss(y_test, y_pred)

report_dict = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

report_df.loc['Weighted F1 Score'] = [weighted_f1, np.nan, np.nan, np.nan]
report_df.loc['Accuracy'] = [accuracy, np.nan, np.nan, np.nan]
report_df.loc['Hamming Loss'] = [hamming, np.nan, np.nan, np.nan]

report_df.to_csv('evaluation-metrics-stratified-samples.csv', index=True, index_label='Label')

print("Classification Report without Threshold Logic:")
print(report_df)

predicted_labels = mlb.inverse_transform(y_pred)
actual_labels = mlb.inverse_transform(y_test)

test_df = X_test.copy()
test_df['Actual Labels'] = ['; '.join(labels) for labels in actual_labels]
test_df['Predicted Labels'] = ['; '.join(labels) for labels in predicted_labels]
# test_df = test_df.merge(df[['cleaned_abstract']], left_index=True, right_index=True, how='left')

test_df.to_csv('data-with-predictions-stratified-samples.csv', index=False)
