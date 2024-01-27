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
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

conn = sqlite3.connect('test-cleaned-abstracts.db')
cursor = conn.cursor()

cursor.execute("SELECT work_abstract, work_names FROM cleaned_limited_abstract")
rows = cursor.fetchall()

conn.close()

try:
    existing_df = pd.read_csv('existing_embeddings.csv')
except FileNotFoundError:
    existing_df = pd.DataFrame()

new_rows = rows[len(existing_df):]

if new_rows:
    new_df = pd.DataFrame(new_rows, columns=['work_abstract', 'work_names'])
    new_df['ada_002_embeddings'] = new_df['work_abstract'].apply(lambda x: get_embedding(x))
    new_df['labels'] = new_df['work_names'].apply(lambda x: x.split(','))
    df = pd.concat([existing_df, new_df])
else:
    df = existing_df

df.to_csv('embedded_data_abstract.csv', index=False)

# df = pd.read_csv('embedded_data_abstract.csv')
# df['ada_002_embeddings'] = df['ada_002_embeddings'].apply(ast.literal_eval)
# df['labels'] = df['work_names'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])

X = pd.DataFrame(df['ada_002_embeddings'].tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = OneVsRestClassifier(xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', eta=0.5, n_estimators=500, max_depth=6))
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

report_df.to_csv('evaluation_metrics.csv', index=True, index_label='Label')

print("Classification Report with Custom Logic:")
print(report_df)

predicted_labels = mlb.inverse_transform(y_pred)
actual_labels = mlb.inverse_transform(y_test)

test_df = X_test.copy()
test_df['Actual Labels'] = ['; '.join(labels) for labels in actual_labels]
test_df['Predicted Labels'] = ['; '.join(labels) for labels in predicted_labels]

test_df.to_csv('test_data_with_predictions_abstract.csv', index=False)

