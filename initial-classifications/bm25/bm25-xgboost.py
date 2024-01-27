import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, f1_score
from sklearn.multiclass import OneVsRestClassifier

bm25_df = pd.read_csv('bm25_features.csv')

mlb = joblib.load('mlb.joblib')
y = pd.read_csv('encoded_labels.csv').values

X_train, X_test, y_train, y_test = train_test_split(bm25_df, y, test_size=0.2, random_state=42)

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

report_df.to_csv('evaluation_metrics_bm25.csv', index=True, index_label='Label')

print("Classification Report with Custom Logic:")
print(report_df)

predicted_labels = mlb.inverse_transform(y_pred)
actual_labels = mlb.inverse_transform(y_test)

test_df = X_test.copy()
test_df['Actual Labels'] = ['; '.join(labels) for labels in actual_labels]
test_df['Predicted Labels'] = ['; '.join(labels) for labels in predicted_labels]

test_df.to_csv('test_data_with_predictions_xgboost_bm25.csv', index=False)