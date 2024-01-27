import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import ast

df = pd.read_csv('embedded_data_abstract.csv')
df['ada_002_embeddings'] = df['ada_002_embeddings'].apply(ast.literal_eval)
df['labels'] = df['work_names'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])

X = pd.DataFrame(df['ada_002_embeddings'].tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'estimator__max_depth': [3, 6, 16, 20],
    'estimator__n_estimators': [50, 100, 500, 1000],
    'estimator__eta':[0.0001, 0.3, 0.5, 0.8]
}

model = OneVsRestClassifier(xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'))

grid_search = GridSearchCV(model, param_grid, scoring='f1_weighted', cv=3, verbose=3)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
accuracy = accuracy_score(y_test, y_pred)
hamming = hamming_loss(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Hamming Loss: {hamming:.2f}")

predicted_labels = mlb.inverse_transform(y_pred)
actual_labels = mlb.inverse_transform(y_test)

test_df = X_test.copy()
test_df['Actual Labels'] = ['; '.join(labels) for labels in actual_labels]
test_df['Predicted Labels'] = ['; '.join(labels) for labels in predicted_labels]

test_df.to_csv('test_data_with_predictions_grid_search.csv', index=False)
