import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import joblib
import os

df = pd.read_csv("customer_satisfaction_sample.csv")

df['satisfied'] = (df['satisfaction_rating'] >= 4).astype(int)
target = 'satisfied'
drop_cols = ['customer_id', 'satisfaction_rating']
df = df.drop(columns=drop_cols)

numerics = df.select_dtypes(include=['int64','float64']).columns.drop(target).tolist()
categoricals = df.select_dtypes(include=['object','category']).columns.tolist()

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_pipe, numerics), ('cat', cat_pipe, categoricals)])

model = Pipeline([
    ('pre', preprocessor),
    ('clf', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss'))
])

X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
print("CV ROC-AUC mean:", cv_scores.mean())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/customer_sat_pipeline.joblib")