# predictive_modeling_titanic.py
# Run: python predictive_modeling_titanic.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# 1) Load dataset
def load_data():
    # Option A: local CSV 'titanic.csv' or 'train.csv' in current folder
    for fn in ['titanic.csv', 'train.csv', 'titanic_train.csv']:
        if os.path.exists(fn):
            print(f"Loading local file: {fn}")
            return pd.read_csv(fn)
    # Option B: try seaborn built-in (requires internet)
    try:
        import seaborn as sns
        print("Loading seaborn 'titanic' dataset")
        return sns.load_dataset('titanic')
    except Exception as e:
        raise RuntimeError("No local CSV found and seaborn load failed (no network). Please provide CSV 'titanic.csv'") from e

df = load_data()

# 2) Quick EDA (print)
print("Dataset shape:", df.shape)
print(df.columns.tolist())
print(df.head())

# 3) Feature engineering
df = df.copy()
# Some CSVs include 'Survived' capitalized, handle both
if 'survived' not in df.columns and 'Survived' in df.columns:
    df['survived'] = df['Survived']
# Create FamilySize, IsAlone
df['FamilySize'] = df.get('sibsp', 0).fillna(0) + df.get('parch', 0).fillna(0) + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Fare and Age bins (handle missing)
df['fare'] = df['fare'].fillna(df['fare'].median()) if 'fare' in df.columns else 0
df['age'] = df['age'].fillna(df['age'].median()) if 'age' in df.columns else 0
df['FareBin'] = pd.qcut(df['fare'], 4, labels=False, duplicates='drop')
df['AgeBin'] = pd.cut(df['age'], bins=[0,12,20,40,60,120], labels=False, include_lowest=True)

# Choose features (modify if CSV column names differ)
features = []
for candidate in ['pclass','sex','age','sibsp','parch','fare','embarked','who','adult_male','deck','alone']:
    if candidate in df.columns:
        features.append(candidate)
# add engineered features
features += ['FamilySize','IsAlone','FareBin','AgeBin']
# Ensure target exists
if 'survived' not in df.columns:
    raise RuntimeError("Target column 'survived' not found in data. Rename target to 'survived' (0/1).")

X = df[features].copy()
y = df['survived'].astype(int).copy()

print("Using features:", features)

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5) Preprocessing pipelines
numeric_features = [c for c in ['age','sibsp','parch','fare','FamilySize','FareBin','AgeBin'] if c in X.columns]
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_features = [c for c in ['pclass','sex','embarked','who','adult_male','deck','IsAlone'] if c in X.columns]
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 6) Model pipelines
lr_pipe = Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
rf_pipe = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))])

# 7) Fit baseline models
print("Training Logistic Regression...")
lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)

print("Training Random Forest...")
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)

def metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

m_lr = metrics(y_test, y_pred_lr)
m_rf = metrics(y_test, y_pred_rf)
print("Logistic Regression metrics:", m_lr)
print("Random Forest (baseline) metrics:", m_rf)

# 8) GridSearchCV (Random Forest tuning)
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}
grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
print("Running GridSearchCV for Random Forest (this may take time)...")
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
print("Best RF params:", grid.best_params_)
y_pred_best = best_rf.predict(X_test)
m_best_rf = metrics(y_test, y_pred_best)
print("Tuned Random Forest metrics:", m_best_rf)

# 9) Feature importance extraction
# get numeric names
num_feats = numeric_features
# get onehot names
ohe = best_rf.named_steps['pre'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = list(num_feats) + cat_feature_names
importances = best_rf.named_steps['clf'].feature_importances_
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
print("Top features:\n", imp_series)

# 10) Plots (save to current directory)
os.makedirs('titanic_artifacts', exist_ok=True)
plt.figure(figsize=(8,4))
imp_series.plot(kind='barh')
plt.title('Top feature importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('titanic_artifacts/feature_importances.png')
plt.close()

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix - Tuned RF')
plt.savefig('titanic_artifacts/confusion_matrix.png')
plt.close()

# 11) Save artifacts: model, processed data, metrics
joblib.dump(best_rf, 'titanic_artifacts/best_rf_pipeline.joblib')
X.assign(survived=y).to_csv('titanic_artifacts/titanic_processed.csv', index=False)
with open('titanic_artifacts/report_summary.txt','w') as f:
    f.write("Logistic Regression metrics:\n"+str(m_lr)+"\n\n")
    f.write("Random Forest baseline metrics:\n"+str(m_rf)+"\n\n")
    f.write("Random Forest (tuned) metrics:\n"+str(m_best_rf)+"\n\n")
    f.write("Best params:\n"+str(grid.best_params_)+"\n")

print("Artifacts saved to folder: titanic_artifacts/")