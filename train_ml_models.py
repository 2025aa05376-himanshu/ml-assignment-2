#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
os.makedirs("model", exist_ok=True)


# In[20]:


import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]

df = pd.read_csv(
    "adult.data",
    header=None,
    names=columns,
    na_values=" ?",
    skipinitialspace=True
)

df.to_csv("adult.csv", index=False)

print("CSV file created successfully!")


# In[21]:


import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# In[22]:


# Load dataset
df = pd.read_csv("adult.csv")

# Handle missing values
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Clean target
df["income"] = df["income"].str.strip()

# Map directly to 0/1
df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

X = df.drop("income", axis=1)
y = df["income"]

# # Encode target
# le = LabelEncoder()
# y = le.fit_transform(y)

# joblib.dump(le, "model/label_encoder.pkl")

# Identify columns
num_cols = X.select_dtypes(include=["int64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns


# In[23]:


# Preprocessing (IMPORTANT: Dense output)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[24]:


# Fit and transform
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

joblib.dump(preprocessor, "model/preprocessor.pkl")

# Define models
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost": XGBClassifier(eval_metric="logloss")
}

results = []

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)
    y_prob = model.predict_proba(X_test_p)[:, 1]

    Accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_prob)
    Precision = precision_score(y_test, y_pred)
    Recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    MCC = matthews_corrcoef(y_test, y_pred)

    results.append([name, Accuracy, AUC, Precision, Recall, F1, MCC])

    joblib.dump(model, f"model/{name}.pkl")

# Show evaluation table
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "AUC", "Precision", "Recall", "F1-Score", "MCC"
])
print("\nModel Performance:")
print(results_df)


# In[ ]:





# In[ ]:




