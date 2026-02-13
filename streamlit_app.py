#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ===============================
# Page Config
# ===============================

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Machine Learning Classification Models")
st.markdown("### UCI Adult Income Dataset")

# Load models
models = {
    "Logistic Regression": joblib.load("model/logistic.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

preprocessor = joblib.load("model/preprocessor.pkl")
# le = joblib.load("model/label_encoder.pkl")

model_option = st.selectbox("Select Model", list(models.keys()))

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "income" in df.columns:
        y_true = df["income"].str.strip().map({
            "<=50K": 0,
            ">50K": 1
        })
        # y_true = le.transform(df["income"])
        X = df.drop("income", axis=1)
    else:
        X = df
        y_true = None

    X_processed = preprocessor.transform(X)

    model = models[model_option]
    y_pred = model.predict(X_processed)

    # predictions = le.inverse_transform(y_pred)

    # st.subheader("Predictions")
    # st.write(predictions)

    # if hasattr(model, "predict_proba"):
    #     y_prob = model.predict_proba(X)[:, 1]
    # else:
    #     y_prob = None

    if y_true is not None:
        st.subheader("Evaluation Metrics")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        # if y_prob is not None:
        #         auc = roc_auc_score(y_true, y_prob)
        # else:
        #         auc = "N/A"

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"],
            "Score": [accuracy, precision, recall, f1, mcc]
        })

        st.table(metrics_df)

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        ax.matshow(cm)

        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j])

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))

