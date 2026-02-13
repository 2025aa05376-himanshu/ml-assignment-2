Machine Learning Assignment 2

BITS Pilani – WILP M.Tech (AIML - 2025aa05376)

1. Problem Statement

	The objective of this project is to develop and compare multiple machine learning classification models to determine whether an individual's annual income exceeds $50,000 using demographic and employment-		related features from the U.S. Census dataset.

2. Dataset Description

	Dataset Name: Adult Income Dataset
	Source: UCI Machine Learning Repository
	Number of Instances: 48,842
	Number of Features: 14 input attributes
	Target Variable: Income (<=50K, >50K)
	Type of Problem: Binary Classification

	The dataset contains both numerical and categorical attributes describing an individual's education level, occupation, working hours, marital status, and other socio-economic characteristics.

3. Data Preprocessing Steps
   
	Removed rows containing missing values
	Converted target variable into binary format (0 and 1)
	Applied One-Hot Encoding to categorical features
	Standardized numerical features
	Split dataset into 80% training and 20% testing sets

4. Machine Learning Models Implemented
   
   The following six classification algorithms were implemented:
		Logistic Regression
		Decision Tree Classifier
		K-Nearest Neighbors
		Gaussian Naive Bayes
		Random Forest (Ensemble)
		XGBoost (Ensemble Boosting)

5. Evaluation Metrics

   Each model was evaluated using:
		Accuracy
		AUC Score
		Precision
		Recall
		F1 Score
		Matthews Correlation Coefficient (MCC)

6. Model Comparison Table

	Model					Accuracy	AUC	Precision	Recall	F1	MCC
	Logistic Regression						
	Decision Tree						
	KNN						
	Naive Bayes						
	Random Forest						
	XGBoost	
					
7. Observations
    
	Logistic Regression provided a strong baseline performance.
	Decision Tree showed moderate accuracy but potential overfitting.
	KNN performed well after feature scaling.
	Naive Bayes was computationally efficient but slightly lower in predictive power.
	Random Forest improved overall stability and generalization.
	XGBoost achieved the best performance across most evaluation metrics.

8. Streamlit Application Features
    
   The deployed web application includes:
		CSV test dataset upload
		Model selection dropdown
		Display of evaluation metrics
		Confusion matrix visualization
		Classification report

9. Deployment

The application has been deployed using Streamlit Community Cloud.

🔥 STEP 1 — Train the Model Locally

Inside your project folder:

ml-assignment-2/
│
├── train_ml_models.py
├── streamlit_app.py
├── adult.csv
├── model/

▶ Run Training Script

Open terminal inside project folder:

cd C:\Users\himanshu\Documents\ML\ml-assignment-2
python train_ml_models.py

If successful, it will generate:

model/
│
├── logistic.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
├── random_forest.pkl
├── xgboost.pkl
├── preprocessor.pkl

If these files exist → ✅ Training done.

🔥 STEP 2 — Test App Locally

Before pushing to GitHub, ALWAYS test locally:

streamlit run streamlit_app.py

If it opens in browser and works → ✅ Ready for deployment.

🔥 STEP 3 — Prepare GitHub Repository

Make sure your folder contains:

streamlit_app.py
train_ml_models.py
requirements.txt
model/ (with all .pkl files)

✅ Create requirements.txt

If not created:

pip freeze > requirements.txt

OR manually create:

streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
xgboost

🔥 STEP 4 — Push to GitHub

If first time:

git add .
git commit -m "Final ML project"
git push origin main


If you get pull error:

git pull origin main --allow-unrelated-histories
git push


After push, check your repo online.

You must see:

model/
streamlit_app.py
requirements.txt

🚀 STEP 5 — Deploy on Streamlit Cloud

Now the IMPORTANT part.

🌐 Go To:

👉 https://share.streamlit.io

Login with GitHub.

Click:

New App

Fill Details:

Repository: ml-assignment-2

Branch: main

Main file path: streamlit_app.py

Click Deploy

