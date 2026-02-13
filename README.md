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

