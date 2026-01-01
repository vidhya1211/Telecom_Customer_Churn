Telecom Customer Churn

Project Overview

This project predicts whether a telecom customer will churn or not using machine learning.

The goal is to help telecom companies identify high-risk customers and reduce customer loss.

Problem Statement

Predict customer churn (Yes / No) based on customer usage, contract, and billing details.

Dataset

•	Source: Kaggle – Telco Customer Churn Dataset

•	Target column: Churn

Important Features

•	Tenure

•	Monthly Charges

•	Total Charges

•	Contract Type

•	Internet Service

•	Paperless Billing

Data Preprocessing

•	Removed unnecessary columns

•	Converted TotalCharges to numeric

•	Filled missing values

•	Encoded categorical variables

•	Scaled numerical features

Model Used

•	Logistic Regression (final model)

•	Random Forest (for comparison)

Logistic Regression was chosen for better churn recall.


Model Performance

•	Accuracy: ~74%

•	ROC-AUC: ~0.75

•	Recall (Churn): ~79%

Threshold Selection

•	Default threshold (0.5) was not used

•	Final threshold fixed at 0.35

•	This helps catch more churn customers

Deployment

•	Built a Streamlit web app

•	User enters customer details

•	App shows churn probability and prediction

Tools & Technologies

•	Python

•	Pandas, NumPy

•	Scikit-learn

•	Streamlit

How to Run

pip install -r requirements.txt

streamlit run Telecust.py

Conclusion

This project shows a complete machine learning workflow from data preprocessing to deployment with business-focused decisions.

Author

Vidhya Ulaganathan
