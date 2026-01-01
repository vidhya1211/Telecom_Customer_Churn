import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle


model=pickle.load(open("churn_model.pkl", "rb"))
scaler=pickle.load(open("scaler.pkl", "rb"))
feature_names=pickle.load(open("feature_names.pkl", "rb"))

st.title("📊 Telecom Customer Churn Prediction")

with st.sidebar:
    select = option_menu(
        "Main Menu",
        ["Introduction", "Churn Analysis", "Conclusion"],
        icons=["info-circle", "graph-up", "check-circle"],
        menu_icon="cast"
    )
if select =="Introduction":
        st.subheader("📌 Project Overview")
        st.write("""
        This application predicts whether a telecom customer is likely to **churn**.
        
        - Model used: **Logistic Regression**
        - Metric focus: **Recall & ROC-AUC**
        - Deployed using **Streamlit**
        """)
        
        st.image(r"D:\project\telecom.jpg")


if select == "Churn Analysis":

    st.subheader("Churn Prediction")
    st.write("Churn Prediction Details:")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("MonthlyCharges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("TotalCharges", 0.0, 10000.0, 2000.0)

    Contract = st.selectbox("Contract", ["Month to Month", "One year", "Two year"])
    InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])

    # 🔍 Debug: training features
    st.write("TRAINING FEATURES:", feature_names)

    # ✅ Create input skeleton
    input_df = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # ✅ Fill numeric features (ONLY if they exist)
    if "tenure" in feature_names:
        input_df["tenure"] = tenure

    if "MonthlyCharges" in feature_names:
        input_df["MonthlyCharges"] = MonthlyCharges

    if "TotalCharges" in feature_names:
        input_df["TotalCharges"] = TotalCharges

    # ✅ Binary feature
    if "PaperlessBilling" in feature_names:
        input_df["PaperlessBilling"] = 1 if PaperlessBilling == "Yes" else 0

    # ✅ One-hot: Contract
    if Contract == "One year" and "Contract_One year" in feature_names:
        input_df["Contract_One year"] = 1
    elif Contract == "Two year" and "Contract_Two year" in feature_names:
        input_df["Contract_Two year"] = 1
    # Month to Month → baseline

    # ✅ One-hot: Internet Service
    if InternetService == "DSL" and "InternetService_DSL" in feature_names:
        input_df["InternetService_DSL"] = 1
    elif InternetService == "Fiber optic" and "InternetService_Fiber optic" in feature_names:
        input_df["InternetService_Fiber optic"] = 1
    # No → baseline

    # 🔍 Debug: extra columns check
    extra_cols = set(input_df.columns) - set(feature_names)
    st.write("EXTRA COLS:", extra_cols)

    # ✅ Scale AFTER input is fully built
    input_scaled = scaler.transform(input_df)

    threshold = st.slider("Churn Probability Threshold", 0.1, 0.9, 0.5, 0.05)

    if st.button("Predict Churn"):
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = "Churn" if prob >= threshold else "No Churn"

        st.subheader(f"Prediction: {prediction}")
        st.write(f"Churn Probability: **{prob:.2f}**")

        if prediction == "Churn":
            st.warning("⚠️ High-risk customer — recommend retention offer")
        else:
            st.success("✅ Low churn risk customer")
        

if select=="Conclusion":
        st.subheader("✅ Conclusion")
        st.write("""
        - Logistic Regression performed best for **churn recall**
        - Random Forest improved accuracy but reduced recall
        - Streamlit deployment ensures real-time prediction
        - Feature consistency between training & inference is critical
        """)