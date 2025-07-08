import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Load the trained model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Teleco-Customer-Churn.h5')

model = load_model()

# --- Load the pre-fitted scaler ---
@st.cache_resource
def load_scaler():
    return joblib.load('Scaling.pkl')

loaded_Scale = load_scaler()

# --- Label Encoders used during training ---
gender_le = LabelEncoder().fit(["Female", "Male"])
partner_le = LabelEncoder().fit(["No", "Yes"])
dependents_le = LabelEncoder().fit(["No", "Yes"])
phone_le = LabelEncoder().fit(["No", "Yes"])
internet_le = LabelEncoder().fit(["DSL", "Fiber optic", "No"])

# --- Streamlit UI ---
st.title("📉 Telco Customer Churn Prediction")
st.write("Fill in the customer details to predict churn probability.")

# --- Input fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

# --- Preprocessing ---
def preprocess_input():
    df = pd.DataFrame([[gender, senior, partner, dependents, tenure,
                        phone_service, internet_service, monthly_charges, total_charges]],
                      columns=["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                               "PhoneService", "InternetService", "MonthlyCharges", "TotalCharges"])

    # Label Encoding
    df["gender"] = gender_le.transform(df["gender"])
    df["Partner"] = partner_le.transform(df["Partner"])
    df["Dependents"] = dependents_le.transform(df["Dependents"])
    df["PhoneService"] = phone_le.transform(df["PhoneService"])
    df["InternetService"] = internet_le.transform(df["InternetService"])

    # Scale numeric columns
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    df_numeric = df[numeric_cols].copy()
    df_scaled_numeric = pd.DataFrame(loaded_Scale.transform(df_numeric), columns=numeric_cols)

    # Rebuild DataFrame in original order:
    df_final = pd.concat([
        df[["gender"]].reset_index(drop=True),
        df_scaled_numeric[["SeniorCitizen"]].reset_index(drop=True),
        df[["Partner", "Dependents", "PhoneService", "InternetService"]].reset_index(drop=True),
        df_scaled_numeric[["tenure", "MonthlyCharges", "TotalCharges"]].reset_index(drop=True)
    ], axis=1)[[
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "InternetService", "MonthlyCharges", "TotalCharges"
    ]]

    return df_final
# 0	1.0	0	0	0.402778	1	0	0.400995	0.196186
# --- Predict ---
if st.button("Predict Churn"):
    input_data = preprocess_input()
    print('Input_data :::',input_data)
    prediction_prob = model.predict(input_data)[0][0]
    print(prediction_prob)
    prediction = "Churn" if prediction_prob > 0.5 else "No Churn"
    print(prediction)

    st.subheader("📊 Prediction Result:")
    if prediction == "Churn":
        st.error(f"⚠️ The customer is likely to CHURN (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"✅ The customer is NOT likely to churn (Probability: {prediction_prob:.2f})")















# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
# import joblib  # Added import for joblib

# # --- Load the saved model ---
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model('Teleco-Customer-Churn.h5')

# model = load_model()

# # --- Load the saved scaler ---
# @st.cache_resource
# def load_scaler():
#     return joblib.load('Scaling.pkl')

# loaded_Scale = load_scaler()

# # --- Label Encoders (same as training) ---
# gender_le = LabelEncoder().fit(["Female", "Male"])
# partner_le = LabelEncoder().fit(["No", "Yes"])
# dependents_le = LabelEncoder().fit(["No", "Yes"])
# phone_le = LabelEncoder().fit(["No", "Yes"])
# internet_le = LabelEncoder().fit(["DSL", "Fiber optic", "No"])

# # --- Streamlit UI ---
# st.title("📉 Telco Customer Churn Prediction")
# st.write("Provide the customer information to predict churn status.")

# # --- Input fields ---
# gender = st.selectbox("Gender", ["Male", "Female"])
# senior = st.selectbox("Senior Citizen", [0, 1])
# partner = st.selectbox("Has Partner", ["Yes", "No"])
# dependents = st.selectbox("Has Dependents", ["Yes", "No"])
# tenure = st.slider("Tenure (months)", 0, 72, 12)
# phone_service = st.selectbox("Phone Service", ["Yes", "No"])
# internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
# monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
# total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

# # --- Preprocess input data ---
# def preprocess_input():
#     # Create raw input DataFrame
#     df = pd.DataFrame([[
#         gender, senior, partner, dependents, tenure,
#         phone_service, internet_service, monthly_charges, total_charges
#     ]], columns=[
#         "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
#         "PhoneService", "InternetService", "MonthlyCharges", "TotalCharges"
#     ])

#     # Label encoding
#     df["gender"] = gender_le.transform(df["gender"])
#     df["Partner"] = partner_le.transform(df["Partner"])
#     df["Dependents"] = dependents_le.transform(df["Dependents"])
#     df["PhoneService"] = phone_le.transform(df["PhoneService"])
#     df["InternetService"] = internet_le.transform(df["InternetService"])

#     # Reorder columns to match scaler’s fit
#     numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
#     df_numeric = df[numeric_cols].copy()
#     df_scaled_numeric = pd.DataFrame(loaded_Scale.transform(df_numeric), columns=numeric_cols)

#     # Combine scaled numeric with encoded categorical
#     df_final = pd.concat([
#         df[["gender", "Partner", "Dependents", "PhoneService", "InternetService"]].reset_index(drop=True),
#         df_scaled_numeric.reset_index(drop=True)
#     ], axis=1)

#     return df_final


# # --- Predict on click ---
# if st.button("Predict Churn"):
#     input_data = preprocess_input()
#     prediction_prob = model.predict(input_data)[0][0]
#     print(prediction_prob)
#     prediction = "Churn" if prediction_prob > 0.5 else "No Churn"
#     print(prediction)

#     st.subheader("📊 Prediction Result:")
#     if prediction == "Churn":
#         st.error(f"⚠️ The customer is likely to CHURN (Probability: {prediction_prob:.2f})")
#     else:
#         st.success(f"✅ The customer is NOT likely to churn (Probability: {prediction_prob:.2f})")













# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# # --- Load the saved model ---
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model('Teleco-Customer-Churn.h5')

# model = load_model()

# # --- Label Encoders (fitted as per training) ---
# gender_le = LabelEncoder().fit(["Female", "Male"])
# partner_le = LabelEncoder().fit(["No", "Yes"])
# dependents_le = LabelEncoder().fit(["No", "Yes"])
# phone_le = LabelEncoder().fit(["No", "Yes"])
# internet_le = LabelEncoder().fit(["DSL", "Fiber optic", "No"])

# # MinMaxScaler instance
# loaded_Scale = joblib.load('Scaling.pkl')

# # --- Streamlit UI ---
# st.title("📉 Telco Customer Churn Prediction")
# st.write("Provide the customer information to predict churn status.")

# # --- Input fields ---
# gender = st.selectbox("Gender", ["Male", "Female"])
# senior = st.selectbox("Senior Citizen", [0, 1])
# partner = st.selectbox("Has Partner", ["Yes", "No"])
# dependents = st.selectbox("Has Dependents", ["Yes", "No"])
# tenure = st.slider("Tenure (months)", 0, 72, 12)
# phone_service = st.selectbox("Phone Service", ["Yes", "No"])
# internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
# monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
# total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

# # --- Preprocess input data ---
# def preprocess_input():
#     df = pd.DataFrame([[
#         gender_le.transform([gender])[0],
#         senior,
#         partner_le.transform([partner])[0],
#         dependents_le.transform([dependents])[0],
#         tenure,
#         phone_le.transform([phone_service])[0],
#         internet_le.transform([internet_service])[0],
#         monthly_charges,
#         total_charges
#     ]], columns=[
#         "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
#         "PhoneService", "InternetService", "MonthlyCharges", "TotalCharges"
#     ])

#     # Normalize only numerical columns
#     for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
#         df[[col]] = scaler.fit_transform(df[[col]])

#     return df

# # --- Predict on click ---
# if st.button("Predict Churn"):
#     input_data = preprocess_input()
#     prediction_prob = model.predict(input_data)[0][0]
#     print(prediction_prob)
#     prediction = "Churn" if prediction_prob > 0.5 else "No Churn"

#     st.subheader("📊 Prediction Result:")
#     if prediction == "Churn":
#         st.error(f"⚠️ The customer is likely to CHURN (Probability: {prediction_prob:.2f})")
        
#     else:
#         st.success(f"✅ The customer is NOT likely to churn (Probability: {prediction_prob:.2f})")
        


# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
# import joblib

# # --- Load the saved model ---
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model('Teleco-Customer-Churn.h5')

# model = load_model()

# # --- Load the saved scaler (rename corrected) ---
# @st.cache_resource
# def load_scaler():
#     return joblib.load("Scaling.pkl")  # <- You saved it as 'Scaling.pkl' in your Jupyter notebook

# scaler = load_scaler()

# # --- Label Encoders (must match training logic) ---
# gender_le = LabelEncoder().fit(["Female", "Male"])
# partner_le = LabelEncoder().fit(["No", "Yes"])
# dependents_le = LabelEncoder().fit(["No", "Yes"])
# phone_le = LabelEncoder().fit(["No", "Yes"])
# internet_le = LabelEncoder().fit(["DSL", "Fiber optic", "No"])

# # --- Streamlit UI ---
# st.title("📉 Telco Customer Churn Prediction")
# st.write("Provide the customer information to predict churn status.")

# # --- Input fields ---
# gender = st.selectbox("Gender", ["Male", "Female"])
# senior = st.selectbox("Senior Citizen", [0, 1])
# partner = st.selectbox("Has Partner", ["Yes", "No"])
# dependents = st.selectbox("Has Dependents", ["Yes", "No"])
# tenure = st.slider("Tenure (months)", 0, 72, 12)
# phone_service = st.selectbox("Phone Service", ["Yes", "No"])
# internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
# monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
# total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

# # --- Preprocess input ---
# def preprocess_input():
#     df = pd.DataFrame([[
#         gender, senior, partner, dependents, tenure,
#         phone_service, internet_service, monthly_charges, total_charges
#     ]], columns=[
#         "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
#         "PhoneService", "InternetService", "MonthlyCharges", "TotalCharges"
#     ])

#     # Encode categorical values
#     df["gender"] = gender_le.transform(df["gender"])
#     df["Partner"] = partner_le.transform(df["Partner"])
#     df["Dependents"] = dependents_le.transform(df["Dependents"])
#     df["PhoneService"] = phone_le.transform(df["PhoneService"])
#     df["InternetService"] = internet_le.transform(df["InternetService"])

#     # Apply scaling
#     df_scaled = scaler.transform(df)

#     return df_scaled

# # --- Prediction ---
# if st.button("Predict Churn"):
#     input_data = preprocess_input()
#     prediction_prob = model.predict(input_data)[0][0]
#     prediction = "Churn" if prediction_prob > 0.5 else "No Churn"

#     st.subheader("📊 Prediction Result:")
#     if prediction == "Churn":
#         st.error(f"⚠️ The customer is likely to CHURN (Probability: {prediction_prob:.2f})")
#     else:
#         st.success(f"✅ The customer is NOT likely to churn (Probability: {prediction_prob:.2f})")
