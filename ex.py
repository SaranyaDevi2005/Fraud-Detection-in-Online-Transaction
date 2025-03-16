import streamlit as st
import pandas as pd
import pickle
import base64
import time
import pyttsx3
from lime.lime_tabular import LimeTabularExplainer


with open('lightgbm_model.pkl', 'rb') as file:
    model, feature_names = pickle.load(file)


def get_base64(img_path):
    with open(img_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

bg_img = get_base64("image.jpg")  # Ensure correct image path


page_bg = f"""
<style>
    .stApp {{
        background: url(data:image/jpg;base64,{bg_img});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
        padding: 20px;
    }}
    .warning-box {{
        background-color: rgba(255, 0, 0, 0.9);
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 16px;
    }}
    .success-box {{
        background-color: rgba(0, 150, 0, 0.8);
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 16px;
    }}
    .stButton>button {{
        background-color: #4169E1 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        width: 100%;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: #1976D2 !important;
    }}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("💳 Fraud Detection System")
st.subheader("Fill in the transaction details below:")


with st.sidebar:
    st.markdown('<h3 style="color:#FFFFFF; text-align:center;">📝 Transaction Details</h3>', unsafe_allow_html=True)

    transaction_type = st.selectbox("💳 Transaction Type", ["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER"], index=None)
    amount = st.number_input("💰 Transaction Amount", min_value=0.01, step=0.01, format="%.2f")
    oldbalanceOrg = st.number_input("🏦 Old Balance (Origin)", min_value=0.0, step=0.01, format="%.2f")
    newbalanceOrig = st.number_input("🏦 New Balance (Origin)", min_value=0.0, step=0.01, format="%.2f")
    oldbalanceDest = st.number_input("🏧 Old Balance (Destination)", min_value=0.0, step=0.01, format="%.2f")
    newbalanceDest = st.number_input("🏧 New Balance (Destination)", min_value=0.0, step=0.01, format="%.2f")


def trigger_alarm():
    engine = pyttsx3.init(driverName='sapi5')
    engine.say("Alert! Fraudulent transaction detected!")
    engine.runAndWait()


if st.button("🔍 Predict Fraud"):
    if transaction_type is None:
        st.markdown('<div class="warning-box">⚠ Please select a transaction type!</div>', unsafe_allow_html=True)
    elif any(val is None for val in [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]):
        st.markdown('<div class="warning-box">⚠ Please fill in all the required fields!</div>', unsafe_allow_html=True)
    else:
        user_input = pd.DataFrame([{ 
            "type": transaction_type, "amount": amount, "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig, "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])
        user_input = pd.get_dummies(user_input)
        for col in feature_names:
            if col not in user_input.columns:
                user_input[col] = 0
        user_input = user_input[feature_names]
        prediction = model.predict(user_input)[0]

        # Assuming the model outputs a probability for fraud (class 1)
        prob_fraud = model.predict_proba(user_input)[0][1]  # Probability of being fraudulent

        # Display the risk score
        st.write(f"*Risk Score (Probability of Fraud):* {prob_fraud:.2f}")

        if prob_fraud > 0.7:  # If the probability is above 70%, block the transaction
            st.markdown('<div class="warning-box">🚨 FRAUDULENT TRANSACTION BLOCKED!</div>', unsafe_allow_html=True)
            trigger_alarm()
            st.subheader("🔍 Why was this transaction flagged and blocked?")

           
            explainer = LimeTabularExplainer(
                training_data=user_input.values,  # Use user input data
                feature_names=feature_names,
                class_names=["Legitimate", "Fraud"],
                mode="classification"
            )

            exp = explainer.explain_instance(user_input.values[0], model.predict_proba, num_features=5)  # Get LIME explanation
            explanation_data = exp.as_list()  # Extract feature contributions

            
            explanation_sentences = []
            for feature_text, weight in explanation_data:
                feature_name = feature_text.split()[0]  # Extract feature name
                
                if feature_name in user_input.columns:  # Check if it exists in user_input
                    feature_value = user_input[feature_name].values[0]
                    
                    # Customize explanation based on feature type
                    if feature_name == "step":
                        explanation_sentences.append(f"- The transaction occurred very early (step = {feature_value:.2f}), which is unusual.")
                    elif feature_name == "amount":
                        explanation_sentences.append(f"- A high amount ({feature_value:.2f}) was transferred, which may be suspicious.")
                    elif feature_name == "oldbalanceOrg":
                        explanation_sentences.append(f"- The sender's balance before transaction was {feature_value:.2f}, which raised suspicion.")
                    elif feature_name == "newbalanceOrig":
                        explanation_sentences.append(f"- After the transaction, the sender's new balance is {feature_value:.2f}, which is very low.")
                    elif feature_name == "oldbalanceDest":
                        explanation_sentences.append(f"- The recipient's account had a high initial balance ({feature_value:.2f}), which is common in fraud cases.")
                    else:
                        explanation_sentences.append(f"- The feature {feature_name} had a value of {feature_value:.2f}, influencing the decision.")

            if explanation_sentences:
                st.markdown("🚨 This transaction was flagged as fraudulent and blocked because:")
                for sentence in explanation_sentences:
                    st.markdown(sentence)
            else:
                st.markdown("✅ No significant factors reduced fraud probability.")
        else:
            st.markdown('<div class="success-box">✅ Transaction is Legitimate.</div>', unsafe_allow_html=True)


st.subheader("📡 Real-Time Monitoring")

with open('lightgbm_model.pkl', 'rb') as file:
    model, feature_names = pickle.load(file)

# ⭐ Added Prerequisite Section
st.markdown("## 📌 File Upload Requirements")
st.markdown(""" 
Please ensure that the uploaded Excel file contains the following required columns:
- type (Transaction Type: "CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER")
- amount (Transaction Amount)
- oldbalanceOrg (Old Balance of Sender)
- newbalanceOrig (New Balance of Sender)
- oldbalanceDest (Old Balance of Receiver)
- newbalanceDest (New Balance of Receiver)

⚠ If any of these columns are missing, the system will reject the file.
""")

# Store file in session state to persist selection
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.file_uploader("📂 Upload an Excel File for Real-Time Monitoring", type=["xlsx"])

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file  # Save the uploaded file to session state

if st.session_state.uploaded_file is not None:
    if st.button("🚨 Start Monitoring"):
        
        def real_time_monitoring(file):
            required_columns = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

            try:
                while True:
                    file.seek(0)  
                    data = pd.read_excel(file)

                    # Validate Required Columns
                    if not all(col in data.columns for col in required_columns):
                        missing_cols = [col for col in required_columns if col not in data.columns]
                        st.error(f"⚠ Missing required columns: {', '.join(missing_cols)}")
                        return  

                    # 🔹 Display Latest Transactions
                    st.subheader("📊 Latest Transactions Being Monitored:")
                    st.write(data.tail(5))  

                    for _, latest_data in data.iterrows():
                        user_input = pd.DataFrame([latest_data])
                        user_input = pd.get_dummies(user_input)

                        for col in feature_names:
                            if col not in user_input.columns:
                                user_input[col] = 0
                        user_input = user_input[feature_names]
                        prediction = model.predict(user_input)[0]

                        # Assuming the model outputs a probability for fraud (class 1)
                        prob_fraud = model.predict_proba(user_input)[0][1]  # Probability of being fraudulent

                        # Display the risk score
                        st.write(f"*Risk Score (Probability of Fraud):* {prob_fraud:.2f}")

                        st.markdown(f"Processing Transaction: {latest_data.to_dict()}")

                        #  Fraud Detection Message
                        if prob_fraud > 0.7:  # If the probability is above 70%, block the transaction
                            st.markdown('<div class="warning-box">🚨 FRAUDULENT TRANSACTION DETECTED!</div>', unsafe_allow_html=True)
                            trigger_alarm()
                        else:
                            st.markdown('<div class="success-box">✅ Transaction is Legitimate.</div>', unsafe_allow_html=True)

                        time.sleep(5)  # ✅ Delay 5 seconds before reading next row

            except Exception as e:
                st.error(f"⚠ Error occurred: {str(e)}")

        real_time_monitoring(st.session_state.uploaded_file)