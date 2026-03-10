import streamlit as st
import pandas as pd
import requests
import json

# ==========================================
# 1. AZURE API CONFIGURATION
# ==========================================
SCORING_URI = "https://insurance-churn-live-api.eastus2.inference.ml.azure.com/score"
AUTH_KEY = "PASTE_KEY_HERE"

# ==========================================
# 2. DASHBOARD UI SETUP
# ==========================================
st.set_page_config(page_title="Insurance Churn Predictor", layout="wide")

st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h1 style='color: #2e86c1;'>Customer Churn Prediction Dashboard</h1>
        <p style='color: #7f8c8d; font-size: 1.1em;'>Powered by Azure Machine Learning MLOps Pipeline</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs([" Single Customer Prediction", " Batch Prediction (CSV Upload)"])

# ==========================================
# TAB 1: MANUAL SINGLE ENTRY
# ==========================================
with tab1:
    st.markdown("### Enter Customer Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=45)
    with col2:
        premium = st.number_input("Monthly Premium ($)", min_value=0.0, max_value=5000.0, value=150.0, step=10.0)
    with col3:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=24)
        
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(" Predict Single Risk", type="primary"):
        with st.spinner("Analyzing customer data via Azure ML..."):
            input_data = {"input_data": [{"age": age, "premium": premium, "tenure": tenure}]}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AUTH_KEY}"}
            
            try:
                response = requests.post(SCORING_URI, headers=headers, json=input_data)
                if response.status_code == 200:
                    prediction = response.json()["predictions"][0]
                    if prediction == 1:
                        st.error(" **HIGH RISK: CHURN PREDICTED** - Recommend immediate retention intervention.")
                    else:
                        st.success(" **LOW RISK: RETENTION PREDICTED** - Customer is stable.")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to Azure: {str(e)}")

# ==========================================
# TAB 2: BATCH CSV UPLOAD
# ==========================================
with tab2:
    st.markdown("### Upload Customer Dataset")
    st.info(" Upload your `synthetic_insurance.csv` here.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.markdown("**Data Preview:**")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button(" Run Batch Prediction on All Rows", type="primary"):
            
            with st.spinner(f"Sending {len(df)} complete records to Azure ML..."):
                
                # We send the ENTIRE dataframe to Azure, not just 3 columns!
                records = df.to_dict(orient="records")
                input_data = {"input_data": records}
                
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AUTH_KEY}"}
                
                try:
                    response = requests.post(SCORING_URI, headers=headers, json=input_data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Catch the exact error if score.py fails
                        if "error" in result:
                            st.error(f" Azure Model Error: {result['error']}")
                        
                        elif "predictions" in result:
                            predictions = result["predictions"]
                            
                            # Add predictions to our dataframe
                            df["Predicted_Churn"] = predictions
                            df["Risk_Label"] = df["Predicted_Churn"].apply(lambda x: "High Risk (Churn)" if x == 1 else "Low Risk (Retain)")
                            
                            st.success(" Batch scoring complete!")
                            
                            col_a, col_b = st.columns(2)
                            churn_count = sum(predictions)
                            
                            col_a.metric("Total Customers Scored", len(df))
                            col_b.metric("Identified High-Risk Customers", churn_count, delta_color="inverse")
                            
                            st.markdown("### Scored Dataset")
                            def highlight_risk(val):
                                color = '#fdedec' if val == "High Risk (Churn)" else ''
                                return f'background-color: {color}'
                                
                            # Show the dataframe with the new predictions at the front
                            cols = ['Risk_Label', 'Predicted_Churn'] + [c for c in df.columns if c not in ['Risk_Label', 'Predicted_Churn']]
                            st.dataframe(df[cols].style.applymap(highlight_risk, subset=['Risk_Label']), use_container_width=True)
                            
                            csv = df[cols].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=" Download Scored Dataset",
                                data=csv,
                                file_name="churn_predictions_scored.csv",
                                mime="text/csv",
                            )
                    else:
                        st.error(f"API HTTP Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to Azure: {str(e)}")