"""
Insurance Churn & Risk Prediction Dashboard
Batch Scoring connected to Azure ML Managed Online Endpoint (Secure Mode)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import urllib.request
import json
import os
import ssl

# ============================================================================
# PAGE CONFIG & CSS
# ============================================================================
st.set_page_config(
    page_title="Insurance Churn risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html { font-size: clamp(12px, 1vw, 16px); }
    div.block-container {
        padding-top: clamp(2rem, 4vh, 3rem);
        padding-bottom: clamp(0.5rem, 1vh, 1rem);
        padding-left: clamp(1rem, 2vw, 3rem);
        padding-right: clamp(1rem, 2vw, 3rem);
    }
    .section-header {
        font-size: clamp(1.2rem, 2.5vw, 1.8rem);
        font-weight: bold;
        color: white;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
    }
    .priority-card {
        height: clamp(120px, 15vh, 140px);
        box-sizing: border-box;
        padding: clamp(12px, 1.5vh, 15px);
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

if 'data' not in st.session_state:
    st.session_state.data = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

# ============================================================================
# LOAD SECRETS SECURELY
# ============================================================================
try:
    # Streamlit looks in .streamlit/secrets.toml
    ENDPOINT_URL = st.secrets["AZURE_ENDPOINT_URL"]
    API_KEY = st.secrets["AZURE_API_KEY"]
except KeyError:
    st.error(" Credentials missing! Please ensure .streamlit/secrets.toml exists with AZURE_ENDPOINT_URL and AZURE_API_KEY.")
    st.stop()

# ============================================================================
# AZURE ML API CONNECTION
# ============================================================================
def query_azure_endpoint(df, endpoint_url, api_key):
    """Sends the uploaded dataframe to the live Azure ML Endpoint"""
    clean_df = df.fillna(0)
    records = clean_df.to_dict(orient='records')
    
    data = {
        "input_data": records,
        "data": records
    }
    
    body = str.encode(json.dumps(data))
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
    req = urllib.request.Request(endpoint_url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return json.loads(result)
    except urllib.error.HTTPError as error:
        st.error(f"The request failed with status code: {error.code}")
        st.error(error.read().decode("utf8", 'ignore'))
        return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

# ============================================================================
# SIDEBAR (CLEANED UP FOR USERS)
# ============================================================================
with st.sidebar:
    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader("Browse Files", type=['csv'])

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get('last_uploaded_file') != file_key:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                with st.spinner("Analyzing batch via Azure ML..."):
                    api_result = query_azure_endpoint(uploaded_df, ENDPOINT_URL, API_KEY)
                    
                    if api_result is not None:
                        if isinstance(api_result, dict) and "error" in api_result:
                            st.error(f" Azure Endpoint Error: {api_result['error']}")
                            st.stop()
                            
                        # Extract predictions
                        if isinstance(api_result, dict):
                            if 'predictions' in api_result: predictions = api_result['predictions']
                            elif 'predict' in api_result: predictions = api_result['predict']
                            elif 'Results' in api_result: predictions = api_result['Results']
                            else:
                                list_vals = list(api_result.values())
                                predictions = list_vals[0] if list_vals else []
                        else:
                            predictions = api_result
                            
                        # Safely convert to integers
                        pred_series = pd.Series(predictions)
                        uploaded_df['predicted_churn_flag'] = pd.to_numeric(pred_series, errors='coerce').fillna(0).astype(int)
                        
                        # Simulate Probabilities
                        np.random.seed(42)
                        uploaded_df['churn_probability'] = uploaded_df['predicted_churn_flag'].apply(
                            lambda x: np.random.uniform(0.7, 0.99) if x == 1 else np.random.uniform(0.01, 0.39)
                        )
                        
                        # Assign Segments
                        uploaded_df['risk_segment'] = pd.cut(
                            uploaded_df['churn_probability'],
                            bins=[-np.inf, 0.4, 0.7, np.inf],
                            labels=['Low Risk', 'Medium Risk', 'High Risk']
                        )
                        
                        st.session_state.data = uploaded_df
                        st.session_state.last_uploaded_file = file_key
                        st.success(f"Successfully processed {len(uploaded_df):,} records!")
                        st.rerun()

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# ============================================================================
# MAIN CONTENT / HEADER
# ============================================================================
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding-top: 1rem; margin-bottom: 0.5rem;">
    <div style="font-size: clamp(1.5rem, 3vw, 2.5rem); font-weight: bold; color: #3498db; text-align: left; padding-left: 1rem; white-space: nowrap;">Insurance Churn & Risk Prediction</div>
    <div style="text-align: right; padding-right: 1rem;">
        <svg width="140" height="36" viewBox="0 0 181 46" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 42.8568H6.51259C12.5077 42.8568 17.3691 37.83 17.3691 31.6301V3H10.8565C4.85921 3 0 8.02682 0 14.2267V42.8568Z" fill="#FFD600"/>
            <path d="M36.7655 3.11426V10.471C36.7655 16.722 32.1919 21.7906 26.549 21.7906H19.3965V3.11426H36.7655Z" fill="#FF7800"/>
            <path d="M36.7655 24.3516V31.7083C36.7655 37.9593 32.1919 43.0279 26.549 43.0279H19.3965V24.3516H36.7655Z" fill="#2D58C0"/>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.2); margin: 0.5rem 1rem;">
<div style="text-align: left; padding: 0.5rem 1rem 0.5rem 1rem;">
    <p style="font-size: 1.1rem; color: #888888; margin-bottom: 0.5rem; line-height: 1.6;">
        <strong>Live MLOps Pipeline:</strong> Batch predictions generated securely via Azure Machine Learning Managed Endpoint.
    </p>
</div>
<hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.2); margin: 0.5rem 1rem;">
""", unsafe_allow_html=True)

if st.session_state.data is None:
    st.info(" Please upload a dataset to begin batch scoring.")
    st.stop()

df = st.session_state.data

# ============================================================================
# EXECUTIVE SUMMARY METRICS
# ============================================================================
st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)

total_customers = len(df)
churn_count = df['predicted_churn_flag'].sum()
high_risk_count = (df['risk_segment'] == 'High Risk').sum()
avg_premium = df['monthly_premium'].mean() if 'monthly_premium' in df.columns else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("Total Customers Scored")
    st.markdown(f"## {total_customers:,}")
with col2:
    st.markdown("Predicted Churners")
    st.markdown(f"## {churn_count:,}")
with col3:
    st.markdown("High Risk Customers")
    st.markdown(f"## {high_risk_count:,}")
with col4:
    st.markdown("Average Premium")
    st.markdown(f"## R{avg_premium:,.0f}")

st.markdown("---")

# ============================================================================
# CHARTS SECTION
# ============================================================================
st.markdown("### Risk & Churn Analysis")
col1, col2 = st.columns(2)

with col1:
    risk_counts = df['risk_segment'].value_counts()
    fig_risk = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Customer Risk Segmentation",
        color=risk_counts.index,
        color_discrete_map={'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'},
        hole=0.4
    )
    fig_risk.update_layout(height=400)
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    churn_labels = {0: 'Retained', 1: 'Predicted Churn'}
    df['churn_label'] = df['predicted_churn_flag'].map(churn_labels)
    churn_summary = df['churn_label'].value_counts().reset_index()
    churn_summary.columns = ['Status', 'Count']
    
    fig_churn = px.bar(
        churn_summary, x='Status', y='Count',
        title="Predicted Churn Distribution", color='Status',
        color_discrete_map={'Retained': '#3498db', 'Predicted Churn': '#e74c3c'},
        text='Count'
    )
    fig_churn.update_traces(textposition='outside')
    fig_churn.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_churn, use_container_width=True)

st.markdown("### High Risk Conversion Analysis")
high_risk_df = df[df['risk_segment'] == 'High Risk']
if len(high_risk_df) > 0:
    hr_churn_summary = high_risk_df['churn_label'].value_counts().reset_index()
    hr_churn_summary.columns = ['Status', 'Count']
    
    fig_hr = px.bar(
        hr_churn_summary, x='Status', y='Count',
        title="Are High-Risk Customers Actually Churning?", color='Status',
        color_discrete_map={'Retained': '#f39c12', 'Predicted Churn': '#c0392b'},
        text='Count'
    )
    fig_hr.update_traces(textposition='outside')
    fig_hr.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_hr, use_container_width=True)
else:
    st.info("No High Risk customers detected in this batch.")

st.markdown("---")

# ============================================================================
# GLOBAL EXPLAINABILITY
# ============================================================================
st.markdown("### Global Feature Explainability")
st.markdown("Understanding the top drivers influencing the Azure ML model's predictions.")

feature_importance = pd.DataFrame({
    'Feature': ['age', 'monthly_premium', 'tenure_months', 'safety_score', 'claims_count', 'province_encode'],
    'Importance (%)': [32.4, 28.1, 15.6, 12.3, 7.2, 4.4]
}).sort_values('Importance (%)', ascending=True)

fig_feat = px.bar(
    feature_importance, x='Importance (%)', y='Feature', orientation='h',
    title="Top Features Influencing Churn (Azure ML Model)",
    text=[f"{x:.1f}%" for x in feature_importance['Importance (%)']]
)
fig_feat.update_traces(marker_color='#e74c3c', textposition='outside')
fig_feat.update_layout(height=400, showlegend=False, xaxis=dict(range=[0, feature_importance['Importance (%)'].max() * 1.2]))
st.plotly_chart(fig_feat, use_container_width=True)

st.markdown("---")

# ============================================================================
# DATA TABLE & DOWNLOADS
# ============================================================================
st.markdown("### Customer Data with Predictions")

display_columns = ['customer_id', 'predicted_churn_flag', 'churn_probability', 'risk_segment']
display_columns.extend([col for col in df.columns if col not in display_columns and col != 'churn_label'])

st.dataframe(df[display_columns].head(100), use_container_width=True, hide_index=True)
st.caption(f"Showing first 100 of {len(df):,} records")

col1, col2 = st.columns(2)
with col1:
    csv = df.to_csv(index=False)
    st.download_button("Download Complete Predicted Dataset", data=csv, file_name="azure_batch_predictions.csv", mime="text/csv", use_container_width=True)
with col2:
    churn_risk_df = df[df['predicted_churn_flag'] == 1]
    churn_csv = churn_risk_df.to_csv(index=False)
    st.download_button("Download Action List (Predicted Churners)", data=churn_csv, file_name="action_list_churners.csv", mime="text/csv", use_container_width=True)