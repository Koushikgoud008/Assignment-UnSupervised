import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.data_preprocessing import load_data
from src.feature_engineering import prepare_features
from src.predict import predict_new_data

st.set_page_config(page_title="Financial Behavior Clustering", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px; border: 1px solid var(--faded-text-10) !important; background-color: var(--secondary-background-color); box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.08);
    }
    .stButton>button { border-radius: 8px; font-weight: 600; border: 1px solid var(--faded-text-20); transition: all 0.3s; }
    .stButton>button:hover { border: 1px solid var(--primary-color); color: var(--primary-color); box-shadow: 0px 4px 12px rgba(0,0,0,0.1); }
    .accent { color: var(--primary-color); }
    .subtext { opacity: 0.7; font-size: 1.1rem; }
    .microtext { opacity: 0.5; font-size: 0.9rem; }
    .desc { opacity: 0.8; padding: 0 30px; font-size: 1.05rem; line-height: 1.6; }
    div[data-testid="stExpander"] div[role="button"] p { font-weight: 600; color: var(--primary-color); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 3rem; margin-bottom: 0;'>Financial<em class='accent'>Clustering (GMM)</em></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext' style='text-align: center; margin-top: 10px;'>Analyze credit behaviors through Gaussian distributions to map customer personas.</p>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

df = load_data()
X_scaled, scaler, feature_cols = prepare_features(df)

if not os.path.exists('models/gmm_model.pkl'):
    from src.train_model import train
    model = train(X_scaled)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
else:
    model = joblib.load('models/gmm_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')

purchases_idx = feature_cols.get_loc('PURCHASES')
balance_idx = feature_cols.get_loc('BALANCE')
centers = scaler.inverse_transform(model.means_)

transactor_cluster = int(centers[:, purchases_idx].argmax())
revolver_cluster = int(centers[:, balance_idx].argmax())

if transactor_cluster == revolver_cluster:
    revolver_cluster = int(centers[:, balance_idx].argsort()[-2])

steady_cluster = list(set([0, 1, 2]) - {transactor_cluster, revolver_cluster})[0]

persona_map = {
    transactor_cluster: {"label": "High-Value Transactors", "icon": "💳", "desc": "High purchase volume and frequency. They use the card heavily for retail but generally pay off their balances."},
    revolver_cluster: {"label": "Revolving Debtors", "icon": "🏦", "desc": "High rolling balances and frequent cash advances. They utilize a large portion of their credit limit and carry debt."},
    steady_cluster: {"label": "Conservative Users", "icon": "🛡️", "desc": "Low overall activity. They keep balances low, make minimal purchases, and interact sparingly with credit lines."}
}

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("<h6 class='subtext' style='font-weight: 600; letter-spacing: 1px; margin-top: 10px;'>✦ TRANSACTION METRICS</h6>", unsafe_allow_html=True)
        st.divider()
        
        with st.form("gmm_form"):
            with st.expander("Account Balances & Limits", expanded=True):
                balance = st.number_input('Balance', min_value=0.0, value=1500.0, step=100.0)
                bal_freq = st.slider('Balance Frequency', 0.0, 1.0, 0.9)
                credit_limit = st.number_input('Credit Limit', min_value=0.0, value=5000.0, step=500.0)
                
            with st.expander("Purchase Behavior"):
                purchases = st.number_input('Total Purchases', min_value=0.0, value=800.0, step=100.0)
                oneoff_purch = st.number_input('One-Off Purchases', min_value=0.0, value=300.0, step=100.0)
                install_purch = st.number_input('Installment Purchases', min_value=0.0, value=500.0, step=100.0)
                purch_freq = st.slider('Purchases Frequency', 0.0, 1.0, 0.5)
                oneoff_freq = st.slider('One-Off Frequency', 0.0, 1.0, 0.2)
                install_freq = st.slider('Installments Frequency', 0.0, 1.0, 0.4)
                purch_trx = st.number_input('Purchases Transactions', min_value=0, value=12, step=1)

            with st.expander("Cash Advance Profile"):
                cash_adv = st.number_input('Cash Advance', min_value=0.0, value=0.0, step=100.0)
                cash_freq = st.slider('Cash Advance Frequency', 0.0, 1.0, 0.0)
                cash_trx = st.number_input('Cash Advance Transactions', min_value=0, value=0, step=1)

            with st.expander("Payment History"):
                payments = st.number_input('Total Payments', min_value=0.0, value=1000.0, step=100.0)
                min_payments = st.number_input('Minimum Payments', min_value=0.0, value=300.0, step=50.0)
                prc_full = st.slider('Percent Full Payment', 0.0, 1.0, 0.1)
                tenure = st.slider('Tenure (Months)', 6, 12, 12)

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.form_submit_button("RUN GMM CLUSTERING", use_container_width=True)

with col2:
    with st.container(border=True):
        if predict_btn:
            input_dict = {
                'BALANCE': balance, 'BALANCE_FREQUENCY': bal_freq, 'PURCHASES': purchases,
                'ONEOFF_PURCHASES': oneoff_purch, 'INSTALLMENTS_PURCHASES': install_purch,
                'CASH_ADVANCE': cash_adv, 'PURCHASES_FREQUENCY': purch_freq,
                'ONEOFF_PURCHASES_FREQUENCY': oneoff_freq, 'PURCHASES_INSTALLMENTS_FREQUENCY': install_freq,
                'CASH_ADVANCE_FREQUENCY': cash_freq, 'CASH_ADVANCE_TRX': cash_trx,
                'PURCHASES_TRX': purch_trx, 'CREDIT_LIMIT': credit_limit, 'PAYMENTS': payments,
                'MINIMUM_PAYMENTS': min_payments, 'PRC_FULL_PAYMENT': prc_full, 'TENURE': tenure
            }
            
            input_df = pd.DataFrame([input_dict], columns=feature_cols)
            
            prediction, probabilities = predict_new_data(input_df, scaler)
            pred_idx = int(prediction[0])
            confidence = probabilities[0][pred_idx] * 100
            
            p_data = persona_map[pred_idx]
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 4rem; transition: all 0.3s ease;'>{p_data['icon']}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 5px; transition: all 0.3s ease;'>Segment: <strong>{p_data['label']}</strong></h3>", unsafe_allow_html=True)
            st.markdown(f"<p class='microtext' style='text-align: center;'>GMM Confidence Match: {confidence:.1f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='desc' style='text-align: center; transition: all 0.3s ease;'>{p_data['desc']}</p>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)
        else:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("<h1 class='microtext' style='text-align: center; font-size: 3rem;'>✦</h1>", unsafe_allow_html=True)
            st.markdown("<h4 class='subtext' style='text-align: center; margin-bottom: 5px;'>Awaiting GMM Pipeline</h4>", unsafe_allow_html=True)
            st.markdown("<p class='microtext' style='text-align: center;'>Configure the 17 financial vectors on the left<br>to map the customer's probabilistic archetype.</p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)