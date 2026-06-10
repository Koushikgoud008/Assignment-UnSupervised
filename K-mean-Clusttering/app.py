import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.data_preprocessing import load_data
from src.feature_engineering import prepare_features

st.set_page_config(page_title="Instagram Audience Clustering", layout="wide", initial_sidebar_state="collapsed")

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
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 3rem; margin-bottom: 0;'>Audience<em class='accent'>Clustering</em></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext' style='text-align: center; margin-top: 10px;'>Analyze Instagram engagement and spending ranks to algorithmically segment users.</p>", unsafe_allow_html=True)
st.markdown("<p class='microtext' style='text-align: center;'>Powered by scikit-learn & Streamlit</p><br><br>", unsafe_allow_html=True)

df = load_data()
X_scaled, scaler = prepare_features(df)

if not os.path.exists('models/kmeans_model.pkl'):
    from src.train_model import train
    model = train(X_scaled)
    joblib.dump(scaler, 'models/scaler.pkl')
else:
    model = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

centers = scaler.inverse_transform(model.cluster_centers_)
vip_cluster = int((centers[:, 0] + centers[:, 1]).argmax())
dormant_cluster = int((centers[:, 0] + centers[:, 1]).argmin())
browser_cluster = list(set([0, 1, 2]) - {vip_cluster, dormant_cluster})[0]

persona_map = {
    vip_cluster: {"label": "Engaged Loyalists", "icon": "💎", "desc": "High profile visitation and top-tier spending. These are your most valuable converting customers."},
    dormant_cluster: {"label": "Dormant Users", "icon": "👻", "desc": "Low visitation and minimal spending. Require significant re-engagement campaigns or should be purged from targeted ad spend."},
    browser_cluster: {"label": "Casual Browsers", "icon": "👀", "desc": "Mixed engagement signals. They may view content frequently without buying, or make occasional purchases without following closely."}
}

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("<h6 class='subtext' style='font-weight: 600; letter-spacing: 1px; margin-top: 10px;'>✦ INPUT METRICS</h6>", unsafe_allow_html=True)
        st.divider()
        
        insta_score = st.number_input('Instagram Visit Score', min_value=0, max_value=200, value=50, step=1)
        spend_rank = st.number_input('Spending Rank (0 to 100)', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("RUN K-MEANS CLUSTERING", use_container_width=True)

with col2:
    with st.container(border=True):
        if predict_btn:
            input_df = pd.DataFrame(
                [[insta_score, spend_rank]], 
                columns=['Instagram visit score', 'Spending_rank(0 to 100)']
            )
            input_scaled = scaler.transform(input_df)
            prediction = int(model.predict(input_scaled)[0])
            
            p_data = persona_map[prediction]
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 3.5rem;'>{p_data['icon']}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 5px;'>Segment: <strong>{p_data['label']}</strong></h3>", unsafe_allow_html=True)
            st.markdown(f"<p class='desc' style='text-align: center;'>{p_data['desc']}</p>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)
        else:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("<h1 class='microtext' style='text-align: center; font-size: 3rem;'>✦</h1>", unsafe_allow_html=True)
            st.markdown("<h4 class='subtext' style='text-align: center; margin-bottom: 5px;'>Awaiting Clustering Pipeline</h4>", unsafe_allow_html=True)
            st.markdown("<p class='microtext' style='text-align: center;'>Configure the engagement metrics on the left to map<br>the user to their behavioral segment.</p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)