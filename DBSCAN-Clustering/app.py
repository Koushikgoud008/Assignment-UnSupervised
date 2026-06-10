import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.data_preprocessing import load_data
from src.feature_engineering import prepare_features

st.set_page_config(page_title="Spatial Density Clustering", layout="wide", initial_sidebar_state="collapsed")

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

st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 3rem; margin-bottom: 0;'>Location<em class='accent'>Clustering (DBSCAN)</em></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext' style='text-align: center; margin-top: 10px;'>Analyze geographic coordinates to identify high-density hubs and isolated outposts.</p>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

df = load_data()
X_scaled, scaler = prepare_features(df)

if not os.path.exists('models/dbscan_knn_model.pkl'):
    from src.train_model import train
    model = train(X_scaled)
    joblib.dump(scaler, 'models/scaler.pkl')
else:
    model = joblib.load('models/dbscan_knn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("<h6 class='subtext' style='font-weight: 600; letter-spacing: 1px; margin-top: 10px;'>✦ SPATIAL COORDINATES</h6>", unsafe_allow_html=True)
        st.divider()
        
        with st.form("dbscan_form"):
            lat = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=35.1922, step=0.0100, format="%.4f")
            lon = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=-111.6430, step=0.0100, format="%.4f")
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.form_submit_button("RUN SPATIAL CLUSTERING", use_container_width=True)

with col2:
    with st.container(border=True):
        if predict_btn:
            input_df = pd.DataFrame([[lat, lon]], columns=['Latitude', 'Longitude'])
            input_scaled = scaler.transform(input_df)
            prediction = int(model.predict(input_scaled)[0])
            
            if prediction == -1:
                p_data = {
                    "label": "Isolated Territory (Outlier)", 
                    "icon": "🏜️", 
                    "desc": "This coordinate sits far outside established high-density zones. It represents an isolated location or a disconnected frontier."
                }
            else:
                p_data = {
                    "label": f"Core Market Hub (Region {prediction})", 
                    "icon": "📍", 
                    "desc": "This location successfully falls within a high-density geographic cluster, indicating strong integration into a regional network."
                }
            
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 4rem; transition: all 0.3s ease;'>{p_data['icon']}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 5px; transition: all 0.3s ease;'>Segment: <strong>{p_data['label']}</strong></h3>", unsafe_allow_html=True)
            st.markdown(f"<p class='desc' style='text-align: center; transition: all 0.3s ease;'>{p_data['desc']}</p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)
        else:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("<h1 class='microtext' style='text-align: center; font-size: 3rem;'>✦</h1>", unsafe_allow_html=True)
            st.markdown("<h4 class='subtext' style='text-align: center; margin-bottom: 5px;'>Awaiting Spatial Pipeline</h4>", unsafe_allow_html=True)
            st.markdown("<p class='microtext' style='text-align: center;'>Input the target coordinates on the left to map<br>the location's geographic density tier.</p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)