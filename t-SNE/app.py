import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.data_preprocessing import load_data
from src.feature_engineering import prepare_features
from src.predict import predict_new_data

st.set_page_config(page_title="t-SNE Dimensionality Reduction", layout="wide", initial_sidebar_state="collapsed")

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

st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 3rem; margin-bottom: 0;'>Retail Data<em class='accent'>t-SNE Mapping</em></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext' style='text-align: center; margin-top: 10px;'>Map multi-dimensional retail metrics into a non-linear local neighborhood space.</p>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

df = load_data()
X_scaled, scaler, feature_cols = prepare_features(df)

if not os.path.exists('models/tsne_knn_model.pkl'):
    from src.train_model import train
    model = train(X_scaled)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
else:
    model = joblib.load('models/tsne_knn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("<h6 class='subtext' style='font-weight: 600; letter-spacing: 1px; margin-top: 10px;'>✦ TRANSACTION VECTORS</h6>", unsafe_allow_html=True)
        st.divider()
        
        with st.form("tsne_form"):
            price = st.number_input('Product Price (mehsul_qiymeti)', min_value=0.0, value=15.50, step=1.0)
            bonus = st.selectbox('Bonus Card Used (bonus_kart)', ['True', 'False'])
            lat = st.number_input('Store Latitude (magaza_lat)', min_value=0.0, max_value=90.0, value=40.48556, step=0.0100, format="%.5f")
            lon = st.number_input('Store Longitude (magaza_lon)', min_value=0.0, max_value=180.0, value=49.94674, step=0.0100, format="%.5f")
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.form_submit_button("RUN t-SNE MAPPING", use_container_width=True)

with col2:
    with st.container(border=True):
        if predict_btn:
            input_dict = {
                'mehsul_qiymeti': price,
                'magaza_lat': lat,
                'magaza_lon': lon,
                'bonus_kart': 1 if bonus == 'True' else 0
            }
            
            input_df = pd.DataFrame([input_dict], columns=feature_cols)
            
            components = predict_new_data(input_df, scaler)
            tsne1 = components[0][0]
            tsne2 = components[0][1]
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center; font-size: 3.5rem; transition: all 0.3s ease;'>🌌</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 5px; transition: all 0.3s ease;'>Non-Linear Space</h3>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='display: flex; justify-content: center; gap: 40px; margin-top: 20px;'>
                <div style='text-align: center;'>
                    <p class='subtext' style='margin-bottom: 0;'>t-SNE Dimension 1</p>
                    <h2 class='accent' style='margin-top: 5px;'>{tsne1:.4f}</h2>
                </div>
                <div style='text-align: center;'>
                    <p class='subtext' style='margin-bottom: 0;'>t-SNE Dimension 2</p>
                    <h2 class='accent' style='margin-top: 5px;'>{tsne2:.4f}</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<p class='microtext' style='text-align: center; margin-top: 25px;'>The transaction has been successfully mapped to its nearest local neighborhood.</p>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)
        else:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("<h1 class='microtext' style='text-align: center; font-size: 3rem;'>✦</h1>", unsafe_allow_html=True)
            st.markdown("<h4 class='subtext' style='text-align: center; margin-bottom: 5px;'>Awaiting t-SNE Pipeline</h4>", unsafe_allow_html=True)
            st.markdown("<p class='microtext' style='text-align: center;'>Submit the raw transaction vectors on the left to map<br>them into the probability distribution space.</p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)