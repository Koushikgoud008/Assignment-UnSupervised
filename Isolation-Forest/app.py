import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.data_preprocessing import load_data
from src.feature_engineering import prepare_features
from src.predict import predict_new_data

st.set_page_config(page_title="Turbine Anomaly Detection", layout="wide", initial_sidebar_state="collapsed")

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

st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 3rem; margin-bottom: 0;'>System<em class='accent'>Anomaly Detection</em></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext' style='text-align: center; margin-top: 10px;'>Analyze turbine telemetry via Isolation Forest to identify critical mechanical deviations.</p>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

df = load_data()
X_scaled, scaler, feature_cols = prepare_features(df)

if not os.path.exists('models/isolation_forest_model.pkl'):
    from src.train_model import train
    model = train(X_scaled)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
else:
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("<h6 class='subtext' style='font-weight: 600; letter-spacing: 1px; margin-top: 10px;'>✦ TURBINE TELEMETRY</h6>", unsafe_allow_html=True)
        st.divider()
        
        with st.form("anomaly_form"):
            with st.expander("Thermal Diagnostics", expanded=True):
                gearbox_c = st.number_input('Gearbox Casing Temp', value=60.50, step=0.10)
                gearbox_b = st.number_input('Gearbox Bearing Temp', value=67.10, step=0.10)
                
            with st.expander("Vibration Frequency"):
                vib_x = st.number_input('Vibration X-Axis', value=0.0125, step=0.0010, format="%.4f")
                vib_y = st.number_input('Vibration Y-Axis', value=0.0115, step=0.0010, format="%.4f")
                vib_z = st.number_input('Vibration Z-Axis', value=0.0145, step=0.0010, format="%.4f")

            with st.expander("Fluid Analytics"):
                oil_press = st.number_input('Oil Pressure', value=4.40, step=0.01)
                particle = st.number_input('Particle Count', value=120, step=1)

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.form_submit_button("RUN DIAGNOSTIC SCAN", use_container_width=True)

with col2:
    with st.container(border=True):
        if predict_btn:
            input_dict = {
                'gearbox_c': gearbox_c,
                'gearbox_b': gearbox_b,
                'vibration_x': vib_x,
                'vibration_y': vib_y,
                'vibration_z': vib_z,
                'oil_pressure': oil_press,
                'particle_count': particle
            }
            
            input_df = pd.DataFrame([input_dict], columns=feature_cols)
            
            prediction, anomaly_score = predict_new_data(input_df, scaler)
            
            is_anomaly = prediction[0] == -1
            score = anomaly_score[0]
            
            if is_anomaly:
                icon = "⚠️"
                label = "CRITICAL ANOMALY DETECTED"
                color = "#E74C3C"
                desc = "Telemetry indicates a severe statistical deviation from nominal operating bounds. Immediate maintenance inspection recommended."
            else:
                icon = "✅"
                label = "SYSTEM NOMINAL"
                color = "#2ECC71"
                desc = "All sensory inputs are operating within expected functional thresholds. No anomalies detected in current machine state."
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 4rem; transition: all 0.3s ease;'>{icon}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: {color}; margin-bottom: 5px; transition: all 0.3s ease;'><strong>{label}</strong></h3>", unsafe_allow_html=True)
            st.markdown(f"<p class='microtext' style='text-align: center;'>Isolation Score: {score:.3f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='desc' style='text-align: center; transition: all 0.3s ease;'>{desc}</p>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)
        else:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("<h1 class='microtext' style='text-align: center; font-size: 3rem;'>✦</h1>", unsafe_allow_html=True)
            st.markdown("<h4 class='subtext' style='text-align: center; margin-bottom: 5px;'>Awaiting Telemetry Pipeline</h4>", unsafe_allow_html=True)
            st.markdown("<p class='microtext' style='text-align: center;'>Configure the mechanical sensors on the left to evaluate<br>the hardware for hidden fault patterns.</p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)