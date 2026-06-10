import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.data_preprocessing import load_data
from src.feature_engineering import prepare_features

st.set_page_config(page_title="Holistic Wellbeing Clustering", layout="wide", initial_sidebar_state="collapsed")

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

st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 3rem; margin-bottom: 0;'>Wellbeing<em class='accent'>Clustering (Hierarchical)</em></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext' style='text-align: center; margin-top: 10px;'>Analyze 23 distinct lifestyle variables to map behavioral archetypes.</p>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

df = load_data()
X_scaled, scaler = prepare_features(df)

if not os.path.exists('models/hc_full_model.pkl'):
    from src.train_model import train
    model, raw_centers = train(X_scaled)
    joblib.dump(scaler, 'models/scaler.pkl')
else:
    model = joblib.load('models/hc_full_model.pkl')
    raw_centers = joblib.load('models/hc_full_centers.pkl')
    scaler = joblib.load('models/scaler.pkl')

wlb_idx = df.drop(columns=['Timestamp'], errors='ignore').columns.get_loc('WORK_LIFE_BALANCE_SCORE')
centers = scaler.inverse_transform(raw_centers)
thriving_cluster = int(centers[:, wlb_idx].argmax())
burnout_cluster = int(centers[:, wlb_idx].argmin())
steady_cluster = list(set([0, 1, 2]) - {thriving_cluster, burnout_cluster})[0]

persona_map = {
    thriving_cluster: {"label": "Optimal Performers", "icon": "🌟", "desc": "High physical activity, strong social integration, and excellent stress management. This archetype represents sustainable, holistic well-being."},
    burnout_cluster: {"label": "At-Risk Overextenders", "icon": "⚠️", "desc": "Characterized by high stress, sleep deprivation, and low passion time. High statistical probability of impending systemic burnout."},
    steady_cluster: {"label": "Baseline Functioners", "icon": "⚖️", "desc": "Moderate metrics across diet, social, and work sectors. Functioning adequately but operating below optimal physiological and psychological capacity."}
}

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("<h6 class='subtext' style='font-weight: 600; letter-spacing: 1px; margin-top: 10px;'>✦ HOLISTIC METRICS</h6>", unsafe_allow_html=True)
        st.divider()
        
        with st.form("clustering_form"):
            with st.expander("Health & Vitality", expanded=True):
                f_veg = st.slider('Fruits & Veggies (Servings)', 0, 5, 3)
                bmi = st.slider('BMI Range (1=Healthy, 2=Over)', 1, 2, 1)
                steps = st.slider('Daily Steps (1k intervals)', 0, 10, 5)
                sleep = st.slider('Sleep Hours', 0, 10, 7)
                meditation = st.slider('Weekly Meditation', 0, 10, 3)
                
            with st.expander("Social & Community"):
                places = st.slider('Places Visited', 0, 10, 5)
                core_c = st.slider('Core Circle (Close Friends)', 0, 10, 4)
                support = st.slider('Supporting Others', 0, 10, 5)
                network = st.slider('Social Network Interactions', 0, 10, 5)
                donation = st.slider('Donations/Volunteering', 0, 5, 2)
                shout = st.slider('Daily Shouting/Anger', 0, 10, 2)

            with st.expander("Work, Stress & Achievement"):
                stress = st.slider('Daily Stress', 0, 5, 3)
                achieve = st.slider('Achievement Satisfaction', 0, 10, 5)
                todo = st.slider('Todo Completed', 0, 10, 6)
                flow = st.slider('Flow State Hours', 0, 10, 4)
                vision = st.slider('Live Vision Focus', 0, 10, 5)
                vacation = st.slider('Lost Vacation Days', 0, 10, 2)
                income = st.slider('Sufficient Income (1=No, 2=Yes)', 1, 2, 2)
                awards = st.slider('Personal Awards', 0, 10, 4)
                passion = st.slider('Time for Passion', 0, 10, 4)
                wlb = st.number_input('Work-Life Balance Score (0-1000)', value=650.0)

            with st.expander("Demographics"):
                age = st.selectbox('Age Group', ['Less than 20', '21 to 35', '36 to 50', '51 or more'], index=1)
                gender = st.selectbox('Gender', ['Female', 'Male'])

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.form_submit_button("RUN HIERARCHICAL CLUSTERING", use_container_width=True)

with col2:
    with st.container(border=True):
        if predict_btn:
            age_mapped = {'Less than 20': 0, '21 to 35': 1, '36 to 50': 2, '51 or more': 3}[age]
            gender_mapped = {'Female': 0, 'Male': 1}[gender]
            
            input_df = pd.DataFrame([[
                f_veg, stress, places, core_c, support, network, achieve, donation, 
                bmi, todo, flow, steps, vision, sleep, vacation, shout, income, 
                awards, passion, meditation, age_mapped, gender_mapped, wlb
            ]], columns=[
                'FRUITS_VEGGIES', 'DAILY_STRESS', 'PLACES_VISITED', 'CORE_CIRCLE', 'SUPPORTING_OTHERS', 
                'SOCIAL_NETWORK', 'ACHIEVEMENT', 'DONATION', 'BMI_RANGE', 'TODO_COMPLETED', 'FLOW', 
                'DAILY_STEPS', 'LIVE_VISION', 'SLEEP_HOURS', 'LOST_VACATION', 'DAILY_SHOUTING', 
                'SUFFICIENT_INCOME', 'PERSONAL_AWARDS', 'TIME_FOR_PASSION', 'WEEKLY_MEDITATION', 
                'AGE', 'GENDER', 'WORK_LIFE_BALANCE_SCORE'
            ])
            
            input_scaled = scaler.transform(input_df)
            prediction = int(model.predict(input_scaled)[0])
            
            p_data = persona_map[prediction]
            
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 4rem; transition: all 0.3s ease;'>{p_data['icon']}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 5px; transition: all 0.3s ease;'>Segment: <strong>{p_data['label']}</strong></h3>", unsafe_allow_html=True)
            st.markdown(f"<p class='desc' style='text-align: center; transition: all 0.3s ease;'>{p_data['desc']}</p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)
        else:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("<h1 class='microtext' style='text-align: center; font-size: 3rem;'>✦</h1>", unsafe_allow_html=True)
            st.markdown("<h4 class='subtext' style='text-align: center; margin-bottom: 5px;'>Awaiting Agglomerative Pipeline</h4>", unsafe_allow_html=True)
            st.markdown("<p class='microtext' style='text-align: center;'>Configure the 23 metrics on the left and run<br>diagnostics to map the behavioral archetype.</p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)