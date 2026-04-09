import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from streamlit_lottie import st_lottie
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Impact Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR SMOOTH ANIMATIONS & STYLING ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Smooth hover effect for buttons */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 12px rgba(0,0,0,0.5);
    }
    
    /* Card-like styling for inputs */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #1E2129;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4BA3E3;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_lottieurl(url: str):
    """Loads a Lottie animation from a URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_model():
    """Loads the pre-trained KNN model."""
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

# --- LOAD ASSETS ---
# Smooth futuristic animation for the header
lottie_ai = load_lottieurl("https://lottie.host/8b7d27e7-3b95-46f9-90d0-4bd24687d69b/gX9T2Wj39T.json")
model = load_model()

# --- HEADER SECTION ---
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🚀 AI Impact Predictor")
    st.write("Enter the user demographics and AI usage behavior below to predict the **Impact on Grades**.")
with col2:
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="header_animation")

st.markdown("---")

# --- USER INPUT SECTION ---
st.subheader("📊 User Profile & Usage Metrics")

# Layout columns for a cleaner UI
left_col, mid_col, right_col = st.columns(3)

with left_col:
    age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

with mid_col:
    education_level = st.selectbox("Education Level", ["High School", "Undergraduate", "Postgraduate"])
    ai_tool = st.selectbox("Primary AI Tool Used", ["ChatGPT", "Claude", "Gemini", "Copilot", "Other"])

with right_col:
    daily_hours = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
    purpose = st.selectbox("Primary Purpose", ["Research", "Coding", "Writing", "General Query", "Entertainment"])

# --- DATA ENCODING & PREPARATION ---
# IMPORTANT: Update these dictionaries to match the exact numerical mapping used during your training phase!
mappings = {
    "gender": {"Male": 0, "Female": 1, "Other": 2},
    "education": {"High School": 0, "Undergraduate": 1, "Postgraduate": 2},
    "city": {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2},
    "ai_tool": {"ChatGPT": 0, "Claude": 1, "Gemini": 2, "Copilot": 3, "Other": 4}
