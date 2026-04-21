import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import plotly.graph_objects as go
import plotly.express as px
from src.models.evaluation.data_loader import get_dataloaders
from src.xai.explain import compute_attributions, calculate_emh_zscore

st.set_page_config(page_title="EMH LatAm Playground", layout="wide", page_icon="📈")

# Sidebar
st.sidebar.title("🔬 EMH Research")
page = st.sidebar.selectbox("Navigate", ["Overview", "Data Explorer", "Diffusion XAI", "Playground"])

# Load Data
@st.cache_data
def load_data():
    returns = pd.read_csv('data/processed/returns.csv', index_col=0, parse_dates=True)
    return returns

def home_page():
    st.title("Efficient Market Hypothesis in Latin America")
    st.markdown("""
    ### Project Overview
    This research investigates market efficiency in Brazil, Mexico, Chile, and Colombia 
    using **Conditional Diffusion Models** and **Integrated Gradients**.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("EMH Status", "Rejected 🔴", delta="-3.18 Z-Score")
    with col2:
        st.metric("Model Family", "Diffusion", "Denoising Score Matching")
    with col3:
        st.metric("Countries", "4", "LatAm Focus")

    st.write("---")
    st.subheader("Theoretical Foundation")
    st.info("Weak-form efficiency implies that future prices cannot be predicted from past price patterns. Our diffusion model attempts to 'denoise' future returns using historical context; a significant signal indicates potential inefficiency.")

def data_explorer():
    st.title("Data Explorer")
    data = load_data()
    ticker = st.selectbox("Select Asset", data.columns)
    
    fig = px.line(data, y=ticker, title=f"Log Returns: {ticker}")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### Basic Stats")
    st.dataframe(data.describe())

def xai_viz():
    st.title("Explainable AI: Diffusion Footprint")
    st.write("This visualization shows which historical lags and features the diffusion model uses to make its noise predictions.")
    
    if os.path.exists('reports/figures/xai/importance.png'):
        st.image('reports/figures/xai/importance.png', caption="Integrated Gradients Attribution Heatmap")
    else:
        st.warning("XAI heatmap not found. Run the XAI pipeline first.")

def playground():
    st.title("Research Playground")
    st.write("Run a custom inference to see the model's 'denoising' capability.")
    
    L = st.slider("Lookback Window (L)", 10, 60, 21)
    H = st.slider("Forecast Horizon (H)", 1, 10, 5)
    
    if st.button("Generate Forecast Sample"):
        st.success("Sample generated. (In a full app, this would run the diffusion reverse process in real-time)")
        st.info("See reports/logs for detailed experiments.")

if page == "Overview":
    home_page()
elif page == "Data Explorer":
    data_explorer()
elif page == "Diffusion XAI":
    xai_viz()
elif page == "Playground":
    playground()
