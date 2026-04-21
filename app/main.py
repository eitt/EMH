import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EMH LatAm Dashboard", layout="wide")

st.sidebar.title("EMH Research")
page = st.sidebar.selectbox(
    "Navigate",
    ["Overview", "Data Explorer", "Diffusion XAI", "Experiment Results"],
)


@st.cache_data
def load_data():
    returns = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
    return returns


def home_page():
    st.title("Efficient Market Hypothesis in Latin America")
    st.markdown(
        """
        ### Project Overview
        This research investigates market efficiency in Brazil, Mexico, Chile, and Colombia
        using **Conditional Diffusion Models** and **Integrated Gradients**.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("EMH Inference", "Benchmark-relative", delta="No causal claim")
    with col2:
        st.metric("Model Family", "Diffusion", "Denoising Score Matching")
    with col3:
        st.metric("Countries", "4", "LatAm Focus")

    st.write("---")
    st.subheader("Theoretical Foundation")
    st.info(
        "Weak-form efficiency implies that future prices cannot be predicted from past "
        "price patterns. Our diffusion model attempts to denoise future returns using "
        "historical context; benchmark-relative forecast gains are interpreted as limited "
        "predictive evidence only, not direct proof of tradable inefficiency."
    )


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
    st.write(
        "This visualization shows which historical lags and features the diffusion model "
        "uses to make its noise predictions."
    )

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("reports/figures/xai/importance_heatmap.png"):
            st.image(
                "reports/figures/xai/importance_heatmap.png",
                caption="Integrated Gradients Attribution Heatmap",
            )
        else:
            st.warning("XAI heatmap not found. Run XAI pipeline first.")

    with col2:
        if os.path.exists("reports/figures/xai/top_features_bar.png"):
            st.image(
                "reports/figures/xai/top_features_bar.png",
                caption="Top 10 Predictors by Magnitude",
            )


def experiments_page():
    st.title("Experiment Results and Statistical Claims")
    st.markdown(
        """
        This section evaluates Conditional Diffusion model against baselines across
        multiple horizons ($H$) and lookback windows ($L$). Statistical significance is
        determined using **Diebold-Mariano test** compared to Random Walk baseline.
        """
    )

    try:
        df = pd.read_csv("reports/tables/experiment_results.csv")

        st.subheader("Hypothesis Testing Table")
        st.write(
            "Negative DM Stat indicates model performs worse than Random Walk in MSE. "
            "Positive DM Stat indicates better performance."
        )
        st.dataframe(
            df.style.format(
                {
                    "RMSE": "{:.4f}",
                    "DM_Stat": "{:.2f}",
                    "P_Value": "{:.4e}",
                }
            )
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RMSE Comparison")
            if os.path.exists("reports/figures/rmse_comparison.png"):
                st.image("reports/figures/rmse_comparison.png", use_container_width=True)

        with col2:
            st.subheader("Significance Heatmap ($p$-values)")
            if os.path.exists("reports/figures/pvalue_heatmap.png"):
                st.image("reports/figures/pvalue_heatmap.png", use_container_width=True)

    except Exception as error:
        st.error(f"Could not load experiment results: {error}. Run experiment loop first.")


if page == "Overview":
    home_page()
elif page == "Data Explorer":
    data_explorer()
elif page == "Diffusion XAI":
    xai_viz()
elif page == "Experiment Results":
    experiments_page()
