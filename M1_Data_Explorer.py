# 🧭 Milestone 3 - M1: Air Quality Data Explorer Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -----------------------------------------------------------
# 1️⃣ PAGE SETUP
# -----------------------------------------------------------
st.set_page_config(page_title="Air Quality Data Explorer", layout="wide")
st.title("🌍 AirAware - Air Quality Data Explorer")
st.markdown("Use filters to explore air quality trends, correlations, and distributions.")

# -----------------------------------------------------------
# 2️⃣ FILE UPLOAD
# -----------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your Air Quality dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # Try to find the date column automatically
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # -----------------------------------------------------------
    # 3️⃣ SIDEBAR FILTERS
    # -----------------------------------------------------------
    st.sidebar.header("🔍 Data Controls")

    # Location filter
    if 'City' in df.columns or 'Location' in df.columns:
        location_col = 'City' if 'City' in df.columns else 'Location'
        locations = df[location_col].dropna().unique().tolist()
        selected_location = st.sidebar.selectbox("Select Location", ["All"] + locations)
    else:
        selected_location = "All"

    # Pollutant filter
    pollutant_columns = [c for c in df.columns if any(p in c.upper() for p in ["PM", "NO", "SO", "CO", "O3", "AQI"])]
    selected_pollutants = st.sidebar.multiselect("Select Pollutants", pollutant_columns, default=pollutant_columns[:3])

    # Date range filter (Safe handling for missing dates)
    if date_col:
        if df[date_col].notna().any():
            min_date = df[date_col].min()
            max_date = df[date_col].max()
        else:
            min_date = pd.Timestamp("2020-01-01")
            max_date = pd.Timestamp("2020-12-31")

        start_date, end_date = st.sidebar.date_input(
            "📆 Select Date Range", [min_date, max_date]
        )
    else:
        start_date, end_date = None, None

    apply_filters = st.sidebar.button("✅ Apply Filters")

    # -----------------------------------------------------------
    # 4️⃣ FILTER DATA BASED ON USER SELECTIONS
    # -----------------------------------------------------------
    if apply_filters:
        filtered_df = df.copy()

        if selected_location != "All" and ('City' in df.columns or 'Location' in df.columns):
            filtered_df = filtered_df[filtered_df[location_col] == selected_location]

        if date_col and start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df[date_col] >= pd.to_datetime(start_date)) &
                (filtered_df[date_col] <= pd.to_datetime(end_date))
            ]

        st.subheader("📊 Filtered Data Preview")
        st.dataframe(filtered_df.head())

        # -----------------------------------------------------------
        # 5️⃣ CHECK DATA QUALITY
        # -----------------------------------------------------------
        st.subheader(" Data Quality Check")
        missing_data = filtered_df.isnull().sum()
        st.write("Missing Values in Each Column:")
        st.write(missing_data)

        completeness = 100 - (missing_data.sum() / (filtered_df.shape[0] * filtered_df.shape[1]) * 100)
        st.write(f"✅ Data Completeness: {completeness:.2f}%")

        # -----------------------------------------------------------
        # 6️⃣ CHARTS & ANALYSIS
        # -----------------------------------------------------------
        st.subheader("📈 Charts & Analysis")

        # Time Series
        if date_col and selected_pollutants:
            for pol in selected_pollutants:
                if pol in filtered_df.columns:
                    fig = px.line(filtered_df, x=date_col, y=pol, title=f"Time Series of {pol}")
                    st.plotly_chart(fig, use_container_width=True)

        # Statistical Summary
        st.subheader("📊 Statistical Summary")
        st.write(filtered_df[selected_pollutants].describe())

        # Correlation Heatmap
        st.subheader(" Pollutant Correlations")
        corr = filtered_df[selected_pollutants].corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Distribution Analysis
        st.subheader("📦 Distribution Analysis")
        for pol in selected_pollutants:
            if pol in filtered_df.columns:
                fig = px.histogram(filtered_df, x=pol, nbins=30, title=f"Distribution of {pol}")
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to begin exploring your air quality data.")
