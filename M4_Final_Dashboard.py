"""
M4_Main_Dashboard.py
AirAware - Final Streamlit Dashboard (combine M1, M2, M3)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="AirAware - Main Dashboard", layout="wide")
st.title("🌍 AirAware — Main Dashboard")
st.markdown("Data Explorer • Forecast Engine • Alert System — combined")

# ---------------------------
# Helpers
# ---------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(name, model):
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))


def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def categorize_aqi(aqi):
    try:
        aqi = float(aqi)
    except:
        return "Unknown", "gray"
    if aqi <= 50:
        return "Good", "green"
    if aqi <= 100:
        return "Moderate", "yellow"
    if aqi <= 200:
        return "Unhealthy for Sensitive", "orange"
    if aqi <= 300:
        return "Unhealthy", "red"
    if aqi <= 400:
        return "Very Unhealthy", "purple"
    return "Hazardous", "maroon"


# ---------------------------
# File upload / load default
# ---------------------------
st.sidebar.header("📁 Data & Controls")

uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])
use_sample = False
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.sidebar.info("No file uploaded — please upload your dataset.")
    df = None

# If user uploaded or df exists, normalize column names
if df is not None:
    df.columns = [c.strip() for c in df.columns]
    # detect date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        st.error("No date column found (column name containing 'date'). Please upload correct file.")
        st.stop()
    # parse dates and sort
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)
    df.rename(columns={date_col: "Date"}, inplace=True)

    # ensure City column (or Station)
    city_col = None
    if "City" in df.columns:
        city_col = "City"
    else:
        for c in df.columns:
            if "station" in c.lower() or "location" in c.lower():
                city_col = c
                break

    # detect common pollutant columns
    pollutant_candidates = [c for c in df.columns if any(p in c.upper() for p in ["PM2.5", "PM2_5", "PM25", "PM10", "O3", "NO2", "SO2", "CO", "AQI"])]
    # normalize column names for PM2.5 variations
    if "PM2_5" in df.columns and "PM2.5" not in df.columns:
        df["PM2.5"] = df["PM2_5"]
    if "PM25" in df.columns and "PM2.5" not in df.columns:
        df["PM2.5"] = df["PM25"]

    # ensure AQI column exists (optional)
    has_aqi = "AQI" in df.columns

    # ---------------------------
    # Sidebar controls
    # ---------------------------
    st.sidebar.markdown("### Filters")
    cities = df[city_col].unique().tolist() if city_col else ["All"]
    selected_city = st.sidebar.selectbox("Station / City", options=cities)
    # timeframe
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input("Time Range", [min_date, max_date])
    # pollutant list (common)
    pollutant_list = [p for p in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"] if p in df.columns]
    if not pollutant_list:
        # fallback: choose numeric cols excluding Date/City/AQI
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        pollutant_list = [c for c in numeric_cols if c not in ["AQI"]][:4]

    selected_pollutant = st.sidebar.selectbox("Pollutant", pollutant_list)
    forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 14, 7)
    refresh = st.sidebar.button("Update / Apply Filters")

    # Filter data when Apply pressed (or initially when app loads)
    # convert date_range to timestamps
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    df_filtered = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    if city_col:
        df_filtered = df_filtered[df_filtered[city_col] == selected_city]

    st.write(f"### Data preview ({selected_city}) — {len(df_filtered)} records")
    st.dataframe(df_filtered.head())

    # ---------------------------
    # Layout: 2 columns top (gauge + quick stats), charts below
    # ---------------------------
    col1, col2 = st.columns([1, 2])

    # AQI Gauge in left column
    with col1:
        st.subheader("🌡️ Current AQI")
        if has_aqi and not df_filtered.empty:
            latest_aqi = df_filtered["AQI"].dropna().iloc[-1]
            status, color = categorize_aqi(latest_aqi)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=float(latest_aqi),
                delta={'reference': df_filtered["AQI"].dropna().iloc[-2] if len(df_filtered["AQI"].dropna()) > 1 else latest_aqi},
                title={'text': f"AQI: {status}"},
                gauge={'axis': {'range': [0, 500]},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgreen"},
                           {'range': [51, 100], 'color': "yellow"},
                           {'range': [101, 200], 'color': "orange"},
                           {'range': [201, 300], 'color': "red"},
                           {'range': [301, 500], 'color': "purple"}
                       ],
                       'bar': {'color': color}}
            ))
            st.plotly_chart(gauge, use_container_width=True)
            st.write(f"Latest AQI: **{latest_aqi:.1f}** — {status}")
        else:
            st.warning("AQI column not available or no data for selected filters.")

    # Quick stats + Alerts in right column
    with col2:
        st.subheader("📊 Quick Stats & Alerts")
        if not df_filtered.empty:
            # Quick stats
            mean_val = df_filtered[selected_pollutant].mean() if selected_pollutant in df_filtered.columns else np.nan
            max_val = df_filtered[selected_pollutant].max() if selected_pollutant in df_filtered.columns else np.nan
            st.metric(label=f"Avg {selected_pollutant}", value=f"{mean_val:.2f}")
            st.metric(label=f"Max {selected_pollutant}", value=f"{max_val:.2f}")

            # Alerts panel: if forecast or latest exceed thresholds
            alerts = []
            # threshold simple example: use WHO short-term or rough thresholds
            thresholds = {"PM2.5": 60, "PM10": 100, "O3": 120}
            if selected_pollutant in thresholds and selected_pollutant in df_filtered.columns:
                if df_filtered[selected_pollutant].max() > thresholds[selected_pollutant]:
                    alerts.append(f"{selected_pollutant} exceeded threshold ({thresholds[selected_pollutant]}) in selected range.")

            if has_aqi and not df_filtered.empty:
                if df_filtered["AQI"].max() > 200:
                    alerts.append("AQI exceeded 200 (very poor/hazardous) in selected range.")

            if alerts:
                for a in alerts:
                    st.error(f"⚠️ {a}")
            else:
                st.success("✅ No active high-level alerts in the selected range.")
        else:
            st.info("No data for selected filters.")

    st.markdown("---")

    # ---------------------------
    # Pollutant Trends (multi-line)
    # ---------------------------
    st.subheader("📈 Pollutant Trends")
    if not df_filtered.empty:
        trend_pollutants = [p for p in ["PM2.5", "PM10", "O3"] if p in df_filtered.columns]
        if trend_pollutants:
            fig_trend = go.Figure()
            for p in trend_pollutants:
                fig_trend.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered[p], mode="lines", name=p))
            # WHO dashed lines (example values)
            who_limits = {"PM2.5": 25, "PM10": 50, "O3": 100}
            # Add horizontal lines as shapes
            for p in trend_pollutants:
                limit = who_limits.get(p, None)
                if limit:
                    fig_trend.add_hline(y=limit, line_dash="dash", line_color="red",
                                        annotation_text=f"WHO {p} limit", annotation_position="top left")
            fig_trend.update_layout(title=f"Pollutant trends ({selected_city})", xaxis_title="Date", yaxis_title="Concentration")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("No standard pollutant columns (PM2.5, PM10, O3) present to plot.")
    else:
        st.info("No data to show pollutant trends.")

    st.markdown("---")

    # ---------------------------
    # Forecast (ARIMA) for PM2.5 (Actual vs Forecast + CI if available)
    # ---------------------------
    st.subheader(f"🔮 Forecast — {selected_pollutant} (Actual vs Forecast)")

    # We'll fallback to ARIMA on selected_pollutant if numeric and has enough data
    if selected_pollutant in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[selected_pollutant]):
        try:
            series = df_filtered.set_index("Date")[selected_pollutant].asfreq("D").fillna(method="ffill")
            # check length
            if len(series.dropna()) < 30:
                st.warning("Not enough data for reliable ARIMA. Need ~30+ days of data.")
            else:
                # try load saved model for this city/pollutant if exists
                model_name = f"arima_{selected_city}_{selected_pollutant}".replace(" ", "_")
                model = load_model(model_name)
                if model is None:
                    # train small ARIMA quickly (2,1,2)
                    with st.spinner("Training ARIMA model (quick)..."):
                        model = ARIMA(series, order=(2, 1, 2)).fit()
                        save_model(model_name, model)
                # forecast
                fh = forecast_horizon
                forecast = model.forecast(steps=fh)
                last_date = series.index.max()
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=fh, freq="D")
                # plot
                figf = go.Figure()
                figf.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Actual"))
                figf.add_trace(go.Scatter(x=future_dates, y=forecast, mode="lines+markers", name="Forecast"))
                figf.update_layout(title=f"{selected_pollutant} — Actual vs Forecast ({selected_city})",
                                   xaxis_title="Date", yaxis_title=selected_pollutant)
                st.plotly_chart(figf, use_container_width=True)
                # show numeric forecast table
                forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": np.round(forecast, 2)})
                st.write("Forecast (next days):")
                st.dataframe(forecast_df.set_index("Date"))
        except Exception as e:
            st.error(f"Forecasting error: {e}")
    else:
        st.info("Selected pollutant not numeric or not present — cannot forecast.")

    st.markdown("---")

    # ---------------------------
    # Admin Mode: Upload new data / retrain models
    # ---------------------------
    st.sidebar.header("⚙️ Admin")
    admin = st.sidebar.checkbox("Enable Admin Mode")
    if admin:
        st.sidebar.subheader("Upload new dataset")
        new_data = st.sidebar.file_uploader("Upload CSV to replace dataset (Admin)", type=["csv"], key="admin_upload")
        if new_data:
            new_df = pd.read_csv(new_data)
            st.sidebar.success("New dataset uploaded. (You may need to restart app to load into memory.)")

        st.sidebar.markdown("### Retrain models")
        if st.sidebar.button("Retrain ARIMA models for all pollutants"):
            # quick retrain loop (train per pollutant per selected_city)
            retrain_pollutants = [p for p in ["PM2.5", "PM10", "O3"] if p in df_filtered.columns]
            with st.spinner("Retraining models..."):
                for p in retrain_pollutants:
                    try:
                        s = df_filtered.set_index("Date")[p].asfreq("D").fillna(method="ffill")
                        if len(s.dropna()) < 30:
                            st.warning(f"Skipping retrain for {p} — not enough data.")
                            continue
                        m = ARIMA(s, order=(2, 1, 2)).fit()
                        save_model(f"arima_{selected_city}_{p}".replace(" ", "_"), m)
                    except Exception as e:
                        st.error(f"Retrain error for {p}: {e}")
                st.success("Retrain complete. Models saved to /models.")
else:
    st.info("Upload dataset from the left sidebar to start.")
