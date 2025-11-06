import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AirAware - Alert System", layout="wide")

st.title("🚨 AirAware - Air Quality Alert System")

# 1️⃣ Upload Dataset
uploaded_file = st.file_uploader("📂 Upload Air Quality Dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Convert date column if available
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            df[col] = pd.to_datetime(df[col])
            break

    # Select city/station if available
    location_col = None
    for col in df.columns:
        if 'city' in col.lower() or 'station' in col.lower():
            location_col = col
            break

    if location_col:
        selected_location = st.selectbox("🏙️ Select Location/Station", df[location_col].unique())
        df = df[df[location_col] == selected_location]

    st.subheader("📊 Current Air Quality Overview")

    # 2️⃣ AQI Donut Chart
    current_aqi = df['AQI'].iloc[-1] if 'AQI' in df.columns else 0

    def categorize_aqi(aqi):
        if aqi <= 50: return "Good", "green"
        elif aqi <= 100: return "Moderate", "yellow"
        elif aqi <= 200: return "Unhealthy for Sensitive", "orange"
        elif aqi <= 300: return "Unhealthy", "red"
        elif aqi <= 400: return "Very Unhealthy", "purple"
        else: return "Hazardous", "maroon"

    aqi_status, color = categorize_aqi(current_aqi)

    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_aqi,
        title={"text": f"Current AQI: {aqi_status}"},
        gauge={"axis": {"range": [0, 500]}, "bar": {"color": color}}
    ))
    st.plotly_chart(fig1, use_container_width=True)

    # 3️⃣ Pollutant Concentrations Chart
    pollutants = ['PM2.5', 'PM10', 'O3']
    available = [p for p in pollutants if p in df.columns]

    if available:
        st.subheader("💨 Pollutant Concentrations vs WHO Limits")
        fig2 = go.Figure()
        for p in available:
            fig2.add_trace(go.Scatter(x=df[date_col], y=df[p], mode='lines', name=p))
        # WHO limits
        limits = {'PM2.5': 25, 'PM10': 50, 'O3': 100}
        for p, limit in limits.items():
            if p in available:
                fig2.add_trace(go.Scatter(
                    x=df[date_col], y=[limit]*len(df),
                    mode='lines', name=f"{p} WHO Limit", line=dict(dash='dash')
                ))
        st.plotly_chart(fig2, use_container_width=True)

    # 4️⃣ 7-Day Forecast (Dummy Example)
    st.subheader("📅 7-Day Forecast AQI Status")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    forecast_aqi = [45, 80, 130, 200, 160, 90, 50]
    forecast_df = pd.DataFrame({'Day': days, 'AQI': forecast_aqi})
    forecast_df['Category'] = forecast_df['AQI'].apply(lambda x: categorize_aqi(x)[0])
    forecast_df['Color'] = forecast_df['AQI'].apply(lambda x: categorize_aqi(x)[1])

    fig3 = px.bar(forecast_df, x='Day', y='AQI', color='Category', color_discrete_map={
        'Good': 'green', 'Moderate': 'yellow', 'Unhealthy for Sensitive': 'orange',
        'Unhealthy': 'red', 'Very Unhealthy': 'purple', 'Hazardous': 'maroon'
    })
    st.plotly_chart(fig3, use_container_width=True)

    # 5️⃣ Active Alerts
    st.subheader("⚠️ Active Alerts")
    alerts = forecast_df[forecast_df['AQI'] > 150]
    if not alerts.empty:
        st.error(f"High pollution alerts for: {', '.join(alerts['Day'].tolist())}")
    else:
        st.success("✅ No high pollution alerts this week!")

else:
    st.info("👆 Upload your air quality dataset to start analysis.")
