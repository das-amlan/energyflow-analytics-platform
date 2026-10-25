import pandas as pd
import plotly.express as px
import streamlit as st

from database import (
    get_historical_demand,
    get_forecast_origins,
    get_forecast,
    get_monitoring,
)


st.set_page_config(
    page_title="EnergyFlow Analytics Platform",
    page_icon="⚡",
    layout="wide",
)

st.title("EnergyFlow Analytics Platform")
st.caption(
    "Electricity demand forecasting and MLOps monitoring "
    "| PJM East | Databricks"
)


@st.cache_data(ttl=300, show_spinner=False)
def load_history():
    return get_historical_demand(3000)


@st.cache_data(ttl=300, show_spinner=False)
def load_origins():
    return get_forecast_origins()


@st.cache_data(ttl=300, show_spinner=False)
def load_forecast(origin):
    return get_forecast(origin)


@st.cache_data(ttl=300, show_spinner=False)
def load_monitoring():
    return get_monitoring()


page = st.sidebar.radio(
    "Navigation",
    [
        "Historical Analysis",
        "Demand Forecasting",
        "Model Monitoring",
    ],
)

st.sidebar.caption(
    "Historical replay based on the PJME electricity dataset."
)


try:
    if page == "Historical Analysis":
        st.header("Historical Electricity Demand")

        df = load_history()
        df["datetime_local"] = pd.to_datetime(
            df["datetime_local"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Demand",
            f"{df['demand_mw'].mean():,.0f} MW",
        )
        col2.metric(
            "Peak Demand",
            f"{df['demand_mw'].max():,.0f} MW",
        )
        col3.metric(
            "Observations",
            f"{len(df):,}",
        )

        fig = px.line(
            df,
            x="datetime_local",
            y="demand_mw",
            title="Historical Electricity Demand",
            labels={
                "datetime_local": "Date",
                "demand_mw": "Demand (MW)",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Hourly Demand Pattern")

        df["hour"] = df["datetime_local"].dt.hour

        hourly = (
            df.groupby("hour", as_index=False)["demand_mw"]
            .mean()
        )

        fig_hourly = px.bar(
            hourly,
            x="hour",
            y="demand_mw",
            title="Average Demand by Hour",
        )

        st.plotly_chart(
            fig_hourly,
            use_container_width=True,
        )

        st.caption(
            "Statistics are calculated from the latest "
            "3,000 historical observations, not the entire dataset."
        )

    elif page == "Demand Forecasting":
        st.header("24-Hour Demand Forecast")

        st.info(
            "These are saved historical forecasts, "
            "not real-time future predictions."
        )

        origins = load_origins()

        if origins.empty:
            st.warning("No forecast data available.")
            st.stop()

        selected = st.selectbox(
            "Select forecast origin",
            options=origins["forecast_origin"].tolist(),
            format_func=lambda x: str(x),
        )

        df = load_forecast(selected)

        if df.empty:
            st.warning("No predictions for this origin.")
            st.stop()

        df["forecast_timestamp"] = pd.to_datetime(
            df["forecast_timestamp"]
        )

        metrics_col1, metrics_col2 = st.columns(2)

        metrics_col1.metric(
            "Forecast MAE",
            f"{df['absolute_error_mw'].mean():,.2f} MW",
        )

        metrics_col2.metric(
            "Peak Predicted Demand",
            f"{df['predicted_mw'].max():,.0f} MW",
        )

        chart_data = df.melt(
            id_vars=["forecast_timestamp"],
            value_vars=["actual_mw", "predicted_mw"],
            var_name="Series",
            value_name="Demand (MW)",
        )

        fig = px.line(
            chart_data,
            x="forecast_timestamp",
            y="Demand (MW)",
            color="Series",
            markers=True,
            title="Predicted vs. Actual Demand",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Hourly Forecast Details")
        st.dataframe(df, use_container_width=True)

    elif page == "Model Monitoring":
        st.header("Model Performance Monitoring")

        df = load_monitoring()

        if df.empty:
            st.warning("No monitoring data available.")
            st.stop()

        df["forecast_date"] = pd.to_datetime(
            df["forecast_date"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Daily MAE",
            f"{df['mae_mw'].mean():,.2f} MW",
        )
        col2.metric(
            "Average Daily RMSE",
            f"{df['rmse_mw'].mean():,.2f} MW",
        )
        col3.metric(
            "Average Daily sMAPE",
            f"{df['smape_pct'].mean():.2f}%",
        )

        fig = px.line(
            df,
            x="forecast_date",
            y=["mae_mw", "rmse_mw"],
            title="Forecast Errors Over Time",
        )

        st.plotly_chart(fig, use_container_width=True)

        fig_smape = px.line(
            df,
            x="forecast_date",
            y="smape_pct",
            title="Daily sMAPE",
        )

        st.plotly_chart(
            fig_smape,
            use_container_width=True,
        )

        st.subheader("Evidently Drift Monitoring")
        st.info(
            "The Evidently report has been generated "
            "in Databricks. Persisted drift statistics "
            "will be integrated into this page next."
        )

except Exception as exc:
    st.error("Unable to load dashboard data.")
    st.exception(exc)
