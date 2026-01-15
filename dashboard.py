"""
Elexon BMRS Data Analysis Dashboard
A Streamlit web application to visualise and analyse UK energy market data. 

Run with: streamlit run dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Elexon BMRS Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - this will be minimal for clarity. I have added some basic styles such as headers and metric cards.
st.markdown("""
<style>
    .main-header{
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header{
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card{
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin:  10px 0;
    }
    .stMetric{
        background-color: #ffffff;
        border-radius: 10px;
        padding:  15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class ElexonAPIClient:
    """Client for fetching data from the Elexon BMRS API with dynamic date ranges"""

    def __init__(self):
        self.base_url = "https://data.elexon.co.uk/bmrs/api/v1"
        self.headers = {"accept": "application/json"}
        st.caption("[ELEXON] Initialised Elexon API Client")

    def fetch_actual_load(self, date_from, date_to, sp_from=1, sp_to=48):
        """Fetch actual total load data"""
        url = f"{self.base_url}/demand/actual/total?from={date_from}&to={date_to}&fromSettlementPeriod={sp_from}&toSettlementPeriod={sp_to}"

        try :
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # filter by settlement periods client-side if needed
                if data and 'data' in data:
                    filtered_data = [entry for entry in data['data'] if sp_from <= int(entry.get('settlementPeriod', 0)) <= sp_to]
                    data['data'] = filtered_data
                return data
            else:
                st.error(f"[ELEXON] Error Fetching Actual Total Load Data: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            st.error(f"[ELEXON] Request Timed out. Please Try again!")
            return None
        except Exception as e:
            st.error(f"[ELEXON] Exception Fetching Actual Total Load Data: {e}")
            return None

    def fetch_day_ahead_forecast(self, date_from, date_to, sp_from=1, sp_to=48):
        """Fetch day-ahead demand forecast (historical/latest)"""
        url = f"{self.base_url}/forecast/demand/day-ahead/latest?from={date_from}&to={date_to}&fromSettlementPeriod={sp_from}&toSettlementPeriod={sp_to}"

        try :
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # filter by settlement periods client-side if needed
                if data and 'data' in data:
                    filtered_data = [entry for entry in data['data'] if sp_from <= int(entry.get('settlementPeriod', 0)) <= sp_to]
                    data['data'] = filtered_data
                return data
            else:
                st.error(f"[ELEXON] Error Fetching Historical Demand Forecast (Day-ahead) Data: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            st.error(f"[ELEXON] Request Timed out. Please Try again!")
            return None
        except Exception as e:
            st.error(f"[ELEXON] Exception Day-ahead Demand Forecast Data: {e}")
            return None

    def fetch_demand_forecast_today(self):
        """Fetch today's demand forecast (no date params needed)"""
        url = f"{self.base_url}/forecast/demand/day-ahead?format=json"

        try :
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                st.error(f"[ELEXON] Error Fetching Latest Demand Forecast (Day-ahead) Data: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            st.error(f"[ELEXON] Request Timed out. Please Try again!")
            return None
        except Exception as e:
            st.error(f"[ELEXON] Exception Fetching Latest Demand Forecast (Day-ahead) Data: {e}")
            return None

    def fetch_indicated_forecast_today(self):
        """Fetch today's indicated forecast (no date params needed)"""
        url = f"{self.base_url}/forecast/indicated/day-ahead?format=json"
        
        try :
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                st.error(f"[ELEXON] Error Fetching Latest Indicated Demand Forecast (Day-ahead) Data: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            st.error(f"[ELEXON] Request Timed out. Please Try again!")
            return None
        except Exception as e:
            st.error(f"[ELEXON] Exception Fetching Latest Indicated Demand Forecast (Day-ahead) Data: {e}")
            return None        


def process_actual_load_data(data):
    """Process actual load data into a clean DataFrame"""
    if not data or 'data' not in data: 
        return None

    df = pd.DataFrame(data['data'])
    if df.empty:
        return None

    if 'startTime' in df.columns:
        df['startTime'] = pd.to_datetime(df['startTime'], utc=True, errors='coerce').dt.tz_convert(None)
        df = df.sort_values('startTime')

    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['settlementPeriod'] = pd.to_numeric(df['settlementPeriod'], errors='coerce')
    df = df.dropna(subset=['quantity', 'settlementPeriod'])

    return df


def process_forecast_data(data):
    """Process day-ahead forecast data into a clean DataFrame"""
    if not data or 'data' not in data:
        return None

    df = pd.DataFrame(data['data'])
    if df.empty:
        return None

    if 'startTime' in df.columns:
        df['startTime'] = pd.to_datetime(df['startTime'], utc=True, errors='coerce').dt.tz_convert(None)
        df = df.sort_values('startTime')

    df['transmissionSystemDemand'] = pd.to_numeric(df['transmissionSystemDemand'], errors='coerce')
    df['settlementPeriod'] = pd.to_numeric(df['settlementPeriod'], errors='coerce')

    if 'nationalDemand' in df.columns:
        df['nationalDemand'] = pd.to_numeric(df['nationalDemand'], errors='coerce')

    df = df.dropna(subset=['transmissionSystemDemand', 'settlementPeriod'])

    # Calculate moving averages - 4-hour moving average (8 periods of 30 mins)
    df['transmissionSystemDemand_MA'] = df['transmissionSystemDemand'].rolling(window=8, min_periods=1).mean()
    if 'nationalDemand' in df.columns:
        df['nationalDemand_MA'] = df['nationalDemand']. rolling(window=8, min_periods=1).mean()

    return df


def process_indicated_forecast(data):
    """Process indicated forecast data"""
    if not data or 'data' not in data: 
        return None

    df = pd.DataFrame(data['data'])
    if df.empty:
        return None

    if 'startTime' in df.columns:
        df['startTime'] = pd.to_datetime(df['startTime'], utc=True, errors='coerce').dt.tz_convert(None)
        df = df. sort_values('startTime')

    df['indicatedGeneration'] = pd. to_numeric(df['indicatedGeneration'], errors='coerce')
    df['indicatedDemand'] = pd.to_numeric(df['indicatedDemand'], errors='coerce')

    # Calculate moving averages - 8-hour moving average (16 periods of 30 mins)
    df['indicatedGeneration_MA'] = df['indicatedGeneration'].rolling(window=16, min_periods=1).mean()
    df['indicatedDemand_MA'] = df['indicatedDemand']. rolling(window=16, min_periods=1).mean()

    return df


def run_load_prediction(df):
    """Run Random Forest prediction model - replicates elexon_actual_total_load_predictor.py"""
    df = df.copy()
    df = df.sort_values('startTime')
    df['lag1'] = df['quantity'].shift(1)
    df = df.dropna()

    if len(df) < 10:
        return None, None, None

    X = df[['settlementPeriod', 'lag1']].values
    y = df['quantity']

    scaler = StandardScaler() # using feature scaling for better model performance
    X_scaled = scaler. fit_transform(X) 

    model = RandomForestRegressor(n_estimators=100, random_state=42) # Using 100 trees for better performance
    model.fit(X_scaled, y)

    # Model metrics (same as your script)
    y_pred_model = model.predict(X_scaled)
    y_persist = df['lag1'].values               # Persistence baseline (lag-1): predict current value as previous value, a common naive benchmark in time series forecasting
    y_true = y.values

    def mae(a, b):                                  # Using mean absolute error to evaluate model. High mae indicates poor performance. Model should have lower mae than persistence baseline.
        return float(np.mean(np.abs(a - b)))

    def rmse(a, b):                             # Using root mean squared error to evaluate model. High rmse indicates poor performance. Model should have lower rmse than persistence baseline.
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def bias(true, pred):                       # Bias indicates whether model tends to over or under predict on average. Positive bias means over prediction, negative means under prediction.
        return float(np. mean(pred - true))

    metrics = {
        'model_mae': mae(y_true, y_pred_model),
        'model_rmse': rmse(y_true, y_pred_model),
        'model_bias': bias(y_true, y_pred_model),
        'persistence_mae': mae(y_true, y_persist),
        'persistence_rmse': rmse(y_true, y_persist),
        'persistence_bias': bias(y_true, y_persist),
        'n_samples': int(len(df))
    }

    # Predict next 48 periods
    predictions = []
    current_lag = df['quantity'].iloc[-1]
    current_sp = int(df['settlementPeriod'].iloc[-1])

    for i in range(48):
        next_sp = (current_sp % 48) + 1
        next_X = scaler.transform(np.array([[next_sp, current_lag]]))
        pred = model.predict(next_X)[0]
        predictions. append(float(pred))
        current_lag = pred
        current_sp = next_sp

    last_time = df['startTime'].max()
    next_times = pd.date_range(start=last_time + pd.Timedelta(minutes=30), periods=48, freq="30min")

    predictions_df = pd. DataFrame({
        'startTime': next_times,
        'settlementPeriod':  list(range(1, 49)),
        'quantity': predictions,
        'is_predicted': True
    })

    return predictions_df, metrics, model. feature_importances_


def run_linear_regression(df_forecast, df_actual):
    """Run linear regression - replicates elexon_linear_regression.py"""
    df_forecast = df_forecast.copy()
    df_actual = df_actual. copy()

    # Determine join keys - We use join keys because the datasets may have different time formats
    join_keys = []
    if "startTime" in df_forecast.columns and "startTime" in df_actual.columns:
        join_keys = ["startTime", "settlementPeriod"]
    elif "settlementDate" in df_forecast.columns and "settlementDate" in df_actual.columns:
        join_keys = ["settlementDate", "settlementPeriod"]
    else:
        st. warning("No suitable join keys found for merging datasets")
        return None, None

    df_merged = pd.merge(
        df_forecast, df_actual,
        on=join_keys,
        how='inner',
        suffixes=('_forecast', '_actual')
    )
    df_merged = df_merged.dropna(subset=["transmissionSystemDemand", "quantity"])
    df_merged = df_merged.sort_values(["startTime", "settlementPeriod"]).reset_index(drop=True)

    if df_merged.empty:
        st.warning("No matching data points found between forecast and actual data")
        return None, None

    # Spread analysis (same as Linear regression script) - Calculate difference between forecast and actual demand to assess accuracy
    df_merged["spread"] = df_merged["quantity"] - df_merged["transmissionSystemDemand"]
    df_merged["abs_spread"] = df_merged["spread"]. abs()

    X = df_merged[['transmissionSystemDemand']].values
    y = df_merged['quantity']. values

    model = LinearRegression()
    model.fit(X, y)

    # Time series cross-validation (same as Linear regression script) - Evaluate model stability and generalisation using time series split
    n_splits = min(5, len(df_merged) // 2)
    if n_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        r2_mean = float(scores.mean())
        r2_std = float(scores.std())
    else:
        r2_mean = float(r2_score(y, model.predict(X)))
        r2_std = 0.0

    results = {
        'slope': float(model.coef_[0]),
        'intercept':  float(model.intercept_),
        'r2_score': r2_mean,
        'r2_std': r2_std,
        'mean_spread': float(df_merged['spread'].mean()),
        'std_spread': float(df_merged['spread'].std()),
        'p05_spread': float(df_merged['spread'].quantile(0.05)),
        'p95_spread': float(df_merged['spread'].quantile(0.95)),
        'min_abs_spread': float(df_merged['abs_spread'].min()),
        'max_abs_spread': float(df_merged['abs_spread'].max()),
        'mean_abs_spread': float(df_merged['abs_spread'].mean()),
        'n_samples': int(len(df_merged))
    }

    return results, df_merged


def interpret_r2(r2_score_val):
    """Interpret R² score - same logic as elexon_linear_regression.py"""
    if r2_score_val < 0:
        return ("⚠️ R² < 0: The transmission demand forecast has no meaningful relationship "
                "with actual load and performs worse than a naïve mean-based estimate.", "red")
    elif r2_score_val < 0.1:
        return ("📉 0 ≤ R² < 0.1: The transmission demand forecast contains almost no information "
                "about actual load.  Knowing the forecast barely improves estimation.", "orange")
    elif r2_score_val < 0.3:
        return ("📊 0.1 ≤ R² < 0.3: The transmission demand forecast contains a weak but real signal.  "
                "Most variation is driven by other factors.", "yellow")
    elif r2_score_val < 0.6:
        return ("📈 0.3 ≤ R² < 0.6: The forecast explains a meaningful portion of variation, "
                "but significant variability remains unexplained.", "lightgreen")
    elif r2_score_val < 0.8:
        return ("✅ 0.6 ≤ R² < 0.8: The forecast has a strong relationship with actual load "
                "and explains most of the observed variation.", "green")
    else: 
        return ("🎯 R² ≥ 0.8: The forecast and actual load are very tightly coupled.  "
                "The forecast is extremely accurate.", "darkgreen")


def main():
    # Header
    st.markdown('<p class="main-header">⚡ Elexon BMRS Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">UK Energy Market Data Analysis | Quantitative Trading Mini Project</p>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.title("📊 Configuration")

    # Initialise API client
    client = ElexonAPIClient()

    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    st.sidebar.caption("Select your analysis period (max 7 days)")

    default_end = datetime.now().date() - timedelta(days=1)
    default_start = default_end - timedelta(days=6)  # 7 days inclusive

    date_from = st.sidebar. date_input(
        "From Date",
        value=default_start,
        max_value=default_end,
        help="Start date for historical data"
    )
    date_to = st.sidebar.date_input(
        "To Date",
        value=default_end,
        max_value=default_end,
        help="End date for historical data"
    )

    # Validate date range
    if date_from > date_to: 
        st.sidebar. error("❌ 'From Date' must be before 'To Date'")
        return

    # Check 7-day limit
    date_range_days = (date_to - date_from).days + 1  # +1 for inclusive
    if date_range_days > 7:
        st. sidebar.error(f"❌ Date range cannot exceed 7 days (currently {date_range_days} days). Please adjust your dates.")
        return
    else:
        st. sidebar.success(f"✅ {date_range_days} day(s) selected")

    # Settlement period selection
    st.sidebar.subheader("⏰ Settlement Periods")
    sp_from, sp_to = st.sidebar.slider(
        "Settlement Period Range (1-48)",
        min_value=1, max_value=48,
        value=(1, 48),
        help="Each settlement period is 30 minutes.  1-48 covers a full day."
    )

    # Analysis type selection
    st. sidebar.subheader("📈 Analysis Type")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        [
            "📈 Load Prediction (Random Forest)",
            "🔗 Regression Analysis (Forecast vs Actual)",
            "📊 Statistical Overview (Today's Forecasts)",
            "🎯 All Analyses"
        ]
    )

    # Display selected parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Selected Parameters:**")
    st.sidebar.code(f"From: {date_from}\nTo: {date_to}\nDays: {date_range_days}\nPeriods: {sp_from}-{sp_to}")

    # Fetch button
    if st. sidebar.button("🔄 Fetch Data & Analyse", type="primary", use_container_width=True):
        st.session_state['fetch_triggered'] = True
        st.session_state['params'] = {
            'date_from':  str(date_from),
            'date_to': str(date_to),
            'sp_from':  sp_from,
            'sp_to':  sp_to
        }
        # Clear previous data
        for key in ['df_actual', 'df_forecast', 'df_demand_today', 'df_indicated_today']: 
            st.session_state. pop(key, None)

    # Process data if fetch was triggered
    if st.session_state. get('fetch_triggered', False):
        params = st.session_state. get('params', {})

        with st.spinner(f"📡 Fetching data from {params. get('date_from')} to {params.get('date_to')}..."):
            # Fetch historical actual load
            actual_data = client.fetch_actual_load(
                params['date_from'], params['date_to'],
                params['sp_from'], params['sp_to']
            )
            if actual_data: 
                df_actual = process_actual_load_data(actual_data)
                if df_actual is not None: 
                    st. session_state['df_actual'] = df_actual
                    st.success(f"✅ Loaded {len(df_actual)} actual load records")

            # Fetch historical day-ahead forecast
            forecast_data = client. fetch_day_ahead_forecast(
                params['date_from'], params['date_to'],
                params['sp_from'], params['sp_to']
            )
            if forecast_data:
                df_forecast = process_forecast_data(forecast_data)
                if df_forecast is not None:
                    st.session_state['df_forecast'] = df_forecast
                    st.success(f"✅ Loaded {len(df_forecast)} day-ahead forecast records")

            # Fetch today's forecasts (for statistical analysis)
            demand_today = client. fetch_demand_forecast_today()
            if demand_today:
                df_demand_today = process_forecast_data(demand_today)
                if df_demand_today is not None: 
                    st.session_state['df_demand_today'] = df_demand_today

            indicated_today = client. fetch_indicated_forecast_today()
            if indicated_today: 
                df_indicated_today = process_indicated_forecast(indicated_today)
                if df_indicated_today is not None:
                    st.session_state['df_indicated_today'] = df_indicated_today

    # Get data from session state
    df_actual = st. session_state.get('df_actual')
    df_forecast = st.session_state.get('df_forecast')
    df_demand_today = st.session_state.get('df_demand_today')
    df_indicated_today = st. session_state.get('df_indicated_today')

    # Display analyses
    if df_actual is not None: 

        # ==================== LOAD PREDICTION ====================
        if analysis_type in ["📈 Load Prediction (Random Forest)", "🎯 All Analyses"]:
            st.markdown("---")
            st.header("📈 Load Prediction Analysis")
            st.markdown("*Using Random Forest Regression to predict the next 48 settlement periods (24 hours)*")

            predictions_df, metrics, feature_importance = run_load_prediction(df_actual)

            if predictions_df is not None:
                # Metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model MAE", f"{metrics['model_mae']:.2f} MW")    # Mean Absolute Error (MAE) indicates average prediction error magnitude. Lower is better.
                with col2:
                    st.metric("Model RMSE", f"{metrics['model_rmse']:.2f} MW")  # Root Mean Squared Error (RMSE) penalises larger errors more than MAE. Lower is better.
                with col3:
                    st. metric("Persistence MAE", f"{metrics['persistence_mae']:.2f} MW")   # Baseline MAE using persistence (lag-1) model for comparison.
                with col4:
                    improvement = (metrics['persistence_mae'] - metrics['model_mae']) / metrics['persistence_mae'] * 100    # Improvement over persistence baseline in percentage. Higher is better because it shows how much better the model is compared to the baseline.
                    st.metric("Improvement vs Baseline", f"{improvement:.1f}%",
                              delta=f"{improvement:.1f}%" if improvement > 0 else None)

                # Additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Bias", f"{metrics['model_bias']:.2f} MW")   # Bias indicates whether model tends to over or under predict on average. Positive bias means over prediction, negative means under prediction.
                with col2:
                    st.metric("Persistence Bias", f"{metrics['persistence_bias']:.2f} MW")   # Bias for persistence baseline model.
                with col3:
                    st.metric("Training Samples", f"{metrics['n_samples']}")    # Number of samples used for training the model.

                # Time series plot
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_actual['startTime'],
                    y=df_actual['quantity'],
                    mode='lines',
                    name='Actual Load',
                    line=dict(color='#1f77b4', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=predictions_df['startTime'],
                    y=predictions_df['quantity'],
                    mode='lines',
                    name='Predicted Load (Next 48 periods)',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))

                fig.update_layout(
                    title="Actual Total Load & Predictions",
                    xaxis_title="Time",
                    yaxis_title="Quantity (MW)",
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Predictions table
                st.columns(1)
                st.subheader("Predicted Values (Next 48 Periods)")
                display_df = predictions_df[['startTime', 'settlementPeriod', 'quantity']].copy()
                display_df['quantity'] = display_df['quantity'].round(2)
                display_df.columns = ['Time', 'Settlement Period', 'Predicted Load (MW)']
                st.dataframe(display_df, use_container_width=True, height=200)

        # ==================== REGRESSION ANALYSIS ====================
        if analysis_type in ["🔗 Regression Analysis (Forecast vs Actual)", "🎯 All Analyses"] and df_forecast is not None:
            st.markdown("---")
            st.header("🔗 Forecast vs Actual Regression Analysis")
            st.markdown("*Analysing the relationship between Day-ahead Demand Forecast and Actual Load*")

            results, df_merged = run_linear_regression(df_forecast, df_actual)

            if results: 
                interpretation, colour = interpret_r2(results['r2_score'])

                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{results['r2_score']:.3f}")
                with col2:
                    st.metric("Slope", f"{results['slope']:.4f}")
                with col3:
                    st.metric("Intercept", f"{results['intercept']:.2f}")
                with col4:
                    st.metric("Samples", f"{results['n_samples']}")

                # Interpretation box
                st.info(f"**Model Interpretation:** {interpretation}")

                # Scatter plot with regression line
                col1, col2 = st.columns(2)

                with col1:
                    fig_scatter = px.scatter(
                        df_merged,
                        x='transmissionSystemDemand',
                        y='quantity',
                        opacity=0.6,
                        title="Scatter Plot: Forecast vs Actual Load"
                    )

                    # Add regression line
                    x_range = np.linspace(
                        df_merged['transmissionSystemDemand'].min(),
                        df_merged['transmissionSystemDemand'].max(),
                        100
                    )
                    y_line = results['slope'] * x_range + results['intercept']

                    fig_scatter.add_trace(go.Scatter(
                        x=x_range, y=y_line,
                        mode='lines',
                        name=f"Linear Fit: y = {results['slope']:.2f}x + {results['intercept']:.2f}",
                        line=dict(color='red', width=2)
                    ))

                    fig_scatter.update_layout(
                        xaxis_title="Transmission System Demand Forecast (MW)",
                        yaxis_title="Actual Quantity (MW)",
                        height=400
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                with col2:
                    fig_hist = px.histogram(
                        df_merged,
                        x='spread',
                        nbins=50,
                        title="Histogram of Forecast Errors (Spread)"
                    )
                    fig_hist.add_vline(
                        x=results['mean_spread'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {results['mean_spread']:.1f}"
                    )
                    fig_hist.update_layout(
                        xaxis_title="Spread (Actual - Forecast) MW",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Spread statistics
                st.subheader("📊 Spread Analysis")
                spread_cols = st.columns(6)
                with spread_cols[0]: 
                    st.metric("Mean Spread", f"{results['mean_spread']:.2f} MW")
                with spread_cols[1]:
                    st.metric("Std Dev", f"{results['std_spread']:.2f} MW")
                with spread_cols[2]: 
                    st.metric("5th Percentile", f"{results['p05_spread']:.2f} MW")
                with spread_cols[3]: 
                    st.metric("95th Percentile", f"{results['p95_spread']:.2f} MW")
                with spread_cols[4]:
                    st.metric("Min |Spread|", f"{results['min_abs_spread']:.2f} MW")
                with spread_cols[5]: 
                    st.metric("Max |Spread|", f"{results['max_abs_spread']:.2f} MW")

        # ==================== STATISTICAL OVERVIEW ====================
        if analysis_type in ["📊 Statistical Overview (Today's Forecasts)", "🎯 All Analyses"]:
            st.markdown("---")
            st.header("📊 Statistical Overview")

            # Historical data statistics
            st.subheader("Historical Actual Load Statistics")
            col1, col2 = st.columns(2)

            with col1:
                stats = df_actual['quantity'].describe()
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50% (Median)', '75%', 'Max'],
                    'Value (MW)': [
                        f"{stats['count']:.0f}",
                        f"{stats['mean']:.2f}",
                        f"{stats['std']:.2f}",
                        f"{stats['min']:.2f}",
                        f"{stats['25%']:.2f}",
                        f"{stats['50%']:.2f}",
                        f"{stats['75%']:.2f}",
                        f"{stats['max']:.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            with col2:
                fig_box = px.box(df_actual, y='quantity', title="Load Distribution")
                fig_box.update_layout(yaxis_title="Quantity (MW)", height=300)
                st.plotly_chart(fig_box, use_container_width=True)

            # Load by settlement period
            st.subheader("Average Load by Settlement Period")
            hourly_avg = df_actual.groupby('settlementPeriod')['quantity'].mean().reset_index()
            fig_hourly = px.bar(
                hourly_avg,
                x='settlementPeriod',
                y='quantity',
                title="Average Load by Settlement Period (30-min intervals)",
                color='quantity',
                color_continuous_scale='Viridis'
            )
            fig_hourly.update_layout(
                xaxis_title="Settlement Period",
                yaxis_title="Average Quantity (MW)",
                height=400
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

            # Time series view
            st.subheader("📈 Load Profile Over Time")
            fig_time = px.line(
                df_actual,
                x='startTime',
                y='quantity',
                title="Actual Total Load Time Series"
            )
            fig_time.update_layout(
                xaxis_title="Time",
                yaxis_title="Quantity (MW)",
                height=400
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # Today's forecasts (like statistical_analysis.py)
            if df_demand_today is not None or df_indicated_today is not None: 
                st.subheader("📅 Today's Forecasts")

                if df_demand_today is not None and df_indicated_today is not None: 
                    fig = make_subplots(rows=2, cols=1,
                                       subplot_titles=('Demand Forecast (Day-ahead)', 'Indicated Forecast (Day-ahead)'),
                                       vertical_spacing=0.20)

                    # Demand forecast
                    if 'transmissionSystemDemand' in df_demand_today.columns:
                        fig.add_trace(
                            go.Scatter(x=df_demand_today['startTime'], y=df_demand_today['transmissionSystemDemand'],
                                      mode='lines', name='Transmission System Demand', line=dict(color='blue', width=2)),
                            row=1, col=1
                        )

                        fig.add_trace(
                            go.Scatter(x=df_demand_today['startTime'], y=df_demand_today['transmissionSystemDemand_MA'],
                                      mode='lines', name='Transmission System Demand MA (8h)', line=dict(color='blue', width=2, dash='dash')),
                            row=1, col=1
                        )
                    if 'nationalDemand' in df_demand_today.columns:
                        fig.add_trace(
                            go.Scatter(x=df_demand_today['startTime'], y=df_demand_today['nationalDemand'],
                                      mode='lines', name='National Demand', line=dict(color='green', width=2)),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df_demand_today['startTime'], y=df_demand_today['nationalDemand_MA'],
                                      mode='lines', name='National Demand MA (8h)', line=dict(color='green', width=2, dash='dash')),
                            row=1, col=1
                        )

                    # Indicated forecast
                    if 'indicatedGeneration' in df_indicated_today.columns:
                        fig.add_trace(
                            go.Scatter(x=df_indicated_today['startTime'], y=df_indicated_today['indicatedGeneration'],
                                      mode='lines', name='Indicated Generation', line=dict(color='red', width=2)),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df_indicated_today['startTime'], y=df_indicated_today['indicatedGeneration_MA'],
                                      mode='lines', name='Indicated Generation MA (8h)', line=dict(color='red', width=2, dash='dash')),
                            row=2, col=1
                        )
                    if 'indicatedDemand' in df_indicated_today.columns:
                        fig.add_trace(
                            go.Scatter(x=df_indicated_today['startTime'], y=df_indicated_today['indicatedDemand'],
                                      mode='lines', name='Indicated Demand', line=dict(color='orange', width=2)),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df_indicated_today['startTime'], y=df_indicated_today['indicatedDemand_MA'],
                                      mode='lines', name='Indicated Demand MA (8h)', line=dict(color='orange', width=2, dash='dash')),
                            row=2, col=1
                        )

                    fig.update_layout(height=700, showlegend=True)
                    fig.update_xaxes(title_text="Time", row=1, col=1)
                    fig.update_yaxes(title_text="Demand (MW)", row=1, col=1)
                    fig.update_xaxes(title_text="Time", row=2, col=1)
                    fig.update_yaxes(title_text="Generation/Demand (MW)", row=2, col=1)

                    st.plotly_chart(fig, use_container_width=True)


    else:
        # Welcome message when no data is loaded
        st.info("👈 **Configure the date range in the sidebar and click 'Fetch Data & Analyse' to get started! **")
    # About section
        st.markdown("""
        ### About This Dashboard

        This interactive dashboard demonstrates quantitative analysis of UK energy market data
        using the **Elexon BMRS API**. It replicates and extends the functionality of: 

        | Script | Dashboard Feature |
        |--------|-------------------|
        | `elexon_actual_total_load_predictor.py` | 📈 **Load Prediction** - Random Forest model |
        | `elexon_linear_regression.py` | 🔗 **Regression Analysis** - Forecast vs Actual |
        | `elexon_statistical_analysis.py` | 📊 **Statistical Overview** - Today's forecasts |

        #### Key Improvements:
        - ✅ **Interactive visualisations** with Plotly (zoom, pan, hover)
        - ✅ **Real-time data fetching** from Elexon API
        - ✅ **All analyses in one place** with easy navigation

        #### API Constraints:
        - ⚠️ **Maximum 7 days** per request (Elexon API limit)

        #### Technical Stack:
        - **Data**: Elexon BMRS API
        - **ML Models**:  Scikit-learn (Random Forest, Linear Regression)
        - **Visualisation**: Plotly
        - **Framework**: Streamlit
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p><strong>Developed by Pavlos Lazarou for Welsh Power Graduate Quantitative Trading Analyst Application</strong></p>
            <p>Data source: <a href="https://bmrs.elexon.co.uk/" target="_blank">Elexon BMRS</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()