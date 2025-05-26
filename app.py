import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go

@st.cache_data
def load_data():
    df = pd.read_excel('R Zonar_Sales data 1.XLSX')
    df['Dispatched'] = pd.to_datetime(df['Dispatched Date & Time-Out'], format='%d.%m.%Y/%H:%M:%S')
    daily = (
        df.set_index('Dispatched')
          .resample('D')['Weight Kg']
          .sum()
          .reset_index()
          .rename(columns={'Dispatched': 'ds', 'Weight Kg': 'y'})
    )
    return df, daily

@st.cache_data
def train_model(daily):
    # define holidays as above...
    sri_lanka_holidays = pd.DataFrame([
    # 2024 holidays
    {'ds': '2024-01-15', 'holiday': 'tamil_thai_pongal'},            # Tamil Thai Pongal Day
    {'ds': '2024-01-25', 'holiday': 'duruthu_full_moon_poya'},       # Duruthu Full Moon Poya Day
    {'ds': '2024-02-04', 'holiday': 'national_day'},                 # National Day
    {'ds': '2024-02-23', 'holiday': 'navam_full_moon_poya'},         # Navam Full Moon Poya Day
    {'ds': '2024-03-08', 'holiday': 'mahasivarathri'},               # Mahasivarathri Day
    {'ds': '2024-03-24', 'holiday': 'madin_full_moon_poya'},         # Madin Full Moon Poya Day
    {'ds': '2024-03-29', 'holiday': 'good_friday'},                  # Good Friday
    {'ds': '2024-04-11', 'holiday': 'id_ul_fitr'},                   # Id Ul-Fitr
    {'ds': '2024-04-12', 'holiday': 'pre_new_year'},                 # Day prior to Sinhala & Tamil New Year
    {'ds': '2024-04-13', 'holiday': 'new_year'},                     # Sinhala & Tamil New Year Day
    {'ds': '2024-04-23', 'holiday': 'bak_full_moon_poya'},           # Bak Full Moon Poya Day
    {'ds': '2024-05-01', 'holiday': 'may_day'},                      # May Day
    {'ds': '2024-05-23', 'holiday': 'vesak_full_moon_poya'},         # Vesak Full Moon Poya Day
    {'ds': '2024-05-24', 'holiday': 'post_vesak'},                   # Day after Vesak
    {'ds': '2024-06-17', 'holiday': 'id_ul_alha'},                   # Id Ul-Alha
    {'ds': '2024-06-21', 'holiday': 'poson_full_moon_poya'},         # Poson Full Moon Poya Day
    {'ds': '2024-07-20', 'holiday': 'esala_full_moon_poya'},         # Esala Full Moon Poya Day
    {'ds': '2024-08-19', 'holiday': 'nikini_full_moon_poya'},        # Nikini Full Moon Poya Day
    {'ds': '2024-09-16', 'holiday': 'milad_un_nabi'},                # Milad un-Nabi
    {'ds': '2024-09-17', 'holiday': 'binara_full_moon_poya'},        # Binara Full Moon Poya Day
    {'ds': '2024-10-17', 'holiday': 'vap_full_moon_poya'},           # Vap Full Moon Poya Day
    {'ds': '2024-10-31', 'holiday': 'deepavali'},                    # Deepavali
    {'ds': '2024-11-15', 'holiday': 'ill_full_moon_poya'},           # Ill Full Moon Poya Day
    {'ds': '2024-12-14', 'holiday': 'unduvap_full_moon_poya'},       # Unduvap Full Moon Poya Day
    {'ds': '2024-12-25', 'holiday': 'christmas'},                    # Christmas Day

    # 2025 holidays
    {'ds': '2025-01-13', 'holiday': 'duruthu_full_moon_poya'},       # Duruthu Full Moon Poya Day
    {'ds': '2025-01-14', 'holiday': 'tamil_thai_pongal'},            # Tamil Thai Pongal Day
    {'ds': '2025-02-04', 'holiday': 'national_day'},                 # National Day
    {'ds': '2025-02-12', 'holiday': 'navam_full_moon_poya'},         # Navam Full Moon Poya Day
    {'ds': '2025-02-26', 'holiday': 'mahasivarathri'},               # Mahasivarathri Day
    {'ds': '2025-03-13', 'holiday': 'madin_full_moon_poya'},         # Madin Full Moon Poya Day
    {'ds': '2025-03-31', 'holiday': 'id_ul_fitr'},                   # Id Ul-Fitr
    {'ds': '2025-04-12', 'holiday': 'bak_full_moon_poya'},           # Bak Full Moon Poya Day
    {'ds': '2025-04-13', 'holiday': 'pre_new_year'},                 # Day prior to Sinhala & Tamil New Year
    {'ds': '2025-04-14', 'holiday': 'new_year'},                     # Sinhala & Tamil New Year Day
    {'ds': '2025-04-18', 'holiday': 'good_friday'},                  # Good Friday
    {'ds': '2025-05-01', 'holiday': 'may_day'},                      # May Day
    {'ds': '2025-05-12', 'holiday': 'vesak_full_moon_poya'},         # Vesak Full Moon Poya Day
    {'ds': '2025-05-13', 'holiday': 'post_vesak'},                   # Day after Vesak
    {'ds': '2025-06-07', 'holiday': 'id_ul_alha'},                   # Id Ul-Alha
    {'ds': '2025-06-10', 'holiday': 'poson_full_moon_poya'},         # Poson Full Moon Poya Day
    {'ds': '2025-07-10', 'holiday': 'esala_full_moon_poya'},         # Esala Full Moon Poya Day
    {'ds': '2025-08-08', 'holiday': 'nikini_full_moon_poya'},        # Nikini Full Moon Poya Day
    {'ds': '2025-09-05', 'holiday': 'milad_un_nabi'},                # Milad un-Nabi
    {'ds': '2025-09-07', 'holiday': 'binara_full_moon_poya'},        # Binara Full Moon Poya Day
    {'ds': '2025-10-06', 'holiday': 'vap_full_moon_poya'},           # Vap Full Moon Poya Day
    {'ds': '2025-10-20', 'holiday': 'deepavali'},                    # Deepavali
    {'ds': '2025-11-05', 'holiday': 'ill_full_moon_poya'},           # Ill Full Moon Poya Day
    {'ds': '2025-12-04', 'holiday': 'unduvap_full_moon_poya'},       # Unduvap Full Moon Poya Day
    {'ds': '2025-12-25', 'holiday': 'christmas'},                    # Christmas Day
])
    m = Prophet(weekly_seasonality=True, holidays=sri_lanka_holidays)
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    m.fit(daily)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    return m, forecast

st.title("Daily Weight Forecast")

# Load
raw_df, daily = load_data()

# Date filter
start_date, end_date = st.sidebar.date_input("Select date range",
                                             [daily['ds'].min(), daily['ds'].max()])
mask = (daily['ds'] >= pd.to_datetime(start_date)) & (daily['ds'] <= pd.to_datetime(end_date))

# Show raw data
st.subheader("Original Data")
st.dataframe(raw_df)

# Train & Forecast
model, forecast = train_model(daily)
mask1 = (forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))
filtered_forecast = forecast[mask1]
# Plot trend
st.subheader("Baseline Trend + Forecast")
# fig1 = model.plot(forecast)
# st.pyplot(fig1)
# Trend and forecast with legend
def plot_forecast_with_legend(forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast (yhat)'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['trend'], mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))
    fig.update_layout(title='Forecast with Trend and Uncertainty', xaxis_title='Date', yaxis_title='Weight (Kg)')
    return fig

st.plotly_chart(plot_forecast_with_legend(filtered_forecast))

# Plot components
st.subheader("Decomposed Components")
# fig2 = model.plot_components(forecast)
# st.pyplot(fig2)
def plot_decomposed_components(forecast_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['weekly'], name='Weekly Seasonality'))
    if 'holidays' in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['holidays'], name='Holiday Effect'))

    fig.update_layout(title='Decomposed Components', xaxis_title='Date', yaxis_title='Effect on Weight')
    return fig

st.plotly_chart(plot_decomposed_components(filtered_forecast))
