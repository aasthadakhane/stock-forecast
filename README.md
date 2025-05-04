# stock-forecast
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import datetime as dt

st.title("📈 AAPL Stock Forecast App (ARIMA)")

# Date input
start_date = st.date_input("Start Date", dt.date(2020, 1, 1))
end_date = st.date_input("End Date", dt.date.today())

p = st.number_input("ARIMA p", min_value=0, max_value=10, value=5)
d = st.number_input("ARIMA d", min_value=0, max_value=2, value=1)
q = st.number_input("ARIMA q", min_value=0, max_value=10, value=0)

if st.button("Run Forecast"):
    data = yf.download('AAPL', start=start_date, end=end_date)
    data = data[['Close']].dropna()

    model = ARIMA(data['Close'], order=(p, d, q))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=10)
    forecast.index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10, freq='B')

    st.line_chart(pd.concat([data['Close'], forecast]))
    st.write("📅 Forecasted Prices:")
    st.write(forecast)
