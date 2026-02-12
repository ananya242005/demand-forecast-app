import streamlit as st
import pandas as pd
from prophet import Prophet

st.title("📊 Demand Forecasting Web App")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url)
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

days = st.slider("Select number of days to forecast", 7, 60, 30)

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)

st.subheader("Forecast Result")
fig = model.plot(forecast)
st.pyplot(fig)
