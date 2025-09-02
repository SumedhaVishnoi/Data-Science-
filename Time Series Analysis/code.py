import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("time_series.csv", parse_dates=['Date'], index_col='Date')
df.plot()
plt.show()


from statsmodels.tsa.stattools import adfuller

result = adfuller(df['value'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df['value'], order=(1,1,1)) # (p,d,q)
model_fit = model.fit()
print(model_fit.summary())


forecast = model_fit.forecast(steps=10)
print(forecast)


from prophet import Prophet

df_prophet = df.reset_index().rename(columns={'Date':'ds','value':'y'})
model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

model.plot(forecast)
plt.show()
