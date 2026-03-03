
from darts import TimeSeries
import pandas as pd
import yfinance as yf

class Model: 
    def __init__(self):
        self.data ={}


    def fit_and_forecast_model(model, series, forecast_horizon):
        """
        Fit a given model and forecast future values.

        Parameters:
            model: A Darts forecasting model.
            series (TimeSeries): The time series data to fit the model on.
            forecast_horizon (int): The number of days to forecast into the future.

        Returns:
            forecast (TimeSeries): The forecasted time series data.
        """
        try:
            model.fit(series)  # Fit on the provided series
            forecast = model.predict(forecast_horizon)  # Predict future values
            return forecast
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
