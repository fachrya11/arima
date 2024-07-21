import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# Load the trained ARIMA model
with open('model/arima_model.pkl', 'rb') as file:
    model_ARIMA = pickle.load(file)

# Streamlit app
st.title("Stock Price Prediction")
st.write("This application uses an ARIMA model to predict stock prices.")

# Date input
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=7))
end_date = st.date_input("End Date", datetime.today())

if st.button('Predict'):
    try:
        # Generate date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Use the ARIMA model to predict the stock prices
        predictions_diff = model_ARIMA.predict(start=len(date_range), end=len(date_range) + (end_date - start_date).days - 1)
        predictions_diff_cumsum = predictions_diff.cumsum()
        last_value = model_ARIMA.fittedvalues[-1]
        predictions = last_value + predictions_diff_cumsum

        # Prepare the results
        results = pd.DataFrame({'Date': date_range, 'Predicted Price': predictions})
        
        # Display results
        st.write(results)

        # Plot the results
        st.line_chart(results.set_index('Date'))
    except Exception as e:
        st.error(f"Error: {str(e)}")
