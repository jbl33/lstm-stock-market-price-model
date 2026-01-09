A simple & fun project that includes an LSTM (Long Short-Term Memory) stock market price predictor model. The accuracy of this model is somewhat low, but it was created with the intent of learning the basics of neural networks and time-series forecasting.

# Project Overview
This project uses historical stock data for the top 50 companies in the S&P 500 to train a deep learning model. It utilizes a sliding window approach, looking at the previous 30 days of data to predict the next day's closing price.

# Key Features
Data Sourcing: Uses yfinance to pull 5 years of historical data.
Feature Engineering: Includes several technical indicators to improve model context:
Simple Moving Averages (SMA 10/50)
Exponential Moving Averages (EMA 10/50)
Relative Strength Index (RSI)
Moving Average Convergence Divergence (MACD)
Bollinger Bands
Volume Analysis (Changes and SMAs)

# Deep Learning Architecture: A multi-layered LSTM built with Keras/TensorFlow.
# Technical Stack
Language: Python
Deep Learning: Keras / TensorFlow
Data Manipulation: NumPy, Pandas, Scikit-Learn
Finance API: yfinance

# Model Architecture
The model is composed of a sequential stack of layers designed to extract temporal patterns:
LSTM Layer (128 units): Returns sequences to pass to the next layer.
LSTM Layer (64 units): Returns sequences.
LSTM Layer (32 units): Collapses the temporal dimension.
Dense Layers (32 & 16 units): Fully connected layers with ReLU activation for non-linearity.
Output Layer (1 unit): Predicts the continuous value of the next day's price.

# How to Use
Install Dependencies:
pip install keras tensorflow yfinance scikit-learn numpy
Train the Model: Run the script to download the data, calculate indicators, and train the stock_price_predictor.keras model.

# Outputs:
stock_price_predictor.keras: The saved neural network.
stock_scalers.pkl: Pickled dictionary of MinMax scalers used for normalizing and inverse-transforming stock data.

# How to Test
Once you have trained the model and generated the .keras and .pkl files, you can use the testing script to predict tomorrow's closing price for any of the top 50 S&P 500 stocks.
To run the prediction for the predefined list of stocks, ensure your saved files are in the same directory and run your testing script: **python test_model.py**
