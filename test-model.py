import keras
from keras.models import load_model
import pickle
import yfinance as yahoofinance
import numpy as np
from datetime import datetime, timedelta

num_days = 30

# Top 50 from S&P 500 (updated 11/16/25)
top_50_stocks = ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'AVGO', 'META', 'TSLA', 'SCHW', 'LLY', 'WMT', 'JPM', 'ORCL', 'V', 'XOM', 'MA', 'JNJ',
                 'ABBV', 'PLTR', 'COST', 'AMD', 'BAC', 'HD', 'PG', 'GE', 'CVX', 'CSCO', 'KO', 'UNH', 'IBM', 'MU', 'WFC', 'CAT', 'MS', 'TMUS',
                 'PM', 'RTX', 'AXP', 'GS', 'MRK', 'ABT', 'CRM', 'MCD', 'PEP', 'LIN', 'ISRG', 'UBER', 'DIS', 'LRCX', 'AMGN']


# Calculating the technical indicators for each stock
# Sources:
# https://wire.insiderfinance.io/calculate-rsi-with-python-and-yahoo-finance-c8fb78b1c199
# https://gregorycernera.medium.com/computing-simple-moving-average-sma-with-python-pandas-yfinance-0458bb0b5d3b
# https://wire.insiderfinance.io/ema-strategy-using-python-and-yfinance-api-5e029223ab01
# https://medium.com/@financial_python/building-a-macd-indicator-in-python-190b2a4c1777
# https://medium.com/@financial_python/how-to-plot-bollinger-bands-in-python-1d7cc95ad9af
def add_technical_indicators(data):
    # Moving averages
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['EMA_10'] = data['Close'].ewm(span = 10, adjust = False).mean()
    data['EMA_50'] = data['Close'].ewm(span = 50, adjust = False).mean()

    # RSI calculation
    delta = data['Close'].diff()
    gain = delta.clip(lower = 0)
    loss = -1 * delta.clip(upper = 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data['Close'].ewm(span = 12, adjust = False).mean()
    ema26 = data['Close'].ewm(span = 26, adjust = False).mean()
    data['MACD'] = ema12 - ema26

    # Bollinger Bands
    data['BB_upper'] = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
    data['BB_lower'] = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()

    # Volume indicators
    data["Volume_Change"] = data["Volume"].pct_change().fillna(0)
    data["Volume_SMA_10"] = data["Volume"].rolling(10).mean().fillna(0)
    data["Volume_SMA_50"] = data["Volume"].rolling(50).mean().fillna(0)

    data.fillna(0, inplace = True) # Replaces NaN values with 0 / Important for avoiding errors
    return data


# Loading the model and scalers
model = load_model('stock_price_predictor.keras')
with open("stock_scalers.pkl", "rb") as scalers_file:
    scalers = pickle.load(scalers_file)

features = ['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower',
            'Volume', 'Volume_Change', 'Volume_SMA_10', 'Volume_SMA_50']

# Predicting the next day's closing price for each stock
for ticker in top_50_stocks:
    data = yahoofinance.download(ticker, start = datetime.now() - timedelta(days = 365),
        end = datetime.now(), interval = "1d", progress = False, auto_adjust = True) # Downloading data for each stock
    data = add_technical_indicators(data)

    scaler = scalers[ticker]
    scaled = scaler.transform(data[features])

    # Getting the last 30 days for prediction
    start_index = len(scaled) - num_days
    X_prediction = scaled[start_index:]
    X_prediction = X_prediction.reshape(1, num_days, len(features))

    # Predicting the price
    price = model.predict(X_prediction)

    # Inversing output value to get the actual dollar value
    inverse_array = np.zeros((1, len(features)))
    inverse_array[0, 0] = price[0, 0]
    predicted_price = scaler.inverse_transform(inverse_array)[0, 0]
    predicted_price = round(predicted_price, 2)
    print(f"Predicted price for ticker ({ticker}): ${predicted_price}")
