import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.metrics import MeanSquaredError, MeanAbsoluteError
from keras.losses import MeanSquaredError
import pickle
import yfinance as yahoofinance
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

    data.fillna(0, inplace = True) # Replaces NaN values with 0
    return data


def create_sequences(data, num_days):
    # Creates 30 day sequences using sliding window approach
    X, y = [], []
    for i in range(num_days, len(data)):
        X.append(data[i - num_days:i])
        y.append(data[i, 0])  # Next day closing price
    return np.array(X), np.array(y)


# Downloading and processing stock data
all_sequences_X = []
all_sequences_y = []
scalers = {}

for ticker in top_50_stocks:
    data = yahoofinance.download(ticker, start = datetime.now() - timedelta(days = 365 * 5),
        end = datetime.now(), interval = "1d", progress = False, auto_adjust = True)
    data = add_technical_indicators(data)

    features = ['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower',
    'Volume', 'Volume_Change', 'Volume_SMA_10', 'Volume_SMA_50']

    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled = scaler.fit_transform(data[features]) # Scales values from 0-1
    scalers[ticker] = scaler

    X, y = create_sequences(scaled, num_days)

    all_sequences_X.append(X)
    all_sequences_y.append(y)

# Combining all sequences
X = np.vstack(all_sequences_X)
y = np.concatenate(all_sequences_y)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Defining the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape = (num_days, X.shape[2])))
model.add(LSTM(64, return_sequences = True))
model.add(LSTM(32))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1))

# Compiling the model
model.compile(optimizer = 'adam', loss = MeanSquaredError(), metrics = [MeanAbsoluteError()])

# Training the model
model.fit(X, y, epochs = 30, batch_size = 64, validation_split = 0.1)

# Saving everything
model.save('stock_price_predictor.keras')
with open("stock_scalers.pkl", "wb") as scalers_file:
    pickle.dump(scalers, scalers_file)

    print(f"The model has been saved")
