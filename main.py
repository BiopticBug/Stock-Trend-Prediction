import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Fetch stock data
def fetch_stock_data(ticker):
    df = yf.download(ticker, start='2000-01-01', end='2025-01-01')
    return df[['Close']].dropna().reset_index(drop=True)

# Prepare data for LSTM
def prepare_data(df, window_size=100):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(df)

    x, y = [], []
    for i in range(window_size, len(data_scaled)):
        x.append(data_scaled[i-window_size:i])
        y.append(data_scaled[i, 0])

    x, y = np.array(x), np.array(y)

    train_size = int(len(x) * 0.7)
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:], scaler

# Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main flow
if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    df = fetch_stock_data(ticker)

    x_train, y_train, x_test, y_test, scaler = prepare_data(df)
    model = build_model((x_train.shape[1], 1))

    model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1)

    y_pred = model.predict(x_test)
    y_pred = y_pred * (1 / scaler.scale_[0])
    y_test = y_test * (1 / scaler.scale_[0])

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
