import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle # import pickle to save scalar

DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_training_data(ticker, is_crypto=False):
    """Load preprocessed stock or crypto data for training."""
    filetype = "crypto" if is_crypto else "stocks"
    filepath = os.path.join(DATA_DIR, f"{ticker}_{filetype}_processed.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data file not found: {filepath}")

    # ✅ Detect correct date column
    date_col = "timestamp" if is_crypto else "date"

    df = pd.read_csv(filepath, parse_dates=[date_col], index_col=date_col)
    
    return df

def prepare_data(df, feature="close", sequence_length=50):
    """Prepare data for LSTM model training."""
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[[feature]])

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i : i + sequence_length])
        y.append(df_scaled[i + sequence_length])

    X, y = np.array(X), np.array(y)

    return X, y, scaler  # Return scaler for inverse transformation later

def build_lstm_model(input_shape):
    """Create an LSTM model for stock price prediction."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(ticker, is_crypto=False, sequence_length=50, epochs=10, batch_size=16):
    """Train an LSTM model on stock/crypto data and save it in Keras format."""
    df = load_training_data(ticker, is_crypto)

    X, y, scaler = prepare_data(df)

    # Split into training & test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_lstm_model((sequence_length, 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.keras")
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # ✅ Save the scaler
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved at {scaler_path}")

    return model, scaler

if __name__ == "__main__":
    train_model("AAPL", is_crypto=False)
    train_model("BTC/USDT", is_crypto=True)
