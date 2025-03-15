import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle # import pickle to save scalar

DATA_DIR = "data/processed"
MODEL_DIR = "models"

def load_processed_data(ticker, is_crypto=False):
    """Load processed stock or crypto data for making predictions."""
    filetype = "crypto" if is_crypto else "stocks"
    filepath = os.path.join(DATA_DIR, f"{ticker}_{filetype}_processed.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data file not found: {filepath}")

    # ✅ Detect correct column name
    date_col = "timestamp" if is_crypto else "date"

    df = pd.read_csv(filepath, parse_dates=[date_col], index_col=date_col)

    return df

def make_prediction(ticker, is_crypto=False):
    """Load trained model and predict next closing price."""
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = load_model(model_path)  # ✅ Load trained model

    # ✅ Load the correct scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df = load_processed_data(ticker, is_crypto)

    # Extract only the last 50 timesteps (as the model was trained with 50-step sequences)
    sequence_length = 50
    feature = "close"

    df_scaled = scaler.transform(df[[feature]])
    last_sequence = df_scaled[-sequence_length:].reshape(1, sequence_length, 1)

    # Predict next closing price (normalized)
    predicted_price_scaled = model.predict(last_sequence)[0][0]

    # ✅ Convert back to actual price using scaler.inverse_transform()
    predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0][0]

    print(f"🔍 Last closing price for {ticker}: ${df['actual_close'].iloc[-1]:.2f}")
    print(f"🎯 Predicted next closing price for {ticker}: ${predicted_price:.2f}")

    return predicted_price


if __name__ == "__main__":
    make_prediction("AAPL", is_crypto=False)
    make_prediction("ETH_USDT", is_crypto=True)
