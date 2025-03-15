import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib  # for saving the scaler, if not already imported

DATA_DIR = "data"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data(filepath, is_crypto=False):
    """Load stock or crypto data from CSV, correctly handling date column names."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read CSV, skipping second row if it's a stock
    df = pd.read_csv(filepath, skiprows=[1] if not is_crypto else None)

    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()

    # Determine correct date column
    date_col = "timestamp" if "timestamp" in df.columns else "date"

    if date_col not in df.columns:
        raise ValueError(f"Expected column '{date_col}' not found in {filepath}, found {df.columns.tolist()}")

    # Convert to datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    print(f"Loaded {filepath}: {df.shape[0]} rows")
    return df

def clean_data(df):
    """Fill missing values and remove anomalies."""
    print("Cleaning data...")

    # Fill missing values using forward-fill method
    df.ffill(inplace=True)

    # Drop any remaining missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    return df

def add_technical_indicators(df):
    """Add technical indicators: Moving Averages, RSI, MACD."""
    print("Adding technical indicators...")

    # Simple Moving Averages (SMA)
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["SMA_50"] = df["close"].rolling(window=50).mean()

    # Relative Strength Index (RSI)
    delta = df["close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

def normalize_data(df):
    # 1. Fit MinMaxScaler on the actual 'close' price values (raw prices)
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_values = df['close'].values.reshape(-1, 1)   # reshape for scaler
    scaler.fit(close_values)
    
    # 2. Transform the 'close' prices to [0, 1] range using the fitted scaler
    df['close_normalized'] = scaler.transform(close_values)
    
    # 3. Debug: print information to verify scaling
    print(f"Close price range before scaling: {df['close'].min()} to {df['close'].max()}")
    print("Sample normalized values:", df['close_normalized'].head().tolist())
    
    # 4. Save the fitted scaler for later use (inverse transform during prediction)
    joblib.dump(scaler, "minmax_scaler.pkl")
    
    return df, scaler

def preprocess_and_save(ticker, is_crypto=False):
    """Load, clean, preprocess, and save data for AI training with actual price included."""
    filetype = "crypto" if is_crypto else "stocks"
    filepath = f"{DATA_DIR}/{ticker}_{filetype}.csv"

    df = load_data(filepath, is_crypto)

    # ✅ Store actual close price before normalization
    df["actual_close"] = df["close"]

    df = clean_data(df)
    df = add_technical_indicators(df)
    df, scaler = normalize_data(df)

    processed_filepath = f"{PROCESSED_DIR}/{ticker}_{filetype}_processed.csv"
    df.to_csv(processed_filepath)

    print(f"✅ Saved processed data with actual prices to {processed_filepath}")

    return df, scaler

if __name__ == "__main__":
    preprocess_and_save("AAPL", is_crypto=False)  # Preprocess Apple stock data
    preprocess_and_save("BTC_USDT", is_crypto=True)  # Preprocess Bitcoin data
