import yfinance as yf
import ccxt
import pandas as pd
import os

# Create a directory to store data if it doesn’t exist
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_stock_data(ticker, start_date="2020-01-01", end_date="2025-01-01"):
    """Fetch historical stock data from Yahoo Finance and save cleanly."""
    print(f"Fetching stock data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Ensure correct column order & reset index
    data.reset_index(inplace=True)

    # Save clean CSV file
    filepath = f"{DATA_DIR}/{ticker}_stocks.csv"
    data.to_csv(filepath, index=False)  # No extra index column

    print(f"Saved stock data to {filepath}")

def fetch_crypto_data(symbol, exchange_name="binanceus", timeframe="1d", limit=1000):
    """Fetch historical crypto data from Binance.US instead of Binance.com"""
    print(f"Fetching crypto data for {symbol} from {exchange_name}...")
    exchange = getattr(ccxt, exchange_name)()  # Use binanceus instead of binance

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(ohlcv, columns=columns)
    
    # Convert timestamp to readable date
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    filepath = f"data/{symbol.replace('/', '_')}_crypto.csv"
    df.to_csv(filepath, index=False)
    print(f"Saved crypto data to {filepath}")

if __name__ == "__main__":
    fetch_stock_data("AAPL")  # Example: Fetch Apple stock data
    fetch_crypto_data("BTC/USDT")  # Example: Fetch Bitcoin data
