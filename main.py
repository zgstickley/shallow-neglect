from scripts.data_collection import fetch_stock_data, fetch_crypto_data
from scripts.data_preprocessing import load_data, preprocess_and_save
from scripts.model_training import train_model
from scripts.model_prediction import make_prediction

# Define which assets to fetch
STOCKS = ["AAPL", "GOOGL", "TSLA", "GBTC", "BITO"]
CRYPTO = ["BTC/USDT", "ETH/USDT"]

def main():
    # Fetch stock data, load, preprocess, train models, and predict
    for stock in STOCKS:
        fetch_stock_data(stock)
        df = load_data(f"data/{stock}_stocks.csv", is_crypto=False)
        preprocess_and_save(stock, is_crypto=False)
        train_model(stock, is_crypto=False)
        make_prediction(stock, is_crypto=False)

    # Fetch crypto data, load, preprocess, train models, and predict
    for crypto in CRYPTO:
        fetch_crypto_data(crypto)
        df = load_data(f"data/{crypto.replace('/', '_')}_crypto.csv", is_crypto=True)
        preprocess_and_save(crypto.replace("/", "_"), is_crypto=True)
        train_model(crypto.replace("/", "_"), is_crypto=True)
        make_prediction(crypto.replace("/", "_"), is_crypto=True)

if __name__ == "__main__":
    main()