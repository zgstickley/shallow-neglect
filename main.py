from data.data_collection import fetch_stock_data, fetch_crypto_data

# Define which assets to fetch
STOCKS = ["AAPL", "GOOGL", "TSLA"]
CRYPTO = ["BTC/USDT", "ETH/USDT"]

def main():
    # Fetch stock data
    for stock in STOCKS:
        fetch_stock_data(stock)

    # Fetch crypto data
    for crypto in CRYPTO:
        fetch_crypto_data(crypto)

if __name__ == "__main__":
    main()
