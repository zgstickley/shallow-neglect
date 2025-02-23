# AI Trading Bot - Shallow Neglect

## Overview
This project is an AI-driven trading bot that:
- **Predicts future stock and crypto prices** using an LSTM model.
- **Executes trades automatically** via Alpaca (for stocks) and Binance (for crypto).
- **Uses historical data** for training and backtesting.

## Features
✅ **LSTM Time-Series Forecasting** - Predicts stock/crypto price movements.
✅ **Rule-Based Trading Execution** - Automates buy/sell decisions.
✅ **Integration with Trading APIs** - Connects to Alpaca & Binance.
✅ **Backtesting & Optimization** - Tests strategies before deploying live.

## Project Structure
```
/shallow-neglect
│── /data                # Raw and processed data (not committed)
│── /models              # Trained AI models
│── /notebooks           # Jupyter notebooks for experiments
│── /scripts             # Python scripts for training & trading
│── /config              # API keys and environment variables (DO NOT PUSH)
│── /logs                # Logs from trading & backtesting
│── main.py              # Main script to run the trading bot
│── requirements.txt     # Python dependencies
│── .gitignore           # Ignore sensitive and unnecessary files
│── README.md            # Documentation
```

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/shallow-neglect.git
cd shallow-neglect
```

### 2. Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### `requirements.txt` Dependencies:
```
numpy
pandas
scikit-learn
tensorflow
torch
matplotlib
yfinance
ccxt
alpaca-trade-api
```

### 4. Set Up API Keys
- **Alpaca (Stocks)**: Get API keys from [Alpaca](https://alpaca.markets/).
- **Binance (Crypto)**: Get API keys from [Binance](https://www.binance.com/).
- Store them in `config/keys.json` (DO NOT PUSH to GitHub).

### 5. Run the Trading Bot
```bash
python main.py
```

## Future Enhancements
🚀 **Enhance AI Model** - Improve LSTM predictions with additional indicators.
🚀 **Add Reinforcement Learning** - Let AI learn trading strategies.
🚀 **Deploy on a Cloud Server** - Run 24/7 trading with AWS/GCP.

---
**📌 Disclaimer:** This project is for educational purposes. Use at your own risk.

