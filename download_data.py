import yfinance as yf
import os

print("Current Working Directory:", os.getcwd())

tickers = ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN"]

output_folder = "data/company_datasets"
os.makedirs(output_folder, exist_ok=True)

start_date = "2010-01-01"
end_date = "2024-01-01"

for ticker in tickers:
    try:
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            print(f"No data found for {ticker}. Skipping...")
            continue
        
        file_path = os.path.join(output_folder, f"{ticker}_historical_data.csv")
        data.to_csv(file_path)
        print(f"Data saved for {ticker} at {file_path}")
    
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")

print("All downloads completed.")