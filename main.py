import pandas as pd
import datetime
import yfinance as yf

sentiment_file_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\twitter_sentiment.csv"
tsla_history_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\TSLA.csv"

tsla_df = pd.read_csv(tsla_history_path, parse_dates=['Date'])
sentiment_df = pd.read_csv(sentiment_file_path, parse_dates=['Date Created'])

start = datetime.datetime(2021,1,1)
end = datetime.datetime.now()

tsla = yf.download("TSLA", start=start, end=end, progress=True)
tsla = pd.DataFrame(tsla)
tsla.to_csv(tsla_history_path)
print("TSLA Financial Data has been updated")





