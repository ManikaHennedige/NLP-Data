import pandas as pd
import datetime
import yfinance as yf
from financial_transformations import perform_prediction
import pickle

sentiment_file_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\twitter_sentiment.csv"
tsla_history_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\TSLA.csv"


sentiment_df = pd.read_csv(sentiment_file_path, parse_dates=['Date Created'])

start = datetime.datetime(2020,1,1)
end = datetime.datetime.now()

tsla = yf.download("TSLA", start=start, end=end, progress=True)
tsla_df = pd.DataFrame(tsla)
print("Latest financial data retrieved")
# tsla["Date"]
tsla_df.to_csv(tsla_history_path)

tsla_df = pd.read_csv(tsla_history_path)

## put joceline's code here ##
twitter_df = sentiment_df
## end joceline's code ##

print("Latest Twitter Sentiment data retrieved")

pred_df, actual_df = perform_prediction(tsla_df, twitter_df)

with open(r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\lstm_dataframes.pkl", 'wb') as f:
    pickle.dump((pred_df, actual_df, tsla_df), f)

print("Updated model data")










