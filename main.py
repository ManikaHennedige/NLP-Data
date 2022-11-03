import pandas as pd
import datetime
import yfinance as yf
from financial_transformations import perform_prediction
import pickle

# this file runs the model on the sentiment and tsla financial information data. This should have been generated through the 'crawling.py' file
sentiment_file_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\twitter_sentiment.csv"
tsla_history_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\TSLA.csv"


sentiment_df = pd.read_csv(sentiment_file_path, parse_dates=['Date Created'])
tsla_df = pd.read_csv(tsla_history_path)


pred_df, actual_df = perform_prediction(tsla_df, sentiment_df)

with open(r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\lstm_dataframes.pkl", 'wb') as f:
    pickle.dump((pred_df, actual_df, tsla_df), f)



print("Updated model data")











