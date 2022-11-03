from datetime import datetime, timedelta
from csv import writer
import snscrape.modules.twitter as sntwitter
import pandas as pd
import bert as bm
import yfinance as yf
import nltk 
import re
import os

# this file is responsible for crawling twitter and yahoo finance for twitter sentiment and tsla stock ticker data
def main():
    if os.path.exists(r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\twitter_sentiment.csv"):
        sentiment_file_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\twitter_sentiment.csv"
    else:
        sentiment_file_path = r"C:\Users\mhenn\OneDrive\NLP Project\data\twitter_sentiment.csv"
        
    def get_latestdate(filename):
        df = pd.read_csv(sentiment_file_path)
        df["Date Created"] = pd.to_datetime(df["Date Created"], format="%d/%m/%Y %H:%M", errors='coerce')
        latest_date = max(df["Date Created"]).strftime("%d/%m/%Y %H:%M")
        return latest_date

    def get_todaydate():
        today = datetime.now()
        tmr = today + timedelta(days=1)
        tmr = tmr.strftime("%d/%m/%Y %H:%M")
        today = today.strftime("%d/%m/%Y %H:%M")
        return today,tmr

    def crawl_tweets(latest_date, tmr):
        latest_date = datetime.strptime(latest_date, "%d/%m/%Y %H:%M")
        start_crawl = latest_date + timedelta(days=1)
        start_crawl = start_crawl.strftime("%Y-%m-%d")
        print(start_crawl)
        tday = datetime.now().strftime("%Y-%m-%d")
        a = pd.date_range(start=start_crawl, end=tday)
        print("Crawling tweets...")
        
        for everyday in a:
            data = []
            attributes_container = []
            current=everyday.date()
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f"$AAPL OR $TSLA OR $GOOG OR $MSFT OR #stocks until:{current} lang:en").get_items()):
                if i>150:
                    break
                attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
            avg_score = bert(attributes_container)
            print("current date is", current)
            data.append(current.strftime("%d/%m/%Y %H:%M"))
            data.append(avg_score)
            append_list_as_row(sentiment_file_path, data)
        print("Tweets successfully crawled")

    def clean_text(text):
        punctuations = '!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        stopword = nltk.corpus.stopwords.words('english')
        text_noreply = re.sub('Replying to', '', text)
        text_nolink = re.sub('(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*', '', text_noreply)
        text_noname = re.sub('(?<!\w)@[\w+]{1,15}', '', text_nolink)
        text_punct = "".join([word.lower() for word in text_noname if word not in punctuations]) # remove puntuation
        text_punct = re.sub('[0-9]+', '', text_punct)
        text_tokenized = re.split('[^(A-Za-z0-9_$)]+', text_punct) # tokenization
        text_nonstop = [word for word in text_tokenized if word not in stopword] # remove stopwords and stemming
        return text_nonstop

    def bert(tweets):
        print("running BERT...")
        score = []
        for i in range(0,len(tweets),1):
            query = clean_text(tweets[i][4])
            query = ' '.join(query)
            # print("Query " + query)
            scoring = bm.run_bert(query)
            # print("Score "+ str(scoring))
            score.append(scoring)
        print("BERT successful")
        print(score)
        avg_score = sum(score)/len(score)
        return avg_score

    def append_list_as_row(filename, list_of_elem):
        #Open file in append mode
        with open(filename, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)

    latest = get_latestdate(sentiment_file_path)
    today, tmr = get_todaydate()
    if datetime.strptime(today, "%d/%m/%Y %H:%M") > datetime.strptime(latest, "%d/%m/%Y %H:%M"):
        print("today is after latest")
        crawl_tweets(latest, today)        
    else:
        print("Tweets are up-to-date")    
            

if __name__ == '__main__':
    if os.path.exists(r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\twitter_sentiment.csv"):
        sentiment_file_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\twitter_sentiment.csv"
        tsla_history_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\TSLA.csv"
    else:
        sentiment_file_path = r"C:\Users\mhenn\OneDrive\NLP Project\data\twitter_sentiment.csv"
        tsla_history_path = r"C:\Users\mhenn\OneDrive\NLP Project\data\TSLA.csv"


    start = datetime(2020,1,1)
    end = datetime.now()

    tsla = yf.download("TSLA", start=start, end=end, progress=True)
    tsla_df = pd.DataFrame(tsla)
    print("Latest financial data retrieved")
    # tsla["Date"]
    tsla_df.to_csv(tsla_history_path)
    main()