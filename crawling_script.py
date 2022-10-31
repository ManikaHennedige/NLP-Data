from datetime import datetime, timedelta
from csv import writer
import snscrape.modules.twitter as sntwitter
import pandas as pd
import bert as bm
import nltk 
import re

def get_latestdate(filename):
    df = pd.read_csv(filename, index_col = False, lineterminator = '\n', parse_dates = ['Date Created'])
    latest_date = max(df["Date Created"])
    latest_date = pd.to_datetime(latest_date)
    latest_date = latest_date.strftime("%Y-%m-%d")
    return latest_date

def get_todaydate():
    today = datetime.now() + timedelta(days=-1)
    tmr = today + timedelta(days=1)
    tmr = tmr.strftime("%Y-%m-%d")
    today = today.strftime("%Y-%m-%d")
    return today,tmr

def crawl_tweets(today, tmr):
    attributes_container = []
    print("Crawling tweets...")
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f"$AAPL OR $TSLA OR $GOOG OR $MSFT OR #stocks since:{today} until:{tmr}").get_items()):
        if i>150:
            break
        attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
    print("Tweets successfully crawled")
    return attributes_container

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
        print("Query " + query)
        scoring = bm.run_bert(query)
        print("Score "+ str(scoring))
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

def main():
    sentiment_file_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\twitter_sentiment.csv"
    latest = get_latestdate(sentiment_file_path)
    today, tmr = get_todaydate()
    if today>latest:
        data = []
        tweets = crawl_tweets(today, tmr)
        avg_score = bert(tweets)
        data.append(today)
        data.append(avg_score)
        append_list_as_row(sentiment_file_path, data)
    else:
        print("Tweets are up-to-date")    

if __name__ == "__main__":
    main()