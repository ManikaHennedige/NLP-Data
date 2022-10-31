
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from urllib.request import urlopen
import matplotlib.pyplot as plt
from keras.models import load_model
import os

import base64
# getting file from OneDrive
def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    print(data_bytes64_String)
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    print(resultUrl)
    return resultUrl

# Calculation of historical moving averages of closing price (10 and 30 days of trading)
def MA(df, period):
    MA = pd.Series(df['Close'].rolling(period, min_periods=period).mean(), name='MA_' + str(period))
    return MA



#calculation of exponential moving average of closing price (10 and 30 days of trading)
def EMA(df, period):
    EMA = pd.Series(df['Close'].ewm(span=period, min_periods=period).mean(), name='EMA_' + str(period))
    return EMA


#Calculation of closing price momentum (10 and 30 days of trading)
def MOM(df, period):   
    MOM = pd.Series(df.diff(period), name='Momentum_' + str(period))   
    return MOM

# function incorporating the three above functions
def calculate_measures(tsla_df):
    tsla_df['MA10'] = MA(tsla_df, 10)
    tsla_df['MA30'] = MA(tsla_df, 30)
    tsla_df['EMA10'] = EMA(tsla_df, 10)
    tsla_df['EMA30'] = EMA(tsla_df, 30)
    tsla_df['MOM10'] = MOM(tsla_df['Close'], 10)
    tsla_df['MOM30'] = MOM(tsla_df['Close'], 30)

    return tsla_df

def perform_prediction(tsla_df, twitter_df):
    tsla_df = calculate_measures(tsla_df)

    tsla_df['Date'] = tsla_df['Date'].astype(str)
    twitter_df = twitter_df.rename(columns={"Date Created": "Date"})
    twitter_df["Date"] = pd.to_datetime(twitter_df["Date"], errors='coerce')
    twitter_df['Date'] = twitter_df['Date'].astype(str)

    merged_data = pd.merge(tsla_df,twitter_df, how='inner', on='Date')
    # Creating two columns SMA and LMA to label our dataset
    # SMA (Short Moving Average)- The average of the closing price from the next five days in the future
    # LMA (Long Moving Average)- The average of the closing price from the last ten days and the next five days in the future

    full_data = merged_data
    full_data['SMA'] = ""
    full_data['LMA'] = ""

    for ind in range(0, (full_data.shape[0]-5)):
        sma_frame = full_data['Close'].iloc[ind+1:ind+6]
        full_data['SMA'].iloc[ind] = sma_frame.mean()
    
        lma_frame_one = full_data['Close'].iloc[ind-10:ind]
        lma_frame_two = full_data['Close'].iloc[ind+1:ind+6]

        if (lma_frame_one.sum() == 0):
            full_data['LMA'].iloc[ind] = np.NaN
        else: 
            full_data['LMA'].iloc[ind] = (lma_frame_one.sum() + lma_frame_two.sum())/15

    full_data['SMA'] = full_data['SMA'].replace('', np.NaN)
    full_data['LMA'] = full_data['LMA'].replace('', np.NaN)

    #Dropping any empty fields of data
    full_data = full_data.dropna(axis=0)

    #Creating target class - Signal
    # The signal on a given trading day represents either 1-Buy or 0-Sell 
    # The signal is calculated by comparing the future SMA and intermediate LMA
    labelled_data = full_data
    labelled_data['signal'] = np.where(labelled_data['SMA'] > labelled_data['LMA'], 1.0, 0.0)

    #Dropping the SMA and LMA columns to avoid data leakage
    labelled_data  = labelled_data.drop(columns = ['SMA', 'LMA'])

    # Creating the MinMaxScaler Object
    scaler = MinMaxScaler()

    temp = labelled_data.drop(['Date', 'signal'], axis=1)
    cols = temp.columns

    #Creating scaled data
    temp = scaler.fit_transform(temp)

    #Generating input_df which will be used for model training and predictions
    input_df = pd.concat([labelled_data['Date'],pd.DataFrame(temp, columns = cols),
                        labelled_data['signal']], 
                        axis=1, ignore_index=False)

    input_df = input_df.dropna(axis=0)
    print(input_df.head())
    print(input_df.tail())

    # Splitting entire data to create Training and Testing Data
    # We will need to split the training and testing data into equivalent 
    # time steps to train the Model

    # Creating Training and Testing indices
    train_data_size = int((0.7)*(input_df.shape[0]))
    test_data_size = int(input_df.shape[0] - train_data_size)

    # Allocating data instances to training and testing sets, excluding the date
    train_data = input_df.iloc[0:train_data_size,1:]
    test_data = input_df.iloc[train_data_size:input_df.shape[0],1:]

    # train_data.to_csv("./TSLA_Full_Training_Data.csv")
    # test_data.to_csv("./TSLA_Full_Test_Data.csv")

    full_training_data = train_data # Used during model prediction
    full_test_data = test_data  # Used during model prediction

    # Creating numpy arrays from dataframes for future processing
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    # Creating X_train and y_train
    # As an example, this function looks back at five days of trading:
    # X - Consists of all features excluding signal from last 5 days 
    # y - Consists of signal from one day ahead

    X_train = []
    y_train = []

    time_step = 5
    label_col = (train_data.shape[1]-1)

    for i in range(time_step, train_data.shape[0]):
        X_train.append(train_data[i-time_step:i,:label_col])
        y_train.append(train_data[i,label_col])
        
    X_train, y_train = np.array(X_train), np.array(y_train)

    # print('\nShapes of X_train and Y_train:\n')
    # print(X_train.shape)
    # print(y_train.shape)

    # print('\nFirst Element in X_train and Y_train:\n')
    # print(X_train[0])
    # print(y_train[0])

    # Creating X_test and y_test
    # As an example, this function looks back at five days of trading:
    # X - Consists of all features excluding signal from last 5 days 
    # y - Consists of signal from one day ahead

    X_test = []
    y_test = []

    # Gathering the last five days of training data as this 
    # will be used to predict the first few labels in y_test
    last_5_days = full_training_data.tail()
    test_data_df = last_5_days.append([full_test_data], ignore_index=True)

    for i in range(time_step, test_data_df.shape[0]):
        X_test.append(test_data_df.iloc[i-time_step:i,:label_col])
        y_test.append(test_data_df.iloc[i, label_col])

    X_test, y_test = np.array(X_test), np.array(y_test)
    # print('\nShapes of X_test and Y_test:\n')
    # print(X_test.shape)
    # print(y_test.shape)

    # print('\nFirst Element in X_test and Y_test:\n')
    # print(X_test[0])
    # print(y_test[0])

    test_indexes = full_test_data.index.tolist()
    d = {"Date": input_df["Date"]}
    map_df = pd.DataFrame(data=d)
    test_dates = map_df.loc[test_indexes[0]:test_indexes[-1], "Date"]
    if os.path.exists(r"C:\Users\Manika Hennedige\OneDrive\NLP Project\data\twitter_sentiment.csv"):
        model_file_path = r"C:\Users\Manika Hennedige\OneDrive\NLP Project\models\lstm.h5"
    else:
        model_file_path = r"C:\Users\mhenn\OneDrive\NLP Project\models\lstm.h5"

    model = load_model(model_file_path)

    pred_df, actual_df = eval_model(model, X_test, y_test, test_dates)

    return pred_df, actual_df

# Model Evaluation and Results Evaluation
def eval_model(m, test_X, test_y, dates):

    y_pred = m.predict(test_X)
    y_pred = y_pred.flatten()
    

    pred_df = pd.DataFrame(data={"Date": dates, "Predictions": y_pred})
    pred_df["Date"] = pd.to_datetime(pred_df["Date"])
    pred_df = pred_df.set_index("Date")
    actual_df = pd.DataFrame(data={"Date": dates, "Actual": test_y})
    actual_df["Date"] = pd.to_datetime(actual_df["Date"])
    actual_df = actual_df.set_index("Date")



    return pred_df, actual_df