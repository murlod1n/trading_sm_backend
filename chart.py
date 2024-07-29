import time
import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import LSTM, Dense
from keras.models import Sequential
from tvDatafeed import TvDatafeed, Interval


def get_chart():
    # Авторизация на TradingView
    tv = TvDatafeed(username="", password="")

    symbol = 'BTCUSDT'
    exchange = 'BINANCE'
    interval = Interval.in_daily

    df = tv.get_hist(symbol, exchange, interval, 300)


    history_data = df.filter(["close"])
    history_data_set = history_data.values

    training_data_len = math.ceil( len(history_data_set) * .8 )

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(history_data_set)
    print(df)
    chart_items = []
    id = 1
    for index, row in df.iterrows():
        chart_items.append(
            {
                "id" : id, 
                "datetime": str(index), 
                "close" : str(row["close"]),
                "open" : str(row["open"]),
                "high" : str(row["high"]),
                "low" : str(row["low"])
            }
        )
        id += 1

    print(chart_items)

    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])


    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60: , :]

    x_test = []
    y_test = history_data_set[training_data_len: , :]

    for i in range(60, len(test_data)): 
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt( np.mean(predictions - y_test) ** 2 )
    
    data = df.filter(["close"])

    data_set = data.values
    data_set_scaled = scaler.transform(data_set)

    x_test = []
    x_test.append(data_set_scaled)

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred_price = model.predict(x_test)
    pred_price = scaler.inverse_transform(pred_price)

    return { 
        "prediction": str(pred_price[0][0]),
        "chart_items": chart_items
    }
    

    

