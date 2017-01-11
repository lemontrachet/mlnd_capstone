from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.metrics import binary_accuracy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from yahoo_finance import Share
from datetime import datetime, timedelta
from collections import namedtuple
from random import shuffle

"""
This is the deployment version of the LSTM-based classifier. It is called by survey_market.py
"""

class Predictor():

    def __init__(self, stock):
        # Ininitialisation, parameters, make scaler
        np.random.seed(2)
        self.stock = stock
        self.seq_length = 9
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    # fetch data from yahoo finance, and prepare data frame
    def _fetch_data(self, base_date, end_date):
        """
        helper function which retrieves data from the yahoo finance api
        """
        print("downloading stock data...")
        try:
            share_getter = Share(self.stock)
            if self.stock[-1] == 'L': comparitor = Share('^FTSE')
            else: comparitor = Share(share_getter.get_stock_exchange())
        except:
            print("network not available")
            comparitor = None
        df, dfb = pd.DataFrame(), pd.DataFrame()
        try:
            df = df.from_dict(share_getter.get_historical(base_date, end_date))
            dfb = dfb.from_dict(comparitor.get_historical(base_date, end_date))
            print("download complete: fetched", len(df.index) +
                len(dfb.index), "records")
            df = df.drop(['Symbol'], axis=1)
            df['comparitor'] = dfb['Adj_Close']
            df['Adj_Close'] = pd.to_numeric(df['Adj_Close'], errors='ignore')
            df['comparitor'] = pd.to_numeric(df['comparitor'], errors='ignore')
            df['comparitor'] = df['Adj_Close'] / df['comparitor']
            df['roll_av'] = df['Adj_Close'].rolling(center=False, window=7).mean()
            return df
        except Exception as e:
            print("error in fetch_data", e)
            return df

    def _scale_series(self, series):
        """
        helper function: takes a pandas series and a scaler object, converts the series to a numpy
        array, applies the scaler to the series, and returns a flattened version of the data
        """
        return self.scaler.fit_transform(np.array(series).reshape(-1, 1)).ravel()

    def predict_stock(self):
        """
        takes the name of a stock, fetches historic pricing data, builds LSTM neural network, and
        predicts whether the stock price is likely to rise or fall in the next 6 trading days
        """
        now = datetime.strftime(datetime.now(), '%Y-%m-%d')
        base_date = datetime.strftime(datetime.now() - timedelta(days=1000), '%Y-%m-%d')
        df = self._fetch_data(base_date, now)
        df.dropna(inplace=True)
        df[['Adj_Close', 'comparitor', 'roll_av']] = df[['Adj_Close', 'comparitor',
            'roll_av']].apply(self._scale_series)
        data = np.array(df['Adj_Close'])#.astype('float32')
        data2 = np.array(df['comparitor'])#.astype('float32')
        data3 = np.array(df['roll_av'])#.astype('float32')

        # make sequences
        X, y = self._make_sequences(data, data2, data3)
        # reshape for LSTM
        X = np.reshape(X, (X.shape[0], self.seq_length, X.shape[1]))
        # build network
        model = Sequential()
        model.add(LSTM(
            75,
        activation='sigmoid',
        input_shape=(X.shape[1:]),
        dropout_W=0.15,
        return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(250, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(X, y, nb_epoch=225, batch_size=1, verbose=5)

        # predictions
        Xp = self._make_sequences(data, data2, data3, train=False)
        Xp = np.reshape(Xp, (Xp.shape[0], self.seq_length, Xp.shape[1]))
        prediction = 1 if model.predict(Xp)[0][0] > 0.5 else -1
        return prediction

    # make sequences and calculate baselines
    def _make_sequences(self, data, data2, data3, train=True):
        """
        helper function to create time-series for each of the features passed as data, data2, data3;
        create the target variable mapped to each window, and the rolling-average at the end of each
        window, for the purposes of comparison
        """
        if train:
            f1_seq, f2_seq, f3_seq, seqs_out = [], [], [], []
            for i in range(6, len(data) - self.seq_length):
                f1_seq.append(data[i:i + self.seq_length])
                f2_seq.append(data2[i:i + self.seq_length])
                f3_seq.append(data3[i:i + self.seq_length])
                seqs_out.append(1) if data[i - 6] >= data[i] else seqs_out.append(0)
            X = np.array(list(zip(f1_seq, f2_seq, f3_seq)))
            y = np.array(seqs_out).reshape(-1, 1)
            return X, y
        else:
            f1_seq, f2_seq, f3_seq = [], [], []
            f1_seq.append(data[:self.seq_length])
            f2_seq.append(data2[:self.seq_length])
            f3_seq.append(data3[:self.seq_length])
            X = np.array(list(zip(f1_seq, f2_seq, f3_seq)))
            return X

if __name__ == '__main__':
    p = Predictor('VOD.L')
    prediction = p.predict_stock()
    print(prediction)

