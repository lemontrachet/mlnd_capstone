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

# Ininitialisation, parameters, make scaler
np.random.seed(2)
seq_length = 9
test_size = 0.25
scaler = MinMaxScaler(feature_range=(0, 1))


# fetch data from yahoo finance, and prepare data frame
def fetch_data(stock, base_date, end_date):
    """
    helper function which retrieves data from the yahoo finance api
    """
    print("downloading stock data...")
    try:
        share_getter = Share(stock)
        if stock[-1] == 'L': comparitor = Share('^FTSE')
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


def scale_series(series, scaler):
    """
    helper function: takes a pandas series and a scaler object, converts the series to a numpy
    array, applies the scaler to the series, and returns a flattened version of the data
    """
    return scaler.fit_transform(np.array(series).reshape(-1, 1)).ravel()


def predict_stock(stock):
    """
    takes the name of a stock, fetches historic pricing data, builds LSTM neural network, and predicts
    whether the stock price is likely to rise or fall in the next 6 trading days 
    """
    now = datetime.strftime(datetime.now(), '%Y-%m-%d')
    base_date = datetime.strftime(datetime.now() - timedelta(days=1000), '%Y-%m-%d')
    df = fetch_data(stock, base_date, now)
    df.dropna(inplace=True)
    df[['Adj_Close', 'comparitor', 'roll_av']] = df[['Adj_Close', 'comparitor',
        'roll_av']].apply(lambda x: scale_series(x, scaler))
    data = np.array(df['Adj_Close']).astype('float32')
    data2 = np.array(df['comparitor']).astype('float32')
    data3 = np.array(df['roll_av']).astype('float32')

    # make sequences
    X, y, roll_avs = make_sequences(data, data2, data3)

    # train-test split
    split = int(len(data) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], seq_length, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], seq_length, X_test.shape[1]))

    # build network
    model = Sequential()
    model.add(LSTM(
        75,
        activation='sigmoid',
        input_shape=(X_train.shape[1:]),
        dropout_W=0.15,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(250, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=0)

    # predictions
    test_predict = model.predict(X_test)
    score, ra_score = 0, 0
    rounded = [1 if x > 0.5 else 0 for x in test_predict]
    for i in range(len(y_test)):
        #print(rounded[i], y_test[i], roll_avs[i])
        if rounded[i] - y_test[i] == 0:
            score += 1
        if roll_avs[i] - y_test[i] == 0:
            ra_score += 1
    score = score / len(y_test)
    baseline_score = ra_score / len(y_test)
    print(score)
    print("baseline: ", baseline_score)
    return score, baseline_score


# make sequences and calculate baselines
def make_sequences(data, data2, data3):
    """
    helper function to create time-series for each of the features passed as data, data2, data3;
    create the target variable mapped to each window, and the rolling-average at the end of each window,
    for the purposes of comparison
    """
    f1_seq, f2_seq, f3_seq, seqs_out, roll_avs = [], [], [], [], []
    for i in range(6, len(data) - seq_length):
        f1_seq.append(data[i:i + seq_length])
        f2_seq.append(data2[i:i + seq_length])
        f3_seq.append(data3[i:i + seq_length])
        seqs_out.append(1) if data[i - 6] >= data[i] else seqs_out.append(0)
        roll_avs.append(1) if np.mean(data[i:i + seq_length]) >= data[i] else roll_avs.append(0)
    X = np.array(list(zip(f1_seq, f2_seq, f3_seq)))
    y = np.array(seqs_out).reshape(-1, 1)
    roll_avs = np.array(roll_avs).reshape(-1, 1)
    return X, y, roll_avs


# random list of FTSE 100 stocks
shares = ['BARC.L', 'BATS.L', 'BDEV.L', 'BKG.L', 'BLND.L', 'BLT.L', 'BNZL.L', 'BP.L', 'BRBY.L', 'GKN.L', 'GLEN.L', 'GSK.L', 'HIK.L', 'HL.L', 'HMSO.L', 'HSBA.L', 'IAG.L', 'III.L', 'IMB.L', 'INF.L', 'INTU.L', 'ITRK.L', 'MNDI.L', 'MRW.L', 'NG.L', 'NXT.L', 'OML.L', 'PFG.L', 'PPB.L', 'PRU.L', 'PSN.L', 'PSON.L', 'RB.L', 'RBS.L', 'SAB.L', 'SBRY.L', 'SDR.L', 'SGE.L', 'SHP.L', 'SKY.L', 'SL.L', 'SN.L', 'SSE.L', 'STAN.L', 'STJ.L', 'SVT.L', 'TPK.L', 'TSCO.L', 'TUI.L']


"""
this generates the baseline predictions:
baseline_scores = [predict_stock(x)[1] for x in shares]
print("-----------------------------------------------")
print(baseline_scores)
av = np.mean(baseline_scores)
print("mean: ", av)
"""

"""
this generates predictions for the stocks above:
"""
predictions = []
failed = []
for s in shares:
    try:
        predictions.append((s, predict_stock(s)))
    except Exception as e:
        print(s, e)
        failed.append(s)

print(predictions)
print(failed)

