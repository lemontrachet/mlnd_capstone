from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from yahoo_finance import Share
from datetime import datetime, timedelta
from collections import namedtuple

# Ininitialisation, parameters, make scaler
np.random.seed(2)
seq_length = 7
test_size = 0.25
scaler = MinMaxScaler(feature_range=(0, 1))


def predict_stock(stock):
    s = Share(stock)
    df = pd.DataFrame()
    now = datetime.strftime(datetime.now(), '%Y-%m-%d')
    base_date = datetime.strftime(datetime.now() - timedelta(days=600), '%Y-%m-%d')

    # fetch data from yahoo finance
    prices = s.get_historical('2014-12-30', '2016-12-30')
    data = np.array([p['Adj_Close'] for p in prices]).astype('float32').reshape(-1, 1)

    # train-test split
    split = int(len(data) * (1 - test_size))
    train = data[:split]
    test = data[split:]

    # scale the data
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # make time-series for the LSTM
    X_train, y_train, _, _ = make_sequences(train_scaled)
    X_test, y_test, baseline, roll_avs = make_sequences(test_scaled)

    # reshape for LSTM: (samples, time-steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # build network
    model = Sequential()
    model.add(LSTM(25, input_dim=1))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.fit(X_train, y_train, nb_epoch=50, batch_size=1, verbose=1)

    # predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # invert predictions and baselines
    test_predict = scaler.inverse_transform(test_predict).reshape(-1, 1)
    y_test = scaler.inverse_transform(y_test).reshape(-1, 1)
    baseline = scaler.inverse_transform(baseline.reshape(-1, 1))
    roll_avs = scaler.inverse_transform(roll_avs.reshape(-1, 1))

    # calculate errors
    metrics = namedtuple('metrics', ['mean_error', 'within_5', 'rmse', 'roll_av_err', 'baseline_err'])
    errors = np.abs(y_test - test_predict) / y_test
    metrics.mean_error = np.mean(errors)
    metrics.within_5 = len([x for x in errors if x <= 0.05]) # / len(errors)
    metrics.rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    metrics.roll_av_err = len([x for x in np.abs(y_test - roll_avs) / y_test if x <= 0.05])
    metrics.baseline_err = len([x for x in np.abs(y_test - baseline) / y_test if x <= 0.05])
    metrics.count = len(y_test)
    return metrics


# make sequences and calculate baselines
def make_sequences(data):
    """
    helper function to create time-series for price data; create the target variable mapped to
    each window, the rolling-average at the end of each window, and the price at the end of each
    window, for the purposes of comparison
    """
    seqs_in, seqs_out, baseline, roll_avs = [], [], [], []
    for i in range(6, len(data) - seq_length):
        seqs_in.append(data[i:i + seq_length])
        seqs_out.append(data[i - 6])
        baseline.append(data[i])
        roll_avs.append(np.mean(data[i:i + seq_length]))
    X = np.array(seqs_in)
    y = np.array(seqs_out).reshape(-1, 1)
    baseline = np.array(baseline).reshape(-1, 1)
    roll_avs = np.array(roll_avs).reshape(-1, 1)
    return X, y, baseline, roll_avs


# random list of FTSE 100 stocks
shares = ['BARC.L', 'BATS.L', 'BDEV.L', 'BKG.L', 'BLND.L', 'BLT.L', 'BNZL.L', 'BP.L', 'BRBY.L', 'GKN.L', 'GLEN.L', 'GSK.L', 'HIK.L', 'HL.L', 'HMSO.L', 'HSBA.L', 'IAG.L', 'III.L', 'IMB.L', 'INF.L', 'INTU.L', 'ITRK.L', 'MNDI.L', 'MRW.L', 'NG.L', 'NXT.L', 'OML.L', 'PFG.L', 'PPB.L', 'PRU.L', 'PSN.L', 'PSON.L', 'RB.L', 'RBS.L', 'SAB.L', 'SBRY.L', 'SDR.L', 'SGE.L', 'SHP.L', 'SKY.L', 'SL.L', 'SN.L', 'SSE.L', 'STAN.L', 'STJ.L', 'SVT.L', 'TPK.L', 'TSCO.L', 'TUI.L']

"""
calculate the accuracy of the model on the stocks above, and compare with the baseline (close price 6 days before estimate) and the 7-day rolling average for the window before the estimate
"""
accuracies = []
num = len(shares)
for i in range(num):
    print(i + 1)
    metrics = predict_stock(shares[i])

    # name of share
    print("stock:", shares[i])

    # mean error across all predictions
    print("mean error:", metrics.mean_error)

    # within 5%
    print("within 5%", metrics.within_5 / metrics.count)

    # RMSE
    print("root mean squared error:", metrics.rmse)

    # rolling average (30 trading days)
    print("rolling average basis error:", metrics.roll_av_err / metrics.count)

    # last known price
    print("last price basis error:", metrics.baseline_err / metrics.count)

    print("done: {} / {}:".format(i + 1, num))
    print()

    accuracies.append([shares[i], metrics])

print("done")

total, total_acc, total_ra, total_baseline = 0, 0, 0, 0
for stock in accuracies:
    print(stock[0])
    print(stock[1].within_5 / stock[1].count)
    print(stock[1].roll_av_err / stock[1].count)
    print(stock[1].baseline_err / stock[1].count)
    total += stock[1].count
    total_acc += stock[1].within_5
    total_ra += stock[1].roll_av_err
    total_baseline += stock[1].baseline_err

print(total_acc / total)
print(total_ra / total)
print(total_baseline / total)

