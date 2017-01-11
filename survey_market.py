from concurrent import futures
import pandas as pd
from datetime import datetime, timedelta
from predictor import Predictor
from yahoo_finance import Share
from random import sample
import get_stock_list
import time
import numpy as np

filename = 'predictions.csv'

def predict(share):
    temp = {}
    s = Share(share)
    p = Predictor(share)
    temp['stock'] = share
    temp['date_made'] = datetime.strftime(datetime.now(), '%Y-%m-%d')
    temp['trigger_date'] = datetime.strftime(datetime.now() + timedelta(days=7), '%Y-%m-%d')
    temp['base_value'] = s.get_price()
    temp['prediction'] = p.predict_stock()
    temp['prediction'] = pd.to_numeric(temp['prediction'], errors='coerce')
    #temp['actual'] = np.nan
    #temp['error'] = np.nan
    return temp

def generate_predictions():
    shares = sample(get_stock_list.get_stocks()[3:], 5)
    with futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(predict, shares)
    return list(results)

def get_error(actual, prediction, base_value):
    if type(actual) == str or type(prediction) == str or type(base_value) == str: return
    actual = 1 if actual >= base_value else 0
    if (actual == 1 and prediction == 1) or (actual == 0 and prediction == 0):
        return 1
    else:
        return 0

def evaluate_predictions(df):
    today = datetime.strftime(datetime.now(), '%Y-%m-%d')
    df['actual'] = df[df['trigger_date'] == today]['stock'].apply(get_adj_price)
    try:
        df['error'] = df.apply(lambda row: get_error(row['actual'], row['prediction'],
            row['base_value']), axis=1)
        print("mean error:", df['error'].mean())
        print("median error:", df['error'].median())
    except Exception as e:
        print(e)

def get_adj_price(stock):
    today = datetime.strftime(datetime.now(), '%Y-%m-%d')
    yesterday = datetime.strftime(datetime.now() - timedelta(days=1), '%Y-%m-%d')
    try:
        s = Share(stock)
        return s.get_historical(yesterday, today)[-1]['Adj_Close']
    except:
        return 0

def load_data():
    try:
        with open(filename, 'r') as f:
            df = pd.read_csv(f)
            f.close()
    except:
        df = pd.DataFrame(columns=['stock', 'date_made', 'trigger_date', 'base_value', 'prediction',
                                   'actual', 'error'])

    #df = df[['stock', 'date_made', 'trigger_date', 'base_value', 'prediction']]
    df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce')
    #df = df.dropna()
    print(df)
    return df

def save_data(df):
    print("saving data...")
    with open(filename, 'w') as f:
        df.to_csv(f)
    f.close()
    print("done")

if __name__ == '__main__':
    df = load_data()
    print("checking forecasts")
    evaluate_predictions(df)

    while True:
        try:
            results = generate_predictions()
            print("done")
        except Exception as e:
            print("error: stopping")
            print(e)
            results = []

        if results != []:
            for r in results:
                df = df.append(r, ignore_index=True)
        df = df[['stock', 'date_made', 'trigger_date', 'base_value', 'prediction', 'actual',    
                     'error']]
        print(df)
        print()
        save_data(df)
        print("sleeping")
        time.sleep(500)


