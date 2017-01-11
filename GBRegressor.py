from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from yahoo_finance import Share
import warnings

class Price_Predictor():

    def __init__(self, stock):
        self.stock = stock
        self.date = datetime.now()

    def predict(self):
        print("building model for", self.stock)

        # fetch data
        self.df = self.fetch_data()
        if self.df.empty:
            return

        # prepare data
        print("training model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prepare_data()
        prediction_data = self.df[0:1:]
        cutoff = self.date - timedelta(days=300)
        self.df = self.df.ix[:cutoff]  # train on 300 days of data
        self.df.dropna(inplace=True)
        self.X = self.df.drop(['price 7 days later'], axis=1)
        self.y = self.df['price 7 days later']


        # make scaler and fit transform
        self.scaler = MinMaxScaler()
        self.X = self.scaler.fit_transform(self.X)

        # make learner and fit training data
        self.p = GradientBoostingRegressor(loss='huber', learning_rate=0.15,
                                          max_depth=6, n_estimators=250, max_features=3,
                                          warm_start=True)
        self.p.fit(self.X, self.y)

        print("making predictions...")
        X = prediction_data.drop(['price 7 days later'], axis=1)
        X = self.scaler.fit_transform(X)
        prediction = round(float(self.p.predict(X.reshape(1, -1))[0]))
        print("predicted price in 7 days' time:", prediction)
        return prediction

    def prepare_data(self):
        df_temp = self.df.copy()
        # convert dates to datetime and reindex
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        df_temp.index = df_temp['Date']

        # get price 7 days later
        df_temp['+ 7'] = df_temp['Date'] + timedelta(days=7)
        df_temp['price 7 days later'] = \
            df_temp['+ 7'].apply(lambda x: self.get_price(x, df_temp))
        df_temp.dropna(axis=0, inplace=True)
        df_temp = df_temp.loc[(df_temp['price 7 days later'] != 0)]

        # get rolling average
        df_temp['roll_av'] = df_temp['Date'].apply(lambda x: self.get_rolling_av(x, df_temp))

        # get market relative price change
        df_temp['mkt_rel_ch'] = df_temp['Date'].apply(lambda x: self.market_rel_price_change(x, df_temp))
        df_temp['mkt_rel_ch'] = df_temp['mkt_rel_ch'].replace(to_replace=0, method='bfill')

        # get metrics
        self.df = df_temp.copy()
        self.df['a'], self.df['b'], self.df['r^2'], self.df['volatility'], \
        self.df['momentum'] = zip(*self.df['Date'].map(self.get_metrics))
        self.df = self.df.drop(['+ 7', 'Date', 'Close', 'High', 'Low', 'Open'], axis=1)

    def fetch_data(self):
        print("fetching")
        stock = self.stock
        start_date = datetime.strftime(self.date - timedelta(days=500), '%Y-%m-%d')
        end_date = datetime.strftime(self.date, '%Y-%m-%d')
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
            df = df.from_dict(share_getter.get_historical(start_date, end_date))
            dfb = dfb.from_dict(comparitor.get_historical(start_date, end_date))
            print("download complete: fetched", len(df.index) +
                len(dfb.index), "records")
            df = df.drop(['Symbol'], axis=1)
            df['comparitor'] = dfb['Adj_Close']
            df['Adj_Close'] = pd.to_numeric(df['Adj_Close'], errors='ignore')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='ignore')
            df['comparitor'] = pd.to_numeric(df['comparitor'], errors='ignore')
            return df
        except Exception as e:
            print("error in fetch_data", e)
            return df

    def get_metrics(self, date):
        returns = self.df.ix[date:date - timedelta(days=365)]
        returns = returns.resample('M').last()
        try:
            dfsm = pd.DataFrame({'s_adjclose': returns['Adj_Close'],
                             'b_adjclose': returns['comparitor']},
                            index=returns.index)
            dfsm[['s_returns', 'b_returns']] = \
            dfsm[['s_adjclose', 'b_adjclose']] / \
            dfsm[['s_adjclose', 'b_adjclose']].shift(1) - 1

        except KeyError as e:
            print("error in get_metrics", e)
            dfsm = pd.DataFrame()
        dfsm = dfsm.dropna()

        covar = np.cov(dfsm["s_returns"], dfsm["b_returns"])
        if len(covar) < 2: return

        beta = covar[0, 1] / covar[1, 1]
        alpha = np.mean(dfsm["s_returns"]) - beta * np.mean(dfsm["b_returns"])
        ypred = alpha + beta * dfsm["b_returns"]
        SS_res = np.sum(np.power(ypred - dfsm["s_returns"], 2))
        SS_tot = covar[0, 0] * (len(dfsm) - 1)
        r_squared = 1.0 - SS_res / SS_tot

         # volatility on entire data; momentum for last 3 months
        volatility = np.sqrt(covar[0, 0])
        momentum = np.prod(1 + dfsm["s_returns"].tail(3).values) - 1

        prd = 12.0  # monthly returns; 12 periods to annualize
        alpha = alpha * prd

        volatility = volatility * np.sqrt(prd)

        return alpha, beta, r_squared, volatility, momentum


    @staticmethod
    def get_price(date, df, market=False):
        date = datetime.strftime(date, '%Y-%m-%d')
        if market == True:
            try:
                price = df['comparitor'][df['Date'] == date].values[0]
                return price
            except Exception:
                #print("error in get_price", e)
                return 0
        try:
            price = df['Adj_Close'][df['Date'] == date].values[0]
            return price
        except Exception:
            #print("error in get_price", e)
            return 0

    @staticmethod
    def get_rolling_av(date, df):
        prices = 0
        count = 0
        for i in range(30):
            new_date = date - timedelta(days=i)
            x = float(Price_Predictor.get_price(new_date, df))
            if x > 0:
                prices += x
                count += 1
        return prices / count

    @staticmethod
    def market_rel_price_change(date, df):
        base_date = date - timedelta(days=30)
        old_price = float(Price_Predictor.get_price(base_date, df))
        old_market = float(Price_Predictor.get_price(base_date, df, market=True))
        current = float(Price_Predictor.get_price(date, df))
        current_market = float(Price_Predictor.get_price(date, df, market=True))
        if old_price > 0 and old_market > 0 and current > 0 and current_market > 0:
            return (current / current_market) - (old_price / old_market)
        return 0


if __name__ == '__main__':
    p = Price_Predictor('ITV.L')
    prediction = p.predict()


