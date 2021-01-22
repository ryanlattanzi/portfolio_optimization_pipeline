import sqlalchemy as sa
import json
from datetime import datetime, timedelta

import yfinance as yf

import pandas as pd
import numpy as np
from functools import reduce
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

#------------------------------------------------------------------------------------------------------------

class DatabaseHandler():

    def __init__(self, db_path):
        self.db_path = db_path
        self.engine  = None
        self.tables  = []

    def create_engine(self):
        self.engine = sa.create_engine(self.db_path)

    def create_table(self, table_name, col_dict):
        col_str = ', '.join([' '.join([k,v]) for (k,v) in col_dict.items()])
        self.engine.execute("""CREATE TABLE IF NOT EXISTS {} ({})""".format(table_name, col_str))

    def find_new_tables(self, ticker_list):
        clean_tickers = [x.replace('-','_') for x in ticker_list]
        self._get_db_tables()
        return list(set(clean_tickers) - set(self.tables))

    def _get_db_tables(self):
        with self.engine.connect() as conn:
            res = conn.execute("""select name from sqlite_master where type = 'table'""")
            for table in res:
                self.tables.append(table[0])

#------------------------------------------------------------------------------------------------------------

class Gatherer():

    def __init__(self, ticker_list):
        self.ticker_list   = ticker_list
        self.data          = {}

        self.clean_tickers     = [x.replace('-','_') for x in self.ticker_list]
        self.clean_ticker_dict = {k:v for (k,v) in zip(self.ticker_list, self.clean_tickers)}

    def get_batch_history(self, period):
        # period (str): 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        for ticker in self.ticker_list:
            tix  = yf.Ticker(ticker)
            hist = tix.history(period = period)
            hist.columns = self._clean_headers(hist)
            hist.index   = self._clean_date_index(hist)
            self.data[self.clean_ticker_dict[ticker]] = hist

    def get_previous_day(self):
        for ticker in self.ticker_list:
            tix  = yf.Ticker(ticker)
            hist = tix.history(period = '5d')
            hist.columns = self._clean_headers(hist)
            hist.index   = self._clean_date_index(hist)
            self.data[self.clean_ticker_dict[ticker]] = hist

    def save_to_db(self, engine):
        for ticker, data in self.data.items():
            dates_tuple = str(tuple(data.index))
            engine.execute('delete from {} where date in {}'.format(ticker, dates_tuple))
            data.to_sql(ticker, con = engine, if_exists = 'append')

    def _clean_headers(self, df):
        clean_headers = [x.lower().replace(' ','_') for x in df.columns]
        return clean_headers

    def _clean_date_index(self, df):
        string_index = df.index.strftime('%Y-%m-%d')
        return string_index

#------------------------------------------------------------------------------------------------------------

class PyPortfolioOptModel():

    def __init__(self, ticker_list, price_history_engine, models_engine, portfolio_value = 10000):
        self.ticker_list          = ticker_list
        self.price_history_engine = price_history_engine
        self.models_engine        = models_engine
        self.portfolio_value      = portfolio_value

        self.price_data_df     = pd.DataFrame()
        self.common_start_date = '0000-00-00'
        self.youngest_ticker   = None
        self.weights           = None
        self.weights_file      = None
        self.allocation_file   = None
        self.allocation        = None
        self.leftover          = None
        self.exp_return        = None
        self.volatility        = None
        self.sharpe_ratio      = None

        self.clean_tickers     = [x.replace('-','_') for x in self.ticker_list]
        self.clean_ticker_dict = {k:v for (k,v) in zip(self.ticker_list, self.clean_tickers)}

    def build_price_data_df(self):
        self._get_common_start_date()
        dfs = []
        temp_date = datetime.today().strftime('%Y-%m-%d')
        for ticker in self.clean_tickers:
            res = self.price_history_engine.execute("""
                    select date, close from {} where date >= '{}' and date < '{}'""".format(ticker, self.common_start_date, temp_date))
            temp_df = pd.DataFrame.from_records(list(res), columns = ['date', ticker])
            dfs.append(temp_df)
        self.price_data_df = reduce(lambda left, right: pd.merge(left, right, on = 'date'), dfs)
        self.price_data_df.set_index('date', inplace=True)

    def _get_common_start_date(self):
        for ticker in self.clean_tickers:
            res = self.price_history_engine.execute("""
                    select min(date) from {}""".format(ticker))
            for item in res:
                if item[0] > self.common_start_date:
                    self.common_start_date = item[0]
                    self.youngest_ticker = ticker

    def train_sharpe_model(self):
        mu      = self._get_mu()
        cov     = self._get_sample_cov()
        self.ef = EfficientFrontier(mu, cov)

        raw_weights  = self.ef.max_sharpe()
        self.weights = dict(self.ef.clean_weights())

        # Ensuring there are no shorts
        assert (np.array(list(self.weights.values())) >= 0).sum() == np.array(list(self.weights.values())).size, \
            'Not all positions are long.'

    def _get_mu(self):
        return expected_returns.mean_historical_return(self.price_data_df)

    def _get_sample_cov(self):
        return risk_models.sample_cov(self.price_data_df)

    def get_discrete_allocation(self):
        latest_prices = get_latest_prices(self.price_data_df)

        da = DiscreteAllocation(self.weights, latest_prices, total_portfolio_value = self.portfolio_value)
        self.allocation, self.leftover = da.lp_portfolio()
        self.allocation = {k:int(v) for k,v in self.allocation.items()}
        print('Discrete allocation: {}'.format(self.allocation))
        print('Funds remaining: ${:.2f}'.format(self.leftover))
        self.exp_return, self.volatility, self.sharpe_ratio = self.ef.portfolio_performance(verbose=True)

    def save_model(self):
        max_id = self._get_max_model_id()
        if max_id is not None:
            model_id = max_id + 1
        else:
            model_id = 0
        self.weights_file    = '../model_reg/pyportfolioopt/weights_{}.json'.format(model_id)
        self.allocation_file = '../model_reg/pyportfolioopt/allocation_{}.json'.format(model_id)
        with open(self.weights_file, 'w') as w, open(self.allocation_file, 'w') as a:
            json.dump(self.weights, w, indent = 2)
            json.dump(self.allocation, a, indent = 2)
        self._save_model_to_db(model_id)
        
    def _save_model_to_db(self, model_id):
        date = datetime.today().strftime('%Y-%m-%d')
        self.models_engine.execute("""
                insert into pyportfolioopt values (?,?,?,?,?,?,?,?,?)""", \
                (model_id,date,len(self.allocation),self.exp_return*100,self.volatility*100,self.sharpe_ratio,self.leftover,self.weights_file,self.allocation_file))

    def _get_max_model_id(self):
        res = self.models_engine.execute("""
                select max(model_id) from pyportfolioopt""")
        for item in res:
            max_id = item[0]
        if max_id is None:
            return None
        else:
            return int(max_id)