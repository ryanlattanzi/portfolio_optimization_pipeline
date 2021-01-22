from objects import DatabaseHandler, Gatherer
import yaml

def initial_run():
    db_handler = DatabaseHandler('../production_dbs/pricehistory.db')
    db_handler.create_engine()

    cols = {'date':'text',
            'open':'real',
            'high':'real',
            'low':'real',
            'close':'real',
            'volume':'integer',
            'dividends':'integer',
            'stock_splits':'integer'}
    db_handler.create_table('voo', cols)
    db_handler.create_table('vtwg', cols)
    db_handler.create_table('vong', cols)
    db_handler.create_table('btc_usd', cols)
    db_handler.create_table('eth_usd', cols)
    db_handler.create_table('aapl', cols)
    db_handler.create_table('icln', cols)
    db_handler.create_table('qcln', cols)

    ticker_list = ['voo','vtwg','vong','btc-usd','eth-usd','aapl','icln','qcln']
    gatherer = Gatherer(ticker_list)
    gatherer.get_batch_history(period = 'max')
    gatherer.save_to_db(db_handler.engine)

def price_history_etl(config):
    price_history_db = config['price_history_db']
    ticker_list      = config['ticker_list']
    table_cols       = config['price_history_table_cols']

    # Creating the engine and getting a list of the tables in the db
    db = DatabaseHandler(price_history_db)
    db.create_engine()

    # Checking to see if new tables need to be created; if so, create them and get historical data
    new_tickers = db.find_new_tables(ticker_list)
    if len(new_tickers) > 0:
        print('ALERT: New Tickers Found.')
        for new_ticker in new_tickers:
            db.create_table(new_ticker, table_cols)
            print('Created table {}'.format(new_ticker))
        new_ticker_gatherer = Gatherer(new_tickers)
        new_ticker_gatherer.get_batch_history(period = 'max')
        new_ticker_gatherer.save_to_db(db.engine)

    # Fetching most recent data and putting into the db
    gatherer = Gatherer(ticker_list)
    gatherer.get_previous_day()
    gatherer.save_to_db(db.engine)

if __name__ == '__main__':
    with open("../config.yaml", 'r') as cfg:
        config = yaml.safe_load(cfg)
    price_history_etl(config)
    