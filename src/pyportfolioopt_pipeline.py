from objects import DatabaseHandler, PyPortfolioOptModel
import yaml

def initial_run(config):
    model      = config['model']
    model_db   = config['model_db']
    table_cols = config['model_table_cols']

    db_handler = DatabaseHandler(model_db)
    db_handler.create_engine()

    if 'pyportfolioopt' in model.lower():
        db_handler.create_table('pyportfolioopt', table_cols)

def pyportfoliooptmodel_pipeline(config):
    model_db    = config['model_db']
    price_db    = config['price_history_db']
    ticker_list = config['ticker_list']

    # Creating necessary engines
    model_db_handler = DatabaseHandler(model_db)
    model_db_handler.create_engine()
    price_db_handler = DatabaseHandler(price_db)
    price_db_handler.create_engine()

    # Building out the best portfolio according to the maximum Sharpe Ratio
    model = PyPortfolioOptModel(ticker_list, price_db_handler.engine, model_db_handler.engine)
    model.build_price_data_df()
    model.train_sharpe_model()
    model.get_discrete_allocation()
    model.save_model()

if __name__ == '__main__':
    with open("../config.yaml", 'r') as cfg:
        config = yaml.safe_load(cfg)
    pyportfoliooptmodel_pipeline(config)