#
# Python Module with Class
# for Vectorized Backtesting
# of Machine Learning-based Strategies
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import numpy as np
import pandas as pd
from sklearn import linear_model


class ScikitVectorBacktester(object):
    ''' Class for the vectorized backtesting of
    Machine Learning-based trading strategies.
    此物件類別可以用向量化的方式回測機器學習型交易策略
    Attributes
    ==========
    symbol: str
        TR RIC (financial instrument) to work with 金融工具RIC代碼
    start: str
        start date for data selection 開始日期
    end: str
        end date for data selection 結束日期
    amount: int, float
        amount to be invested at the beginning 投入資本額
    tc: float
        proportional transaction costs (e.g. 0.5% = 0.005) per trade 交易成本佔比
    model: str
        either 'regression' or 'logistic' 不是迴歸就是邏輯

    Methods
    =======
    get_data:
        retrieves and prepares the base data set 基礎資料集的檢索與準備
    select_data:
        selects a sub-set of the data 選出一組子集合資料
    prepare_features:
        prepares the features data for the model fitting 準備好模型套入時所需的特徵資料
    fit_model:
        implements the fitting step 套入步驟
    run_strategy:
        runs the backtest for the regression-based strategy 執行回測
    plot_results:
        plots the performance of the strategy compared to the symbol 畫出策略績效，與原投資比較
    '''

    def __init__(self, symbol, start, end, amount, tc, model):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        if model == 'regression':
            self.model = linear_model.LinearRegression()
        elif model == 'logistic':
            self.model = linear_model.LogisticRegression(C=1e6,
                solver='lbfgs', multi_class='ovr', max_iter=1000)
        else:
            raise ValueError('Model not known or not yet implemented.')
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        資料檢索準備
        '''
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
                          index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['returns'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data.
        選出一組金融數據資料的子集合
        '''
        data = self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()
        return data

    def prepare_features(self, start, end):
        ''' Prepares the feature columns for the regression and prediction steps.
        針對回歸與預測步驟，準備好一些特徵資料縱列
        '''
        self.data_subset = self.select_data(start, end)
        self.feature_columns = []
        for lag in range(1, self.lags + 1):
            col = 'lag_{}'.format(lag)
            self.data_subset[col] = self.data_subset['returns'].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)

    def fit_model(self, start, end):
        ''' Implements the fitting step.
        實作套入步驟
        '''
        self.prepare_features(start, end)
        self.model.fit(self.data_subset[self.feature_columns],
                       np.sign(self.data_subset['returns']))

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        ''' Backtests the trading strategy.
        回測交易策略
        '''
        self.lags = lags
        self.fit_model(start_in, end_in)
        # data = self.select_data(start_out, end_out)
        self.prepare_features(start_out, end_out)
        prediction = self.model.predict(
            self.data_subset[self.feature_columns])
        self.data_subset['prediction'] = prediction
        self.data_subset['strategy'] = (self.data_subset['prediction'] *
                                        self.data_subset['returns'])
        # determine when a trade takes place 判斷何時做交易
        trades = self.data_subset['prediction'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place 進行交易時，要先從報酬中扣除掉交易成本
        self.data_subset['strategy'][trades] -= self.tc
        self.data_subset['creturns'] = (self.amount *
                        self.data_subset['returns'].cumsum().apply(np.exp))
        self.data_subset['cstrategy'] = (self.amount *
                        self.data_subset['strategy'].cumsum().apply(np.exp))
        self.results = self.data_subset


        # absolute performance of the strategy 策略的絕對績效表現
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy 較原投資工具，此策略表現更好/更差的程度
        operf = aperf - self.results['creturns'].iloc[-1]
        return self.results['cstrategy'],'\n',self.results['creturns'],round(aperf, 2), round(operf, 2) # 

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        畫出交易策略的累積績效表現，並與原投資工具本身進行比較
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10, 6))


if __name__ == '__main__':
    scibt = ScikitVectorBacktester('.SPX', '2010-1-1', '2019-12-31',
                                   10000, 0.0, 'regression')
    print(scibt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31'))
    print(scibt.run_strategy('2010-1-1', '2016-12-31',
                             '2017-1-1', '2019-12-31'))
    scibt = ScikitVectorBacktester('.SPX', '2010-1-1', '2019-12-31',
                                   10000, 0.0, 'logistic')
    print(scibt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31'))
    print(scibt.run_strategy('2010-1-1', '2016-12-31',
                             '2017-1-1', '2019-12-31'))
    scibt = ScikitVectorBacktester('.SPX', '2010-1-1', '2019-12-31',
                                   10000, 0.001, 'logistic')
    print(scibt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31', lags=15))
    print(scibt.run_strategy('2010-1-1', '2013-12-31',
                             '2014-1-1', '2019-12-31', lags=15))
