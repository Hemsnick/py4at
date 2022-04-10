#
# Python Module with Class
# for Vectorized Backtesting
# of Linear Regression-based Strategies
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import numpy as np
import pandas as pd


class LRVectorBacktester(object):
    ''' 此物件類別可以用向量化的方式回測線性迴歸交易策略
    Attributes
    ==========
    symbol: str
       TR RIC (financial instrument) to work with
       要處理的金融投資工具代碼
    start: str
        start date for data selection 開始日期
    end: str
        end date for data selection 結束日期
    amount: int, float
        amount to be invested at the beginning 一剛開始要投入的資本額
    tc: float
        proportional transaction costs (e.g. 0.5% = 0.005) per trade
        交易成本，在每一筆交易中所佔的比例 (例如:0.5%=0.005)

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
        基礎資料集的檢索與準備
    select_data:
        selects a sub-set of the data
        選出一組子集合資料
    prepare_lags:
        prepares the lagged data for the regression
        準備好回歸所需的滯後量資料
    fit_model:
        implements the regression step
        實作迴歸步驟
    run_strategy:
        runs the backtest for the regression-based strategy
        針對策略執行回測
    plot_results:
        plots the performance of the strategy compared to the symbol
        畫出策略的績效表現,並與原投資工具本身進行比較
    '''

    def __init__(self, symbol, start, end, amount, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data. 資料檢索與準備
        '''
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
                          index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['returns'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data. 選出一組金融數據資料的子集合
        '''
        data = self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()
        return data

    def prepare_lags(self, start, end):
        ''' Prepares the lagged data for the regression and prediction steps.
            針對迴歸與預測步驟,準備好滯後量資料
        '''
        data = self.select_data(start, end)
        self.cols = []
        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            data[col] = data['returns'].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)
        self.lagged_data = data

    def fit_model(self, start, end):
        ''' Implements the regression step.
            實作迴歸步驟
        '''
        self.prepare_lags(start, end)
        reg = np.linalg.lstsq(self.lagged_data[self.cols],
                              np.sign(self.lagged_data['returns']),
                              rcond=None)[0]
        self.reg = reg

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        ''' Backtests the trading strategy.
            迴歸交易策略
        '''
        self.lags = lags
        self.fit_model(start_in, end_in)
        self.results = self.select_data(start_out, end_out).iloc[lags:]
        self.prepare_lags(start_out, end_out)
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.reg))
        self.results['prediction'] = prediction
        self.results['strategy'] = self.results['prediction'] * \
                                   self.results['returns']
        # determine when a trade takes place 判斷何時該進行交易
        trades = self.results['prediction'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place 進行交易時，要先從報酬中扣掉交易成本
        self.results['strategy'][trades] -= self.tc
        self.results['creturns'] = self.amount * \
                        self.results['returns'].cumsum().apply(np.exp)
        self.results['cstrategy'] = self.amount * \
                        self.results['strategy'].cumsum().apply(np.exp)
        # gross performance of the strategy 策略的總體績效表現
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy 相較於原投資工具，此策略表現更好/更差的程度
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        畫出交易策略的累積績效表現,並與原投資工具本身進行比較
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10, 6))


if __name__ == '__main__':
    lrbt = LRVectorBacktester('.SPX', '2010-1-1', '2018-06-29', 10000, 0.0)
    print(lrbt.run_strategy('2010-1-1', '2019-12-31',
                            '2010-1-1', '2019-12-31'))
    print(lrbt.run_strategy('2010-1-1', '2015-12-31',
                            '2016-1-1', '2019-12-31'))
    lrbt = LRVectorBacktester('GDX', '2010-1-1', '2019-12-31', 10000, 0.001)
    print(lrbt.run_strategy('2010-1-1', '2019-12-31',
                            '2010-1-1', '2019-12-31', lags=5))
    print(lrbt.run_strategy('2010-1-1', '2016-12-31',
                            '2017-1-1', '2019-12-31', lags=5))
