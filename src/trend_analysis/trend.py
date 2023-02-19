from statsmodels.tsa.stattools import adfuller, pp
import matplotlib.pyplot as plt
import talib
from statsmodels.tsa.stattools import adfuller, ppca


class trend:
    def __init__(self, df):
        pass
        """
        There are several statistical tests that can be used to test for trends in time series data. Here's one approach using Python:

        Load the cryptocurrency data into a pandas dataframe.
        Create a new column in the dataframe that contains the log returns of the closing price and the logarithm of the volume.
        Use the Augmented Dickey-Fuller (ADF) test from the statsmodels package to test for stationarity in the log return series.
        Use the Phillips-Perron (PP) test from the statsmodels package to test for stationarity in the logarithm of volume series.
        Interpret the results of the tests to determine whether the data shows evidence of a trend.
        Here's some example code to get started:
        """

    def trend_analyzer(self,df):
        # Define significance level
        alpha = 0.05

        # Test for trend in close price
        close_price_test = adfuller(df['Close'])
        if close_price_test[1] < alpha:
            close_price_trend = "Stationary"
        else:
            close_price_test = ppca(np.array(df['Close']))
            if close_price_test[1] < alpha:
                close_price_trend = "Stationary"
            else:
                close_price_trend = "Trend"

        # Test for trend in volume
        volume_test = adfuller(df['Volume'])
        if volume_test[1] < alpha:
            volume_trend = "Stationary"
        else:
            volume_test = ppca(np.array(df['Volume']))
            if volume_test[1] < alpha:
                volume_trend = "Stationary"
            else:
                volume_trend = "Trend"

        print("Close price trend:", close_price_trend)
        print("Volume trend:", volume_trend)


    def apply_MACD(self,df):
        macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        signal = []
        for i in range(len(macd)):
            if macd[i] > macdsignal[i]:
                signal.append(1)
            else:
                signal.append(0)
        df = df.assign(signal=signal)
        df.plot(x='date', y='close')
        plt.plot(df[df['signal'] == 1]['date'], df[df['signal'] == 1]['close'], '^', markersize=10, color='g')
        plt.plot(df[df['signal'] == 0]['date'], df[df['signal'] == 0]['close'], 'v', markersize=10, color='r')
        plt.show()
    

    def apply_price_based_indicators(self,df):
        def apply_bollinger_bands(self,df):
            upper_band, middle_band, lower_band = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            signal = []
            for i in range(len(df)):
                if df['close'][i] < lower_band[i]:
                    signal.append(1)
                elif df['close'][i] > upper_band[i]:
                    signal.append(-1)
                else:
                    signal.append(0)
            df = df.assign(signal=signal)
            df.plot(x='date', y='close')
            plt.plot(df[df['signal'] == 1]['date'], df[df['signal'] == 1]['close'], '^', markersize=10, color='g')
            plt.plot(df[df['signal'] == -1]['date'], df[df['signal'] == -1]['close'], 'v', markersize=10, color='r')
            plt.show()
 

    def apply_volume_based_indicators(self,df):
        def apply_Chaikin_Money_Flow(self,df):
            cmf = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
            signal = []
            for i in range(len(df)):
                if cmf[i] > 0:
                    signal.append(1)
                elif cmf[i] < 0:
                    signal.append(-1)
                else:
                    signal.append(0)
            df = df.assign(signal=signal)
            df.plot(x='date', y='close')
            plt.plot(df[df['signal'] == 1]['date'], df[df['signal'] == 1]['close'], '^', markersize=10, color='g')
            plt.plot(df[df['signal'] == -1]['date'], df[df['signal'] == -1]['close'], 'v', markersize=10, color='r')
            plt.show()

        def apply_onBalance_Volume(self,df):
            obv = talib.OBV(df['close'], df['volume'])
            signal = []
            for i in range(len(df)):
                if obv[i] > obv[i-1]:
                    signal.append(1)
                elif obv[i] < obv[i-1]:
                    signal.append(-1)
                else:
                    signal.append(0)
            df = df.assign(signal=signal)
            df.plot(x='date', y='close')
            plt.plot(df[df['signal'] == 1]['date'], df[df['signal'] == 1]['close'], '^', markersize=10, color='g')
            plt.plot(df[df['signal'] == -1]['date'], df[df['signal'] == -1]['close'], 'v', markersize=10, color='r')
            plt.show()
    
        
    def apply_RSI(self,df):
        rsi = talib.RSI(df['close'], timeperiod=14)
        signal = []
        for i in range(len(df)):
            if rsi[i] < 30:
                signal.append(1)
            elif rsi[i] > 70:
                signal.append(-1)
            else:
                signal.append(0)
        df = df.assign(signal=signal)
        df.plot(x='date', y='close')
        plt.plot(df[df['signal'] == 1]['date'], df[df['signal'] == 1]['close'], '^', markersize=10, color='g')
        plt.plot(df[df['signal'] == -1]['date'], df[df['signal'] == -1]['close'], 'v', markersize=10, color='r')
        plt.show()


    def apply_Volatility_Indiacators(self,df):
        def apply_atr(self,df):
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            buy_level = df['close'] - (atr * 2)
            sell_level = df['close'] + (atr * 2)
            signal = []
            for i in range(len(df)):
                if df['close'][i] < buy_level[i]:
                    signal.append(1)
                elif df['close'][i] > sell_level[i]:
                    signal.append(-1)
                else:
                    signal.append(0)
            df = df.assign(signal=signal)
            df.plot(x='date', y='close')
            plt.plot(df[df['signal'] == 1]['date'], df[df['signal'] == 1]['close'], '^', markersize=10, color='g')
            plt.plot(df[df['signal'] == -1]['date'], df[df['signal'] == -1]['close'], 'v', markersize=10, color='r')
            plt.show()


        def apply_vix(self,df):
            signal = []
            for i in range(len(df)):
                if df['Close'][i] > 30:
                    signal.append(1)
                elif df['Close'][i] < 20:
                    signal.append(-1)
                else:
                    signal.append(0)
            df = df.assign(signal=signal)
            df.plot(x='Date', y='Close')
            plt.plot(df[df['signal'] == 1]['Date'], df[df['signal'] == 1]['Close'], '^', markersize=10, color='g')
            plt.plot(df[df['signal'] == -1]['Date'], df[df['signal'] == -1]['Close'], 'v', markersize=10, color='r')
            plt.show()

