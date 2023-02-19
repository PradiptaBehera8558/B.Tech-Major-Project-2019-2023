import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_plot
import torch
from torch.utils.data import DataLoader
from ts.data.dataset import TimeSeriesDataset
from ts.model.nbeats import NBEATS
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from sklearn.metrics import mean_squared_error, r2_score
from elm import ELM
from tftb.processing import Segment
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import TCN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import BayesianRidge


class TimeSeriesModel:
    res_ls = []
    def __init__(self):
        self.res_dict = {}

##############################################################################

    def apply_prophet(self, df):
        # process data
        data1 = pd.DataFrame({'ds': df.index, 'y': df['Log_Close']})
        data2 = pd.DataFrame({'ds': df.index, 'y': df['Log_Volume']})

        model1 = Prophet()
        model1.fit(data1)
        model2 = Prophet()
        model2.fit(data2)

        future1 = model1.make_future_dataframe(periods=365)
        future2 = model2.make_future_dataframe(periods=365)
        forecost1 = model1.predict(future1)
        forecost2 = model2.predict(future2)

        fig1 = model1.plot(forecost1)
        fig2 = model2.plot(forecost2)

#########################################################################################

    def apply_deepAR(self,df):
        time_series = []
        for col in df.columns:
            time_series.append({'start': df.index[0], 'target': df[col].values,'feat_dynamic_real':None})
        dataset = ListDataset(time_series, freq='1D')
        estimator = DeepAREstimator(prediction_length=30,context_length=90,num_layers=2, num_cells=40,trainer=Trainer(epochs=10))
        predictor = estimator.train(dataset)
        forecast_it, ts_it = make_evaluation_predictions(dataset=predictor.dataset, predictor=predictor, num_samples=100)
        forecasts = list(forecast_it)
        make_evaluation_plot(dataset=predictor.dataset, forecasts=forecasts, num_samples=100)

###########################################################################################

    def apply_NBeats(self,df):
        # need to change the train and test data
        train_data = data.loc[:'2021-01-01']
        test_data = data.loc['2021-01-01':]
        train_dataset = TimeSeriesDataset(train_data.values)
        test_dataset = TimeSeriesDataset(test_data.values)
        model = NBEATS(
                        input_size=train_dataset.dim,
                        output_size=train_dataset.dim,
                        stack_types=(NBEATS.GENERIC_BLOCK, NBEATS.GENERIC_BLOCK),
                        stack_sizes=(128, 128),
                        nb_blocks_per_stack=2,
                        sharing=False,
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    )
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_func = torch.nn.MSELoss()
        num_epochs = 100

        for epoch in range(num_epochs):
            for i, batch in enumerate(train_loader):
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                loss.backward()
                optimizer.step()

        test_loader = DataLoader(test_dataset, batch_size=1)
        predictions = []
        for i, batch in enumerate(test_loader):
            inputs, targets = batch
            outputs = model(inputs)
            predictions.append(outputs[0])

        plt.plot(test_data.values)
        plt.plot(predictions)
        plt.legend(['Actual', 'Predicted'])
        plt.show()

#################################################################################

    def apply_Temporal_Fusion_Transformer(self,df):
        data = df[['Close', 'Volume']]
        data = np.array(data)

        # Create segments
        seg = Segment(data)
        segments = seg.run()
        num_inputs = segments.shape[1]
        num_outputs = segments.shape[2]
        train_size = int(0.8 * segments.shape[0])
        x_train = segments[:train_size]
        x_test = segments[train_size:]
        from tftb.models import TemporalFusionTransformer

        # Define model
        model = TemporalFusionTransformer(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            context_length=30,
            output_length=1,
            dropout_rate=0.1,
            hidden_layer_size=30,
            num_heads=4,
            num_forecasts=10,
            loss='mae'
        )

        # Train model
        model.fit(
            x=x_train,
            epochs=100,
            batch_size=32,
            verbose=1,
            validation_data=x_test
        )
        # Make predictions
        predictions = model.predict(x_test)

        # Get daily returns and volume predictions
        daily_returns_pred = predictions[:, :, 0]
        volume_pred = predictions[:, :, 1]

        plt.plot(daily_returns_pred, label='Predicted Daily Returns')
        plt.plot(volume_pred, label='Predicted Volume')
        plt.legend()
        plt.show()

######################################################################################
    def apply_AR(self,df):
        y1 = df['Log_Close']
        y2 = df['Log_Volume']
        train_size = int(len(y) * 0.8)
        train, test = y[0:train_size], y[train_size:len(y)]

        # Fit the AR model
        model = sm.tsa.AR(train, freq='D').fit(maxlag=30, method='mle')

        # Print the model summary
        print(model.summary())

        # Predict the test set
        start = len(train)
        end = len(train) + len(test) - 1
        predictions = model.predict(start=start, end=end, dynamic=False)

        # Plot the results
        plt.plot(train, label='Train')
        plt.plot(test, label='Test')
        plt.plot(predictions, label='AR Predictions')
        plt.legend()
        plt.show()

#################################################################################

    def apply_arima(self,df):
        # Create a new dataframe with just the date and closing price columns
        data = df[['date', 'close']].copy()

        # Set the date as the index
        data.set_index('date', inplace=True)

        # Resample to daily data
        data = data.resample('D').last()

        # Fill missing values using linear interpolation
        data = data.interpolate(method='linear')
        # Plot the time series to check for stationarity
        data.plot(figsize=(12, 6))
        plt.show()

        # Perform Dickey-Fuller test to check for stationarity
        result = sm.tsa.stattools.adfuller(data['close'])
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')

        # If the series is non-stationary, difference it
        diff = data.diff().dropna()
        # Use grid search to find optimal parameters for ARIMA model
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        aic_results = []
        for param in pdq:
            try:
                model = sm.tsa.ARIMA(diff, order=param)
                results = model.fit()
                aic_results.append((param, results.aic))
            except:
                continue
        optimal_params = min(aic_results, key=lambda x: x[1])[0]
        print(f'Optimal parameters: {optimal_params}')
        # Fit ARIMA model with optimal parameters
        model = sm.tsa.ARIMA(diff, order=optimal_params)
        results = model.fit()

        # Make predictions on the test data
        preds = results.predict(start='2019-12-01', end='2020-06-30', dynamic=False)

        # Add back the differenced values to get the original scale
        preds = preds.add(data['close'].shift(), fill_value=0)
        # Plot the actual vs predicted values
        plt.figure(figsize=(12, 6))
        plt.plot(data['close'], label='Actual')
        plt.plot(preds, label='Predicted')
        plt.legend()
        plt.show()

#######################################################################################

    def apply_boxJenkins(self,df):
        # Plot data
        plt.plot(df)
        plt.show()

        # Identify model order
        model = ARIMA(df, order=(p, d, q)).fit()

        # Diagnostic checking
        residuals = model.resid
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(residuals)
        ax[0].set_title('Residuals')
        ax[1].set(ylim=[-5, 5])
        ax[1].set_title('ACF of Residuals')
        _ = sm.graphics.tsa.plot_acf(residuals, ax=ax[1])

##########################################################################################

    def apply_CNN(self,df):
        # Split data into train and test sets
        train_data = data.iloc[:1000]
        test_data = data.iloc[1000:]

        # Convert train and test sets to arrays
        train_array = np.array(train_data)
        test_array = np.array(test_data)

        # Define window size and number of features
        window_size = 30
        num_features = 1

        # Define CNN model
        model = tf.keras.Sequential([
            layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, num_features)),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        # Compile model
        model.compile(optimizer='adam', loss='mse')

        # Fit model
        model.fit(x=train_array, y=train_array[:, 0], batch_size=32, epochs=100, verbose=1)

        # Predict on test data
        predictions = model.predict(test_array[:, :-1].reshape(-1, window_size, num_features))

        # Calculate MSE
        mse = np.mean(np.square(predictions - test_array[:, -1]))

        print(f'MSE: {mse}')

###############################################################################################

    def apply_exponentialSmoothing(self,df):
        # Create a time series plot of the closing prices
        plt.plot(data['close'])
        plt.title('Closing Prices')
        plt.show()

        # Split the data into training and testing sets
        train_data = data.iloc[:-100]
        test_data = data.iloc[-100:]

        # Train the model
        model = ExponentialSmoothing(train_data['close'], trend='add', seasonal='add', seasonal_periods=7).fit()

        # Generate predictions for the test data
        predictions = model.forecast(100)

        # Plot the predicted values against the actual values
        plt.plot(test_data['close'], label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.title('Exponential Smoothing Forecast')
        plt.legend()
        plt.show()

###############################################################################################

    def apply_gru(self,df):
        # Split into train and test sets
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size,:]
        test_data = df.iloc[train_size:,:]

        # Scale the data
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        # Create X and y for train and test sets
        def create_dataset(data, look_back=1):
            X, y = [], []
            for i in range(look_back, len(data)):
                X.append(data[i-look_back:i, :])
                y.append(data[i, :])
            return np.array(X), np.array(y)

        look_back = 60  # number of time steps to look back
        train_X, train_y = create_dataset(train_data, look_back)
        test_X, test_y = create_dataset(test_data, look_back)

        # Build the model
        model = Sequential()
        model.add(GRU(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

        # Fit the model
        history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

        # Plot the training loss and validation loss over time
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        # Make predictions
        train_predict = model.predict(train_X)
        test_predict = model.predict(test_X)

        # Invert the scaling of the predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_y = scaler.inverse_transform(train_y)
        test_predict = scaler.inverse_transform(test_predict)
        test_y = scaler.inverse_transform(test_y)

        # Plot the predicted values against the actual values
        train_predict_plot = np.empty_like(df)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
        test_predict_plot = np.empty_like(df)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(train_predict)+(look_back*2)+1:len(df)-1, :] = test_predict
        plt.plot(df)
        plt.plot(train_predict_plot)
        plt.plot(test_predict_plot)
        plt.show()

################################################################################################

    def apply_lstm(self,df):
        # Create a new DataFrame with only the 'Close' column
        df_close = df[['Close']]

        # Split the data into training and testing sets
        train_size = int(len(df_close) * 0.7)
        train_data = df_close.iloc[:train_size]
        test_data = df_close.iloc[train_size:]

        # Normalize the data
        train_data = (train_data - train_data.mean()) / train_data.std()
        test_data = (test_data - test_data.mean()) / test_data.std()

        # Convert the data to numpy arrays
        X_train = np.array(train_data)
        y_train = np.array(train_data.shift(-1))
        X_test = np.array(test_data)
        y_test = np.array(test_data.shift(-1))

        # Reshape the data to be 3-dimensional
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # Create the LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

        # Make predictions on the test data
        predictions = model.predict(X_test)

        # Evaluate the model
        mse = model.evaluate(X_test, y_test, verbose=0)
        print('Mean Squared Error:', mse)

###########################################################################################

    def apply_sarima(self,df):
        ts = data["Close"]

        # Fit the SARIMA model
        model = sm.tsa.statespace.SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
        results = model.fit()

        # Generate predictions
        start_date = "2022-01-01"
        end_date = "2022-12-31"
        predictions = results.get_prediction(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), dynamic=False)

        # Plot the results
        plt.plot(ts)
        plt.plot(predictions.predicted_mean)
        plt.fill_between(predictions.conf_int().index, predictions.conf_int()["lower Close"], predictions.conf_int()["upper Close"], alpha=0.2)
        plt.legend(["Actual", "Predicted"])
        plt.show()

###############################################################################################

    def apply_sma(self,df):
        rolling_mean = df['Close'].rolling(window=7).mean()
        # Plot the original data and the rolling mean
        plt.plot(df.index, df['Close'], label='Original Data')
        plt.plot(rolling_mean.index, rolling_mean, label='SMA (7 days)')
        plt.legend()
        plt.show()

################################################################################################

    def apply_StateSpace(self,df):
        mod = sm.tsa.statespace.SARIMAX(df['price'], order=(1,0,0), trend='c')
        # Fit the model
        res = mod.fit()

        # Make predictions
        predictions = res.predict(start='2022-01-01', end='2022-01-31')

        # Print the predictions
        print(predictions)

################################################################################################

    def apply_var(self,df):
        def adf_test(series, signif=0.05):
            dftest = adfuller(series, autolag='AIC')
            adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags Used','Number of Observations Used'])
            for key,value in dftest[4].items():
                adf['Critical Value (%s)'%key] = value
            p = adf['p-value']
            if p <= signif:
                print(f"Series is Stationary ({round(p, 4)})")
            else:
                print(f"Series is Non-Stationary ({round(p, 4)})")
                
        for name in df.columns:
            adf_test(df[name])
        df_diff = df.diff().dropna()

        for name in df_diff.columns:
            adf_test(df_diff[name])
        nobs = 30
        train, test = df_diff[0:-nobs], df_diff[-nobs:]
        model = VAR(train)
        lags = model.select_order(maxlags=12)
        results = model.fit(lags.aic)
        results.k_ar
        lag_order = results.k_ar
        predictions = results.forecast(train.values[-lag_order:], len(test))
        predictions = pd.DataFrame(predictions, index=df_diff.index[-len(predictions):], columns=df_diff.columns)
        predictions = predictions.cumsum()
        predictions = predictions.add(df.iloc[-nobs - 1])
        import matplotlib.pyplot as plt

        plt.plot(df[-nobs:])
        plt.plot(predictions)
        plt.legend(['Actual', 'Predicted'])
        plt.show()

  ####################################################################################################

    def apply_varma(self,df):
        model = sm.tsa.VARMAX(df, order=(1, 1))
        # Fit the model
        results = model.fit()

        # Make predictions
        predictions = results.forecast(steps=10)

        # Print the predictions
        print(predictions)

   ##################################################################################################

    def apply_wma(self,df):
        # Define the weighting function
        def weight_func(x):
            n = len(x)
            weights = np.arange(1, n + 1)
            return np.dot(weights, x) / weights.sum()

        # Calculate the weighted moving average using the rolling method and the defined weighting function
        window_size = 10
        df['wma'] = df['close'].rolling(window_size).apply(weight_func)

        # Print the DataFrame with the added 'wma' column
        print(df)

####################################################################################################

    def apply_elm(self,df):
        # Split the data into training and testing sets
        train_size = int(len(df) * 0.8)
        train_set = df.iloc[:train_size]
        test_set = df.iloc[train_size:]
        # Scale the input features to a range of 0 to 1
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_set[['Open', 'High', 'Low', 'Close', 'Volume']])
        X_test = scaler.transform(test_set[['Open', 'High', 'Low', 'Close', 'Volume']])
        # Define the output feature for training set
        y_train = train_set['Return'].to_numpy()

        # Define the output feature for testing set
        y_test = test_set['Return'].to_numpy()
        # Create the ELM model
        elm = ELM(hidden_units=50)

        # Fit the model to the training data
        elm.fit(X_train, y_train)
        # Make predictions on the testing data
        y_pred = elm.predict(X_test)
        # Calculate the mean squared error
        mse = mean_squared_error(y_test, y_pred)

        # Calculate the R-squared score
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error: ", mse)
        print("R-squared: ", r2)

################################################################################################
    
    def apply_rnn(self,df):
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # Define the number of time steps for the RNN
        time_steps = 30

        # Define the input and output data for the RNN
        X_train = []
        y_train_volume = []
        y_train_close = []
        for i in range(time_steps, len(train_data)):
            X_train.append(train_data.iloc[i-time_steps:i, :].values)
            y_train_volume.append(train_data.iloc[i, 1])
            y_train_close.append(train_data.iloc[i, 3])
        X_train, y_train_volume, y_train_close = np.array(X_train), np.array(y_train_volume), np.array(y_train_close)

        X_test = []
        y_test_volume = []
        y_test_close = []
        for i in range(time_steps, len(test_data)):
            X_test.append(test_data.iloc[i-time_steps:i, :].values)
            y_test_volume.append(test_data.iloc[i, 1])
            y_test_close.append(test_data.iloc[i, 3])
        X_test, y_test_volume, y_test_close = np.array(X_test), np.array(y_test_volume), np.array(y_test_close)

        # Build the RNN model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train_volume, epochs=50, batch_size=32)
        model.fit(X_train, y_train_close, epochs=50, batch_size=32)

        # Make predictions on the test data
        y_pred_volume = model.predict(X_test)
        y_pred_close = model.predict(X_test)

        # Plot the predictions against the actual data
        import matplotlib.pyplot as plt
        plt.plot(y_test_volume, color='red', label='Actual Volume')
        plt.plot(y_pred_volume, color='blue', label='Predicted Volume')
        plt.legend()
        plt.show()

        plt.plot(y_test_close, color='red', label='Actual Close')
        plt.plot(y_pred_close, color='blue', label='Predicted Close')
        plt.legend()
        plt.show()

##############################################################################################

    def apply_neuralNetwork(self,df):
        scaler = MinMaxScaler()
        data[features] = scaler.fit_transform(data[features])
        # Split the data into training and testing sets
        train_data = data.iloc[:5000]
        test_data = data.iloc[5000:]
        # Define the neural network architecture
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=len(features)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2))

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(train_data[features], train_data[['Close', 'Volume']], epochs=50, batch_size=32)
        # Make predictions on the testing set
        predictions = model.predict(test_data[features])

        # Inverse transform the predictions to get the original scale
        predictions = scaler.inverse_transform(predictions)

        # Calculate the mean squared error
        mse = np.mean(np.square(predictions - test_data[['Close', 'Volume']].values))

        print(f"Mean Squared Error: {mse}")

##################################################################################################

    def apply_tcn(self,df):
        # Define the model architecture
        """
        In this example, window_size is the number of time steps used as input to the model, 
        num_features is the number of features in the data (e.g. volume and close price),
         nb_filters is the number of filters in the convolutional layers, kernel_size is the size of the convolutional kernel, 
         and dilations is a list of dilation factors for the convolutional layers. The model is trained using the mean squared 
         error loss function and the Adam optimizer.
        """
        model = Sequential()
        model.add(TCN(input_shape=(window_size, num_features),
                    nb_filters=64,
                    kernel_size=3,
                    dilations=[1, 2, 4, 8],
                    activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2))

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32)

        # Evaluate the model on the test set
        score = model.evaluate(X_test, y_test, batch_size=32)

        # Make predictions on new data
        y_pred = model.predict(X_new)


        def apply_linearRegression(self,df):
            # Load the data
            data = pd.read_csv('crypto_data.csv')

            # Split the data into train and test sets
            train, test = train_test_split(data, test_size=0.2)

            # Prepare the features and target variables
            train_X = train.drop(['Date', 'Close', 'Volume', 'Market Cap'], axis=1)
            train_y_returns = train['Close'].pct_change().shift(-1).dropna()
            train_y_volume = train['Volume'].shift(-1).dropna()

            test_X = test.drop(['Date', 'Close', 'Volume', 'Market Cap'], axis=1)
            test_y_returns = test['Close'].pct_change().shift(-1).dropna()
            test_y_volume = test['Volume'].shift(-1).dropna()

            # Fit the linear regression model for daily returns
            lr_returns = LinearRegression()
            lr_returns.fit(train_X, train_y_returns)

            # Predict the daily returns on the test set
            pred_y_returns = lr_returns.predict(test_X)

            # Evaluate the model performance for daily returns
            r2_score_returns = r2_score(test_y_returns, pred_y_returns)
            mse_returns = mean_squared_error(test_y_returns, pred_y_returns)

            print('R-squared score for daily returns:', r2_score_returns)
            print('Mean squared error for daily returns:', mse_returns)

            # Fit the linear regression model for daily volume
            lr_volume = LinearRegression()
            lr_volume.fit(train_X, train_y_volume)

            # Predict the daily volume on the test set
            pred_y_volume = lr_volume.predict(test_X)

            # Evaluate the model performance for daily volume
            r2_score_volume = r2_score(test_y_volume, pred_y_volume)
            mse_volume = mean_squared_error(test_y_volume, pred_y_volume)

            print('R-squared score for daily volume:', r2_score_volume)
            print('Mean squared error for daily volume:', mse_volume)


    def apply_randomForestRegressor(self,df):
        # load cryptocurrency data into a pandas dataframe
        crypto_df = pd.read_csv('crypto_data.csv')

        # extract relevant features and target variable
        X = crypto_df[['Open', 'High', 'Low', 'Volume']]
        y = crypto_df['Close']

        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # create a Random Forest Regressor model with 100 trees
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # fit the model to the training data
        rf.fit(X_train, y_train)

        # use the trained model to make predictions on the test data
        y_pred = rf.predict(X_test)

        # evaluate the model using mean squared error and R-squared
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print('Mean Squared Error:', mse)
        print('R-squared:', r2)


    def apply_supportVectorRegressor(self,df):
        # load the data
        data = pd.read_csv('crypto_data.csv', index_col=0)

        # split the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

        # separate the input (X) and output (y) variables for both training and testing sets
        X_train = train_data[['Open', 'High', 'Low', 'Volume']]
        y_train_returns = train_data['Daily Returns']
        y_train_volume = train_data['Volume']

        X_test = test_data[['Open', 'High', 'Low', 'Volume']]
        y_test_returns = test_data['Daily Returns']
        y_test_volume = test_data['Volume']

        # create an SVR model for returns prediction
        svr_returns = SVR(kernel='rbf')
        svr_returns.fit(X_train, y_train_returns)

        # create an SVR model for volume prediction
        svr_volume = SVR(kernel='rbf')
        svr_volume.fit(X_train, y_train_volume)

        # make predictions for both returns and volume using the testing set
        y_pred_returns = svr_returns.predict(X_test)
        y_pred_volume = svr_volume.predict(X_test)

        # calculate the evaluation metrics for both returns and volume predictions
        r2_returns = r2_score(y_test_returns, y_pred_returns)
        mse_returns = mean_squared_error(y_test_returns, y_pred_returns)

        r2_volume = r2_score(y_test_volume, y_pred_volume)
        mse_volume = mean_squared_error(y_test_volume, y_pred_volume)

        print(f'R2 Score for Returns: {r2_returns:.4f}')
        print(f'Mean Squared Error for Returns: {mse_returns:.4f}')
        print(f'R2 Score for Volume: {r2_volume:.4f}')
        print(f'Mean Squared Error for Volume: {mse_volume:.4f}')



    def apply_decisionTreeRegressor(self,df):
        # Load the crypto currency data
        data = pd.read_csv('crypto_data.csv')

        # Select the relevant features for prediction
        X = data[['Open', 'High', 'Low', 'Volume']]
        y_return = data['Daily_Return']
        y_volume = data['Volume']

        # Split the data into training and testing sets
        X_train, X_test, y_return_train, y_return_test = train_test_split(X, y_return, test_size=0.2, random_state=0)
        _, _, y_volume_train, y_volume_test = train_test_split(X, y_volume, test_size=0.2, random_state=0)

        # Create the decision tree regressor model for returns
        model_return = DecisionTreeRegressor(random_state=0)
        model_return.fit(X_train, y_return_train)

        # Predict the returns on the test set
        y_return_pred = model_return.predict(X_test)

        # Evaluate the model for returns
        mse_return = mean_squared_error(y_return_test, y_return_pred)
        r2_return = r2_score(y_return_test, y_return_pred)

        # Create the decision tree regressor model for volume
        model_volume = DecisionTreeRegressor(random_state=0)
        model_volume.fit(X_train, y_volume_train)

        # Predict the volume on the test set
        y_volume_pred = model_volume.predict(X_test)

        # Evaluate the model for volume
        mse_volume = mean_squared_error(y_volume_test, y_volume_pred)
        r2_volume = r2_score(y_volume_test, y_volume_pred)

        # Print the evaluation metrics for both models
        print('Decision Tree Regressor model for Daily Returns - Mean Squared Error: ', mse_return)
        print('Decision Tree Regressor model for Daily Returns - R-squared: ', r2_return)
        print('Decision Tree Regressor model for Volume - Mean Squared Error: ', mse_volume)
        print('Decision Tree Regressor model for Volume - R-squared: ', r2_volume)


    def apply_Gradient_Boosting_Regression(self,df):
        # Load data
        df = pd.read_csv('crypto_data.csv')

        # Preprocessing
        X = df.drop(['Date', 'Volume', 'Close'], axis=1)
        y = df[['Close', 'Volume']]
        y_return = (y.pct_change() + 1).fillna(1).apply(lambda x: np.log(x))
        y_return.columns = ['Close_return', 'Volume_return']
        y_return = y_return.reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y_return, test_size=0.2, random_state=42)

        # Create model
        model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=100, max_depth=3, 
                                        random_state=42, criterion='mse')

        # Fit model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate model
        print('MSE: ', mean_squared_error(y_test, y_pred))
        print('R-squared: ', r2_score(y_test, y_pred))


    def apply_Gaussian_Process_Regression(self,df):
        # Load the data
        data = np.genfromtxt('crypto_data.csv', delimiter=',', skip_header=1)

        # Split the data into features and target
        X = data[:, 1:-2]
        y = data[:, -2:-1]

        # Split the data into training and testing sets
        split = int(0.8 * len(data))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        # Define the kernel
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))

        # Create the Gaussian process regressor
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
        # Fit the model
        gp.fit(X_train, y_train)

        # Make predictions
        y_pred, sigma = gp.predict(X_test, return_std=True)

    def apply_Bayesian_Ridge_Regression(self,df):
        # Split data into features and target
        X = df.drop(['Date', 'Close', 'Volume'], axis=1)
        y_return = df['Close'].pct_change().shift(-1).dropna()
        y_volume = df['Volume'].shift(-1).dropna()

        # Split data into train and test sets
        X_train, X_test, y_return_train, y_return_test, y_volume_train, y_volume_test = train_test_split(X, y_return, y_volume, test_size=0.2, shuffle=False)

        # Define and fit the model for returns
        return_model = BayesianRidge()
        return_model.fit(X_train, y_return_train)

        # Define and fit the model for volume
        volume_model = BayesianRidge()
        volume_model.fit(X_train, y_volume_train)

        # Predict the test set for returns
        y_return_pred = return_model.predict(X_test)

        # Predict the test set for volume
        y_volume_pred = volume_model.predict(X_test)

        # Calculate and print the root mean squared error for returns
        rmse_return = np.sqrt(mean_squared_error(y_return_test, y_return_pred))
        print('Root Mean Squared Error for Returns: {:.2f}'.format(rmse_return))

        # Calculate and print the root mean squared error for volume
        rmse_volume = np.sqrt(mean_squared_error(y_volume_test, y_volume_pred))
        print('Root Mean Squared Error for Volume: {:.2f}'.format(rmse_volume))

        # Calculate and print the R2 score for returns
        r2_return = r2_score(y_return_test, y_return_pred)
        print('R2 Score for Returns: {:.2f}'.format(r2_return))

        # Calculate and print the R2 score for volume
        r2_volume = r2_score(y_volume_test, y_volume_pred)
        print('R2 Score for Volume: {:.2f}'.format(r2_volume))
