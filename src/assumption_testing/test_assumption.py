from arch import arch_model
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.stats.diagnostic as smd
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import breusch_godfrey

class AssumptionValidator:
    """
    Contains all the analysis of the data that included:
    - desc stats
    - distribution
    - distribution graph
    """
    def __init__(self, df):
        self.df = df
        self.status = {
            "auto_colinearity": {
                "test1":sdfgdrt 
            }
        }
        

    def validate_heteroscedascity(self):
        def apply_WhitesTest(self,df):
            # Fit the regression model
            model = sm.OLS(y, X).fit()
            # Run White's test
            white_test = het_white(model.resid, model.model.exog)

            # Extract the p-value from the test results
            p_value = white_test[1]

            # Check if p-value is below the significance level (e.g. 0.05)
            if p_value < 0.05:
                print("Heteroscedasticity is present")
            else:
                print("Heteroscedasticity is not present")

        def apply_Breusch_Pagan(self,df):
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import het_breuschpagan

            # Fit the regression model
            model = sm.OLS(y, X).fit()

            # Run Breusch-Pagan test
            bp_test = het_breuschpagan(model.resid, model.model.exog)

            # Extract the p-value from the test results
            p_value = bp_test[1]

            # Check if p-value is below the significance level (e.g. 0.05)
            if p_value < 0.05:
                print("Heteroscedasticity is present")
            else:
                print("Heteroscedasticity is not present")

        def apply_Goldfeld_Quandt_Test(self,df):
            # Split the data into training and testing sets
            n_obs = len(y)
            n_train = int(n_obs * 0.8)
            train_y, test_y = y[:n_train], y[n_train:]
            train_X, test_X = X[:n_train], X[n_train:]

            # Fit the regression model to the training set
            model = sm.OLS(train_y, train_X)
            results = model.fit()

            # Predict the values of the dependent variable for the testing set
            y_pred = results.predict(test_X)

            # Calculate the residuals for the testing set
            residuals = test_y - y_pred

            # Sort the residuals in ascending order
            residuals_sorted = np.sort(np.abs(residuals))

            # Divide the residuals into two equal-sized groups
            n = len(residuals_sorted)
            n_half = n // 2
            residuals_low = residuals_sorted[:n_half]
            residuals_high = residuals_sorted[n_half:]

            # Regress the absolute values of the residuals on the predictor variable
            X_low = test_X[:n_half]
            X_high = test_X[n_half:]
            y_low = residuals_low**2
            y_high = residuals_high**2
            model_low = sm.OLS(y_low, X_low)
            model_high = sm.OLS(y_high, X_high)
            results_low = model_low.fit()
            results_high = model_high.fit()

            # Calculate the F-statistic for the regression
            F = results_low.ssr / results_high.ssr

            # Compare the F-statistic to the critical values from an F-distribution
            df_low = n_half - test_X.shape[1] - 1
            df_high = n - n_half - test_X.shape[1] - 1
            crit_low = sm.stats.f.cdf(0.95, df_low, df_high)
            crit_high = sm.stats.f.ppf(0.95, df_low, df_high)
            if F < crit_low or F > crit_high:
                print('Heteroscedasticity detected')
            else:
                print('No evidence of heteroscedasticity')

        def apply_Park_test(self,df):
            # Assuming that the time series data is stored in a variable called "data"

            # Define a significance level
            alpha = 0.05

            # Perform the Park test
            _, p_value, _, _ = smd.het_goldfeldquandt(data, alternative='two-sided')

            # Check if the p-value is less than the significance level
            if p_value < alpha:
                print("The Park test rejects the null hypothesis of homoscedasticity. Heteroscedasticity is present in the data.")
            else:
                print("The Park test fails to reject the null hypothesis of homoscedasticity. Heteroscedasticity is not present in the data.")

        def apply_arch_lm(self,df):
            # Load data
            df = pd.read_csv('crypto_data.csv', index_col=0, parse_dates=True)

            # Define model
            model = arch_model(df['Close'], vol='GARCH', p=1, o=0, q=1, dist='Normal')

            # Fit model
            results = model.fit()

            # Perform ARCH LM test
            resid = results.resid
            arch_lm_test = results.arch_lm_test(lags=12)

            # Check if heteroskedasticity is present
            if arch_lm_test.pvalue < 0.05:
                print('Heteroskedasticity is present.')
            else:
                print('Heteroskedasticity is not present.')


# =============================================================================================#


    def testing_Volatility(self,data):
        # Estimate GARCH model
        garch_model = arch.arch_model(data, vol='GARCH', p=1, q=1)
        garch_result = garch_model.fit(disp='off')
        # Extract volatility series
        volatility = garch_result.conditional_volatility
        volume_std = data['Volume'].std()
        # Set volatility threshold to 1.5 times standard deviation
        volatility_threshold = 1.5 * volume_std

        if volatility.max() > volatility_threshold:
            print('Volatility detected')
        else:
            print('No volatility detected')


    def testing_AutoCorrelation(self,df):
        """
        It is important to examine autocorrelation both before and after building a time series model.
        Before building a model, examining the autocorrelation function (ACF) and partial autocorrelation 
        function (PACF) plots can help identify the nature of the time series data. This can inform the selection 
        of appropriate modeling techniques and parameter choices.
        After building a model, examining the autocorrelation of the residuals can help assess whether the model 
        adequately captures the structure of the data. Significant autocorrelation in the residuals can indicate
        that the model is missing important patterns in the data and may need to be revised. This is commonly done 
        using statistical tests such as the Ljung-Box test or the Durbin-Watson test.
        """
        
        def apply_Durbin_Watson(self,df):
            # load data into a pandas dataframe
            df = pd.read_csv('crypto_currency_data.csv')

            # extract the target variable (close price) and the predictors (volume)
            y = df['close_price']
            X = df['volume']

            # add a constant to the predictors
            X = sm.add_constant(X)

            # fit a linear regression model to the data
            model = sm.OLS(y, X).fit()

            # compute the residuals
            residuals = model.resid

            # perform the Durbin-Watson test
            dw_test = durbin_watson(residuals)

            # print the test statistic and the result of the test
            if dw_test < 1.5:
                print('The residuals exhibit positive autocorrelation.')
            elif dw_test > 2.5:
                print('The residuals exhibit negative autocorrelation.')
            else:
                print('The residuals do not exhibit significant autocorrelation.')

        
        def apply_ljung_box(self,df):
            from statsmodels.stats.diagnostic import acorr_ljungbox

            # assume `residuals` is the residuals of the time series model
            lbtest = acorr_ljungbox(residuals, lags=[10])
            p_value = lbtest[1][0]

            if p_value < 0.05:
                print("The residuals are significantly autocorrelated.")
            else:
                print("The residuals are not significantly autocorrelated.")

            """
            The Ljung-Box test is a statistical test for the presence of autocorrelation in a time series. 
            The test calculates the Q statistic, which is the sum of the squared autocorrelation coefficients 
            up to a given lag, and the associated p-value for each lag. If the p-value is less than a chosen 
            significance level (e.g., 0.05), the null hypothesis of no autocorrelation is rejected.
            Interpret the results of the Ljung-Box test to determine if there is significant autocorrelation 
            in the cryptocurrency data. If the p-value is less than the significance level for one or more lags,
             this indicates the presence of autocorrelation.
            By performing the Ljung-Box test on cryptocurrency data, we can assess the degree of autocorrelation 
            in the time series and potentially use this information to inform trading strategies or other investment 
            decisions.
            """
            if <>:
                pass
            else:
                pass

        def apply_Breusch_Godfrey(self,df):
            # Fit a linear regression model
            model = sm.OLS(y, X).fit()

            # Perform the Breusch-Godfrey test
            test_statistic, p_value, f_statistic, f_p_value = breusch_godfrey(model, nlags=5)

            # Check the p-value against a significance level
            alpha = 0.05

            if p_value < alpha:
                print("Autocorrelation is present in the model")
            else:
                print("No evidence of autocorrelation")

            """
            The Breusch-Godfrey test is a statistical test for autocorrelation in the residuals of a regression model. 
            The test calculates the test statistic and the associated p-value, and compares them to a chosen significance
            level (e.g., 0.05). If the p-value is less than the significance level, the null hypothesis of no autocorrelation is rejected.
            Interpret the results of the Breusch-Godfrey test to determine if there is significant autocorrelation in the residuals of the 
            regression model. If the p-value is less than the significance level, this indicates the presence of autocorrelation.
            By performing the Breusch-Godfrey test on cryptocurrency data, we can assess the degree of autocorrelation in the residuals of the regression 
            model and potentially use this information to improve the model's accuracy.
            """


        def plot_using_acf(self,df):
            plt.plot(data)
            plt.title('Cryptocurrency Time Series')
            plt.show()
            plot_acf(data, lags=30)
            plt.title('Autocorrelation Function (ACF)')
            plt.show()

        def plot_using_pacf(self,df):
            plt.plot(data)
            plt.title('Cryptocurrency Time Series')
            plt.show()
            plot_pacf(data, lags=30)
            plt.title('Partial Autocorrelation Function (PACF)')
            plt.show()