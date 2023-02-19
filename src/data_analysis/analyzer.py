import pandas as pd
import numpy as np
import seaborn as sns
from tabulate import tabulate
from fitter import Fitter, get_common_distributions, get_distributions


class Analyzer:
    """
    Contains all the analysis of the data that included:
    - descriptive stats
    - distribution
    - distribution graph
    """
    def __init__(self):
        self.analysis_res = {}

    def col_meta_store(self, data):
        df = pd.DataFrame(data.dtypes).reset_index()
        df.columns = ['field_name', 'current_datatype']
        df.to_csv("../data/meta_data.csv")

    def diff_metrics_calc(self,data):
        data.set_index('Date', inplace=True)

        #================ Calculating Daily Returns ================#
        data['Log_Close'] = np.log(data['Close'])
        data['Daily_Return'] = data['Close'].pct_change()
        data['Daily_Log_Returns'] = np.log(data['Close']).pct_change()
        #===========================================================#

        #=========================== Log Volume =============================#
        data['Log_Volume'] = np.log(data['Volume'])
        data['Daily_Log_Volumetric_Change'] = np.log(data['Log_Volume']).pct_change()
        #====================================================================#
    
        return data.iloc[1:]
    
    def get_descriptive_stats(self, data):
        q1, q3, iqr = self.getIQR(data)
        stats = pd.concat([data.describe().T,
                   data.mad().to_frame('MAD'),
                   data.skew().to_frame('Skew'),
                   data.kurtosis().to_frame('Kurtosis'),
                   data.sem().to_frame('SEM'),
                   iqr.to_frame('IQR'),
                   data.var().to_frame('Variance')
                ],
                  axis=1)
        self.analysis_res['DescStat'] = tabulate(stats, headers='keys', tablefmt='psql')


    def get_distribution(self, data):
        DailyLogReturn = data["Daily_Log_Return"].values
        dist = Fitter(DailyLogReturn,timeout=10000)
        dist.fit()
        self.analysis_res['distSum'] = dist.summary()

    def plot_distribution(self, data):
        sns.set_style('white')
        sns.set_context("paper", font_scale = 2)
        sns.displot(data=data, x="Volume", kind="hist", bins = 100, aspect = 1.0)
        