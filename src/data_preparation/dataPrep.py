import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

class Preparetor:
    """
    Contains all the analysis of the data that included:
    - desc stats
    - distribution
    - distribution graph
    """
    def __init__(self):
        self.Prep_Res = {}
        
    def data_modelling(self,data):
        def clean_columns(col):
            if col.name.lower().startswith('date'):
                return pd.to_datetime(col)
            return pd.to_numeric(col.str.replace(',', '').str.replace('$', ''), errors='coerce')
        data = data.apply(clean_columns)
        return data

    def get_basicInfo(self,data):
        self.Prep_Res['Shape'] = data.shape
        self.Prep_Res['info'] = data.info() 
        self.Prep_Res['datatype'] = data.dtypes       
        self.Prep_Res['nullVal_count'] = data.isnull().sum()

    def get_IQR(self,data):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        self.Prep_Res['iqr'] = iqr

    def outliers_handling(self,data):
        q1, q3, iqr = self.getIQR(data)
        outliers = (data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))
        outlier_rows = np.where(outliers.any(axis=1))[0]
        if len(outlier_rows) < 1:
            return len(outlier_rows)
        else:
            # Winsorization
            cols_to_winsorize = data.columns
            data[cols_to_winsorize] = data[cols_to_winsorize].apply(lambda x: winsorize(x, limits=(0.05, 0.05)))
            return data

        