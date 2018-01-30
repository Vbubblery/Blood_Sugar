import time
import datetime
import pandas as pd
import numpy as np
from dateutil.parser import parse
class FeatureExtractor():
    def clean_dataset(self,X_df):
        assert isinstance(X_df, pd.DataFrame), "df needs to be a pd.DataFrame"
        #X_df.dropna(inplace=True)
        #indices_to_keep = ~X_df.isin([np.nan, np.inf, -np.inf]).any(1)
        return X_df.iloc[:,:].astype(np.float64)
    
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X_df['sex'] = X_df['sex'].map({'M':1,'F':0})
        X_df['date'] = (pd.to_datetime(X_df['date']) - parse('2018-01-26')).dt.days
        X_df.fillna(X_df.median(axis=0),inplace=True)
        X_df = self.clean_dataset(X_df)
        return X_df.values
    