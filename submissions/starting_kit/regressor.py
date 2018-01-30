from sklearn.base import BaseEstimator
import lightgbm as lgb
#import xgboost as xgb

#class Regressor(BaseEstimator):
#    def __init__(self):
#        pass
#    def fit(self, X, y):
#        params = {"objective": "reg:linear", "booster":"gblinear"}
#        T_train_xgb = xgb.DMatrix(X,y)
#        self.reg = xgb.train(dtrain=T_train_xgb,params=params)
#    def predict(self, X):
#        XX = xgb.DMatrix(X)
#        return self.reg.predict(XX)
#from sklearn.ensemble import RandomForestRegressor

class Regressor(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y):
        params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'sub_feature': 0.7,
        'num_leaves': 60,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.7,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': -1,
         }
        self.reg = lgb.train(params,
                    lgb.Dataset(X,y),
                    num_boost_round=300,
                        )
    def predict(self, X):
        return self.reg.predict(X)