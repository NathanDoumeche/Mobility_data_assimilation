# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
def fit_lr(data_train, features=None, target=None):
    x_train, y_train = data_train[features], data_train[target]
    model_lr = LinearRegression(n_jobs=-1).fit(x_train, y_train)
    return model_lr


def predict_lr(df_to_predict, model_lr, features, date):
    X_to_predict, y_to_predict  = df_to_predict[features], df_to_predict.loc[:, date]
    y_to_predict["pred"] = model_lr.predict(X_to_predict)
    y_pred = y_to_predict.sort_values(by='date')
    return (y_pred["pred"])
