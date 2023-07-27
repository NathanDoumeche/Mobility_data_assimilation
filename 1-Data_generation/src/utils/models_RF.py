# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from multiprocessing.pool import ThreadPool as Pool  #optional because rf are already well optimized with n_jobs=-1

np.random.seed(1000)

def fit_rf(data_train, parameters=None, features=None, target=None):
    parameters = {"n_estimators": 50, "random_state": 50} if parameters is None else parameters
    x_train, y_train = data_train[features], data_train[target]
    model_rf = RandomForestRegressor(n_estimators=parameters["n_estimators"], random_state=parameters["random_state"],
                                     n_jobs=-1).fit(x_train, y_train)
    return model_rf


def predict_rf(df_to_predict, model_rf, features):
    return model_rf.predict(df_to_predict[features])


def fit_rf_by(data_train, parameters=None, features=None, target=None, by_feature="tod"):
    parameters = {"n_estimators": 50, "random_state": 50} if parameters is None else parameters
    split_model_values = sorted(list(set(data_train[by_feature])))
    pool = Pool()
    models_rf = {value: model for model, value in zip(pool.map(
        lambda value: fit_rf(data_train.loc[data_train[by_feature] == value], parameters=parameters, features=features,
                             target=target), split_model_values), split_model_values)}
    # optional if multiprocessing won't work on other os
    # models_rf = {value:fit_rf(data_train.loc[data_train[by_feature]==value], parameters=parameters, features=features,
    #                         target=target) for value in split_model_values}

    return models_rf


def predict_rf_return_dataframe(df_to_predict, model_rf, features):
    df_to_predict["rf_prediction"] = predict_rf(df_to_predict, model_rf, features)
    return df_to_predict


def predict_rf_by(df_to_predict, models_rf, features, by_feature="tod"):

    if len(df_to_predict)>0:
        split_model_values = sorted(list(set(df_to_predict[by_feature])))
        pool = Pool()
        predictions = pool.map(lambda value: predict_rf_return_dataframe(
            df_to_predict=df_to_predict.loc[df_to_predict[by_feature] == value].reset_index(), model_rf=models_rf[value],
            features=features), split_model_values)

        predictions = pd.concat(list(predictions))
        del (predictions["index"])
        predictions = predictions.sort_values(by="date").reset_index()

        return predictions["rf_prediction"]

    else:
        return np.nan
