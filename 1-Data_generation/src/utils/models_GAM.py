# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pygam import LinearGAM, s

"""
from pygam import LinearGAM,l,s, f, te
l() linear terms
s() spline terms
f() factor terms
te() tensor products
"""
from multiprocessing.pool import ThreadPool as Pool


def fit_gam(data_train, parameters=None, features=None, target=None):
    parameters = {"formule": s(0)} if parameters is None else parameters
    x_train, y_train = data_train[features], data_train[target]
    model_gam = LinearGAM(parameters["formule"], max_iter=1000).fit(x_train, y_train)
    return model_gam


def predict_gam(df_to_predict, model_gam, features):
    return model_gam.predict(df_to_predict[features])


def predict_gam_with_terms(df_to_predict, model_gam, features):
    terms = []
    for i in range(len(model_gam.terms) - 1):
        df_to_predict["term_" + str(i)] = model_gam.partial_dependence(term=i, X=df_to_predict[features])
        terms.append("term_" + str(i))
    return df_to_predict[terms]


def fit_gam_by(data_train, parameters=None, features=None, target=None, by_feature="tod"):
    parameters = {"formule": s(0)} if parameters is None else parameters
    split_model_values = sorted(list(set(data_train[by_feature])))
    pool = Pool()
    models_gam = {value: model for model, value in zip(pool.map(
        lambda value: fit_gam(data_train.loc[data_train[by_feature] == value], parameters=parameters, features=features,
                              target=target), split_model_values), split_model_values)}
    # optional if multiprocessing won't work on other os
    # models_gam = {value:fit_gam(data_train.loc[data_train[by_feature]==value], parameters=parameters,
    #                         features=features, target=target) for value in split_model_values}

    return models_gam


def predict_gam_return_dataframe(df_to_predict, model_gam, features):
    df_to_predict["gam_prediction"] = predict_gam(df_to_predict, model_gam, features)
    return df_to_predict


def predict_gam_by(df_to_predict, models_gam, features, by_feature="tod", with_terms=False):
    if len(df_to_predict) > 0:
        split_model_values = sorted(list(set(df_to_predict[by_feature])))
        pool = Pool()

        predictions = pool.map(lambda value: predict_gam_return_dataframe(
            df_to_predict=df_to_predict.loc[df_to_predict[by_feature] == value].reset_index(),
            model_gam=models_gam[value],
            features=features), split_model_values)

        if with_terms:
            predictions = pool.map(lambda value: pd.concat([predictions[value],
                                                            predict_gam_with_terms(
                                                                df_to_predict=df_to_predict.loc[
                                                                    df_to_predict[by_feature] == value].reset_index(),
                                                                model_gam=models_gam[value],
                                                                features=features)], axis=1), split_model_values)

        predictions = pd.concat(list(predictions))
        del(predictions["index"])
        predictions = predictions.sort_values(by="date").reset_index()

        if with_terms:
            return predictions[["gam_prediction"]+[col for col in predictions.columns if 'term' in col]]
        else:
            return predictions["gam_prediction"]
    else:
        return np.nan
