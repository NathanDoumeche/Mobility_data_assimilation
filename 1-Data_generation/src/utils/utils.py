# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging

def Rsummary(df):
    for c in df.columns:
        logging.info("_______________")
        new_col = df[c][~df[c].isna()]
        categories = list(set(new_col))
        dtype_serie = new_col.dtype
        info = str(c) + "-" + "NA:" + str(df[c].isna().sum()) + ' || '

        if len(categories)<50 :
            for category in categories :
                info+=str(category)+' : '+str(new_col.isin([category]).sum(axis=0))+' || '
        elif(dtype_serie == np.float64 or dtype_serie == np.int64 or "datetime64" in str(dtype_serie)):
            info+="min"+' : '+str(new_col.min())+' || '+"Q1"+' : '+str(new_col.quantile(0.25))+' || '+"median"+' : '
            info+=str(new_col.quantile(0.5))+' || '+"mean"+' : '+str(new_col.mean())+' || '+"Q3"+' : '
            info+=str(new_col.quantile(0.75))+' || '+"max"+' : '+str(new_col.max())+' || '
        else:
            info+="not summarizable for now !"+" || "+str(dtype_serie)
        logging.info(info)

def info_na(data, var, title):
    data = data[["date", var]].set_index(["date"])
    is_missing = pd.isna(data[var]) * 1
    diff_is_missing = is_missing.diff()
    diff_is_missing.loc[diff_is_missing.isna(),] = is_missing.loc[diff_is_missing.isna(),]  # first na management
    consecutive_nans = []
    count = 1
    countmax = 1
    for value in diff_is_missing:
        if value == 0:
            consecutive_nans.append(count)
            if count > 1:
                count += 1
        elif value == -1:
            consecutive_nans.append(count)
            count = 1
        else:
            consecutive_nans.append(count)
            count += 1
        countmax = max(count, countmax)

    return (f'\n  . {title} :  max consecutive na number {countmax - 1} et total na values {is_missing.sum()}')