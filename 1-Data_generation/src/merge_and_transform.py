# -*- coding: utf-8 -*-
import pandas as pd
import logging
import datetime as dt
from pygam import l, s, te
from src.utils.models_GAM import fit_gam_by, predict_gam_by
from src.utils.models_RF import fit_rf_by, predict_rf_by
from src.utils.utils import Rsummary, info_na


def exponential_smoothing(input_vector, smoothing_coefficient):
    input_vector = list(input_vector)
    output_vector = input_vector.copy()
    for n in range(1, len(output_vector)):
        output_vector[n] = (1 - smoothing_coefficient) * input_vector[n] + smoothing_coefficient * output_vector[n - 1]
    return output_vector


def interpolation_with_prevision(data, column_with_na, prevcolumn):
    data = data.copy()
    data["z_i"] = data[prevcolumn]
    data.loc[~data[column_with_na].isna(), "z_i_with_na"] = data.loc[~data[column_with_na].isna(), "z_i"]

    data["z_0"] = data["z_i_with_na"].fillna(method='ffill')
    data["z_n-1"] = data["z_i_with_na"].fillna(method='bfill')

    data["y_0"] = data[column_with_na].fillna(method='ffill')
    data["y_n-1"] = data[column_with_na].fillna(method='bfill')

    data.loc[~data[column_with_na].isna(), "z_i"] = data.loc[~data[column_with_na].isna(), column_with_na]

    data["NA"] = pd.isna(data[column_with_na])

    i_value = 0
    i_values = []
    i_nan = []
    n_1_i_values = []
    n_1_values = []

    for na_check in data["NA"]:
        if na_check:
            i_value = i_value + 1
            i_nan.append(i_value)
        else:
            if i_value > 0:
                i_values = i_values + i_nan
                n_1_i_values = n_1_i_values + list(reversed(i_nan))
                n_1_values = n_1_values + [max(i_nan) + 1 for i in i_nan]
                i_nan = []
                i_value = 0
            i_values.append(i_value)
            n_1_i_values.append(i_value)
            n_1_values.append(1)

    if i_value > 0:
        i_values = i_values + i_nan
        n_1_i_values = n_1_i_values + list(reversed(i_nan))
        n_1_values = n_1_values + [max(i_nan) + 1 for i in i_nan]

    data["i"] = i_values
    data["n_1_i"] = n_1_i_values
    data["n_1"] = n_1_values
    data["yi"] = data["z_i"] + (data["y_0"] - data["z_0"]) * data["n_1_i"] / data["n_1"] + (
            data["y_n-1"] - data["z_n-1"]) * data["i"] / data["n_1"]

    return data["yi"]


def rte_na_completion_with_gamrfres_and_interpolation(regional_merge, regions):
    regional_merge.reset_index(inplace=True, drop=True)
    print("Completion of the NA electricity load data:")
    for region in regions:
        print("   . "+region[1:], end=" : ") if region!="" else print("   . france", end=" : ")
        features = ['temperature' + region, 'temperature' + region + '_smooth_990',
                    'temperature' + region + '_smooth_950', 'toy', 'day_type_week', 'day_type_week_jf',
                    'period_hour_changed', 'period_christmas', 'period_summer', 'period_holiday_zone_a',
                    'period_holiday_zone_b', 'period_holiday_zone_c', 'day_type_ljf', 'day_type_vjf', 'Date']
        tr = {features[i]: i for i in range(len(features))}
        parameters = {
            "formule": s(tr['temperature' + region]) + s(tr['temperature' + region + '_smooth_990']) + s(
                tr['temperature' + region + '_smooth_950']) + te(tr['toy'], tr['day_type_week']) + te(
                tr['day_type_week_jf'], tr['period_hour_changed']) + l(tr['period_christmas']) + l(
                tr['period_summer']) + l(tr['period_holiday_zone_a']) + l(tr['period_holiday_zone_b']) + l(
                tr['period_holiday_zone_c']) + l(tr['day_type_ljf']) + l(tr['day_type_vjf']) + s(tr['Date'])}
        target = 'Load' + region

        print("gam training", end=" / ")
        models_gam = fit_gam_by(data_train=regional_merge.loc[regional_merge["DayValidity"] == 1][
            features + ["date", "tod", target]].dropna(),
                                parameters=parameters, features=features, target=target, by_feature="tod")

        print("gam forecast", end=" / ")
        prediction_and_terms = predict_gam_by(df_to_predict=regional_merge, models_gam=models_gam, features=features,
                                              by_feature="tod", with_terms=True)
        print("rfres training", end=" / ")
        prediction_and_terms.columns = [col + region for col in prediction_and_terms.columns]
        prediction_and_terms = pd.concat(
            [prediction_and_terms, regional_merge[["date", "tod", target, "Date", "day_type_jf", "day_type_week"]]],
            axis=1)

        prediction_and_terms["ResLoad" + region] = prediction_and_terms[target] - prediction_and_terms[
            "gam_prediction" + region]

        features_rf = ["Date", "day_type_jf", "day_type_week"] + [col for col in prediction_and_terms.columns if
                                                                  'term' in col]
        target_rf = "ResLoad" + region

        models_rf = fit_rf_by(data_train=prediction_and_terms[features_rf + ["date", "tod", target_rf]].dropna(),
                              parameters={"n_estimators": 50, "random_state": 50}, features=features_rf,
                              target=target_rf, by_feature="tod")

        print("rfres forecast", end=" / ")
        prediction_and_terms['PrevResLoad' + region] = predict_rf_by(df_to_predict=prediction_and_terms,
                                                                     models_rf=models_rf, features=features_rf,
                                                                     by_feature="tod")

        prediction_and_terms['PrevLoad' + region] = prediction_and_terms["gam_prediction" + region] + \
                                                    prediction_and_terms['PrevResLoad' + region]

        regional_merge = pd.concat([regional_merge, prediction_and_terms['PrevLoad' + region]], axis=1)

        print("interpolation")
        logging.info(f'\n*** RTE data - {target} - region : {region}  ***')
        logging.info(info_na(data=regional_merge, var=target, title="before completion"))
        Rsummary(regional_merge[[target]])
        regional_merge[target] = interpolation_with_prevision(data=regional_merge, column_with_na=target,
                                                              prevcolumn='PrevLoad' + region)
        logging.info(info_na(data=regional_merge, var=target, title="after completion"))
        Rsummary(regional_merge[[target]])
        del (regional_merge['PrevLoad' + region])

    return regional_merge


def merge_and_transform(national_or_regional_features="national"):
    """
    A function used to merge three datasets:
    . the half-hourly calendar (see generate_calendar function),
    . the eco2mix Power-related data (see process_rte_raw_datas function) and
    . the Meteorological dataset (see process_synop_temperature_datas function)

    This function is filling NA values for the eco2mix Power-related data.
    ...

    Parameters ---------- national_or_regional_features : str or list of str, optional --- possible str values :
    "regional" or "national"
    Returns
    -------
    csv file
        dataset_[...].csv

    """
    logging.info('\n\nMERGING DATA AT '+str(national_or_regional_features)+' LEVEL(S)')
    print("\nMerging the data:")
    admissible_regions = ["auvergne_rhone_alpes", "bourgogne_franche_comte", "bretagne", "centre_val_de_loire",
                          "grand_est", "haut_de_france", "ile_de_france", "normandie", "nouvelle_aquitaine",
                          "occitanie", "paca", "pays_de_loire"]

    # coherence / selection level(s) of extraction
    regions = []
    name_file = ""
    if national_or_regional_features in ["regional", "national"]:
        name_file = "_"+national_or_regional_features
        if national_or_regional_features == "regional":
            regions = ["_" + region for region in admissible_regions]
        elif national_or_regional_features == "national":
            regions = [""]

    # read and merge datasets
    rte = pd.read_feather("Outputs/rte.feather")
    calendar = pd.read_feather("Outputs/calendar.feather")
    synop = pd.read_feather("Outputs/synop.feather")
    regional_merge = (calendar.merge(synop, on='date', how='left')).merge(rte, on='date', how='left')

    # smoothing and lags
    features_to_keep = ["date", "Date", "tod"]
    regional_merge['day_type_jf_d1'] = regional_merge['day_type_jf'].shift(1 * 48)
    regional_merge['day_type_jf_d7'] = regional_merge['day_type_jf'].shift(7 * 48)
    for region in regions:
        regional_merge['temperature' + region + "_smooth_990"] = exponential_smoothing(
            input_vector=regional_merge['temperature' + region], smoothing_coefficient=0.990)
        regional_merge['temperature' + region + "_smooth_950"] = exponential_smoothing(
            input_vector=regional_merge['temperature' + region], smoothing_coefficient=0.950)
        regional_merge['temperature' + region + "_max_smooth_990"] = regional_merge['temperature' + region + "_smooth_990"].rolling(48).max()
        regional_merge['temperature' + region + "_min_smooth_990"] = regional_merge['temperature' + region + "_smooth_990"].rolling(48).min()
        regional_merge['temperature' + region + "_max_smooth_950"] = regional_merge[
            'temperature' + region + "_smooth_950"].rolling(48).max()
        regional_merge['temperature' + region + "_min_smooth_950"] = regional_merge[
            'temperature' + region + "_smooth_950"].rolling(48).min()
        regional_merge['Load' + region + '_d1'] = regional_merge['Load' + region].shift(1 * 48)
        regional_merge['Load' + region + '_d7'] = regional_merge['Load' + region].shift(7 * 48)
        features_to_keep = features_to_keep + ['Load' + region, 'Load' + region + '_d1', 'Load' + region + '_d7',
                                               'temperature' + region, 'temperature' + region + "_smooth_990",
                                               'temperature' + region + "_smooth_950", 'temperature' + region + "_max_smooth_990",
                                               'temperature' + region + "_max_smooth_950", 'temperature' + region + "_min_smooth_990",
                                               'temperature' + region + "_min_smooth_950"]

    features_to_keep = features_to_keep + [col for col in calendar.columns if col not in ["date", "Date", "tod"]]
    regional_merge = regional_merge[features_to_keep]

    # min/max dates : valid observations management
    # rte_kept_features na removal before min/max dates calculation
    rte_na = rte[[f for f in features_to_keep if f in rte.columns]].dropna()
    # max of 7 day lag +
    # timing for perfect smoothing 260 - 312 K * pow(0.995, 48 tods * 60 days) -> 0.00014 - 0.00017
    date_min = max(rte_na['date'].min() + dt.timedelta(days=7), synop['date'].min() + dt.timedelta(days=60))
    date_max = min(rte_na['date'].max(), synop["date"].max())
    if calendar["date"].min() > date_min:
        print("Warning - check calendar min date")
    if calendar["date"].max() < date_max:
        print("Warning - check calendar max date")

    # rte completion data except if there is na values at the beginning or the end
    regional_merge = rte_na_completion_with_gamrfres_and_interpolation(regional_merge=regional_merge[
        ((regional_merge['date'] >= rte_na['date'].min()) & (regional_merge['date'] <= rte_na['date'].max()))],
                                                                       regions=regions)

    # selection of valid observations
    regional_merge = regional_merge[((regional_merge['date'] >= date_min) & (regional_merge['date'] <= date_max))]
    regional_merge.to_csv("Outputs/dataset" + name_file + ".csv")
    Rsummary(regional_merge)
