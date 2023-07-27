# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import datetime as dt
from src.utils.utils import Rsummary
from unidecode import unidecode
from pathlib import Path

# global variable
regions = ["national", "auvergne_rhone_alpes", "bourgogne_franche_comte", "bretagne", "centre_val_de_loire",
           "grand_est", "haut_de_france", "ile_de_france", "normandie", "nouvelle_aquitaine",
           "occitanie", "paca", "pays_de_loire"]
map_regions_encoding = {region: region_encoding for region, region_encoding in
                        zip(regions, ["", "Auvergne-Rhône-Alpes_", "Bourgogne-Franche-Comté_",
                                      "Bretagne_", "Centre-Val-de-Loire_",
                                      "Grand-Est_", "Hauts-de-France_", "Ile-de-France_",
                                      "Normandie_", "Nouvelle-Aquitaine_",
                                      "Occitanie_", "PACA_", "Pays-de-la-Loire_"])}


def read_load(eco2mix_load_path):
    eco2mix_load = pd.read_csv(eco2mix_load_path, encoding='windows-1252', sep="\t", engine="python",
                               usecols=['Date', 'Heures', 'Consommation'], index_col=False)
    eco2mix_load["minutes"] = eco2mix_load["Heures"].str[3:5]
    eco2mix_load = eco2mix_load[eco2mix_load["minutes"].isin(['00', '30'])].reset_index(drop=True)
    eco2mix_load["Date"] = pd.to_datetime(eco2mix_load["Date"], utc=True, dayfirst=True)
    return eco2mix_load


def create_nat_or_reg_data_load(region):
    print('\t' + region)
    logging.info(region)
    region_encoding = map_regions_encoding[region]
    eco2mix_load_real_time = read_load(
        eco2mix_load_path="Inputs/rte/eCO2mix_RTE_" + unidecode(region_encoding) + "En-cours-TR.xls")
    eco2mix_load_consolidated = read_load(
        eco2mix_load_path="Inputs/rte/eCO2mix_RTE_" + region_encoding + "En-cours-Consolide.xls")
    nat_or_reg_data_load = pd.concat([eco2mix_load_real_time, eco2mix_load_consolidated], ignore_index=True)

    year = 2012 if region == "national" else 2013
    eco2mix_load_path_root = "Inputs/rte/conso_nat/" if region == "national" else "Inputs/rte/conso_regions/" + region + "/"
    while Path(eco2mix_load_path_root + "eCO2mix_RTE_" + region_encoding + "Annuel-Definitif_" + str(
            year) + ".xls").is_file():
        eco2mix_load_final = read_load(
            eco2mix_load_path=eco2mix_load_path_root + "eCO2mix_RTE_" + region_encoding + "Annuel-Definitif_" + str(
                year) + ".xls")
        nat_or_reg_data_load = pd.concat([nat_or_reg_data_load, eco2mix_load_final],
                                         ignore_index=True)
        year = year + 1

    nat_or_reg_data_load = nat_or_reg_data_load.rename(
        columns={"Date": "date_jour", "Heures": "heures", "Consommation": "Load"})
    nat_or_reg_data_load["date"] = nat_or_reg_data_load["date_jour"].astype(str).str[:11] + nat_or_reg_data_load[
        "heures"]
    nat_or_reg_data_load = nat_or_reg_data_load.loc[:, ["date", "Load"]]
    nat_or_reg_data_load["date"] = pd.to_datetime(nat_or_reg_data_load["date"], utc=True)
    nat_or_reg_data_load = nat_or_reg_data_load.sort_values(by='date').reset_index(drop=True)
    nat_or_reg_data_load.loc[nat_or_reg_data_load['Load'] == "ND", 'Load'] = np.nan
    nat_or_reg_data_load['Load'] = pd.to_numeric(nat_or_reg_data_load['Load'])
    return nat_or_reg_data_load


def interpolation_data_load(data, var='Load'):
    #logging.info("avant interpolation")
    #Rsummary(data[[var]])

    data_interpolated_one_na = data.copy()
    data_interpolated_one_na[var + "_lag1"] = data_interpolated_one_na[var].shift(1)
    data_interpolated_one_na[var + "_lead1"] = data_interpolated_one_na[var].shift(-1)

    data_interpolated_one_na[var + "_interpolate"] = data_interpolated_one_na[var].interpolate(method='linear', limit=1,
                                                                                               limit_direction="both")

    mask_only_one_consecutive_na = (data_interpolated_one_na[var + "_lag1"].notna()) & (
        data_interpolated_one_na[var].isna()) & (data_interpolated_one_na[var + "_lead1"].notna())
    data_interpolated_one_na.loc[mask_only_one_consecutive_na, var] = data_interpolated_one_na.loc[
        mask_only_one_consecutive_na, var + '_interpolate']

    del (data_interpolated_one_na[var + "_lag1"], data_interpolated_one_na[var + "_lead1"],
         data_interpolated_one_na[var + "_interpolate"])

    #logging.info("après interpolation")
    #Rsummary(data_interpolated_one_na[[var]])

    return data_interpolated_one_na


def process_rte_raw_datas(date_start=dt.datetime(2013, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
                          date_end=dt.datetime(2023, 2, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
                          return_data_load=False):
    """
    A function used to process the eco2mix Power-related data (*) (real-time, consolidated and final): this data
    concerns France, its 12 administrative regions and its 21 metropolitan areas.
        real-time data : real-time smart devices data + estimation when data non available
        consolidated data : rte and enedis measured data (available after one month)
        final data : rte and local distribution entities (enedis, municipalities, etc.) measured data (available after
        one year)
    The "Load" or "Load_region" extracted is the national or regional power-related french electricity consumption,
    unit of measure = MW
    (*) https://www.rte-france.com/en/eco2mix/download-indicators

    ...

    Parameters
    ----------
    date_start : datetime64[ns, tz], optional
        The beginning date of the rte dataset (default is dt.datetime(2013, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc))
    date_end : datetime64[ns, tz], optional
        The end date of the rte dataset (default is dt.datetime(2023, 12, 31, 23, 30, 0, tzinfo=dt.timezone.utc))
    return_data_load : bool, optional
        (default is False)

    Returns
    -------
    Feather file
        rte.feather
    DataFrame, optional
        rte data_load

    """
    print("Processing the electricity load data:")
    logging.info("ELECTRICITY DATA LOADING")

    data_load = pd.DataFrame()
    for region in regions:
        nat_or_reg_data_load = create_nat_or_reg_data_load(region=region)

        nat_or_reg_data_load = interpolation_data_load(nat_or_reg_data_load)

        nat_or_reg_data_load.columns = ['date', 'Load'] if region == "national" else ['date', 'Load_' + region]

        nat_or_reg_data_load = nat_or_reg_data_load[
            ((nat_or_reg_data_load['date'] >= date_start) &
             (nat_or_reg_data_load['date'] <= date_end))]

        data_load = nat_or_reg_data_load if region == "national" else data_load.merge(
            nat_or_reg_data_load, on="date", how='left')

    Rsummary(data_load)

    # save rte data
    data_load.to_feather("Outputs/rte.feather")

    if return_data_load:
        return data_load
