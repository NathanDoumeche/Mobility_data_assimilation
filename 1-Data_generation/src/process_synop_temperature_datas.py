# -*- coding: utf-8 -*-
import datetime as dt
import logging
import math
import pandas as pd
from src.utils.utils import Rsummary, info_na
from src.utils.models_LR import fit_lr, predict_lr
from pathlib import Path

regions = ["auvergne_rhone_alpes", "bourgogne_franche_comte", "bretagne", "centre_val_de_loire", "grand_est",
           "haut_de_france", "ile_de_france", "normandie", "nouvelle_aquitaine", "occitanie", "paca",
           "pays_de_loire"]
region_file = ['Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne', 'Centre-Val de Loire', 'Grand Est',
               'Hauts-de-France', 'Île-de-France', 'Normandie', 'Nouvelle-Aquitaine', 'Occitanie',
               "Provence-Alpes-Côte d'Azur", 'Pays de la Loire']


def hav(phi): return math.pow(math.sin(phi / 2), 2)  # the haversine : half a versine of the angle phi


def ahav(h): return 2 * math.asin(math.sqrt(h))  # the archaversine : inverse to h = hav(phi)


def haversine_formula(latitude1, longitude1, latitude2, longitude2):  #see https://en.wikipedia.org/wiki/Haversine_formula
    radius_of_earth = 6371000  #see https://en.wikipedia.org/wiki/Earth

    lat1 = latitude1 * math.pi / 180
    long1 = longitude1 * math.pi / 180
    lat2 = latitude2 * math.pi / 180
    long2 = longitude2 * math.pi / 180
    delta_lat = lat2 - lat1
    delta_long = long2 - long1

    haversine_of_central_angle = hav(delta_lat) + math.cos(lat1) * math.cos(lat2) * hav(delta_long)
    distance_between_two_points = radius_of_earth * ahav(haversine_of_central_angle)
    return distance_between_two_points


def extract_synop_data():
    synop_data = None
    year = 2020
    while Path('Inputs/obs_meteo_france_synop/' + str(year) + '_donnees-synop-essentielles-omm.csv').is_file():
        print(year)
        synop_extract = pd.read_csv(
            filepath_or_buffer='Inputs/obs_meteo_france_synop/' + str(year) + '_donnees-synop-essentielles-omm.csv',
            sep=";", engine='pyarrow',
            usecols=["Date", "Température", "Vitesse du vent moyen 10 mn", "Nebulosité totale",
                     "communes (name)", "region (name)", "Latitude", "Longitude", "Altitude"])
        synop_extract = synop_extract[synop_extract["region (name)"].isin(region_file)]
        synop_data = pd.concat([synop_data, synop_extract], ignore_index=True)
        year = year + 1

    synop_data.columns = ["date", "temperature", "wind", "nebulosity", "city", "region", "latitude", "longitude",
                          "altitude"]

    # for each station, sort a list of closest meteo stations
    regions_cities = sorted(list(
        set(zip(synop_data["region"], synop_data["city"], synop_data["latitude"], synop_data["longitude"],
                synop_data["altitude"]))))

    closest_stations_by_stations = {}
    for region1, city1, latitude1, longitude1, altitude1 in regions_cities:
        closest_stations_by_stations[(region1, city1)] = []
        cities_checked = []
        for region2, city2, latitude2, longitude2, altitude2 in regions_cities:
            if city1 != city2 and city2 not in cities_checked:
                delta_altitude_level = abs(round((altitude2 - altitude1) / 300))
                distance = haversine_formula(latitude1=latitude1, longitude1=longitude1, latitude2=latitude2,
                                             longitude2=longitude2)
                closest_stations_by_stations[(region1, city1)].append((delta_altitude_level, distance, region2, city2))
                cities_checked.append(city2)
        closest_stations_by_stations[(region1, city1)] = sorted(closest_stations_by_stations[(region1, city1)])

    del (synop_data["latitude"], synop_data["longitude"], synop_data["altitude"])
    synop_data["date"] = synop_data["date"].dt.to_pydatetime()
    synop_data = synop_data.sort_values(by='date')

    return synop_data, closest_stations_by_stations


def data_generation_meteo_per_station_with_na(synop_data, data_calendar, closest_stations_by_stations):
    regions_cities = sorted(closest_stations_by_stations.keys())
    data_meteo_by_station = pd.DataFrame()
    for region, city in regions_cities:
        synop_data_station = synop_data.loc[
            ((synop_data['city'] == city) & (synop_data['region'] == region))].reset_index(drop=True)
        synop_data_station = pd.merge(data_calendar, synop_data_station.drop_duplicates(subset=["date"]), how="left",
                                      on="date")
        synop_data_station['city'] = city
        synop_data_station['region'] = region
        data_meteo_by_station = pd.concat([data_meteo_by_station, synop_data_station])

    data_meteo_by_station = data_meteo_by_station.sort_values(by='date').reset_index(drop=True)

    return data_meteo_by_station


def fill_only_one_consecutive_na(data, var):
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

    return data_interpolated_one_na


def fill_with_closest_station(data, data_near, var):
    data_near = data_near[["date", var]]
    data_near.columns = ["date", var + "_near"]

    data_completed = pd.merge(data, data_near.drop_duplicates(subset=["date"]), how="left", on="date")

    features = [var + "_near", "tod", "toy", "year"]
    data_train = data_completed[[var] + features].dropna()
    if len(data_train) > 100:
        model_lr = fit_lr(data_train=data_train, features=features, target=var)

        mask = ~data_completed[["date"] + features].isna().any(axis=1)
        data_completed.loc[mask, var + "_lr"] = predict_lr(data_completed.loc[mask], model_lr=model_lr,
                                                           features=features, date=["date"])

        mask_var_isna = data_completed[var].isna()
        data_completed.loc[mask_var_isna, var] = data_completed.loc[mask_var_isna, var + "_lr"]

        del (data_completed[var + "_lr"])
    del (data_completed[var + "_near"])

    return data_completed


def fill_by_tod_interpolation(data, var):
    data_interpolated_by_tod = pd.DataFrame()

    for tod in set(data["tod"]):
        data_tod = data[data["tod"] == tod].reset_index(drop=True)
        data_tod[var + "_interpolate"] = data[var].interpolate(method='linear', limit_direction="both")
        data_interpolated_by_tod = pd.concat([data_interpolated_by_tod, data_tod])

    data_interpolated_by_tod = data_interpolated_by_tod.sort_values(by='date').reset_index(drop=True)
    mask_var_isna = data_interpolated_by_tod[var].isna()
    data_interpolated_by_tod.loc[mask_var_isna, var] = data_interpolated_by_tod.loc[mask_var_isna, var + '_interpolate']

    del (data_interpolated_by_tod[var + "_interpolate"])

    return data_interpolated_by_tod


def fill_missing_data_by_station(data_meteo_by_station, var, closest_stations_by_stations):
    data_meteo_by_station[var + "_closest_with_lr"] = data_meteo_by_station[var]
    var = var + "_closest_with_lr"

    regions_cities = sorted(closest_stations_by_stations.keys())

    data_per_station = {}
    log_per_station = {}

    for region, city in regions_cities:
        log_per_station[(region, city)] = f'\n*** SYNOP meteo station - {var} - city : {city} / region {region}  ***'
        data_per_station[(region, city)] = data_meteo_by_station.loc[
            ((data_meteo_by_station['city'] == city) & (data_meteo_by_station['region'] == region))].reset_index(
            drop=True)
        log_per_station[(region, city)] = log_per_station[(region, city)] + info_na(
            data=data_per_station[(region, city)], var=var, title="step 0 - raw data")

        data_per_station[(region, city)] = fill_only_one_consecutive_na(data=data_per_station[(region, city)], var=var)
        log_per_station[(region, city)] = log_per_station[(region, city)] + info_na(
            data=data_per_station[(region, city)], var=var, title="step 1 - one consecutive fill")

    for i in range(len(regions_cities) - 1):
        for region, city in regions_cities:
            if data_per_station[(region, city)][var].isna().sum() > 0:
                delta_altitude_level, distance, region_near, city_near = closest_stations_by_stations[(region, city)][i]
                is_missing_diff = pd.isna(data_per_station[(region, city)][var]) * 1 - pd.isna(
                    data_per_station[(region_near, city_near)][var]) * 1
                if is_missing_diff.max() == 1:
                    data_per_station[(region, city)] = fill_with_closest_station(data=data_per_station[(region, city)],
                                                                                 data_near=data_per_station[
                                                                                     (region_near, city_near)], var=var)
                    log_per_station[(region, city)] = log_per_station[(region, city)] + info_na(
                        data=data_per_station[(region, city)], var=var,
                        title="step 2." + str(i) + " - fill lr with " + city_near)

    data_meteo_by_station_completed = pd.DataFrame()
    for region, city in regions_cities:
        if data_per_station[(region, city)][var].isna().sum() > 0:
            data_per_station[(region, city)] = fill_by_tod_interpolation(data=data_per_station[(region, city)], var=var)
            log_per_station[(region, city)] = log_per_station[(region, city)] + info_na(
                data=data_per_station[(region, city)], var=var, title="step 3 - interpolation by tod")
        data_meteo_by_station_completed = pd.concat([data_meteo_by_station_completed, data_per_station[(region, city)]])
        logging.info(log_per_station[(region, city)])

    data_meteo_by_station_completed = data_meteo_by_station_completed.sort_values(by='date').reset_index(drop=True)

    return data_meteo_by_station_completed


def data_generation_meteo_per_region(data_meteo_by_station, data_calendar):
    data_meteo_by_region = pd.DataFrame({"date": data_calendar["date"]})

    for region, region_id in zip(regions, region_file):
        for var in ["temperature", "wind", "nebulosity"]:
            data_meteo_one_region_by_station = data_meteo_by_station.loc[
                data_meteo_by_station['region'] == region_id, ["date", var]]
            data_meteo_one_region = data_meteo_one_region_by_station[["date", var]].dropna().groupby(
                "date").mean().reset_index()
            data_meteo_one_region.columns = ['date', var + "_" + region]
            data_meteo_by_region = data_meteo_by_region.merge(data_meteo_one_region, on='date', how='left')

    return data_meteo_by_region


def meteo_3h_to_30min_interpolation(data_meteo_by_region, data_calendar_3h, data_calendar_30min):
    data_meteo_by_region = data_calendar_3h[["date", "Date", "tod"]].merge(data_meteo_by_region, on='date', how='left')
    del (data_meteo_by_region["date"])
    data_meteo = data_calendar_30min[["date", "Date", "tod"]].merge(data_meteo_by_region, on=["Date", "tod"],
                                                                    how='left')

    for region in regions:
        print('\t' + region)
        logging.info(region)
        for var in ["temperature", "wind", "nebulosity"]:
            logging.info("\n\nbefore interpolation")
            Rsummary(data_meteo[[var + "_" + region]])
            data_meteo[var + "_" + region] = data_meteo[var + "_" + region].interpolate(method='linear',
                                                                                        limit_direction="both")

            logging.info("\n\nafter interpolation")
            Rsummary(data_meteo[[var + "_" + region]])

    del (data_meteo["Date"], data_meteo["tod"])
    return data_meteo


def national_data_meteo_by_electrical_load_weight(data_meteo_by_region):
    data_loads = pd.read_feather("Outputs/rte.feather")

    date_end_train = dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    data_loads = data_loads[data_loads.date < date_end_train].dropna()

    data_loads['Load_sum'] = data_loads[['Load_' + region for region in regions]].values.sum(axis=1)
    total_load_sum = data_loads['Load_sum'].values.sum(axis=0)
    electrical_weight_by_region = {r: data_loads['Load_' + r].values.sum(axis=0) / total_load_sum for r in regions}

    data_loads["error_Load_sum"] = data_loads['Load'] - data_loads['Load_sum']
    logging.info("Is there a difference between Load and sum(Load_regions) ? Should be equal to 0 : error_value=" + str(
        data_loads["error_Load_sum"].max() - data_loads["error_Load_sum"].min()))

    data_meteo_by_region_and_national = data_meteo_by_region.copy()
    for var in ["temperature", "wind", "nebulosity"]:
        data_meteo_by_region_and_national[var] = 0
        for region in regions:
            data_meteo_by_region_and_national[var] = data_meteo_by_region_and_national[var] + \
                                                     electrical_weight_by_region[region] * \
                                                     data_meteo_by_region_and_national[var + '_' + region]

    logging.info("Poids Conso : " + str(electrical_weight_by_region))

    return data_meteo_by_region_and_national


def process_synop_temperature_datas():
    print("Processing the meteorological data:")
    logging.info('METEOROLOGICAL DATA LOADING')

    synop_data, closest_stations_by_stations = extract_synop_data()

    data_calendar_30min = pd.read_feather("Outputs/calendar.feather")
    data_calendar_30min = data_calendar_30min[["date", "Date", "tod", "toy", "year"]]
    data_calendar_3h = data_calendar_30min.loc[data_calendar_30min["tod"] % 6 == 0]

    data_meteo_by_station = data_generation_meteo_per_station_with_na(synop_data=synop_data,
                                                                      data_calendar=data_calendar_3h,
                                                                      closest_stations_by_stations=closest_stations_by_stations)

    for var in ["temperature", "wind", "nebulosity"]:
        data_meteo_by_station = fill_missing_data_by_station(data_meteo_by_station=data_meteo_by_station, var=var,
                                                             closest_stations_by_stations=closest_stations_by_stations)

    data_meteo_by_region = data_generation_meteo_per_region(data_meteo_by_station=data_meteo_by_station,
                                                            data_calendar=data_calendar_3h)

    data_meteo_by_region = meteo_3h_to_30min_interpolation(data_meteo_by_region=data_meteo_by_region,
                                                           data_calendar_3h=data_calendar_3h,
                                                           data_calendar_30min=data_calendar_30min)

    Rsummary(data_meteo_by_region)

    data_meteo_by_region_and_national = national_data_meteo_by_electrical_load_weight(
        data_meteo_by_region=data_meteo_by_region)
    Rsummary(data_meteo_by_region_and_national)

    data_meteo_by_region_and_national.to_feather("Outputs/synop.feather")
    return data_meteo_by_region_and_national
