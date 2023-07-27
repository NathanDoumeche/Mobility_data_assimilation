# -*- coding: utf-8 -*-
import datetime as dt
import pandas as pd
import numpy as np
import logging
from src.utils.utils import Rsummary


def read_calendar_datasets():
    data_holidays = pd.read_csv(filepath_or_buffer="Inputs/cal/data_vacances.csv", sep=",")
    data_jours_feries = pd.read_csv(filepath_or_buffer="Inputs/cal/jours_feries_metropole.csv", sep=",",
                                    usecols=["date"])
    return data_holidays, data_jours_feries


def left_join_holidays_processing(data_calendar, holidays):
    holidays["date"] = pd.to_datetime(holidays["date"], utc=True)

    holidays["period_holiday"] = 0
    holidays.loc[(holidays["vacances_zone_a"]) | (holidays["vacances_zone_b"]) | (
        holidays["vacances_zone_c"]), "period_holiday"] = 1

    holidays["period_holiday_zone_a"] = 0
    holidays.loc[holidays["vacances_zone_a"], "period_holiday_zone_a"] = 1
    holidays["period_holiday_zone_b"] = 0
    holidays.loc[holidays["vacances_zone_b"], "period_holiday_zone_b"] = 1
    holidays["period_holiday_zone_c"] = 0
    holidays.loc[holidays["vacances_zone_c"], "period_holiday_zone_c"] = 1

    holidays["period_christmas"] = 0
    holidays.loc[holidays["nom_vacances"] == "Vacances de Noël", "period_christmas"] = 1

    holidays["period_summer"] = 0
    holidays.loc[holidays["nom_vacances"] == "Vacances d'été", "period_summer"] = 1

    holidays = holidays[["date", "period_holiday", "period_holiday_zone_a", "period_holiday_zone_b",
                         "period_holiday_zone_c", "period_christmas", "period_summer"]]

    data_calendar = pd.merge(data_calendar, holidays, how="left", on="date")

    return data_calendar


def left_join_calendar_processing(data_calendar, bank_holidays):
    bank_holidays["date"] = pd.to_datetime(bank_holidays["date"], utc=True)

    day_after_bank_holidays = pd.DataFrame({"date": bank_holidays.date + pd.Timedelta(days=1)})

    day_before_bank_holidays = pd.DataFrame({"date": bank_holidays.date - pd.Timedelta(days=1)})

    # create features "jf" bank holidays, "ljf" day after bank holidays, "vjf" day before bank holidays
    bank_holidays["day_type_jf"] = 1
    day_after_bank_holidays["day_type_ljf"] = 1
    day_before_bank_holidays["day_type_vjf"] = 1

    data_cal = pd.merge(data_calendar, bank_holidays, how="left", on="date")
    data_cal = pd.merge(data_cal, day_after_bank_holidays, how="left", on="date")
    data_cal = pd.merge(data_cal, day_before_bank_holidays, how="left", on="date")

    data_cal.replace(np.nan, 0, inplace=True)

    data_cal["day_type_week"] = data_cal[
        "date"].dt.dayofweek  # Monday 0 Tuesday 1 Wednesday 2 Thursday 3 Friday 4 Saturday 5 Sunday 6

    data_cal["day_type_week_jf"] = data_cal["day_type_jf"]
    data_cal.loc[((data_cal["day_type_week"] == 5) | (data_cal["day_type_week"] == 6)), "day_type_week_jf"] = 1

    data_cal["month"] = data_cal["date"].dt.month
    data_cal["year"] = data_cal["date"].dt.year
    data_cal["Date"] = data_cal["year"] * 10000 + data_cal["month"] * 100 + data_cal["date"].dt.day
    data_cal["toy"] = data_cal["date"].dt.day_of_year / data_cal["date"].dt.day_of_year.max()
    data_cal["toy"] = data_cal["toy"].round(3)

    # days with hour changed
    """Since 1998, the winter time starts the last Sunday of October and the summer time starts the last Sunday of March
    Source : https://www.service-public.fr/particuliers/actualites/A15539"""

    data_hc = data_cal[["date", "month", "year", "day_type_week"]]

    data_hc = data_hc[((data_hc["day_type_week"] == 6) & ((data_hc["month"] == 3) | (data_hc["month"] == 10)))]
    del (data_hc["day_type_week"])

    # initialised hour changed dataset
    data_hour_changed = pd.DataFrame()

    # add only the last Sunday of March & October for each year to the hour changed dataset
    for y in set(data_hc["year"]):
        data_hc_mars = data_hc[((data_hc["year"] == y) & (data_hc["month"] == 3))]
        data_hc_mars = data_hc_mars[data_hc_mars["date"] == data_hc_mars["date"].max()]
        data_hc_oct = data_hc[((data_hc["year"] == y) & (data_hc["month"] == 10))]
        data_hc_oct = data_hc_oct[data_hc_oct["date"] == data_hc_oct["date"].max()]
        data_hour_changed = pd.concat([data_hour_changed, data_hc_mars, data_hc_oct])

    # create features "day_type_hc" hour changed
    data_hour_changed["day_type_hc"] = 1

    # create features "period_hour_changed" - 1 summer hour 0 winter hour
    data_hour_changed["period_hour_changed"] = 0
    data_hour_changed.loc[data_hour_changed["month"] == 3, "period_hour_changed"] = 1

    data_cal = pd.merge(data_cal, data_hour_changed[["date", "day_type_hc", "period_hour_changed"]], how="left",
                        on="date")

    data_cal["day_type_hc"].replace(np.nan, 0, inplace=True)

    # initialise hour change period
    if list(data_cal["period_hour_changed"].dropna())[0] == 0:
        data_cal.loc[0, "period_hour_changed"] = 1
    else:
        data_cal.loc[0, "period_hour_changed"] = 0

    # propagate hour change period
    data_cal["period_hour_changed"].fillna(method="ffill", inplace=True)

    return data_cal


def create_other_calendar_features(data_calendar):
    data_calendar["DayValidity"] = 1 - data_calendar["day_type_hc"]
    covid_periods = [(20200317, 20200510), (20201030, 20201214), (20210403, 20210502)]
    for start_date, end_date in covid_periods:
        data_calendar.loc[
            (data_calendar['Date'] >= start_date) & (data_calendar['Date'] <= end_date), "DayValidity"] = 0

    data_calendar["day_type_week_period_hour_changed"] = data_calendar["day_type_week"] * 10 + data_calendar[
        "period_hour_changed"]

    data_calendar["day_type_week_jf_period_holiday"] = data_calendar["day_type_week_jf"] * 10 + data_calendar[
        "period_holiday"]

    data_calendar["week_number"] = data_calendar["year"] * 100 + data_calendar["date"].dt.isocalendar().week
    data_calendar.loc[
        ((data_calendar["month"] == 1) & (data_calendar["date"].dt.isocalendar().week > 50)), "week_number"] = \
        data_calendar["week_number"] - 100  # week_number coherence between consecutive year

    return data_calendar


def expand_calendar_from_daily_to_half_hourly(data_calendar, date_start, date_end):
    # initialise a 30min calendar dataset
    data_calendar_tod = pd.DataFrame({"date": pd.date_range(start=date_start, end=date_end, freq='30min')})

    # create the integer Date and the time of day "tod" features
    data_calendar_tod["Date"] = data_calendar_tod["date"].dt.year * 10000 + data_calendar_tod["date"].dt.month * 100 + \
                                data_calendar_tod["date"].dt.day
    data_calendar_tod["tod"] = data_calendar_tod["date"].dt.hour * 2 + data_calendar_tod["date"].dt.minute / 30
    data_calendar_tod["tod"] = data_calendar_tod["tod"].astype('int')

    # delete date from daily dataset before merging
    del (data_calendar["date"])

    # transform data to a 30 minutes frequency dataset
    data_calendar_tod = pd.merge(data_calendar_tod, data_calendar, how="left", on="Date")

    # reorder columns
    data_calendar_tod = data_calendar_tod[['date', 'Date', 'tod', 'month', 'year', 'toy', 'day_type_jf', 'day_type_ljf',
                                           'day_type_vjf', 'day_type_week', 'day_type_week_jf', 'day_type_hc',
                                           'period_hour_changed', 'period_holiday', 'period_holiday_zone_a',
                                           'period_holiday_zone_b', 'period_holiday_zone_c', 'period_christmas',
                                           'period_summer', "day_type_week_period_hour_changed",
                                           "day_type_week_jf_period_holiday", "week_number", "DayValidity"]]

    return data_calendar_tod


def generate_calendar(date_start=dt.datetime(2010, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
                      date_end=dt.datetime(2023, 12, 31, 23, 30, 0, tzinfo=dt.timezone.utc),
                      return_calendar=False):
    """
    A function used to create a half-hourly calendar with these features :

    date  --> date - format yyyy-mm-dd HH:MM:SS
    Date --> date - format yyyymmdd tod --> time of day 0 = 00:30, 1 = 00:30, ..., 46 = 23:30, 47 = 00:00
    month --> month - format mm
    year --> year - format yy
    toy --> time of year 0 to 1
    day_type_jf --> 1 bank holiday, else 0
    day_type_ljf --> 1 day after bank holiday, else 0
    day_type_vjf --> 1 day before bank holiday, else 0
    day_type_week --> 0=monday, 1=tuesday, 2=wednesday, 3=thursday, 4=friday, 5=saturday, 6=sunday
    day_type_week_jf --> 1 bank holiday or week-end, else 0
    day_type_hc --> 1 hour changed day, else 0
    period_hour_changed --> 1 summer time 0 winter time
    period_holiday --> 1 french holidays, else 0
    period_holiday_zone_a --> 1 zone a french holidays, else 0
    period_holiday_zone_b --> 1 zone b french holidays, else 0
    period_holiday_zone_c --> 1 zone c french holidays, else 0
    period_christmas --> 1 Christmas holidays period, else 0
    period_summer --> 1 summer holidays period, else 0
    day_type_week_period_hour_changed -> summer time: (01=monday, 11=tuesday, 21=wednesday, 31=thursday, 41=friday,
    51=saturday, 61=sunday), winter time: (00=monday, 10=tuesday, 20=wednesday, 30=thursday, 40=friday, 50=saturday,
    60=sunday)
    day_type_week_jf_period_holiday -> holidays : (11 bank holiday or week-end, else 01), no holidays : (10 bank
    holiday or week-end, else 00)
    week_number --> format yyyyww, ww week / yyyy year or year-1 for the first days of the year
    DayValidity --> 0 by default except if the day is an hour changed day or within the covid period The
    covid period is between : . the 17th of March 2020 and the 10th of May 2020, . the 30th of October 2020 and the
    14th of October 2020, . the 3rd of April 2021 and the 2nd of May 2021.

    Sources :
    https://www.data.gouv.fr/fr/datasets/vacances-scolaires-par-zones/
    https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/
    ...

    Parameters
    ----------
    date_start : datetime64[ns, tz], optional
        The beginning date of the calendar dataset (default is dt.datetime(2010, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc))
    date_end : datetime64[ns, tz], optional
        The end date of the calendar dataset (default is dt.datetime(2023, 12, 31, 23, 30, 0, tzinfo=dt.timezone.utc))
    return_calendar : bool, optional (default is False)

    Returns
    -------
    Feather file
        calendar.feather
    DataFrame, optional
        The half-hourly calendar

    """

    print("Processing the calendar data:")
    logging.info("CALENDAR DATA LOADING")

    # French government datasets (see sources above)
    holidays, bank_holidays = read_calendar_datasets()

    # initialise a daily calendar dataset
    data_calendar = pd.DataFrame({"date": pd.date_range(start=date_start, end=date_end, freq='D')})

    # calendar processing
    data_calendar = left_join_holidays_processing(data_calendar=data_calendar, holidays=holidays)
    data_calendar = left_join_calendar_processing(data_calendar, bank_holidays=bank_holidays)
    data_calendar = create_other_calendar_features(data_calendar)
    data_calendar = expand_calendar_from_daily_to_half_hourly(data_calendar, date_start=date_start, date_end=date_end)

    # check data
    Rsummary(data_calendar)

    # save calendar
    data_calendar.to_feather("Outputs/calendar.feather")

    if return_calendar:
        return data_calendar
