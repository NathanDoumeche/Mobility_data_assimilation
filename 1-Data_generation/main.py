# -*- coding: utf-8 -*-
import datetime as dt
import logging
import os
import pathlib
from src.generate_calendar import generate_calendar
from src.process_rte_raw_datas import process_rte_raw_datas
from src.process_synop_temperature_datas import process_synop_temperature_datas
from src.merge_and_transform import merge_and_transform

os.chdir(pathlib.Path(__file__).parent)
if not os.path.exists("Logs"):
    os.makedirs("Logs")
logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename='Logs/log.txt', filemode='w')

date_end = dt.datetime(2023, 3, 1, 0, 0, 0, tzinfo=dt.timezone.utc)

if __name__ == "__main__":
    generate_calendar(date_start=dt.datetime(2010, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc), date_end=date_end)
    process_rte_raw_datas(date_start=dt.datetime(2013, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc), date_end=date_end)
    process_synop_temperature_datas()
    merge_and_transform(national_or_regional_features="national")
    merge_and_transform(national_or_regional_features="regional")
