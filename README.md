# Mobility data assimilation to forecast electricity demand in a sobriety context in France

## Overview

**You can directly download the final dataset called *dataset_national.csv* spanning from 2013-01-08 to 2023-03-01 
at https://drive.google.com/file/d/1iMLPkr9nzXOF9cORLBsICDBLPcg0a7m5/view?usp=drive_link.**

## Citation

    @article{doumeche2023human,
      title={Human spatial dynamics for electricity demand forecasting: the case of France during the 2022 energy crisis},
      author={Doumèche, Nathan and Allioux, Yann and Goude, Yannig and Rubrichi, Stefania},
      journal={arXiv:2309.16238},
      volume={},
      pages={},
      year={2023},
      publisher={}
    }

## 1. Installation guide

### Add the heavy data
Go to the folder 1-Data_generation and unzip the Input.zip file available at https://drive.google.com/file/d/1CWKafkx8yXZLLlzOdq1AXNny-mLax_Wj/view?usp=drive_link.

### Build the dataset
Install a Python IDE (Pycharm, Visual Studio Code...) and use it to open the folder 1-Data_generation. 
Next, create a virtual environment using Python 3.9.13.
You may download it on https://www.python.org/downloads/release/python-3913/. 
In the terminal, execute the following instructions to install the proper packages.

    cd 1-Data_generation
    pip3 install -r requirements.txt

Then, you may run the code with the following command.

    python main.py

The dataset will then be created in the _1-Data_generation/Outputs_ folder as _dataset_national.csv_ (for national 
load) and _dataset_regional.csv_ (for regional loads).

### Run the benchmark
Install a R IDE (Pycharm, Rstudio...) and use it to open the folder 2-Benchmark.
In the terminal, execute the following instructions to install the proper packages.

## 2. Dataset description

### Calendar data


Each dataset contains the following features:
- date: date (yyyy-mm-dd HH:MM:SS)
- Date: date (yyyymmdd) 
- day_type_jf: 1 on bank holidays, else 0
- day_type_ljf: 1 on days after bank holidays, else 0
- day_type_vjf: 1 on day before bank holidays, else 0
- day_type_week: 0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday, 5 = Saturday, 6 = Sunday
- day_type_week_jf: 1  for bank holidays or weekends, else 0
- day_type_week_period_hour_changed: 
  * summer time: 01 = Monday, 11 = Tuesday, 21 = Wednesday, 31 = Thursday, 41 = Friday,
      51 = Saturday, 61 = Sunday, 
  * winter time: 00 = Monday, 10 = Tuesday, 20 = Wednesday, 30 = Thursday, 40 = Friday, 50 = Saturday,
      60 = Sunday
- day_type_week_jf_period_holiday : 
  * holidays: 11 on bank holidays or weekends, else 01, 
  * no holidays: 10 on bank holidays or weekends, else 00)
- day_type_hc: 1 on hour-changed day, else 0
- DayValidity: 0 by default, 1 on an hour-changed day and on the Covid period (between the 17th of March 2020 and the 10th of May 2020,
between the 30th of October 2020 and the 14th of October 2020, and between the 3rd of April 2021 and the 2nd of May 2021)
- month: month (mm)
- period_hour_changed: 1 during summer and during 0 winter
- period_holiday: 1 on French national holidays (zones a,b, and c), else 0
- period_holiday_zone_a: 1 on zone a holidays, else 0
- period_holiday_zone_b: 1 on zone b holidays, else 0
- period_holiday_zone_c: 1 on zone c holidays, else 0
- period_christmas: 1 on Christmas holidays, else 0
- period_summer: 1 on summer holidays, else 0
- tod: time of day (0 = 00:30, 1 = 01:00, ..., 46 = 23:30, 47 = 00:00)
- toy: time of year (from 0 on January 01, to 1 on December 31)
- week_number  
- year: year (yy)

Sources :
- https://www.data.gouv.fr/fr/datasets/vacances-scolaires-par-zones/
- https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/

### Electricity data

This data concerns France, its 12 administrative regions and its 21 metropolitan areas.
The "Load" or "Load_region" feature is the national or regional power-related French electricity consumption
measured in MegaWatt (MW).
According to the delay, the accuracy of the data differs.
- Real-time data : real-time smart devices data + estimation when data non available
- Consolidated data (available after one month): RTE and Enedis measured data 
- Final data  (available after one year): RTE and local distribution entities (Enedis, municipalities, etc.) measured data


Source : https://www.rte-france.com/en/eco2mix/download-indicators


### Meteorological data

Climatic features :
    Données d'observations issues des messages internationaux d’observation en surface (SYNOP)
    circulant sur le système mondial de télécommunication (SMT) de l’Organisation Météorologique Mondiale (OMM).
    Paramètres atmosphériques mesurés (température, humidité, direction et force du vent, pression atmosphérique, hauteur de précipitations)
    ou observés (temps sensible, description des nuages, visibilité) depuis la surface terrestre.
    Selon instrumentation et spécificités locales, d'autres paramètres peuvent être disponibles (hauteur de neige, état du sol, etc.)
        temperature --> French temperature - ponderate by regional electrical weight (K)
        temperature_"r" --> regional temperature - ponderate by regional electrical weight (K)

Source : https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm/export/?sort=date

## 3. Benchmark 