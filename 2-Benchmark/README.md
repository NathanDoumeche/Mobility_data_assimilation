---
title: "Mobility data assimilation to forecast electricity demand in a sobriety context in France"
output: html_document
---

## R setup

R 3.6.3.
requirements
Temps exécution

## Data

Data files "dataset_national.csv", "data_regional.csv", and "departements-region.csv" generated from the Python repo must be stored in a "Data/Input" folder.

## Sobriety period

The "Clean_codes/1-Sobriety" allows to recreate figures ... ("drift.pdf") and ...  ("change_point.pdf") by running the "graphs_sober.R" file.

## Models
The folder "Clean_codes/Models" contains the benchmark models and parallelize along the 48 half-hours. 

*  "agg_utils.R" regroups the aggregation of experts,
*  "gam_utils.R" regroups the generalized additive models, as well the the static and dynamic Kalman filters,
*  "vikings.R" contains the viking Kalman algorithm
*  "predict_offline_discriminate.R" allows to evaluate the 48 models parallelized along the 48 half-hours.

## State-of-the-art
The "Clean_codes/2-Benchmark" file regroups state-of-the-art algorithms. 

* Running "arima.R" computes traditional time-series methods (persistence and ARIMA)
* Running "benchmark_gam_kalman.R" computes GAM, Static Kalman, Dynamic Kalman, Viking, the agregation of experts. 

In both these files, one can set the variable HOLIDAYS to TRUE to keep holidays in the training and test periods.
Performance is then stored in the "Clean_codes/2-Benchmark/Results/benchmark_nat_perfs_holidays.RDS" file. 
One can set the variable HOLIDAYS to FALSE to exclude holidays from the training and test periods.
Performance is then stored in the "Clean_codes/2-Benchmark/Results/benchmark_nat_perfs.RDS" file.

## New estimator using Orange data
Models used in the benchmark with Orange data are stored in the "Clean_codes/3-Orange estimator" foder.
Since Orange data is private, one cannot run these codes.

* "0-nationalDataOrange.R" and "0-nationalDataOrange_agg.R" are usefull to agregate the Orange variables to the national scale.
* "nationalResidualsCompleteKalman.R" contains all the models using Orange data in benchmarks from tables ...

One can set the variable HOLIDAYS to TRUE to keep holidays in the training and test periods.
Performance is then stored in the "Clean_codes/3-Orange estimator/Results/nationalResidualsCompleteHolidays.RDS" file. 
One can set the variable HOLIDAYS to FALSE to exclude holidays from the training and test periods.
Performance is then stored in the "Clean_codes/3-Orange estimator/Results/nationalResidualsComplete.RDS" file.

Other techniques to integrate the Orange Data are investigated.