#Dans le terminal, exécuter: R CMD INSTALL shaff_0.1.1.tar.gz
library(nnls)
library(shaff)
library(tidyverse)
set.seed(123)

Data_nat <- readRDS("Data/Input/Data_orange_national.RDS")

Data_nat$period_hour_changed%>%unique
Data_nat$day_type_jf%>%unique
Data_nat$day_type_week = as.factor(Data_nat$day_type_week)
Data_nat$period_hour_changed = as.factor(Data_nat$period_hour_changed)

begin_train_kalman = as.POSIXct(strptime("2021-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
end_train = as.POSIXct(strptime("2022-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
train <- which(Data_nat$date < end_train)

data_test_mobility = Data_nat[train,][c("Load", "toy", "day_type_week", "day_type_jf", "temperature_smooth_950",
                                        "Exc_rec_aggl", "Residents_aggl", "Tourists_aggl")]
data_test_mobility$day_type_week = as.numeric(data_test_mobility$day_type_week)
data_test_mobility = data_test_mobility %>% rename(Y = Load)
features = c( "toy", "day_type_week", "day_type_jf", "temperature_smooth_950",
"Exc_rec_aggl", "Residents_aggl", "Tourists_aggl")
values1 = shaff(data_test_mobility[sample(1:length(data_test_mobility$Y), 1000, replace=FALSE),])
values2 = shaff(data_test_mobility[sample(1:length(data_test_mobility$Y), 1000, replace=FALSE),])
values3 = shaff(data_test_mobility[sample(1:length(data_test_mobility$Y), 1000, replace=FALSE),])
shapley_values = data.frame(features, values1, values2, values3)

library(mgcv)
source("Clean_codes/Models/gam_utils.R")

eq_temp <- Load ~ s(temperature_smooth_950) 
gams_temp <- gam_discriminate(eq_temp, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam_temp = predict_gam_discriminate(gams_temp, Data_nat, Data_nat$tod)
data_test_mobility$Y =  Data_nat$Load[train]  -forecast_gam_temp[train]
values1 = shaff(data_test_mobility[sample(1:length(data_test_mobility$Y), 1000, replace=FALSE),])
values2 = shaff(data_test_mobility[sample(1:length(data_test_mobility$Y), 1000, replace=FALSE),])
values3 = shaff(data_test_mobility[sample(1:length(data_test_mobility$Y), 1000, replace=FALSE),])
shapley_values_temp = data.frame(features, values1, values2, values3)


eq_mob <- Load ~ 
  s(Exc_rec_aggl, bs = "bs", m = 1, k=3)+ 
  s(temperature_smooth_950)
gams_mob <- gam_discriminate(eq_mob, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam_mob = predict_gam_discriminate(gams_mob, Data_nat, Data_nat$tod)
data_test_mobility$Y =  Data_nat$Load[train]  -forecast_gam_mob[train]
values1 = shaff(data_test_mobility[sample(1:length(data_test_mobility$Y), 1000, replace=FALSE),])
values2 = shaff(data_test_mobility[sample(1:length(data_test_mobility$Y), 1000, replace=FALSE),])
values3 = shaff(data_test_mobility[sample(1:length(data_test_mobility$Y), 1000, replace=FALSE),])
shapley_values_mob = data.frame(features, values1, values2, values3)

print("Shapley values for  Load")
print(data.frame(features, mean = rowMeans(shapley_values[,-c(1)]), sd = apply(shapley_values[,-c(1)], 1, sd)))
print("Shapley values for  Load corrected from temperature")
print(data.frame(features, mean = rowMeans(shapley_values_temp[,-c(1)]), sd = apply(shapley_values_temp[,-c(1)], 1, sd)))
print("Shapley values for  Load corrected from mobility")
print(data.frame(features, mean = rowMeans(shapley_values_mob[,-c(1)]), sd = apply(shapley_values_mob[,-c(1)], 1, sd)))
