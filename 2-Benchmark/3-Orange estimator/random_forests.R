library(mgcv)
library(ranger)
library(tidyverse)
library(rangerts)
source("Clean_codes/Models/gam_utils.R")

set.seed(123)
HOLIDAYS = FALSE

initialize_perf = function(){
  columns_perf = c("Model","RMSE","MAPE") 
  performances_df = data.frame(matrix(nrow = 0, ncol = length(columns_perf))) 
  colnames(performances_df) = columns_perf
  return(performances_df)}

RMSE = function(ground_truth, prediction, test){
  return(sqrt(mean(((ground_truth-prediction)[test])^2)))}

MAPE = function(ground_truth, prediction, test){
  return(mean(abs((ground_truth-prediction)[test]) / ground_truth[test]))}

performances = function(model_name, ground_truth, prediction, test){
  return(data.frame(Model = model_name,
                    RMSE = RMSE(ground_truth, prediction, test),
                    MAPE = MAPE(ground_truth, prediction, test)))}

Data_nat <- read_csv("Data/Input/dataset_national.csv")

Data_nat$period_hour_changed%>%unique
Data_nat$day_type_jf%>%unique
Data_nat$day_type_week = as.factor(Data_nat$day_type_week)
Data_nat$period_hour_changed = as.factor(Data_nat$period_hour_changed)

#Exclude bank holidays
if(HOLIDAYS==FALSE)
{
holidays <- which(Data_nat$day_type_jf==1)
holidays_24h_window <- c(holidays-48, holidays, holidays+48)
outliers <- pmax(c(holidays_24h_window, which(Data_nat$period_summer!=0), which(Data_nat$period_christmas!=0)),1)
Data_nat <- Data_nat[-outliers,]}

#subsets
begin_train_kalman = as.POSIXct(strptime("2021-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
end_train = as.POSIXct(strptime("2022-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
train <- which(Data_nat$date < end_train)
test <- which(Data_nat$date >=  end_train)
performances_df = initialize_perf()

#RF_elec
cov <-  c("temperature", "tod", "toy", "day_type_week", "day_type_jf", "temperature_smooth_950", "temperature_max_smooth_950", "period_hour_changed", "Load_d1", "Load_d7")
rf_formula  <- paste('Load~',paste(cov, collapse='+'))
rf_elec <- ranger(rf_formula, data=Data_nat[train,], importance ='permutation')
rf_elec.forecast <- predict(rf_elec, data=Data_nat)$prediction

#RF_mv_elec
set.seed(123)
nb_trees <- 1000
mtry <- floor(sqrt(ncol(Data_nat)))
block_size <- 5*48
cov <-  c("temperature", "tod", "toy", "day_type_week", "day_type_jf", "temperature_smooth_950", "temperature_max_smooth_950", "period_hour_changed", "Load_d1", "Load_d7")
rf_formula_elec  <- paste('Load~',paste(cov, collapse='+'))

rf_mv_elec <- rangerts::rangerts(rf_formula_elec, data = Data_nat[train,],
                            num.trees = nb_trees,
                            mtry = mtry,
                            replace = T, # default = T too
                            seed = 1,
                            bootstrap.ts = "moving",
                            block.size = block_size)

rf_mv_elec.forecast <- predict(rf_mv_elec, data=Data_nat)$prediction
performances_df = rbind(performances_df, performances("FR_MV_elec", Data_nat$Load, rf_mv_elec.forecast, test))
performances_df

#Loading Orange data
Data_nat$residuals = Data_nat$Load - rf_mv_elec.forecast 

Data_Orange <- readRDS("Data/Input/Data_orange_national.RDS")
Data_Orange$period_hour_changed%>%unique
Data_Orange$day_type_jf%>%unique
Data_Orange$day_type_week = as.factor(Data_Orange$day_type_week)
Data_Orange$period_hour_changed = as.factor(Data_Orange$period_hour_changed)

#Exclude bank holidays for Orange Data
if(HOLIDAYS == FALSE)
{
holidays <- which(Data_Orange$day_type_jf==1)
holidays_24h_window <- c(holidays-48, holidays, holidays+48)
outliers <- pmax(c(holidays_24h_window, which(Data_Orange$period_summer!=0), which(Data_Orange$period_christmas!=0)),1)
Data_Orange <- Data_Orange[-outliers,]}

train_Orange <- which(Data_Orange$date < end_train)
test_Orange <- which(Data_Orange$date >=  end_train)
Data_Orange$residuals = Data_nat$residuals[which(Data_nat$date %in% Data_Orange$date)]


### Random forests 
cov <-  c("temperature", "tod", "toy", "day_type_week", "day_type_jf", "temperature_smooth_950", "temperature_max_smooth_950", "period_hour_changed", "Load_d1", "Load_d7",
          "Exc_rec_aggl", "Residents_aggl")
rf_formula  <- paste('residuals~',paste(cov, collapse='+'))
rf <- ranger(rf_formula, data=Data_Orange[train_Orange,], importance ='permutation')
rf.forecast <- predict(rf, data=Data_Orange)$prediction

imp <- rf$variable.importance
nom <- names(imp)
o <- order(imp, decreasing=T)
plot(c(1:length(imp)), imp[o], type='h', ylim = c(0, max(imp) + max(imp)/5), xlab='', ylab='Importance (permutation)')
K <- length(imp)
text(tail(c(1:length(imp)), K), tail(imp[o]+max(imp/8), K), labels= tail(nom[o], K), pos=3, srt=90, adj=1)
points(c(1:length(imp)), imp[o], pch=20)

RMSE(Data_Orange$residuals, rf.forecast, test_Orange)
MAPE(Data_Orange$Load, rf_mv_elec.forecast[which(Data_nat$date %in% Data_Orange$date)]+  rf.forecast, test_Orange)

### Random forests + bootstrap
nb_trees <- 1000
mtry <- floor(sqrt(ncol(Data_Orange)))
block_size <- 5*48

rf_mv <- rangerts::rangerts(rf_formula, data = Data_Orange[train_Orange,],
                            num.trees = nb_trees,
                            mtry = mtry,
                            replace = T, # default = T too
                            seed = 1,
                            bootstrap.ts = "moving",
                            block.size = block_size)

rf_mv.forecast <- predict(rf_mv, data=Data_Orange)$prediction

RMSE(Data_Orange$residuals, rf_mv.forecast, test_Orange)
MAPE(Data_Orange$Load, rf_mv_elec.forecast[which(Data_nat$date %in% Data_Orange$date)]+  rf_mv.forecast, test_Orange)


#Estimators
estimators <- cbind(Data_Orange$Load, 
                    rf_elec.forecast[which(Data_nat$date %in% Data_Orange$date)],
                    rf_mv_elec.forecast[which(Data_nat$date %in% Data_Orange$date)],
                    rf_mv_elec.forecast[which(Data_nat$date %in% Data_Orange$date)]+  rf.forecast,
                    rf_mv_elec.forecast[which(Data_nat$date %in% Data_Orange$date)]+  rf_mv.forecast)
estimators = estimators[test_Orange,]
colnames(estimators) = c("Load", "RF_elec", "RF_bootstrap_elec", "RF", "RF_bootstrap")

if(HOLIDAYS)
{
  saveRDS(estimators, "Results/estimatorResidualsRFHolidays.RDS")
}else{
  saveRDS(estimators, "Results/estimatorResidualsRF.RDS")
}
