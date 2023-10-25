library(mgcv)
library(ranger)
library(tidyverse)
source("Clean_codes/Models/gam_utils.R")
source("Clean_codes/Models/viking_utils.R")

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

#subsets
begin_train = as.POSIXct(strptime("2019-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
end_train = as.POSIXct(strptime("2022-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
train <- which((Data_nat$date < end_train)&(Data_nat$date >= begin_train))
test <- which(Data_nat$date >=  end_train)

#Exclude bank holidays
test_norm = test
if(HOLIDAYS == FALSE)
{
holidays <- which(Data_nat[test,]$day_type_jf==1)
holidays_24h_window <- c(holidays-48, holidays, holidays+48)
outliers <- pmax(c(holidays_24h_window, which(Data_nat[test,]$period_summer!=0), which(Data_nat[test,]$period_christmas!=0)),1)
test_norm = test[-outliers]

train = which((Data_nat$date < end_train)&(Data_nat$day_type_jf==0))
}

### Random forests 
cov <-  c("temperature", "tod", "toy", "day_type_week", "day_type_jf", "temperature_smooth_950", "temperature_max_smooth_950", "period_hour_changed", "Load_d1", "Load_d7")
rf_formula  <- paste('Load~',paste(cov, collapse='+'))
rf <- ranger(rf_formula, data=Data_nat[train,], importance ='permutation')
rf.forecast <- predict(rf, data=Data_nat)$prediction

imp <- rf$variable.importance
nom <- names(imp)
o <- order(imp, decreasing=T)
plot(c(1:length(imp)), imp[o], type='h', ylim = c(0, max(imp) + max(imp)/5), xlab='', ylab='Importance (permutation)')
K <- length(imp)
text(tail(c(1:length(imp)), K), tail(imp[o]+max(imp/8), K), labels= tail(nom[o], K), pos=3, srt=90, adj=1)
points(c(1:length(imp)), imp[o], pch=20)

RMSE(Data_nat$Load, rf.forecast, test_norm)
MAPE(Data_nat$Load, rf.forecast, test_norm)

### Random forests + bootstrap
library(rangerts)
nb_trees <- 1000
mtry <- floor(sqrt(ncol(Data_nat)))
block_size <- 5*48

rf_mv <- rangerts::rangerts(rf_formula, data = Data_nat[train,],
                            num.trees = nb_trees,
                            mtry = mtry,
                            replace = T, # default = T too
                            seed = 1,
                            bootstrap.ts = "moving",
                            block.size = block_size)

rf_mv.forecast <- predict(rf_mv, data=Data_nat)$prediction

RMSE(Data_nat$Load, rf_mv.forecast, test_norm)
MAPE(Data_nat$Load, rf_mv.forecast, test_norm)

