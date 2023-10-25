library(mgcv)
library(tidyverse)

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

#SARIMA
arima_discriminate <- function(data, discr, ncores=1) {
  model <- list()
  model$values <- unique(discr)
  model$arima <- parallel::mclapply(1:length(model$values), function(i) {
    time_serie = ts(data[which(discr == model$values[i])], frequency = 7)
    return(forecast::auto.arima(time_serie))
  },
  mc.cores=ncores)
  model
}

predict_arima_discriminate <- function(model, newdata, discr) {
  prediction <- rep(NA, length(newdata))
  for (val in unique(discr))
  {
    prediction[which(discr == val)] <-
      forecast::Arima(newdata[which(discr == val)],
                      model=model$arima[[which(model$values == val)]])$fitted
  }
  return(prediction)}

fit = arima_discriminate(Data_nat$Load[train], Data_nat$tod[train], ncores=20)
refit = predict_arima_discriminate(fit, Data_nat$Load, Data_nat$tod)

#Exclude bank holidays
train_rf = train
test_norm = test
if(HOLIDAYS == FALSE)
{
holidays <- which(Data_nat[test,]$day_type_jf==1)
holidays_24h_window <- c(holidays-48, holidays, holidays+48)
outliers <- pmax(c(holidays_24h_window, which(Data_nat[test,]$period_summer!=0), which(Data_nat[test,]$period_christmas!=0)),1)
test_norm = test[-outliers]
}

#Persistence
persistence = Data_nat$Load_d1

#Performances
RMSE(Data_nat$Load, refit, test_norm)
MAPE(Data_nat$Load, refit, test_norm)
RMSE(Data_nat$Load, persistence, test_norm)
MAPE(Data_nat$Load, persistence, test_norm)

#Estimators
estimators <- cbind(Data_nat$Load[test_norm], persistence[test_norm], refit[test_norm])
colnames(estimators) = c("Load", "Persistence", "ARIMA")

if(HOLIDAYS)
{
  saveRDS(estimators, "Results/estimatorARIMAHolidays.RDS")
}else{
  saveRDS(estimators, "Results/estimatorARIMA.RDS")
}
