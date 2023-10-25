library(gbm)
library(tidyverse)
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
begin_train = as.POSIXct(strptime("2019-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
end_train = as.POSIXct(strptime("2022-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
train <- which((Data_nat$date < end_train)&(Data_nat$date >= begin_train))
test <- which(Data_nat$date >=  end_train)
performances_df = initialize_perf()



### gb model

gb_discriminate <- function(data, formula, discr, ncores=1) {
  model <- list()
  model$values <- unique(discr)
  model$gb <- parallel::mclapply(1:length(model$values), function(i) {
    gbm(formula, 
        data = data[which(discr == model$values[i]),], 
        distribution = "gaussian", n.trees = 1000, shrinkage = 0.005, n.cores=1) 
  },
  mc.cores=ncores)
  model
}


predict_gb_discriminate <- function(model, newdata, discr) {
  prediction <- rep(NA, length(newdata))
  for (val in unique(discr))
  {
    prediction[which(discr == val)] <-
      predict(model$gb[[which(model$values == val)]], newdata[which(discr == val),], n.trees = 1000)
  }
  return(prediction)}

if(HOLIDAYS)
{
  formula = as.formula("Load~temperature+toy+day_type_week+day_type_jf+
temperature_smooth_950 + temperature_max_smooth_950 + temperature_min_smooth_950 + 
                     period_hour_changed+ Load_d1 + Load_d7")
}else{
  formula = as.formula("Load~temperature+toy+day_type_week+
temperature_smooth_950 + temperature_max_smooth_950 + temperature_min_smooth_950 + 
                     period_hour_changed+ Load_d1 + Load_d7")
}
fit = gb_discriminate(Data_nat[train,], formula, Data_nat$tod[train], ncores = 20)
gbm_forecast_elec = predict_gb_discriminate(fit, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GB_elec", Data_nat$Load, gbm_forecast_elec, test))
performances_df

#Loading Orange data
Data_nat$residuals = Data_nat$Load - gbm_forecast_elec 

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


#Gradient boosting
gb_discriminate_2 <- function(data, formula, discr, ncores=1) {
  model <- list()
  model$values <- unique(discr)
  model$gb <- parallel::mclapply(1:length(model$values), function(i) {
    gbm(formula, 
        data = data[which(discr == model$values[i]),], 
        distribution = "gaussian", n.trees = 1000, shrinkage = 0.005, n.cores=1) 
  },
  mc.cores=ncores)
  model
}
cov <-  c("temperature", "tod", "toy", "day_type_week", "day_type_jf", "temperature_smooth_950", "temperature_max_smooth_950", "period_hour_changed", "Load_d1", "Load_d7",
            "Exc_rec_aggl", "Residents_aggl")
formula  <- as.formula(paste('Load~',paste(cov, collapse='+')))
fit = gb_discriminate_2(Data_Orange[train_Orange,], formula, Data_Orange$tod[train_Orange], ncores = 20)
gbm_forecast = predict_gb_discriminate(fit, Data_Orange, Data_Orange$tod)
performances_df = rbind(performances_df, performances("GB", Data_Orange$Load, 
                                                      #gbm_forecast_elec[which(Data_nat$date %in% Data_Orange$date)]+
                                                      gbm_forecast, 
                                                      test_Orange))
performances_df

#Estimators
estimators <- cbind(Data_Orange$residuals, gbm_forecast_elec[which(Data_nat$date %in% Data_Orange$date)], 
                    gbm_forecast_elec[which(Data_nat$date %in% Data_Orange$date)]+gbm_forecast)
colnames(estimators) = c("Residuals", "GB_elec", "GB")

if(HOLIDAYS)
{
  saveRDS(estimators, "Results/estimatorResidualsGBHolidays.RDS")
}else{
  saveRDS(estimators, "Results/estimatorResidualsGB.RDS")
}