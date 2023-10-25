library(mgcv)
library(tidyverse)
source("Clean_codes/Models/agg_utils.R")
source("Clean_codes/Models/gam_utils.R")
source("Clean_codes/Models/viking_utils.R")

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
  Data_nat <- Data_nat[-outliers,]
}

#subsets
begin_train_kalman = as.POSIXct(strptime("2021-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
end_train = as.POSIXct(strptime("2022-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
train <- which(Data_nat$date < end_train)
train_kalman <- which((Data_nat$date < end_train)&(Data_nat$date > begin_train_kalman))
test <- which(Data_nat$date >=  end_train)
performances_df = initialize_perf()

#Obst 2021 gam equation
eq_gam <- Load ~ s(Date, k=3) + day_type_week:period_hour_changed + s(toy, k=20, bs='cc') + s(Date, temperature, k=c(3,5)) +
  s(temperature_smooth_950, k=5) + s(temperature_smooth_990, k=5) + s(temperature_min_smooth_990, temperature_max_smooth_990) +
  Load_d1:day_type_week + Load_d7 
gams <- gam_discriminate(eq_gam, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam = predict_gam_discriminate(gams, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_Obst_2021", Data_nat$Load, forecast_gam, test))

#Kalman static
X <- prep_data_gam(gams, Data_nat, Data_nat$tod, train_kalman)
d <- dim(X)[2]

ssm <- static_discriminate(X, Data_nat$Load, discr=Data_nat$tod, ncores=20)
forecast_static <- predict_static_discriminate(ssm, discr=Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_Kalman_static", Data_nat$Load, forecast_static, test))

#Kalman dynamic
ssm_dyn <- dynamic_discriminate(ssm, X[train_kalman,], Data_nat$Load[train_kalman], discr=Data_nat$tod[train_kalman], 
                                q_list = 2^(-30:0), p1 = 1, ncores=10)
forecast_dynamic <- predict_viking_discriminate(ssm_dyn,  X, Data_nat$Load, discr=Data_nat$tod, model_type="dynamic")
performances_df = rbind(performances_df, performances("GAM_Kalman_dynamic", Data_nat$Load, forecast_dynamic, test))

#Kalman Viking
ssm_vik <- viking_discriminate(ssm_dyn, X, Data_nat$Load, discr=Data_nat$tod, ncores=10, seed=c(1:48))
forecast_viking <- predict_viking_discriminate(ssm_vik,X, Data_nat$Load, discr=Data_nat$tod, model_type="viking")
performances_df = rbind(performances_df, performances("GAM_Kalman_viking", Data_nat$Load, forecast_viking, test))

#Aggregation
experts <- cbind(forecast_gam, forecast_static, forecast_dynamic, forecast_viking)
#experts <- cbind(experts, experts[,1]-1000, experts[,2]-1000, experts[,3]-1000, experts[,4]-1000)

agg <- agg_discriminate(Data_nat$Load[test], experts[test,], discr=Data_nat[test,]$tod)
forecast_agg <- predict_agg_discriminate(agg, discr=Data_nat[test,]$tod) 
performances_df = rbind(performances_df, performances("Aggregation", Data_nat$Load[test], forecast_agg, 1:length(test)))

#Exporting results
if(HOLIDAYS)
{
  saveRDS(performances_df, "Clean_codes/2-Benchmark/Results/benchmark_nat_perfs_holidays.RDS")
}else{
saveRDS(performances_df, "Clean_codes/2-Benchmark/Results/benchmark_nat_perfs.RDS")
}

#Estimators
estimators <- cbind(Data_nat$Load, forecast_gam, forecast_static, forecast_dynamic, forecast_viking)
estimators = cbind(estimators[test,], forecast_agg)
colnames(estimators) = c("Load", "GAM", "Kalman_static", "Kalman_dynamic", "Viking", "Aggregation")

if(HOLIDAYS)
{
  saveRDS(estimators, "Results/estimatorNationalHolidays.RDS")
}else{
  saveRDS(estimators, "Results/estimatorNational.RDS")
}

