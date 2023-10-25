library(mgcv)
library(tidyverse)
source("Clean_codes/Models/agg_utils.R")
source("Clean_codes/Models/gam_utils.R")
source("Clean_codes/Models/viking_utils.R")

HOLIDAYS = TRUE

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
if(HOLIDAYS == FALSE)
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
test <- which(Data_nat$date >=  end_train)
performances_df = initialize_perf()

#Obst 2021 gam equation
eq_gam_elec <- Load ~ s(Date, k=3) + day_type_week:period_hour_changed + s(toy, k=20, bs='cc') + s(Date, temperature, k=3) +
  s(temperature_smooth_950, k=5) + s(temperature_smooth_990, k=5) + s(temperature_min_smooth_990, temperature_max_smooth_990) +
  Load_d1:day_type_week + Load_d7 

gams_elec <- gam_discriminate(eq_gam_elec, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam_elec = predict_gam_discriminate(gams_elec, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_Obst_2021", Data_nat$Load, forecast_gam_elec, test))

#Kalman static
train_kalman_elec <- which((Data_nat$date < end_train)&(Data_nat$date > begin_train_kalman))
X_elec <- prep_data_gam(gams_elec, Data_nat, Data_nat$tod, train_kalman_elec)
ssm_elec <- static_discriminate(X_elec, Data_nat$Load, discr=Data_nat$tod, ncores=20)
forecast_static_elec <- predict_static_discriminate(ssm_elec, discr=Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_Kalman_static_elec", Data_nat$Load, forecast_static_elec, test))

#Kalman dynamic
ssm_dyn_elec <- dynamic_discriminate(ssm_elec, X_elec[train_kalman_elec,], Data_nat$Load[train_kalman_elec], discr=Data_nat$tod[train_kalman_elec], 
                                q_list = 2^(-30:0), p1 = 1, ncores=10)
forecast_dynamic_elec <- predict_viking_discriminate(ssm_dyn_elec,  X_elec, Data_nat$Load, discr=Data_nat$tod, model_type="dynamic")
performances_df = rbind(performances_df, performances("GAM_Kalman_dynamic_elec", Data_nat$Load, forecast_dynamic_elec, test))

#Kalman Viking
ssm_vik_elec <- viking_discriminate(ssm_dyn_elec, X_elec, Data_nat$Load, discr=Data_nat$tod, ncores=10, seed=c(1:48))
forecast_viking_elec <- predict_viking_discriminate(ssm_vik_elec, X_elec, Data_nat$Load, discr=Data_nat$tod, model_type="viking")
performances_df = rbind(performances_df, performances("GAM_Kalman_viking_elec", Data_nat$Load, forecast_viking_elec, test))


#Loading Orange data
Data_nat$residuals = Data_nat$Load - forecast_gam_elec

Data_Orange <- readRDS("Data/Input/Data_orange_national.RDS")
Data_Orange$period_hour_changed%>%unique
Data_Orange$day_type_jf%>%unique
Data_Orange$day_type_week = as.factor(Data_Orange$day_type_week)
Data_Orange$period_hour_changed = as.factor(Data_Orange$period_hour_changed)

#Exclude bank holidays for Orange Data
if(HOLIDAYS==FALSE)
  {
  holidays <- which(Data_Orange$day_type_jf==1)
holidays_24h_window <- c(holidays-48, holidays, holidays+48)
outliers <- pmax(c(holidays_24h_window, which(Data_Orange$period_summer!=0), which(Data_Orange$period_christmas!=0)),1)
Data_Orange <- Data_Orange[-outliers,]
}

train_Orange <- which(Data_Orange$date < end_train)
test_Orange <- which(Data_Orange$date >=  end_train)
Data_Orange$residuals = Data_nat$residuals[which(Data_nat$date %in% Data_Orange$date)]

#GAM residual
eq_gam_Orange <- residuals ~ #s(Date, k=3) + 
  day_type_week + 
  #s(toy, k=20, bs='cc') + 
  #s(Date, temperature, k=3) +
  #s(temperature_smooth_950, k=5) + 
  #s(temperature_smooth_990, k=5) + 
  #s(temperature_min_smooth_990, temperature_max_smooth_990) +
  #Load_d1 + 
  #Load_d7 +
  #s(Tourists_aggl) + 
  s(Residents_aggl) +
  s(Exc_rec_aggl) #+ s(Exc_aggl)
common_terms = c(2)
gams <- gam_discriminate(eq_gam_Orange, Data_Orange[train_Orange,], Data_Orange$tod[train_Orange], ncores=20)#, select=TRUE, gamma=0.5)
forecast_gam = predict_gam_discriminate(gams, Data_Orange, Data_Orange$tod)
performances_df = rbind(performances_df, performances("ResidualGAM", Data_Orange$Load, forecast_gam_elec[which(Data_nat$date %in% Data_Orange$date)]+forecast_gam, test_Orange))
performances_df

#Evaluate p-value
sumgamLin =  summary(gams$gams[[1]])$p.table
sumgamNonLin = summary(gams$gams[[1]])$s.table
for(i in 2:48)
{
  sumgamLin = sumgamLin + summary(gams$gams[[i]])$p.table
  sumgamNonLin = sumgamNonLin + summary(gams$gams[[i]])$s.table
}
sumgamLin/48
sumgamNonLin/48
#La moyenne reflète mal le phénomène car on s'en fout que l'incertitude soit grande quand l'effet est négligeable.

#Kalman static 
train_kalman <- which((Data_Orange$date < end_train)&(Data_Orange$date > begin_train_kalman))
X <- prep_data_gam_residuals(gams_elec, gams, Data_Orange, Data_Orange$tod, train_kalman, common_terms = common_terms)
ssm <- static_discriminate(X, Data_Orange$Load, discr=Data_Orange$tod, ncores=20)
forecast_static <- predict_static_discriminate(ssm, discr=Data_Orange$tod)
performances_df = rbind(performances_df, performances("GAM_Kalman_static", Data_Orange$Load, forecast_static, test_Orange))
performances_df

#Kalman dynamic
ssm_dyn <- dynamic_discriminate(ssm, X[train_kalman,], Data_Orange$Load[train_kalman], discr=Data_Orange$tod[train_kalman], 
                                q_list = 2^(-30:0), p1 = 1, ncores=10)
forecast_dynamic <- predict_viking_discriminate(ssm_dyn,  X, Data_Orange$Load, discr=Data_Orange$tod, model_type="dynamic")
performances_df = rbind(performances_df, performances("GAM_Kalman_dynamic",   Data_Orange$Load, forecast_dynamic, test_Orange))
performances_df

#Kalman Viking
ssm_vik <- viking_discriminate(ssm_dyn, X, Data_Orange$Load, discr=Data_Orange$tod, 
                               ncores=20, seed=c(1:48))
forecast_viking <- predict_viking_discriminate (ssm_vik,X, Data_Orange$Load, discr=Data_Orange$tod, model_type="viking")
performances_df = rbind(performances_df, performances("GAM_Kalman_viking", Data_Orange$Load, forecast_viking, test_Orange))
performances_df

#Aggregation elec
test_elec = which(Data_nat$date %in% Data_Orange$date)
experts_elec = cbind(forecast_gam_elec[test_elec], forecast_static_elec[test_elec], forecast_dynamic_elec[test_elec],
                forecast_viking_elec[test_elec])
agg_residuals_elec <- agg_discriminate(Data_Orange$Load[test_Orange], experts_elec[test_Orange,], discr=Data_Orange[test_Orange,]$tod)
forecast_agg_residuals <- predict_agg_discriminate(agg_residuals_elec, discr=Data_Orange[test_Orange,]$tod) 
performances_df = rbind(performances_df, performances("Aggregation elec", Data_Orange$Load[test_Orange], forecast_agg_residuals, 1:length(test_Orange)))


#Aggregation 
experts <- cbind(forecast_gam_elec[test_elec]+ forecast_gam, forecast_static, forecast_dynamic, forecast_viking)
experts = cbind(experts, forecast_gam_elec[test_elec], forecast_static_elec[test_elec], forecast_dynamic_elec[test_elec],
                 forecast_viking_elec[test_elec])
agg_residuals <- agg_discriminate(Data_Orange$Load[test_Orange], experts[test_Orange,], discr=Data_Orange[test_Orange,]$tod)
forecast_agg_residuals <- predict_agg_discriminate(agg_residuals, discr=Data_Orange[test_Orange,]$tod) 
performances_df = rbind(performances_df, performances("Aggregation", Data_Orange$Load[test_Orange], forecast_agg_residuals, 1:length(test_Orange)))

if(HOLIDAYS)
{
  saveRDS(performances_df, "Results/nationalResidualsCompleteHolidays.RDS")
}else{
  saveRDS(performances_df, "Results/nationalResidualsComplete.RDS")
}

#Estimators
estimators <- cbind(Data_Orange$Load, forecast_gam_elec[which(Data_nat$date %in% Data_Orange$date)]+ forecast_gam, forecast_static, forecast_dynamic, forecast_viking )
estimators = cbind(estimators[test_Orange,], forecast_agg_residuals)
colnames(estimators)[c(1,2, 6)] = c("Load", "GAM", "Aggregation")

if(HOLIDAYS)
{
  saveRDS(estimators, "Results/estimatorResidualsCompleteHolidays.RDS")
}else{
  saveRDS(estimators, "Results/estimatorResidualsComplete.RDS")
}
