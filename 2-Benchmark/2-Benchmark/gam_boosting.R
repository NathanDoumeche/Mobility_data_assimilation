library(mboost)
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

### gamboost model
if(HOLIDAYS)
{gb_formula  <- as.formula("Load~
                          bbs(temperature, knots=20, df=4)+
                          bbs(toy, knots=20, df=4) +
                          bols(day_type_week) +
                          bols(day_type_jf) +
                          bbs(temperature_smooth_950, knots=20, df=4)+
                          bbs(temperature_max_smooth_950, knots=20, df=4)+
                          bols(period_hour_changed) +
                          bbs(Load_d1,  knots=20, df=4) +
                          bbs(Load_d7,  knots=20, df=4)")
}else{
  gb_formula  <- as.formula("Load~
                          bbs(temperature, knots=20, df=4)+
                          bbs(toy, knots=20, df=4) +
                          bols(day_type_week) +
                          bbs(temperature_smooth_950, knots=20, df=4)+
                          bbs(temperature_max_smooth_950, knots=20, df=4)+
                          bols(period_hour_changed) +
                          bbs(Load_d1,  knots=20, df=4) +
                          bbs(Load_d7,  knots=20, df=4)")
}

gb = gamboost(gb_formula, data = Data_nat[train,],
              control = boost_control(nu = 0.15, mstop = 50))


### 5-fold cross-validation
cv5f <- cv(model.weights(gb), type = "kfold", B=5)
cvm <- cvrisk(gb, folds = cv5f, papply = mclapply)
plot(cvm)
mstop=50
abline(v=mstop)

gb_cv =  gamboost(gb_formula, data = Data_nat[train,], 
                  control = boost_control(nu = 0.15, mstop = mstop))
gb_forecast = predict(gb_cv, Data_nat[test_norm,])

RMSE(Data_nat$Load[test_norm], gb_forecast)
MAPE(Data_nat$Load[test_norm], gb_forecast)


