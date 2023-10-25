library(mgcv)
library(tidyverse)
source("Clean_codes/Models/agg_utils.R")
source("Clean_codes/Models/gam_utils.R")
source("Clean_codes/Models/viking_utils.R")

SOBRIETY = TRUE

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

moving_average = function(time_series, period){
  filter_period = rep(1/period, period)
  return(stats::filter(time_series, filter_period, method="convolution", sides=1))}

Data_nat <- readRDS("Data/Input/Data_orange_national.RDS")

Data_nat$period_hour_changed%>%unique
Data_nat$day_type_jf%>%unique
Data_nat$day_type_week = as.factor(Data_nat$day_type_week)
Data_nat$period_hour_changed = as.factor(Data_nat$period_hour_changed)


#subsets
if(SOBRIETY)
{
  begin_train_kalman = as.POSIXct(strptime("2021-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
  end_train = as.POSIXct(strptime("2022-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
  train <- which(Data_nat$date < end_train)
  train_kalman <- which((Data_nat$date < end_train)&(Data_nat$date > begin_train_kalman))
  test <- which(Data_nat$date >=  end_train)
}else{
  begin_train_kalman = as.POSIXct(strptime("2021-03-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
  end_train = as.POSIXct(strptime("2022-03-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
  end_test =  as.POSIXct(strptime("2022-09-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")
  train <- which(Data_nat$date < end_train)
  train_kalman <- which((Data_nat$date < end_train)&(Data_nat$date > begin_train_kalman))
  test <- which((Data_nat$date >=  end_train)& Data_nat$date <= end_test)
}
performances_df = initialize_perf()

persistence = Data_nat$Load_d1
performances_df = rbind(performances_df, performances("Persistence", Data_nat$Load, persistence, test))

#Temperature
eq_temp <- Load ~ s(temperature_smooth_950) 
gams_temp <- gam_discriminate(eq_temp, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam_temp = predict_gam_discriminate(gams_temp, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_Temp", Data_nat$Load, forecast_gam_temp, test))
performances_df

#Visualisation
#2d histogram
hh = 20
half_hour = which((Data_nat$tod==hh)&(Data_nat$date <=end_train))
Data_plot = Data_nat[half_hour,]
Data_plot$y = Data_nat$Load[half_hour]-forecast_gam_temp[half_hour]

pdf(file="Clean_codes/4-Explainability/histogram.pdf")
ggplot(Data_plot, aes(x=Exc_rec_aggl, y=y) ) +
  geom_bin2d(bins = 17) +
  scale_fill_continuous(type = "viridis") +
  theme_bw() +
  ylab("Load - effect of temperature")
dev.off()

#Specific day
pdf(file="Clean_codes/4-Explainability/wednesday_regular.pdf")
half_hour = which((Data_nat$tod==hh)&(Data_nat$date <=end_train)&(Data_nat$day_type_week ==3))
Data_plot = Data_nat[half_hour,]
Data_plot$Exc_rec_aggl = (Data_nat$Exc_rec_aggl[half_hour]-mean(Data_nat$Exc_rec_aggl[half_hour]))/sd(Data_nat$Exc_rec_aggl[half_hour])
Data_plot$y = (Data_nat$Load[half_hour]-forecast_gam_temp[half_hour])/sd(Data_nat$Load[half_hour]-forecast_gam_temp[half_hour])
ggplot(Data_plot, aes(x=Exc_rec_aggl, y=y) ) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", colour="white")+
  ylab("Load - effect of temperature")+
  xlab("Work index")
dev.off()

pdf(file="Clean_codes/4-Explainability/wednesday_all_days.pdf")
half_hour = which((Data_nat$tod==hh)&(Data_nat$day_type_week ==3))
Data_plot = Data_nat[half_hour,]
Data_plot$Exc_rec_aggl = (Data_nat$Exc_rec_aggl[half_hour]-mean(Data_nat$Exc_rec_aggl[half_hour]))/sd(Data_nat$Exc_rec_aggl[half_hour])
Data_plot$y = (Data_nat$Load[half_hour]-forecast_gam_temp[half_hour])/sd(Data_nat$Load[half_hour]-forecast_gam_temp[half_hour])
ggplot(Data_plot, aes(x=Exc_rec_aggl, y=y) ) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", colour="white")+
  ylab("Load - effect of temperature")
dev.off()


#Sobriety dependency
half_hour = which(Data_nat$tod==hh)
pdf(file="Clean_codes/4-Explainability/sobriety_dependency.pdf")
plot(Data_nat$Exc_rec_aggl[half_hour], 
     Data_nat$Load[half_hour]-forecast_gam_temp[half_hour], 
     xlab = "Remote working indicator", ylab="Load - effect of temperature",
     col=1+as.integer(Data_nat$date[half_hour] >=end_train), 
     pch=4)
legend("topleft",   c("Normal", "Sobriety"), 
       col=1:2, cex=1, pch=4)
dev.off() 


#Year drift
half_hour = which((Data_nat$tod==hh)&(Data_nat$date <=end_train))
pdf(file="Clean_codes/4-Explainability/year_dependency.pdf")
basel_col <- yarrr::piratepal(palette="basel")
col <- adjustcolor(basel_col[Data_nat$year[half_hour]%>%as.numeric - 2018], alpha.f = 1)
col_tr <- adjustcolor(basel_col[Data_nat$year[half_hour]%>%as.numeric - 2018], alpha.f = 0.5)
symb <- Data_nat$year[half_hour]%>%as.numeric-2018+14

plot(Data_nat$Exc_rec_aggl[half_hour], Data_nat$Load[half_hour]-forecast_gam_temp[half_hour], 
     xlab = "Work index", ylab="Load - effect of temperature",
     col=col_tr, pch=symb)
legend("topleft",   c("2019", "2020", "2021", "2022"), 
       col=unique(col), cex=1, pch=unique(symb))
dev.off()


#Day dependency
half_hour = which((Data_nat$tod==hh)&(Data_nat$date <=end_train))
pdf(file="Clean_codes/4-Explainability/day_dependency.pdf")

basel_col <- yarrr::piratepal(palette="basel")
col <- adjustcolor(basel_col[Data_nat$day_type_week[half_hour]%>%as.numeric], alpha.f = 1)
col_tr <- adjustcolor(basel_col[Data_nat$day_type_week[half_hour]%>%as.numeric], alpha.f = 0.5)
symb <- Data_nat$day_type_week[half_hour]%>%as.numeric+14

plot(Data_nat$Exc_rec_aggl[half_hour], Data_nat$Load[half_hour]-forecast_gam_temp[half_hour], 
     xlab = "Work index", ylab="Load - effect of temperature",
     col=col_tr, pch=symb)
legend("topleft",   c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"), 
       col=col[1:length(Data_nat$day_type_week[half_hour])], cex=1, pch=unique(symb))
dev.off()




#Holiday dependency
pdf(file="Clean_codes/4-Explainability/holiday_dependency.pdf")
col <- adjustcolor(c("black", "red"), alpha.f = 1)
col_tr <- adjustcolor(c("black", "red"), alpha.f = 0.5)
symb <- unique(1+as.integer(Data_nat$day_type_jf))+17

plot(Data_nat$Exc_rec_aggl[half_hour], Data_nat$Load[half_hour]-forecast_gam_temp[half_hour], 
       xlab = "Work index", ylab="Load - effect of temperature",
       col=col_tr[1+as.integer(Data_nat$day_type_jf[half_hour])], 
       pch=symb[1+as.integer(Data_nat$day_type_jf[half_hour])])
legend("topleft",   c("Regular day", "Holiday"), col=col, cex=1, pch=unique(symb))
dev.off()


#Temp + time
eq_time <- Load ~ day_type_week:period_hour_changed + s(toy, k=20, bs='cc') + day_type_jf + s(temperature_smooth_950) 
gams <- gam_discriminate(eq_time, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam = predict_gam_discriminate(gams, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_TempTime", Data_nat$Load, forecast_gam, test))

#Temp + Mobility
eq_mob <- Load ~ 
  s(Exc_rec_aggl, bs = "bs", m = 1, k=3)+ 
  s(temperature_smooth_950)
gams_mob <- gam_discriminate(eq_mob, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam_mob = predict_gam_discriminate(gams_mob, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_TempMob", Data_nat$Load, forecast_gam_mob, test))

#Evaluate p-value
performances_df
sumgamLin =  summary(gams_mob$gams[[1]])$p.table
sumgamNonLin = summary(gams_mob$gams[[1]])$s.table
for(i in 2:48)
{
  sumgamLin = sumgamLin + summary(gams_mob$gams[[i]])$p.table
  sumgamNonLin = sumgamNonLin + summary(gams_mob$gams[[i]])$s.table
}
sumgamLin/48
sumgamNonLin/48

#Temp + Mobility + Time
eq_time <- Load ~  day_type_week:period_hour_changed + s(toy, k=20, bs='cc') + day_type_jf+
    s(temperature_smooth_950) + s(Exc_rec_aggl, bs = "bs", m = 1, k=3) 
gams <- gam_discriminate(eq_time, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam = predict_gam_discriminate(gams, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_TempTimeMob", Data_nat$Load, forecast_gam, test))

#Time + Loads
eq_gam_elec <- Load ~  day_type_week:period_hour_changed + s(toy, k=20, bs='cc') +day_type_jf+
  s(temperature) + Load_d1:day_type_week + Load_d7 
gams_elec <- gam_discriminate(eq_gam_elec, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam_elec = predict_gam_discriminate(gams_elec, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_Lags", Data_nat$Load, forecast_gam_elec, test))

# Mobility + Loads
eq_gam_mob <- Load ~  s(Exc_rec_aggl, bs = "bs", m = 1, k=3) + s(temperature) + 
  Load_d1:day_type_week + Load_d7   
gams <- gam_discriminate(eq_gam_mob, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam = predict_gam_discriminate(gams, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_LagsMob", Data_nat$Load, forecast_gam, test))

# All
eq_gam_mob <- Load ~  day_type_week:period_hour_changed + s(toy, k=20, bs='cc') +day_type_jf+
  s(Exc_rec_aggl, bs = "bs", m = 1, k=3) + s(temperature) + 
  Load_d1:day_type_week + Load_d7 
gams <- gam_discriminate(eq_gam_mob, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam = predict_gam_discriminate(gams, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_All", Data_nat$Load, forecast_gam, test))


library(mgcViz)
hh <- 20
half_hour = which(Data_nat$tod==hh)
plot(gams$gams[[hh+1]])
gams$gams[[hh+1]]%>%summary

pdf(file="Clean_codes/4-Explainability/gam_work_indicator_effect.pdf")
b <- gams_mob$gams[[hh+1]]
b <- getViz(b)
o <- plot( sm(b, 1))
o + l_fitLine(colour = "red") + l_rug(mapping = aes(x=x, y=y), alpha = 0.1) +
  l_ciLine(mul = 5, colour = "salmon", linetype = 2) + 
  l_points(shape = 19, size = 0.5, alpha = 0.8) + theme_classic() + labs(y = "Work effect", x = "Work index")
dev.off()

pdf(file="Clean_codes/4-Explainability/gam_temp_effect.pdf")
b <- gams_mob$gams[[hh+1]]
b <- getViz(b)
o <- plot( sm(b, 2))
o + l_fitLine(colour = "red") + l_rug(mapping = aes(x=x, y=y), alpha = 0.1) +
  l_ciLine(mul = 5, colour = "salmon", linetype = 2) + 
  l_points(shape = 19, size = 0.5, alpha = 0.8) + theme_classic() + labs(y = "Temperature effect", x = "Temperature")
dev.off()


pvalue <- lapply(gams$gams, function(x){s <- summary(x); s$s.table["s(Exc_rec_aggl)",4]})%>%unlist
pvalue ##tte égales à 0


#Linear analysis 
#Temp + Mobility
eq_mob <- Load ~ 
  Exc_rec_aggl+ 
  s(temperature_smooth_950)
gams <- gam_discriminate(eq_mob, Data_nat[train,], Data_nat$tod[train], ncores=20)
forecast_gam = predict_gam_discriminate(gams, Data_nat, Data_nat$tod)
performances_df = rbind(performances_df, performances("GAM_TempMoblin", Data_nat$Load, forecast_gam, test))

#Evaluate p-value
coeff_reg = c()
for(i in 1:48)
{
  coeff_reg[i]= gams$gams[[i]]$coefficients["Exc_rec_aggl"]
}

pdf(file="Clean_codes/4-Explainability/regression_coefficient.pdf")
#plot(0:47/2, coeff_reg, xlab="Time of day", ylab="Linear regression coefficient", ylim=c(0, max(coeff_reg)))
plot(0:47/2, coeff_reg, xlab="Time of day", ylab="Linear regression coefficient", ylim=c(0, max(coeff_reg)),
     type='h', pch=20)
points(0:47/2, coeff_reg, pch=20)
dev.off()


#Dependency test
library(Hmisc)
hoeffding_test = hoeffd(cbind(Data_nat$Load[train], 
                              Data_nat$Load[train] - forecast_gam_temp[train],
                              Data_nat$Load[train] - forecast_gam_mob[train],
                              Data_nat$temperature[train], 
                              Data_nat$temperature_smooth_950[train],
                              Data_nat$Exc_rec_aggl[train], 
                              Data_nat$Residents_aggl[train], 
                              Data_nat$Tourists_aggl[train], 
                              Data_nat$toy[train], 
                              Data_nat$day_type_week[train], 
                              Data_nat$day_type_jf[train]))
hoeffding_test$D #La statistique de Hoeffding, entre -0.5 et 1. Plus elle est >0, plus les variables sont dépendantes.
hoeffding_test$P #On teste la dépendance:  petite p-value implique grande dépendance

#Stationarity test
library(ks)
year1 = 2020
year2 = 2021
kg_test_1 = which((Data_nat$year==year1)&(Data_nat$tod==30))
kg_test_2 = which((Data_nat$year==year2)&(Data_nat$tod==30))
Data_test = Data_nat
Data_test$residuals = Data_test$Load - forecast_gam_temp
Data_test$Exc_rec_aggl[kg_test_2] = Data_test$Exc_rec_aggl[kg_test_2]  - 0.01e+7 #0.12e+7
Data_test$residuals[kg_test_2] = Data_test$residuals[kg_test_2]  - 100 #800
plot(Data_test$Exc_rec_aggl[kg_test_1], Data_test$residuals[kg_test_1])
lines(Data_test$Exc_rec_aggl[kg_test_2], Data_test$residuals[kg_test_2], col='red',  type='p')
legend("topleft",   as.character(c(year1, year2)),  col=1:2, cex=1, pch=4)
kde.test(x1=Data_test[c("Exc_rec_aggl", "residuals")][kg_test_1,],
         x2=Data_test[c("Exc_rec_aggl", "residuals")][kg_test_2,])$pvalue

#Minimum Redundancy, Maximum Relevance (mRMR)
library(mRMRe)
set.thread.count(10)

#Without mobility data
data_test = Data_nat[train,][c("Load", 
                                        "tod", "toy", "day_type_week", "day_type_jf",
                                        "temperature", "temperature_smooth_950", "temperature_min_smooth_950", "temperature_max_smooth_950")]
data_test$day_type_week = as.numeric(data_test$day_type_week)
data_mrmr = mRMR.data(data = data_test)
mrmr_test = mRMR.classic(data = data_mrmr, target_indices = c(1), feature_count = 4)
#The most important variables are, by increasing order of importance,
mrmr_test@feature_names[mrmr_test@filters$`1`]
mrmr_test@scores

#With mobility data
data_test_mobility = Data_nat[train,][c("Load", 
                               "tod", "toy", "day_type_week", "day_type_jf",
                               "temperature", "temperature_smooth_950",
                               "Exc_rec_aggl", "Residents_aggl", "Tourists_aggl", "Exc_aggl")]
data_test_mobility$day_type_week = as.numeric(data_test_mobility$day_type_week)
data_mrmr = mRMR.data(data = data_test_mobility)
mrmr_test = mRMR.classic(data = data_mrmr, target_indices = c(1), feature_count = 3)
#The most important variables are, by increasing order of importance,
mrmr_test@feature_names[mrmr_test@filters$`1`]
mrmr_test@scores
