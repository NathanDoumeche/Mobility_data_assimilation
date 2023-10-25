library(mgcv)
library(tidyverse)
library(stats)
library(changepoint)
source("Clean_codes/Models/gam_utils.R")

moving_average = function(time_series, period){
  filter_period = rep(1/period, period)
  return(stats::filter(time_series, filter_period, method="convolution", sides=1))}

subsample = function(data, level){
  return(data[level*(1:(length(data)/level))])}

plot_descriptive = function(date, ground_truth, prediction, reference_period, title){
  residuals_APE =  (ground_truth-prediction)/ground_truth*100
  subsample_level = 10
  
  plot(subsample(date, subsample_level), subsample(residuals_APE, subsample_level), 
       type='l', xlab='Date', ylab='Residual (%)', main=title)
  pred_averaged = moving_average(data.frame(residuals_APE), 48*15)
  lines(subsample(date,subsample_level), subsample(pred_averaged, subsample_level), col='red', lwd=2)
  
  MAPE_reference = mean(residuals_APE[1:reference_period])
  sd_MAPE_reference = sd(residuals_APE[1:reference_period])
  abline(h=MAPE_reference + 2*sd_MAPE_reference, col='blue', lwd=2)
  abline(h=MAPE_reference - 2*sd_MAPE_reference, col='blue', lwd=2)
  
  abline(v= subsample(date, subsample_level)[max(which(subsample(date, subsample_level) <= as.Date("2020-03-17 00:00:00 UTC")))], col='green', lty=2, lwd=2)
  abline(v= subsample(date, subsample_level)[max(which(subsample(date, subsample_level) <= as.Date("2020-05-11 00:00:00 UTC")))], col='purple', lty=2, lwd=2)
  
  abline(v= subsample(date, subsample_level)[max(which(subsample(date, subsample_level) <= as.Date("2020-10-30 00:00:00 UTC")))], col='green', lty=2, lwd=2)
  abline(v= subsample(date, subsample_level)[max(which(subsample(date, subsample_level) <= as.Date("2020-12-15 00:00:00 UTC")))], col='purple', lty=2, lwd=2)
  
  abline(v= subsample(date, subsample_level)[max(which(subsample(date, subsample_level) <= as.Date("2021-04-03 00:00:00 UTC")))], col='green', lty=2, lwd=2)
  abline(v= subsample(date, subsample_level)[max(which(subsample(date, subsample_level) <= as.Date("2021-05-13 00:00:00 UTC")))], col='purple', lty=2, lwd=2)
  
  legend("topleft", 
         legend=c("Residual", "15-day rolling average", 
                  "2-sigma interval (2018-2020)",
                  "Beginning of lockdown",
                  "End of lockdown"), 
         col=c("black", "red", "blue", "green", "purple"), 
         lty=c(1,1,  1, 2,2), 
         lwd = c(1, 2,2,2,2),
         cex=0.8,
         bg="white")
  }

plot_change_point = function(date, ground_truth, prediction, change_points){
  residuals =  ground_truth-prediction
  
  changes = 0*residuals
  index_changes = c(1, cpts(change_points), length(residuals))
  for(i in 1:length(coef(change_points)$mean)){
    changes[index_changes[i]: index_changes[i+1]]= coef(change_points)$mean[i]
  }
  
  subsample_level = 10
  
  plot(subsample(date, subsample_level), subsample(residuals, subsample_level), 
       type='l', xlab='Date', ylab='Residuals (MW)')
  lines(subsample(date, subsample_level), subsample(changes, subsample_level), col='red', lwd=3)
  }

Data_nat = read_csv("Data/Input/dataset_national.csv")
Data_nat$day_type_week = as.factor(Data_nat$day_type_week)
Data_nat$period_hour_changed = as.factor(Data_nat$period_hour_changed)

#subsets
train = which((Data_nat$date >= as.POSIXct(strptime("2014-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC"))&
                (Data_nat$date < as.POSIXct(strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")))
test = which(Data_nat$date >= as.POSIXct(strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC"))
test_reference = which((Data_nat$date >= as.POSIXct(strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC"))&
                      (Data_nat$date < as.POSIXct(strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")))

#GAM model
eq_gam = Load ~  day_type_jf + day_type_week:period_hour_changed + s(toy, k=20, bs='cc') +
  s(temperature_smooth_950, k=5) + s(temperature_smooth_990, k=5) + s(temperature_min_smooth_990, temperature_max_smooth_990)

gams = gam_discriminate(eq_gam, Data_nat[train,], Data_nat$tod[train], ncores=20)
pred = predict_gam_discriminate(gams, Data_nat, Data_nat$tod)

pdf(file="Clean_codes/1-Sobriety/drift.pdf")
plot_descriptive(Data_nat[test,]$date, Data_nat[test,]$Load, pred[test], length(test_reference),'')
dev.off()

#Error distribution
library(nortest)
test_ref_normal =  which((Data_nat$date >= as.POSIXct(strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC"))&
                           (Data_nat$date < as.POSIXct(strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC")))
residual_error = Data_nat[test_ref_normal,]$Load-pred[test_ref_normal]
breaks = as.integer(sqrt(length(test_ref_normal)))
pdf("Clean_codes/1-Sobriety/histogram_residuals.pdf")
hist(residual_error, breaks=breaks, main =" ", xlab = "Residual")
dev.off()
pdf("Clean_codes/1-Sobriety/ACF.pdf")
acf(ts(residual_error, frequency = 48),lag.max = 60*48,
    xlab = "Lag in days", ylab = 'Autocorrelation function', main=' ')
dev.off()
print(paste0("Mean of residuals: ",mean(residual_error)))
print(paste0("Standard deviation of residuals: ",sd(residual_error)))
print(paste0("Mean MAPE of residuals: ",mean(abs(residual_error/Data_nat[test_ref_normal,]$Load))))
print(paste0("Mean MAPE of residuals: ",sqrt(mean(residual_error**2))))
print(t.test(residual_error))
ad.test(residual_error)
Box.test(residual_error, lag=48, type="Ljung-Box")
print(paste0("Ratio of standard deviations: ",mean(residual_error)/sd(Data_nat$Load[train])))
print(paste0("Ratio of standard deviations: ",sd(residual_error)/sd(Data_nat$Load[train])))

eq_obst = Load ~  s(Date, k=3) + day_type_week:period_hour_changed + s(toy, k=20, bs='cc') + s(Date, temperature, k=c(3,5)) +
  s(temperature_smooth_950, k=5) + s(temperature_smooth_990, k=5) + s(temperature_min_smooth_990, temperature_max_smooth_990) +
  Load_d1:day_type_week + Load_d7 + day_type_jf
gams_obst = gam_discriminate(eq_obst, Data_nat[train,], Data_nat$tod[train], ncores=20)
pred_obst = predict_gam_discriminate(gams_obst, Data_nat, Data_nat$tod)
residual_obst = Data_nat[test_ref_normal,]$Load-pred_obst[test_ref_normal]
print(paste0("Mean MAPE of Obst residuals: ",mean(abs(residual_obst/Data_nat[test_ref_normal,]$Load))))
print(paste0("Mean MAPE of Obst residuals: ",sqrt(mean(residual_obst**2))))

#Change point detection
m.bsm <- cpt.mean(Data_nat[test,]$Load-pred[test], method = "BinSeg",Q=11)
pdf(file="Clean_codes/1-Sobriety/change_point.pdf")
plot_change_point(Data_nat[test,]$date, Data_nat[test,]$Load, pred[test], m.bsm)
dev.off()

#Quantifying sobriety
m.bsm
print("The most important changepoints are, by order of importance, ")
print(Data_nat$date[test][cpts.full(m.bsm)[11,]])
coef(m.bsm)
sobriety = test[cpts(m.bsm)[11]:length(test)]
mean_MAPE = mean((Data_nat[sobriety,]$Load-pred[sobriety])/pred[sobriety]*100)
error = sd((Data_nat[sobriety,]$Load-pred[sobriety])/pred[sobriety]*100)/sqrt(length(sobriety))

