library("tseries")
library(tidyverse)

#mat_res <- read_csv("/Users/Yannig/Documents/These_Nathan/Mobility_data_review/Results/prediction_RNN_newdates.txt")
mat_res <- read_csv("/Results/prediction_MLP_newdate_lag.txt")

#with orange data:
#mat_res_orange <- read_csv("/Users/Yannig/Documents/These_Nathan/Mobility_data_review/Results/prediction_RNN_orange_residuals.txt")
mat_res_orange <- read_csv("/Results/prediction_MLP_orange_residuals_lag.txt")

mat_res_orange = left_join(mat_res_orange, mat_res[, c('date', 'Load')])
mat_res_orange[, 2:6] = mat_res_orange[,2:6]+mat_res_orange$Load
mat_res = mat_res_orange[,-7]
#

mat_res$date
a = which(mat_res$date=="2022-09-01 23:30:00 UTC")
mat_res <- as.matrix(mat_res[-c(1:a),-1])
sqrt(mean((mat_res[, 1]-mat_res[, 2])^2))

set.seed(948)
boost_load <- tsbootstrap(mat_res[, 1], nb = 1, type = c("block"))
rmse_mean <- c(2:ncol(mat_res))
rmse_sd <- c(2:ncol(mat_res))
mape_mean <- c(2:ncol(mat_res))
mape_sd <- c(2:ncol(mat_res))

for(j in c(2: ncol(mat_res)))
{
  boost_load <- tsbootstrap(mat_res[, 1]-mat_res[, j] , nb = 1000, type = c("stationary"), b=48)
  rmse_boot <- apply(boost_load, 2, function(x){sqrt(mean((x^2)))})
  rmse_mean[j-1] <- mean(rmse_boot)
  rmse_sd[j-1] <- sd(rmse_boot)
  
  boost_load_relative <- tsbootstrap((mat_res[, 1]-mat_res[, j])/mat_res[, 1]*100, nb = 1000, type = c("stationary"), b=48)
  mape_boot <- apply(boost_load_relative, 2, function(x){mean(abs(x))})
  mape_mean[j-1] <- mean(mape_boot)
  mape_sd[j-1] <- sd(mape_boot)
}

names(rmse_sd) <- colnames(mat_res)[-1]
names(rmse_mean) <- colnames(mat_res)[-1]
names(mape_sd) <- colnames(mat_res)[-1]
names(mape_mean) <- colnames(mat_res)[-1]
rbind(rmse_mean, rmse_sd, mape_mean, mape_sd)



