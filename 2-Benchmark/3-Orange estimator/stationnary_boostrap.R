library("tseries")
set.seed(948)

#mat_res <- readRDS("Results/estimatorARIMA.RDS")
#mat_res <- readRDS("Results/estimatorARIMAHolidays.RDS")
#mat_res <- readRDS("Results/estimatorResidualsComplete.RDS")
#mat_res <- readRDS("Results/estimatorResidualsCompleteHolidays.RDS")
#mat_res <- readRDS("Results/estimatorNational.RDS")
#mat_res <- readRDS("Results/estimatorNationalHolidays.RDS")
#mat_res <- readRDS("Results/estimatorDirectResiduals.RDS")
#mat_res <- readRDS("Results/estimatorGamBoostingHolidays.RDS")
#mat_res <- readRDS("Results/estimatorGamBoostingComplete.RDS")
#mat_res <- readRDS("Results/estimatorResidualsRFHolidays.RDS")
mat_res <- readRDS("Results/estimatorResidualsRF.RDS")


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
