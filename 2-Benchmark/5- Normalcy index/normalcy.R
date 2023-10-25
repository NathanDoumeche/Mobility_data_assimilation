library(dplyr)
library(lubridate)
library(ggplot2)

Data_Orange <- readRDS("Data/Input/Data_orange_national.RDS")
holidays <- which((Data_Orange$day_type_jf==1))
Data_Orange <- Data_Orange[-holidays,]

week_ends = which(Data_Orange$day_type_week %in% c(5, 6))
Data_Orange <- Data_Orange[-week_ends,]
Data_Orange=Data_Orange[c("date", "Exc_rec_aggl")]

#7-day lag
Data_Orange$date = as.Date(Data_Orange$date)
Data_Orange = Data_Orange %>% group_by(date) %>% summarise(Exc_rec = mean(Exc_rec_aggl))
Data_Orange$date = Data_Orange$date+7


Data_normalcy <- read.csv("Data/normalcy-index.csv")
Data_n = Data_normalcy[c("date", "office_occupancy")][which(Data_normalcy$iso3c == "FRA"),]
Data_n$date = as.Date(Data_n$date)

data = merge(Data_Orange, Data_n, by="date")
data_normalized = data[which(data$date >= as.Date("2020-06-30")), ]
data_normalized$Exc_rec = scale(data_normalized$Exc_rec)
data_normalized$office_occupancy = scale(data_normalized$office_occupancy)

period1 =which(data_normalized$date < as.Date("2021-05-01"))
period2 =which((data_normalized$date > as.Date("2021-05-01")) & (data_normalized$date < as.Date("2022-04-01")))
period3 =which((data_normalized$date > as.Date("2022-04-01")))

pdf("Normalcy.pdf", width=9, height=7)
plot(data_normalized$date[period1], data_normalized$Exc_rec[period1], type = 'l',
     xlim = range(data_normalized$date), ylim = range(data_normalized$Exc_rec),
     xlab="Date", ylab="Normalized indices",
     col='royalblue2', cex.lab=1.5, cex.axis = 1.5)
lines(data_normalized$date[period2], data_normalized$Exc_rec[period2],
      col='royalblue2', cex.lab=1.5, cex.axis = 1.5)
lines(data_normalized$date[period3], data_normalized$Exc_rec[period3],
      col='royalblue2', cex.lab=1.5, cex.axis = 1.5)
lines(data_normalized$date[period1], data_normalized$office_occupancy[period1], col='red2', cex.lab=1.5, cex.axis = 1.5)
lines(data_normalized$date[period2], data_normalized$office_occupancy[period2], col='red2', cex.lab=1.5, cex.axis = 1.5)
lines(data_normalized$date[period3], data_normalized$office_occupancy[period3], col='red2', cex.lab=1.5, cex.axis = 1.5)
legend("topleft", c("Lagged work index", "Office occupancy index"), bty="n", col=c('royalblue2', 'red2'), lty=1, 
       cex=1.5)
dev.off()

cor(data_normalized$Exc_rec, data_normalized$office_occupancy)






