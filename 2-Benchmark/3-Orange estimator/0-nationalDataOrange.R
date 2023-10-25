library(lubridate)
library(tidyverse)
library(magrittr)
library(progress)
library(dplyr)

#setwd("~/orange/Data/Input")
#setwd("~/orange/")


##################################################################################################################
#impot des données oranges
name_file = "Juillet2019-Fevrier2023"
Data_dep = data.frame()
ldir <- list.dirs(path="Data/Input")
ldir <- ldir[grep(name_file, ldir)]
lfiles <- ldir%>%list.files()

for(i in c(1:length(lfiles)))
{
  path <- file.path(ldir[grep(name_file, ldir)], lfiles[i])
  print(path)
  Orange <- read_delim(file=path, delim=';')
  Data_dep = rbind(Data_dep, Orange)
}

Data_dep$CategorieVisiteur <- forcats::fct_recode(Data_dep$CategorieVisiteur,  
                                                  "Excur_rec"= "Excursionniste recurrent",
                                                  "Hab_pres" = "Habituellement present")
Data_elec <- read_csv("Data/Input/dataset_national.csv")

#Consommateurs
Data_nat_sum = Data_dep %>% group_by(Date, CategorieVisiteur) %>% summarize(Volume_reg = sum(Volume))
consommateurs = c("Excursionniste", "Excur_rec", "Hab_pres", "Resident" , "Touriste")
data_tot = Data_nat_sum[which(Data_nat_sum$CategorieVisiteur %in% consommateurs),] %>% group_by(Date) %>% summarize(Consumers = sum(Volume_reg))
presence = function(date){return(data_tot$Consumers[which(data_tot$Date == as.Date(date))])}
Data= Data_elec[which(as.Date(Data_elec$date) %in% data_tot$Date),]
Data["Consumers"] = sapply(Data$date, presence)

#Indicateurs Stefania
Data_nat_sum = Data_dep %>% group_by(Date, CategorieVisiteur, Provenance) %>% summarize(Volume_reg = sum(Volume))

residents_aggl = Data_nat_sum[which(Data_nat_sum$CategorieVisiteur %in% c("Resident", "Hab_pres")),] %>% group_by(Date) %>% summarize(Residents_aggl = sum(Volume_reg))
presence_res = function(date){return(residents_aggl$Residents_aggl[which(residents_aggl$Date == as.Date(date))]+Data_nat_sum$Volume_reg[which((Data_nat_sum$Date== as.Date(date))& (Data_nat_sum$CategorieVisiteur=="Touriste")& (Data_nat_sum$Provenance=='Local'))])}
Data["Residents_aggl"] = sapply(Data$date, presence_res)

tourist = Data_nat_sum[which((Data_nat_sum$CategorieVisiteur=="Touriste")& (Data_nat_sum$Provenance %in% c('Etranger', 'NonLocal'))),] %>% group_by(Date) %>% summarize(Tourist_aggl = sum(Volume_reg))
presence_tourist = function(date){return(tourist$Tourist_aggl[which((tourist$Date== as.Date(date)))])}
Data["Tourists_aggl"] = sapply(Data$date, presence_tourist)

exc_rec = Data_nat_sum[which((Data_nat_sum$CategorieVisiteur=="Excur_rec")),] %>% group_by(Date) %>% summarize(Exc_rec_aggl = sum(Volume_reg))
presence_exc_rec = function(date){return(exc_rec$Exc_rec_aggl[which(exc_rec$Date== as.Date(date))])}
Data["Exc_rec_aggl"] = sapply(Data$date, presence_exc_rec)

exc = Data_nat_sum[which((Data_nat_sum$CategorieVisiteur=="Excursionniste")),] %>% group_by(Date) %>% summarize(Exc_aggl = sum(Volume_reg))
presence_exc = function(date){return(exc$Exc_aggl[which(exc$Date== as.Date(date))])}
Data["Exc_aggl"] = sapply(Data$date, presence_exc)

saveRDS(Data, paste0("Data/Input/Data_orange_national.RDS"))
 
