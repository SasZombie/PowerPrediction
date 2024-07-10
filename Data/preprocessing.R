# code by @jorgeklz -------------------------------------------------------

# Load libraries ----------------------------------------------------------
library("tidyverse")
library("magrittr")
library("dplyr")
library("ggplot2")
library("GGally")



# Read raw.data -----------------------------------------------------------
datos <- readxl::read_xlsx("Datos/db_raw_final.xlsx")
datos <- datos %>% select(Código, Area, Fecha, Municipio, Uso, Estrato, ConsumoDis)


# Preprocessing -----------------------------------------------------------
datos <- datos %>% rename(code=Código, area=Area, date=Fecha, municipality=Municipio, use=Uso, stratum=Estrato, consumption=ConsumoDis)
ds <- datos %>% filter(use!="Alumbrado P.") #only 2 cases
ds <- ds %>% mutate(use= str_replace_all(use, 'Especial', 'Special'))
ds <- ds %>% mutate(use= str_replace_all(use, 'Residencial', 'Residential'))
ds <- ds %>% mutate(use= str_replace_all(use, 'Comercial', 'Commercial')) 
ds <- ds %>% mutate(use= str_replace_all(use, 'Oficial', 'Official')) 

# Save processed data -----------------------------------------------------
write.csv(ds, "Datos/db_power_consumption_zeros.csv", row.names = FALSE, quote = FALSE)
ds <- ds %>% filter(consumption>=0) 
write.csv(ds, "Datos/db_power_consumption.csv", row.names = FALSE, quote = FALSE)


