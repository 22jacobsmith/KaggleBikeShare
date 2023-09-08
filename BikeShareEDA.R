##
## Bike Share EDA Code
##

## libraries
library(tidyverse)
library(vroom)

# read in the data
getwd()
setwd('C:/Users/22jac/OneDrive/STAT348')
bike <- vroom("./train.csv")
