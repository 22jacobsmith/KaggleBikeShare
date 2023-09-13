# Bike Share Analysis
# cleaning section: cleaning step with dplyr
# feature engineering: 2 steps using recipe (factors, time of day)

### libraries and reading in data ###
# libraries
library(tidyverse)
library(vroom)
library(skimr)
library(DataExplorer)
library(patchwork)
library(tidymodels)
# read in data

setwd('C:/Users/22jac/OneDrive/Desktop/STAT348/KaggleBikeShare')
bike <- vroom("train.csv")

# attach the bike data set
attach(bike)

##### Data cleaning #####

# change weather level 4 (heavy snow) to 3 (light snow) since 
# there is only one observation with level 4
bike <- bike %>% mutate(weather = ifelse(weather == 4, 3, weather))
# make holiday, workingday, weather, and season factor variables
bike <- bike %>% mutate(holiday = factor(holiday))
bike <- bike %>% mutate(workingday = factor(workingday))
bike <- bike %>% mutate(weather = factor(weather))
bike <- bike %>% mutate(season = factor(season))

# initialize the bike_clean data set

bike_clean <- bike
  
# Feature engineering



my_recipe <- recipe(count ~ ., data=bike) %>% # Set model formula
  step_time(datetime, features=c("hour")) %>% # create hour variable
  step_select(-c(casual,registered, datetime)) #selects columns
prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataS
bake(prepped_recipe, new_data = bike_clean)
bike_clean <- bake(prepped_recipe, new_data = bike_clean)


bike_clean %>% head(10)
bike_clean %>% View()
bike %>% head(10)





