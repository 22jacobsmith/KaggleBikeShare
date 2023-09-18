# Libraries and dependencies
library(tidyverse)
library(tidymodels)
library(vroom)
# Read in the Bike Sharing demand data set, test and training data
bike_train <- vroom("train.csv")
bike_test <- vroom("test.csv")
# Remove casual and registered from the data set
bike_train <- bike_train %>%
  select(-casual, -registered)

### Data cleaning, feature engineering
bike_recipe <- recipe(count~., data=bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime) # remove the original datetime column
bike_recipe <- prep(bike_recipe)
train_baked <- bake(bike_recipe, new_data = bike_train)
test_baked <- bake(bike_recipe, new_data = bike_test)

### try a linear regression model

my_mod <- linear_reg() %>% #type of model
  set_engine("lm") # engine = what r function to use

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_train) # fit the workflow


# View the fitted linear regression model
extract_fit_engine(bike_workflow) %>%
  summary()
## Get Predictions for test set AND format for Kaggle
test_preds <- predict(bike_workflow, new_data = bike_test) %>%
  bind_cols(., bike_test) %>% # combine predicted values with test data
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write prediction file to a CSV for submission
vroom_write(x=test_preds, file="TestPreds.csv", delim=",")









