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
  step_poly(temp, degree = 2) %>%
  step_poly(atemp, degree = 2) %>%
  step_poly(humidity, degree = 2) %>%
  step_poly(windspeed, degree = 2) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime) # remove the original datetime column

# bake the recipe, make sure it works on test and train set

bike_recipe <- prep(bike_recipe)
train_baked <- bake(bike_recipe, new_data = bike_train)
test_baked <- bake(bike_recipe, new_data = bike_test)

### try a linear regression model ####

my_mod <- linear_reg() %>% #type of model
  set_engine("lm") # engine = what r function to use

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_train) # fit the workflow


# View the fitted linear regression model
extract_fit_engine(bike_workflow) %>%
  summary()
# view the model in a tidy data frame
extract_fit_engine(bike_workflow) %>%
  tidy()

## Get Predictions for test set, format for Kaggle
test_preds <- predict(bike_workflow, new_data = bike_test) %>%
  bind_cols(., bike_test) %>% # combine predicted values with test data
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle

## Write prediction file to a CSV for submission
vroom_write(x=test_preds, file="TestPreds.csv", delim=",")

#####  ####### ###### ##### ###
# try a Poisson regression model
###### ###### ##### ###### ######

library(poissonreg)

pois_mod <- poisson_reg() %>%
  set_engine("glm")

bike_pois_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = bike_train)

bike_predictions_pois <-
  predict(bike_pois_workflow, new_data = bike_test)



# View the fitted poisson regression model
extract_fit_engine(bike_pois_workflow) %>%
  summary()
# view the model in a tidy data frame
extract_fit_engine(bike_pois_workflow) %>%
  tidy()

## Get Predictions for test set, format for Kaggle
pois_test_preds <- predict(bike_pois_workflow, new_data = bike_test) %>%
  bind_cols(., bike_test) %>% # combine predicted values with test data
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle


## Write prediction file to a CSV for submission
vroom_write(x=pois_test_preds, file="PoisTestPreds.csv", delim=",")

pois_test_preds
test_preds



############################
### Penalized Regression ###
############################

library(tidymodels)
library(poissonreg)

log_bike_train <- bike_train %>% mutate(count = log(count))
## create a new recipe for penalized regression
bike_pen_recipe <-
  recipe(count~., data=log_bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime) %>%
  step_rm(holiday) %>%
  step_rm(workingday) %>%
  #step_rm(atemp) %>%
  step_poly(temp, degree = 2) %>%
  step_poly(atemp, degree = 2) %>%
  step_poly(humidity, degree = 2) %>%
  step_poly(windspeed, degree = 2) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())


# set preg model
preg_model <- linear_reg(penalty = 0, mixture = 0) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
add_recipe(bike_pen_recipe) %>%
add_model(preg_model) %>%
fit(data=log_bike_train)

#exp(predict(preg_wf, new_data=bike_test)) %>% View()
extract_fit_engine(preg_wf) %>%
  tidy()

preg_preds <- exp(predict(preg_wf, new_data=bike_test)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle



## Write prediction file to a CSV for submission
vroom_write(x=preg_preds, file="PRegTestPreds.csv", delim=",")


######## Tuning models for penalized regression

library(tidymodels)
library(poissonreg)

# initialize the model
tune_preg_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

## set up workflow
 
tune_preg_wf <- workflow() %>%
  add_recipe(bike_pen_recipe) %>%
  add_model(tune_preg_model)


## get a grid of values to tune over
tuning_grid <-
  grid_regular(penalty(),
               mixture(),
               levels = 5)

## split the data into K folds
folds <- vfold_cv(log_bike_train, v = 5, repeats = 1)

## run the cross validation
CV_results <-
  tune_preg_wf %>% tune_grid(resamples = folds,
                             grid = tuning_grid,
                             metrics = metric_set(rmse, mae, rsq))

## plot cv results
collect_metrics(CV_results) %>%
  filter(.metric == 'rmse') %>%
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()

## find best tuning parameters

best_tune <- CV_results %>%
  select_best('rmse')

## finalize workflow

final_wf <-
  tune_preg_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_bike_train)

## get predictions on log scale

final_wf %>% predict(new_data = bike_train)


### export to kaggle
tune_preg_preds <- exp(predict(final_wf, new_data=bike_test)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle



## Write prediction file to a CSV for submission
vroom_write(x=tune_preg_preds, file="TunePRegTestPreds.csv", delim=",")


##################################################
### Make the best model possible

log_bike_train <- bike_train %>% mutate(count = log(count))

bike_recipe_b <- recipe(count~., data=log_bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_poly(temp, degree = 2) %>%
  step_poly(atemp, degree = 2) %>%
  step_poly(humidity, degree = 2) %>%
  step_poly(windspeed, degree = 2) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime)

# bake the recipe, make sure it works on test and train set

bike_recipe <- prep(bike_recipe_b)


### try a linear regression model ####



my_mod <- linear_reg() %>% #type of model
  set_engine("glm") # engine = what r function to use

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe_b) %>%
  add_model(my_mod) %>%
  fit(data = bike_train) # fit the workflow



## Get Predictions for test set, format for Kaggle
test_preds <- exp(predict(bike_workflow, new_data = bike_test)) %>%
  bind_cols(., bike_test) %>% # combine predicted values with test data
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle

## Write prediction file to a CSV for submission
vroom_write(x=test_preds, file="BestTestPreds.csv", delim=",")



### rf ####################################


bike_recipe_rf <- recipe(count~., data=bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  # step_poly(temp, degree = 4) %>%
  # step_poly(atemp, degree = 4) %>%
  # step_poly(humidity, degree = 4) %>%
  # step_poly(windspeed, degree = 4) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime)

# bake the recipe, make sure it works on test and train set

bike_recipe_rf <- prep(bike_recipe_rf)


### try a random forest model ####


library(ranger)
library(randomForest)
my_mod <- rand_forest(engine = "ranger", mode = "regression",
                      mtry = 9, trees = 5000) # engine = what r function to use

bike_workflow_rf <- workflow() %>%
  add_recipe(bike_recipe_rf) %>%
  add_model(my_mod) %>%
  fit(data = bike_train) # fit the workflow



## Get Predictions for test set, format for Kaggle
test_preds <- predict(bike_workflow_rf, new_data = bike_test) %>%
  bind_cols(., bike_test) %>% # combine predicted values with test data
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle

## Write prediction file to a CSV for submission
vroom_write(x=test_preds, file="BestTestPreds.csv", delim=",")
