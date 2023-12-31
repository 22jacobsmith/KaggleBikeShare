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
               levels = 10)

## split the data into K folds
folds <- vfold_cv(log_bike_train, v = 10, repeats = 1)

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


##### regression trees ############



library(rpart)

dtree_rec <-
  recipe(count~., data=log_bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime)

my_tree_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")
                        
### create a workflow with model & recipe

dtree_wf <- workflow() %>%
  add_recipe(dtree_rec) %>%
  add_model(my_tree_mod)


### set up a grid of tuning values

tuning_grid <-
  grid_regular(tree_depth(),
               cost_complexity(),
               min_n(),
               levels = 10)

### set up the k-fold cv
folds <- vfold_cv(log_bike_train, v = 5, repeats = 1)

## run the cross validation
CV_results <-
  dtree_wf %>% tune_grid(resamples = folds,
                             grid = tuning_grid,
                             metrics = metric_set(rmse, mae, rsq))



### find best tuning parameters
best_tune <- CV_results %>%
  select_best('rmse')

## finalize workflow

final_wf <-
  dtree_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_bike_train)

### predict

dtree_preds <- exp(predict(final_wf, new_data=bike_test)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle



## Write prediction file to a CSV for submission
vroom_write(x=dtree_preds, file="DTreeTestPreds.csv", delim=",")





########## RANDOM FOREST MODELS ##################


library(rpart)
library(ranger)


dtree_rec <-
  recipe(count~., data=log_bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime)

rf_mod <- rand_forest(mtry = tune(),
                             min_n = tune(),
                             trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

### create a workflow with model & recipe

rf_wf <- workflow() %>%
  add_recipe(dtree_rec) %>%
  add_model(rf_mod)


### set up a grid of tuning values

tuning_grid <-
  grid_regular(mtry(range = c(1,9)),
               min_n(),
               levels = 10)

### set up the k-fold cv
folds <- vfold_cv(log_bike_train, v = 5, repeats = 1)

## run the cross validation
rf_CV_results <-
  rf_wf %>% tune_grid(resamples = folds,
                         grid = tuning_grid,
                         metrics = metric_set(rmse, mae, rsq))



### find best tuning parameters
rf_best_tune <- rf_CV_results %>%
  select_best('rmse')

## finalize workflow

rf_final_wf <-
  rf_wf %>%
  finalize_workflow(rf_best_tune) %>%
  fit(data = log_bike_train)

### predict

rf_preds <- exp(predict(rf_final_wf, new_data=bike_test)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle


## Write prediction file to a CSV for submission
vroom_write(x=rf_preds, file="RFTestPreds.csv", delim=",")

rf_preds %>% View()




########## STACKING ###############


## stacking many models can outperform the base learners. Performs best
## for uncorrelated predictions

library(stacks)


##### set up recipe

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
  step_rm(datetime) %>% # remove the original datetime column
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

## cv folds

folds <- vfold_cv(bike_train, v = 5)


## control settings for stacking models
untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()






### penalized regression model





# Penalized regression model10
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

preg_wf <-
  workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5) ## L^2 total tuning possibilities


## Run the CV
preg_models <- preg_wf %>%
tune_grid(resamples=folds,
          grid=preg_tuning_grid,
          metrics=metric_set(rmse, mae, rsq),
          control = untuned_model) # including the control grid in the tuning ensures you can
# call on it later in the stacked model


### set up stacked linear model ###

lin_model <- linear_reg() %>%
  set_engine("lm")


## set up workflow
linreg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model)


## fit to folds
lin_reg_model <-
  fit_resamples(linreg_wf,
                resamples = folds,
                metrics = metric_set(rmse),
                control = tuned_model)


### add a decision tree to the stack
library(rpart)

dtree_rec <-
  recipe(count~., data=bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime)




my_tree_mod <- decision_tree(tree_depth = tune(),
                             cost_complexity = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

### create a workflow with model & recipe

dtree_wf <- workflow() %>%
  add_recipe(dtree_rec) %>%
  add_model(my_tree_mod)


### set up a grid of tuning values

dt_tuning_grid <-
  grid_regular(tree_depth(),
               cost_complexity(),

               levels = 5)




## Run the CV
dtree_models <- dtree_wf %>%
  tune_grid(resamples=folds,
            grid=dt_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untuned_model) # including the control grid in the tuning ensures you can
# call on it later in the stacked model




my_stack <-
  stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(dtree_models) %>%
  add_candidates(lin_reg_model)

stack_mod <-
  my_stack %>%
  blend_predictions() %>%
  fit_members()

#predict(stack_mod, new_data = bike_test)


stacked_preds <- predict(stack_mod, new_data=bike_test) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle


## Write prediction file to a CSV for submission
vroom_write(x=stacked_preds, file="stackTestPreds.csv", delim=",")


## try just stacking d trees


my_stack2 <-
  stacks() %>%
  add_candidates(dtree_models)

stack_mod2 <-
  my_stack %>%
  blend_predictions() %>%
  fit_members()

#predict(stack_mod, new_data = bike_test)


stacked_preds2 <- predict(stack_mod2, new_data=bike_test) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle


## Write prediction file to a CSV for submission
vroom_write(x=stacked_preds2, file="stackTestPreds2.csv", delim=",")


################# try stacking just rf models ##################################

library(rpart)
library(ranger)
log_bike_train <- bike_train %>% mutate(count = log(count))
rf_rec <-
  recipe(count~., data=log_bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime)

rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

### create a workflow with model & recipe

rf_wf <- workflow() %>%
  add_recipe(rf_rec) %>%
  add_model(rf_mod)


### set up a grid of tuning values

rf_tuning_grid <-
  grid_regular(mtry(range = c(1,9)),
               min_n(),
               levels = 5)

### set up the k-fold cv
folds <- vfold_cv(log_bike_train, v = 5, repeats = 1)

## run the cross validation
rf_CV_results <-
  rf_wf %>% tune_grid(resamples = folds,
                      grid = rf_tuning_grid,
                      metrics = metric_set(rmse, mae, rsq),
                      control = untuned_model)



my_stack3 <-
  stacks() %>%
  add_candidates(rf_CV_results)

stack_mod3 <-
  my_stack3 %>%
  blend_predictions() %>%
  fit_members()

#predict(stack_mod, new_data = bike_test)


stacked_preds3 <- exp(predict(stack_mod3, new_data=bike_test)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle


## Write prediction file to a CSV for submission
vroom_write(x=stacked_preds3, file="stackTestPreds3.csv", delim=",")



###### Make the best model possible using tidymodels
library(tidymodels)
library(kknn)



knn_rec <-
  recipe(count~., data=log_bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime)

knn_mod <- nearest_neighbor(mode = 'regression',
                           engine = 'kknn')

### create a workflow with model & recipe

knn_wf <- workflow() %>%
  add_recipe(knn_rec) %>%
  add_model(knn_mod)

knn_final_wf <-
  knn_wf %>%
  fit(data = log_bike_train)


### prepare to export to kaggle
knn_preds <- exp(predict(knn_final_wf, new_data=bike_test)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle



## Write prediction file to a CSV for submission
vroom_write(x=knn_preds, file="KNNRegTestPreds.csv", delim=",")


### try an xgboost model



boost_rec <-
  recipe(count~., data=log_bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors())

boost_mod <- boost_tree(mode = 'regression',
                            engine = 'xgboost',
                        tree_depth = tune(),
                        learn_rate = tune(),
                        trees = tune())

### create a workflow with model & recipe

boost_wf <- workflow() %>%
  add_recipe(boost_rec) %>%
  add_model(boost_mod)

# run the CV

boost_tuning_grid <-
  grid_regular(tree_depth(),
               learn_rate(),
               trees(),
               levels = 5)

### set up the k-fold cv
folds <- vfold_cv(log_bike_train, v = 5, repeats = 1)

## run the cross validation
boost_CV_results <-
  boost_wf %>% tune_grid(resamples = folds,
                      grid = boost_tuning_grid,
                      metrics = metric_set(rmse, mae, rsq))


### find best tuning parameters
best_tune <- boost_CV_results %>%
  select_best('rmse')

## finalize workflow

boost_final_wf <-
  boost_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_bike_train)


### predict

boost_preds <- exp(predict(boost_final_wf, new_data=bike_test)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle



## Write prediction file to a CSV for submission
vroom_write(x=boost_preds, file="xgBoostTestPreds.csv", delim=",")





############ try an auto_ml model
library(agua)
library(h2o)

h2o.init()




auto_mod <-
  auto_ml(mode = "regression", engine = "h2o")

auto_wf <-
  workflow() %>%
  add_recipe(boost_rec) %>%
  add_model(auto_mod)

auto_final_wf <-
  auto_wf %>%
  fit(data = log_bike_train)


### predict

auto_preds <- exp(predict(auto_final_wf, new_data=bike_test)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle


## Write prediction file to a CSV for submission
vroom_write(x=auto_preds, file="AutoMLTestPreds.csv", delim=",")









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
                      mtry = tune(), trees = tune()) # engine = what r function to use

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

### try a tuned rf model #####################################
library(tidymodels)
library(rpart)
library(tune)
library(randomForest)
rf_mod <- rand_forest(mtry = 3,
                             min_n = tune(),
                             trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

### create a workflow with model & recipe

rf_wf <- workflow() %>%
  add_recipe(dtree_rec) %>%
  add_model(rf_mod)


### set up a grid of tuning values

tuning_grid <-
  grid_regular(min_n(),
              # mtry(),
               levels = 5)


### set up the k-fold cv
folds <- vfold_cv(log_bike_train, v = 5, repeats = 1)

## run the cross validation
CV_results <-
  rf_wf %>% tune_grid(resamples = folds,
                         grid = tuning_grid,
                         metrics = metric_set(rmse, mae, rsq))



### find best tuning parameters
best_tune <- CV_results %>%
  select_best('rmse')

## finalize workflow

final_wf <-
  rf_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_bike_train)

### predict

tune_preg_preds <- exp(predict(final_wf, new_data=bike_test)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle



## Write prediction file to a CSV for submission
vroom_write(x=tune_preg_preds, file="RFTestPreds.csv", delim=",")





#### try an xgboost model


library(xgboost)

boost_rec <-
  recipe(count~., data=log_bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  #step_mutate(datetime_hour = factor(datetime_hour)) %>% 
  step_rm(datetime) %>%
  # step_rm(holiday) %>%
  # step_rm(workingday) %>%
  #step_rm(atemp) %>%
  # step_poly(temp, degree = 2) %>%
  # step_poly(atemp, degree = 2) %>%
  # step_poly(humidity, degree = 2) %>%
  # step_poly(windspeed, degree = 2) %>%
  step_dummy(all_nominal_predictors())

#boost_rec 

my_mod <- boost_tree(tree_depth = 3,
                     learn_rate = 0.1,
                     trees = 700) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

bike_workflow_boost <- workflow() %>%
  add_recipe(boost_rec) %>%
  add_model(my_mod) %>%
  fit(data = log_bike_train) # fit the workflow








## Get Predictions for test set, format for Kaggle
test_preds <- exp(predict(bike_workflow_boost, new_data = bike_test)) %>%
  bind_cols(., bike_test) %>% # combine predicted values with test data
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle

## Write prediction file to a CSV for submission
vroom_write(x=test_preds, file="BoostTestPreds.csv", delim=",")




#### try a bart model

library(tidymodels)
library(dbarts)
library(parsnip)

log_bike_train <- bike_train %>% mutate(count = log(count))


bart_rec <-
  recipe(count~., data=log_bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  #step_mutate(datetime_hour = factor(datetime_hour)) %>% 
  step_rm(datetime) %>%
  # step_rm(holiday) %>%
  # step_rm(workingday) %>%
  #step_rm(atemp) %>%
  # step_poly(temp, degree = 2) %>%
  # step_poly(atemp, degree = 2) %>%
  # step_poly(humidity, degree = 2) %>%
  # step_poly(windspeed, degree = 2) %>%
  step_dummy(all_nominal_predictors())



my_mod <- bart(
  trees = 15
) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression") %>% 
  translate()

bike_workflow_bart <- workflow() %>%
  add_recipe(bart_rec) %>%
  add_model(my_mod) %>%
  fit(data = log_bike_train) # fit the workflow








## Get Predictions for test set, format for Kaggle
test_preds <- exp(predict(bike_workflow_bart, new_data = bike_test)) %>%
  bind_cols(., bike_test) %>% # combine predicted values with test data
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle

## Write prediction file to a CSV for submission
vroom_write(x=test_preds, file="BARTTestPreds.csv", delim=",")


