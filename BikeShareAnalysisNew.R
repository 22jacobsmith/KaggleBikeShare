## Libraries I am going to need
library(tidyverse)
library(tidymodels)
library(vroom)
## Read in the data
bike_train <- vroom("train.csv")
bike_test <- vroom("test.csv")
## Remove casual and registered because we can't use them to predict
bike_train <- bike_train %>%
  select(-casual, - registered)
## Cleaning & Feature Engineering
bike_recipe <- recipe(count~., data=bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_num2factor(weather, levels=c("Sunny", "Mist", "Rain")) %>%
  step_num2factor(season, levels=c("Spring", "Summer", "Fall", "Winter")) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime)
bike_recipe <- prep(bike_recipe)
train_baked <- bake(bike_recipe, new_data = bikeTrain)
test_baked <- bake(bike_recipe, new_data = bikeTest)

### try a linear regression model

my_mod <- linear_reg() %>% #type of model
  set_engine("lm") # engine = what r function to use

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_train) # fit the workflow

# obtain the predictions
bike_predictions <- predict(bike_workflow,
                            new_data = bike_test)
# round any negative predictions up to 0
bike_predictions[bike_predictions < 0] <- 0



# prepare the csv file
lin_reg_output <- tibble(bike_test$datetime, bike_predictions)
names(lin_reg_output) <- c("datetime", "count")
write_csv(lin_reg_output, "linearRegressionPreds.csv")


