##
## Bike Share EDA Code
##

## libraries
library(tidyverse)
library(vroom)
library(skimr)
library(DataExplorer)
library(patchwork)

# set working directory and read in the data
getwd()
setwd('C:/Users/22jac/OneDrive/Desktop/STAT348/KaggleBikeShare')
bike <- vroom("train.csv")


# exploratory data analysis
glimpse(bike)
skim(bike)
plot_intro(bike)
# look at DataExplorer plots
bike_corr <- plot_correlation(bike)
bike_corr
plot_bar(bike)
plot_histogram(bike)

# make season a factor
bike$season <- as.factor(bike$season)
summary(bike)

# create 4 ggplots
plot1 <- ggplot(data = bike) +
  geom_histogram(aes(x = count), fill = 'blue')
plot2 <- ggplot(data = bike) +
  geom_point(mapping = aes(x = temp, y = count)) +
  geom_smooth(mapping = aes(x = temp, y = count), se = FALSE)
plot3 <- ggplot(data = bike) +
  geom_boxplot(aes(x = season, y = count))
plot4 <- ggplot(data = bike) +
  geom_point(mapping = aes(x = humidity, y = count)) +
  geom_smooth(mapping = aes(x = humidity, y = count), se = FALSE)

plot1
plot3
plot4
plot2

# prepare patchwork plot for saving

(plot1 + plot2) / (plot3 + plot4)
