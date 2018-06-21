---
title: "My First Kaggle Competition"
date: "2018-06-18T21:49:57-07:00"
---

For this Kaggle project, we are attempting to predict whether or not people will be able to repay their loans or not. To estimate this, we use a training and testing data to build a machine learning model. In my submission, I used a linear regression (glm) model. To increase my score, I used variables in the training data that have higher weights, which I found using the VAR IMP() function. This function showed me variables in my model have the highest importance, for example, ext source 2 has a very high weight and by adding it to my model, ROC which measures my accuracy improved my score a lot. Next, I replaced all the "NA" values in my data with the mean of each column, so that I could still use those columns in my predicition. This allowed me to use a lot more columns then I originally could.

Here is the code for my highest submission:


```r
library(readr)
library(ISLR)
library(ROCR)
library(Epi)
library(vcdExtra)
library(MASS)
library(ggplot2)
library(dplyr)
library(mlbench)
library(digest)
library(Metrics)
library(caret)
library(rpart)

library(dplyr)
library(ROCR)
load_application <- function(fileName) {
  read.csv(fileName) %>% setNames(names(.) %>% tolower())
}

test <- load_application("~/Desktop/iXperience/home-credit/home_credit/application_test.csv")
train <- load_application("~/Desktop/iXperience/home-credit/home_credit/application_train.csv")

head(train)

train$amt_annuity[is.na(train$amt_annuity)] <- mean(na.exclude(train$amt_annuity))
test$amt_annuity[is.na(test$amt_annuity)] <- mean(na.exclude(test$amt_annuity))
train$ext_source_1[is.na(train$ext_source_1)] <- mean(na.exclude(train$ext_source_1))
test$ext_source_1[is.na(test$ext_source_1)] <- mean(na.exclude(test$ext_source_1))
train$ext_source_2[is.na(train$ext_source_2)] <- mean(na.exclude(train$ext_source_2))
test$ext_source_2[is.na(test$ext_source_2)] <- mean(na.exclude(test$ext_source_2))
train$ext_source_3[is.na(train$ext_source_3)] <- mean(na.exclude(train$ext_source_3))
test$ext_source_3[is.na(test$ext_source_3)] <- mean(na.exclude(test$ext_source_3))


train$target <- gsub(" ", "", as.character(train$target))
train$target <- gsub("\n", "", as.character(train$target))
change_factor <- function(column) {
  column <- factor(column, labels = c("no", "yes"))
}
train$target <- change_factor(train$target)


fit <- glm(target ~  ext_source_1 + ext_source_3 + amt_annuity + region_population_relative + days_id_publish + days_registration + ext_source_2 + amt_income_total + 
             amt_credit + amt_goods_price + days_birth + days_employed + flag_cont_mobile + region_rating_client_w_city 
           + days_last_phone_change + amt_income_total*flag_own_car + region_population_relative + cnt_children + amt_income_total*flag_own_car
           + amt_income_total*flag_own_realty + days_employed*name_housing_type + amt_income_total*name_contract_type, train, family = binomial)
summary(fit)
(importance <- varImp(fit, scale = FALSE))
fit_1 <- train(target ~ ext_source_1 + ext_source_3 + amt_annuity + region_population_relative + days_id_publish + days_registration + ext_source_2 + amt_income_total + 
                 amt_credit + amt_goods_price + days_birth + days_employed + flag_cont_mobile + region_rating_client_w_city 
               + days_last_phone_change + flag_own_car + region_population_relative + cnt_children + amt_income_total*flag_own_car
               + amt_income_total*flag_own_realty + days_employed*name_housing_type + amt_income_total*name_contract_type,
             data = application_train, method = "glm",
             metric = "Sens",
             na.action = na.pass,
             trControl = trainControl(
               method = "cv",
               number = 5,
               classProbs = TRUE,
               summaryFunction = twoClassSummary,
               verboseIter = TRUE
             ))
fit_1
fit_1$results
(importance <- varImp(fit_1, scale = FALSE))
test$TARGET = predict(fit, test, type = "response")
test$TARGET = predict(fit_1, test, type = "prob")
possibility <- data.frame(SK_ID_CURR = test$sk_id_curr, TARGET = test$TARGET)
head(possibility)
write.csv(possibility, "possibility.csv", row.names = FALSE)

```
After I made my model using the varimp function, and by replacing the NAs with the mean, I used the caret function to check my ROC so that I wouldn't have to repeatedly submit onto Kaggle. Now, I plan to use feature engineering to implement the other data sets in my model as well. 