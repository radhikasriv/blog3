---
title: "My First Kaggle Competition"
date: "2018-06-18T21:49:57-07:00"
---
For my first Kaggle submission, I joined the Home Credit Default Risk competition. 
For this project, we are attempting to predict how capable a client is at repaying a loan. To estimate this, we use a training and testing data to build a machine learning model to give as accurate of an estimate we can as to whether the loan will be paid back or not. Here are the steps for my highest submission: 
#In my submission, I used a linear regression (glm) model. To increase my score, I used variables in the training data that have higher weights, which I found using the VAR IMP() function. This function showed me variables in my model have the highest importance, for example, ext source 2 has a very high weight and by adding it to my model, ROC which measures my accuracy improved my score a lot. Next, I replaced all the "NA" values in my data with the mean of each column, so that I could still use those columns in my predicition. This allowed me to use a lot more columns then I originally could.

#Here is the code for my highest submission:

First I loaded up all the packages I would need to make my model and test it. I added several packages as I worked for added functionality.For example, in order to plot the data, I loaded ggplot2, and to test my model, I loaded the caret package. 
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
```
I then loaded the training and testing data which was given to me, so I can now work with the data in R.
```r
load_application <- function(fileName) {
  read.csv(fileName) %>% setNames(names(.) %>% tolower())
}

test <- load_application("~/Desktop/iXperience/home-credit/home_credit/application_test.csv")
train <- load_application("~/Desktop/iXperience/home-credit/home_credit/application_train.csv")
head(train)
```
Next, I took several columns with NA values in them, and replaced them with the mean of the data, so I could use those columns in my model:

```r
train$amt_annuity[is.na(train$amt_annuity)] <- mean(na.exclude(train$amt_annuity))
test$amt_annuity[is.na(test$amt_annuity)] <- mean(na.exclude(test$amt_annuity))
train$ext_source_1[is.na(train$ext_source_1)] <- mean(na.exclude(train$ext_source_1))
test$ext_source_1[is.na(test$ext_source_1)] <- mean(na.exclude(test$ext_source_1))
train$ext_source_2[is.na(train$ext_source_2)] <- mean(na.exclude(train$ext_source_2))
test$ext_source_2[is.na(test$ext_source_2)] <- mean(na.exclude(test$ext_source_2))
train$ext_source_3[is.na(train$ext_source_3)] <- mean(na.exclude(train$ext_source_3))
test$ext_source_3[is.na(test$ext_source_3)] <- mean(na.exclude(test$ext_source_3))
```
With that, I made my first model using linear regression (glm), and continued on in my project. 
```r
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
Then I made a data frame of th IDs and the Target column:
```r
possibility <- data.frame(SK_ID_CURR = test$sk_id_curr, TARGET = test$TARGET)
head(possibility)
```
Finally, I wrote my model labeled "possbility" to a csv so I could finally submit it to Kaggle. 
```r

write.csv(possibility, "possibility.csv", row.names = FALSE)

```
```

This is just one example of a Kaggle competition, but there are so many more you can participate in to practice your machine learning skills. Here is a link to a list of some competitions: 
<https://www.kaggle.com/competitions>


If you are just getting started with Kaggle, a good competition to enter is the [Titanic competition](https://www.kaggle.com/c/titanic). 
This is a good competition to learn more about Kaggle basics and machine learning, as there are several resources on how to build your model. 
