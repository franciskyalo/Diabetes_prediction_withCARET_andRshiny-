# DIABETES PREDICTION WITH SHINY APP

![bandicam 2023-10-10 18-49-17-304 (1)](https://github.com/franciskyalo/Diabetes_prediction_withCARET_andRshiny-/assets/94622826/14014f86-8456-4397-9b0c-826009c06107)

## Overview
Diabetes is a chronic medical condition characterized by elevated levels of glucose (sugar) in the blood. This occurs due to a deficiency in insulin, a hormone produced by the pancreas that regulates blood sugar levels.

Early diagnosis of diabetes is crucial for several reasons. Firstly, it allows for prompt management and treatment, helping to prevent complications such as heart disease, kidney failure, nerve damage, and vision problems. Secondly, early intervention can help individuals adopt healthier lifestyles, including dietary changes and increased physical activity, which can effectively manage the condition. Moreover, timely diagnosis provides an opportunity to monitor blood sugar levels and adjust treatment plans, optimizing long-term health and quality of life for those affected by diabetes.

### Main objective
The primary goal of the project is to create a predictive model that can accurately pinpoint individuals at high risk of developing diabetes.

## Data understanding
This dataset comprises 768 instances and encompasses 9 columns. It pertains to medical information regarding individuals, encompassing variables such as pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, age, and outcome.

## Modelling
Modelling involved the following steps:

Centering and scaling numeric features for better model training

Over-sampling training set to deal with class imbalance for the target variable

Fitting a random forest model to the data

Fitting a Support Vector Machine to the data

Comparing the fitted model in terms of accuracy and recall to determine the best model to use for deployment

## Evaluation
The models were tested for the performance on new unseen data and the random forest model outperformed the support vector machine model with an accuracy of 79% and a recall of 84% on the test set.

## Deployment
The model would be deployed as a shiny app (the UI OF THE APP CREATED IS DEMONSTRATED IN THE GIF ABOVE) where users can input different variables and get the prediction on whether a patient is either diabetic or not.



