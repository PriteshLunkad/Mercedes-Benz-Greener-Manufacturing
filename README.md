# Mercedes-Benz-Greener-Manufacturing
# Problem Statement:
In this problem, we have to predict the time spent by each Mercedes-Benz cars on the test bench. This will help Mercedes-Benz improve the time taken for testing and therefore reducing the carbon dioxide emissions without reducing their standards. 
# Source of Data:
The dataset is provided by Daimler for the Kaggle competition, and this can be downloaded from here, 

# About the Dataset:
This dataset contains an anonymized set of variables, each representing a custom feature in a Mercedes car. For example, a variable could be 4WD, added air suspension, or a head-up display. 
The ground truth is labeled 'y' and represents the time (in seconds) that the car took to pass testing for each variable.
There are 378 features in the training dataset (including the y variable) and 4209 rows and the test dataset contains 4209 rows and 377 columns.
Machine Learning Problem Formulation:
In this problem we have to predict the time taken in seconds by each car to pass testing and testing is based on 377 different features and the predicted variable y is the time taken in seconds which is continuous so this problem can be considered as a regression problem. 
# Performance Metric:
The performance metric used for this problem is R2 score (Coefficient of Determination). It is the proportion of the variance in the dependent variable that is predictable from the independent variable. 

For more detailed information refer this blog, 
https://priteshlunkad1999.medium.com/mercedes-benz-greener-manufacturing-b18855a35991 
