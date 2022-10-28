# Identifying Outliers in California High Schools

## Overview
This project is designed to take various sources of data on the 2021 cohort of California's high schools and combine them in an user-friendly, easily accessible resource. 
The bulk of the data wrangling and analysis is done in a Jupyter notebook, with results exported as pandas DataFrames in the form of .pkl files. 
The second component of this project is to provide a user-friendly Python module to visualize the data and predictions. The goal is for an user to be able to rapidly 
screen through the high schools and quickly identify under or overperforming high schools for future case studies. 

## Data Sources
Data comes from various sources, as linked in [Links to Data](/Links%20to%20Data) and take the form of either demographic information about the graduating cohort, 
class sizes and types, socioeconomic data on the student population as a whole and the surrounding neighborhood, or salaries of teachers and administrators. 
Metrics that are provided by the CA Department of Education include, but are not limited to, the graduation rate of a specific cohort and the UC/CSU preparation rate (abbreviated
as the college preparation rate). 

## Data Analysis
Two types of machine learning approaches are utilized in the analysis. 
### Random Forest Regressors
To predict the expected value of the metric, a Random Forest Regressor is fitted with a subset of the data, then used to predict on the unfitted subset of the data. This procedure
is done again to come up with a predicted metric using a regressor that has not seen the test data. These predicted values are then compared to the actual, measured value for further analysis. 
### k-Nearest Neighbors
To identify which schools are considered similar in profile to each other, a Nearest Neighbors approach is taken. First, the feautures are scaled with a Quartile Scaler prior to
dimensionality reduction, which is accomplished via Principal Component Analysis. I selected a number of features such that 90% of the explained variance can be captured. 
Then, the distance between each school is calculated using the Nearest Neighbors algorithm with Eucledian distances, and sorted from shortest to longest distance to identify a list of 
schools that are similar to a target school. 

 
