# Identifying Outliers in California High Schools

## Overview
This project is designed to take various sources of data on the 2021 cohort of California's high schools and combine them in an user-friendly, easily accessible resource. 
The bulk of the data wrangling and analysis is done in a Jupyter notebook, with results exported as pandas DataFrames in the form of .pkl files. 
The second component of this project is to provide a user-friendly Python module to visualize the data and predictions. The goal is for an user to be able to rapidly 
screen through the high schools and quickly identify under or overperforming high schools for future case studies. 

## Data Sources
Data comes from various sources, as linked in [Links to Data](/Links%20to%20Data) and take the form of either demographic information about the graduating cohort, 
class sizes and types, socioeconomic data on the student population as a whole and the surrounding neighborhood, or salaries of teachers and administrators. 
Metrics that are provided by the CA Department of Education (CA DoE) include, but are not limited to, the graduation rate of a specific cohort and the UC/CSU preparation rate (abbreviated
as the college preparation rate). The data was joined on two primary keys, the County-District-School (CDS) code assigned by the state of California, and the National Center of Education Statistics (NCES) code
assigned by the US Department of Education. In certain cases, the NCES code may be out of date, due to changes in school district, which necessitates a secondary merge on NCES School code and School Names. 
Finally, incomplete geographic information was obtained via the Nominatim geocoding API. 

# Data Analysis
Two types of machine learning approaches are utilized in the analysis. 
### Random Forest Regressors
To predict the expected value of the metric, a Random Forest Regressor is fitted with fifty percent of the data, then used to predict on the unfitted subset of the data. Hyperparameter tuning on the
max depth, the number of features considered for a split, the number of samples considered for a given tree in the Random Forest, the number of samples at a leaf, and the number of trees in the forest
were all tuned using a cross-validated, grid search approach. 
Subsequently, the other half of the data is used to fit a second regressor, and the results are combined to come up with a predicted metric using a regressor that has not seen the test data. 
These predicted values are then compared to the actual, measured value for further analysis. 
As of this analysis, the graduation rate and the college preparation rate have been analyzed in this fashion.

To obtain feature importances from the Random Forest Regressors, features were first correlated through a correlation matrix and features that were shown to be highly correlated were removed
prior to fitting. This ensures that relative feature importances would not be divided among highly correlated features and provide a measure of importance in subsequent analysis. 
 
### k-Nearest Neighbors
To identify which schools are considered similar in profile to each other, a Nearest Neighbors approach is taken. First, the feautures are scaled with a Quartile Scaler prior to
dimensionality reduction, which is accomplished via Principal Component Analysis. I selected a number of features such that 90% of the explained variance can be captured (with $n = 22$ dimensions 
meeting this criteria). 
Then, the distance between each school is calculated using the Nearest Neighbors algorithm with Eucledian distances, and sorted from shortest to longest distance to identify a list of 
schools that are similar to a target school. 

### Analyzed Data
For appropriate demographic information, a decision was made to select only schools with more than 30 students in the graduating cohort, resulting in an observation set of 1440 high schools. This number was 
chosen due to how California reports its demographic information by not reporting any percentages of demographics that have less than 10 students. The choice of 30 students as a cutoff was to ensure that most
schools analyzed would at least have information on the gender demographics (male/female) as reported by the CA DoE. 

## Results
### Random Forest Regression Fit Quality
In the case of college preparation, the double Random Forest Regressor model yields a $R^2$ score of $R^2_{college} = 0.602$, which indicates moderate correlation between the predicted college preparation rates and
the expected college preparation rates. Outliers were identified as being one standard deviation $\sigma$ away from perfect agreement, with $\sigma_{college} = 15.47$. 

A similar analysis could be conducted on graduation rates, with $R^2_{graduation} = 0.370$ and $\sigma_{graduation} = 6.28$. This worse agreement could be due to the relatively higher degree of bunching for graduation rates 
at higher values. A similar procedure (1 standard deviation) can then be used to identify outliers in overperforming or underperforming schools. 

### Comparsion to Other Models
A Ridge Regression model was also used as comparision. The $R^2$ values for the Ridge Regression were lower when compared to the Random Forest Regressor model, at $R^2_{college} = 0.537$ and $R^2_{graduation} = 0.292$. 
However, the Ridge model provides not only feature importance, but the positive/negative correlation of a feature, which might make this feature appealing for subsequent analyses despite the worse overall fit. 
Another concern is that the different models could predict outliers in different directions (i.e., one model could predict an overperforming school but the second model could predict it as underperforming). To check this, 
the residuals of all schools for both Ridge Regression and Random Forest Regression were plotted on a scatter plot, as seen in 
![Scatter plot of residuals from linear regression and random forest regression](/Figures/LR_RF_College_residuals.png). Over 80% of the data points fall in the first and third quadrants, indicating that the sign of the residual
is the same, while the data points that fall in the second and fourth quadrants have relatively low magnitudes, indicating that outliers would not be contraindicated in the two models. 

### Feature Importance
Because an explanable model is used, we also have insights into the relative importance of the various features used in Random Forest Regression. 
![Graduation feature importance](/Figures/Graduation_Features.png)
![College preparation feature importance](/Figures/College_Prep_Features.png)



