# Multiple Linear Regression on 50 Startups Dataset

This repository implements a Multiple Linear Regression model to predict the profit of startups based on R&D Spend, Administration, Marketing Spend, and State using the `50_Startups.csv` dataset.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Modeling](#modeling)
- [Results](#results)
- [Backward Elimination](#backward-elimination)
- [License](#license)

## Installation

To run this project locally, you will need to have Python installed along with the following packages:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float64)
x = x[:, 1:]  # Avoiding the dummy variable trap
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
import statsmodels.api as sm

x = np.append(arr=np.ones((x.shape[0], 1)).astype(int), values=x, axis=1)
x_opt = x[:, [0, 3]]  # Select statistically significant variables
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.947
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     849.8
Date:                Sun, 13 Oct 2024
Time:                        06:47:05
No. Observations:                  50
AIC:                             1059.
Df Residuals:                      48
BIC:                             1063.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
==============================================================================
Omnibus:                       13.727   Durbin-Watson:                   1.116
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               18.536
Skew:                          -0.911   Prob(JB):                     9.44e-05
Kurtosis:                       5.361   Cond. No.                     1.65e+05
==============================================================================

This `README.md` provides an overview of the project, instructions for installation, an explanation of the modeling approach, and a summary of the results.
