# Multiple Linear Regression with One-Hot Encoding

This project implements multiple linear regression using a dataset of 50 startups to predict profit based on various features. The code includes the use of one-hot encoding to handle categorical data (State), as well as a backward elimination process using **Ordinary Least Squares (OLS)** regression to optimize the model.

## Project Overview

In this project, you will find:
1. Importing and preprocessing of data (handling categorical variables using one-hot encoding).
2. Building a multiple linear regression model to predict the profit of startups.
3. Performing backward elimination to remove insignificant variables.
4. Splitting the data into training and test sets, and evaluating model performance.

## Dataset

The dataset used for this project is `50_Startups.csv`, which contains the following columns:
- R&D Spend
- Administration
- Marketing Spend
- State (categorical)
- Profit (target variable)

## Steps in the Code

1. **Data Import and Preprocessing**  
   - The dataset is loaded using `pandas`.
   - The categorical variable (State) is handled using one-hot encoding.
   - To avoid the dummy variable trap, one of the encoded variables is removed.

2. **Splitting the Dataset**  
   - The dataset is split into training and test sets using `train_test_split` from `sklearn`.

3. **Building the Multiple Linear Regression Model**  
   - A multiple linear regression model is fitted on the training data using `LinearRegression` from `sklearn`.
   - The model is evaluated on both the training and test sets.

4. **Backward Elimination (Feature Selection)**  
   - The backward elimination process is implemented using `statsmodels` to identify statistically significant features based on p-values.

5. **Model Evaluation**  
   - The performance of the model is evaluated using the R-squared score on both training and test sets.
   - The OLS regression results from the backward elimination are displayed.

## Requirements

To run the code, you need the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels




