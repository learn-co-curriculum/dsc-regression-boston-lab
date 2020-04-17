
# Project - Regression Modeling with the Ames Housing Dataset

## Introduction

In this lab, you'll apply the regression analysis and diagnostics techniques covered in this section to the "Ames Housing" dataset. You performed a detailed EDA for this dataset earlier on, and hopefully, you more or less recall how this data is structured! In this lab, you'll use some of the features in this dataset to create a linear model to predict the house price!

## Objectives
You will be able to:
* Perform a linear regression using statsmodels
* Determine if a particular set of data exhibits the assumptions of linear regression
* Evaluate a linear regression model by using statistical performance metrics pertaining to overall model and specific parameters
* Use the coefficient of determination to determine model performance
* Interpret the parameters of a simple linear regression model in relation to what they signify for specific data


## Let's get started

### Import necessary libraries and load 'ames.csv' as a pandas dataframe


```python
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')
ames = pd.read_csv('ames.csv')

subset = ['YrSold', 'MoSold', 'Fireplaces', 'TotRmsAbvGrd', 'GrLivArea',
          'FullBath', 'YearRemodAdd', 'YearBuilt', 'OverallCond', 'OverallQual', 'LotArea', 'SalePrice']

data = ames.loc[:, subset]
```

The columns in the Ames housing data represent the dependent and independent variables. We have taken a subset of all columns available to focus on feature interpretation rather than preprocessing steps. The dependent variable here is the sale price of a house `SalePrice`. The description of the other variables is available on [KAGGLE](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). 

### Inspect the columns of the dataset and comment on type of variables present


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YrSold</th>
      <th>MoSold</th>
      <th>Fireplaces</th>
      <th>TotRmsAbvGrd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>YearRemodAdd</th>
      <th>YearBuilt</th>
      <th>OverallCond</th>
      <th>OverallQual</th>
      <th>LotArea</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>2</td>
      <td>0</td>
      <td>8</td>
      <td>1710</td>
      <td>2</td>
      <td>2003</td>
      <td>2003</td>
      <td>5</td>
      <td>7</td>
      <td>8450</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>1262</td>
      <td>2</td>
      <td>1976</td>
      <td>1976</td>
      <td>8</td>
      <td>6</td>
      <td>9600</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008</td>
      <td>9</td>
      <td>1</td>
      <td>6</td>
      <td>1786</td>
      <td>2</td>
      <td>2002</td>
      <td>2001</td>
      <td>5</td>
      <td>7</td>
      <td>11250</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2006</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>1717</td>
      <td>1</td>
      <td>1970</td>
      <td>1915</td>
      <td>5</td>
      <td>7</td>
      <td>9550</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
      <td>9</td>
      <td>2198</td>
      <td>2</td>
      <td>2000</td>
      <td>2000</td>
      <td>5</td>
      <td>8</td>
      <td>14260</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Record your observations here 

# The dataset contains a mix of continuous and categorical data
# OverallCond and OverallQual look to be ordered categoricals
# There are several time based columns in YrSold, MoSold, YearRemodAdd, and YearBuilt
```

### Create histograms for all variables in the dataset and comment on their shape (uniform or not?)


```python
data.hist(figsize=(18,15), bins='auto');
```


![png](index_files/index_6_0.png)



```python
# You observations here 

# LotArea, SalePrice, GrLivArea are all continuous and appear to be log normally distributed.
# Most values are bunched towards the lower end while there are a few very large values
# From the TotRmsAbvGrd feature it looks like most houses have around 6 rooms above ground
# We can see that there is an increase in the number of houses built as time goes on. Most houses sold were built in the 2000s
```

### Check the linearity assumption for all chosen features with target variable using scatter plots


```python
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(16,15), sharey=True)

for ax, column in zip(axes.flatten(), data.columns):
    ax.scatter(data[column], data['SalePrice'] / 100_000, label=column, alpha=.1)
    ax.set_title(f'Sale Price vs {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Sale Price in $100,000')

fig.tight_layout()
```


![png](index_files/index_9_0.png)


Clearly, your data needs a lot of preprocessing to improve the results. This key behind a Kaggle competition is to process the data in such a way that you can identify the relationships and make predictions in the best possible way. For now, we'll use the dataset untouched and just move on with the regression. The assumptions are not _exactly_ all fulfilled, but they still hold to a level that we can move on. 

### Let's do Regression 

Now, let's perform a number of simple regression experiments between the chosen independent variables and the dependent variable (price). You'll do this in a loop and in every iteration, you should pick one of the independent variables. Perform the following steps:

* Run a simple OLS regression between independent and dependent variables
* Plot the residuals using `sm.graphics.plot_regress_exog()`
* Plot a Q-Q plot for regression residuals normality test 
* Store following values in array for each iteration:
    * Independent Variable
    * r_squared'
    * intercept'
    * 'slope'
    * 'p-value'
    * 'normality (JB)' 
* Comment on each output 


```python
# import libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import statsmodels.stats.api as sms


results = []
for idx, column in enumerate(data.columns):
    print (f"Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~{column}")
    print ("-------------------------------------------------------------------------------------")

    f = f'SalePrice~{column}'
    model = smf.ols(formula=f, data=data).fit()
    
    fig, axes = plt.subplots(figsize=(15,12))
    fig = sm.graphics.plot_regress_exog(model, column, fig=fig)
    fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)
    fig.tight_layout()
    plt.show()
    
    results.append([column, model.rsquared, model.params[0], model.params[1], model.pvalues[1], sms.jarque_bera(model.resid)[0]])
    input("Press Enter to continue...")
```

    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~YrSold
    -------------------------------------------------------------------------------------



![png](index_files/index_11_1.png)



![png](index_files/index_11_2.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~MoSold
    -------------------------------------------------------------------------------------



![png](index_files/index_11_4.png)



![png](index_files/index_11_5.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~Fireplaces
    -------------------------------------------------------------------------------------



![png](index_files/index_11_7.png)



![png](index_files/index_11_8.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~TotRmsAbvGrd
    -------------------------------------------------------------------------------------



![png](index_files/index_11_10.png)



![png](index_files/index_11_11.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~GrLivArea
    -------------------------------------------------------------------------------------



![png](index_files/index_11_13.png)



![png](index_files/index_11_14.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~FullBath
    -------------------------------------------------------------------------------------



![png](index_files/index_11_16.png)



![png](index_files/index_11_17.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~YearRemodAdd
    -------------------------------------------------------------------------------------



![png](index_files/index_11_19.png)



![png](index_files/index_11_20.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~YearBuilt
    -------------------------------------------------------------------------------------



![png](index_files/index_11_22.png)



![png](index_files/index_11_23.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~OverallCond
    -------------------------------------------------------------------------------------



![png](index_files/index_11_25.png)



![png](index_files/index_11_26.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~OverallQual
    -------------------------------------------------------------------------------------



![png](index_files/index_11_28.png)



![png](index_files/index_11_29.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~LotArea
    -------------------------------------------------------------------------------------



![png](index_files/index_11_31.png)



![png](index_files/index_11_32.png)


    Press Enter to continue...
    Ames Housing DataSet - Regression Analysis and Diagnostics for SalePrice~SalePrice
    -------------------------------------------------------------------------------------



![png](index_files/index_11_34.png)



![png](index_files/index_11_35.png)


    Press Enter to continue...



```python
pd.DataFrame(results, columns=['ind_var', 'r_squared', 'intercept', 'slope', 'p-value', 'normality (JB)' ])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ind_var</th>
      <th>r_squared</th>
      <th>intercept</th>
      <th>slope</th>
      <th>p-value</th>
      <th>normality (JB)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YrSold</td>
      <td>0.000837</td>
      <td>3.654560e+06</td>
      <td>-1730.058729</td>
      <td>2.694132e-01</td>
      <td>3432.757805</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MoSold</td>
      <td>0.002156</td>
      <td>1.722959e+05</td>
      <td>1364.350502</td>
      <td>7.612758e-02</td>
      <td>3588.247231</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fireplaces</td>
      <td>0.218023</td>
      <td>1.456485e+05</td>
      <td>57539.831838</td>
      <td>6.141487e-80</td>
      <td>3092.993348</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TotRmsAbvGrd</td>
      <td>0.284860</td>
      <td>1.089647e+04</td>
      <td>26086.180847</td>
      <td>2.772281e-108</td>
      <td>2240.440266</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GrLivArea</td>
      <td>0.502149</td>
      <td>1.856903e+04</td>
      <td>107.130359</td>
      <td>4.518034e-223</td>
      <td>3432.286565</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FullBath</td>
      <td>0.314344</td>
      <td>5.438828e+04</td>
      <td>80848.166787</td>
      <td>1.236470e-121</td>
      <td>3515.004761</td>
    </tr>
    <tr>
      <th>6</th>
      <td>YearRemodAdd</td>
      <td>0.257151</td>
      <td>-3.692146e+06</td>
      <td>1951.299406</td>
      <td>3.164948e-96</td>
      <td>5931.459064</td>
    </tr>
    <tr>
      <th>7</th>
      <td>YearBuilt</td>
      <td>0.273422</td>
      <td>-2.530308e+06</td>
      <td>1375.373468</td>
      <td>2.990229e-103</td>
      <td>6856.947031</td>
    </tr>
    <tr>
      <th>8</th>
      <td>OverallCond</td>
      <td>0.006062</td>
      <td>2.119096e+05</td>
      <td>-5558.115361</td>
      <td>2.912351e-03</td>
      <td>3406.240879</td>
    </tr>
    <tr>
      <th>9</th>
      <td>OverallQual</td>
      <td>0.625652</td>
      <td>-9.620608e+04</td>
      <td>45435.802593</td>
      <td>2.185675e-313</td>
      <td>5872.097631</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LotArea</td>
      <td>0.069613</td>
      <td>1.588362e+05</td>
      <td>2.099972</td>
      <td>1.123139e-24</td>
      <td>3374.002795</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SalePrice</td>
      <td>1.000000</td>
      <td>3.365130e-11</td>
      <td>1.000000</td>
      <td>0.000000e+00</td>
      <td>9028.040730</td>
    </tr>
  </tbody>
</table>
</div>



Clearly, the results are not very reliable. The best R-Squared is witnessed with `OverallQual`, so in this analysis, this is our best predictor. 

### How can you improve these results?
1. Preprocessing 

This is where the preprocessing of data comes in. Dealing with outliers, normalizing data, scaling values etc. can help regression analysis get more meaningful results from the given data. 

2. Advanced Analytical Methods

Simple regression is a very basic analysis technique and trying to fit a straight line solution to complex analytical questions may prove to be very inefficient. Later on, you'll explore multiple regression where you can use multiple features **at once** to define a relationship with the outcome. You'll also look at some preprocessing and data simplification techniques and revisit the Ames dataset with an improved toolkit. 

## Level up - Optional 

Apply some data wrangling skills that you have learned in the previous section to pre-process the set of independent variables we chose above. You can start off with outliers and think of a way to deal with them. See how it affects the goodness of fit. 

## Summary 

In this lab, you applied your skills learned so far on a new data set. You looked at the outcome of your analysis and realized that the data might need some preprocessing to see a clear improvement in the results. You'll pick this back up later on, after learning about more preprocessing techniques and advanced modeling techniques.
