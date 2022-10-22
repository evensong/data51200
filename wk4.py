# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

data = pd.read_excel('/home/evensong/Downloads/HBAT(8).xls')
trim_data = data[['x6', 'x7', 'x9', 'x11', 'x12', 'x16']]
target = data['x19']

X_train, X_test, y_train, y_test = train_test_split(trim_data, target, test_size=0.33)
linreg = LinearRegression()

linreg.fit(X_train, y_train)
score = linreg.score(X_test, y_test)
linreg.get_params()
linreg.coef_
intercept = linreg.intercept_

trim_data = sm.add_constant(trim_data)
smlinreg = sm.OLS(endog = target, exog = trim_data, hasconst=True)
smresults = smlinreg.fit()
smresults.summary()

sns.pairplot(trim_data.loc[:, trim_data.columns != 'const'], kind = 'scatter', diag_kind = 'kde')