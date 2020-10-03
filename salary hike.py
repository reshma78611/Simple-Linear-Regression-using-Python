# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:29:09 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sal_hike=pd.read_csv('C:/Users/HP/Desktop/assignments submission/simple linear regression/Salary_Data.csv')
plt.scatter(sal_hike.YearsExperience,sal_hike.Salary)
np.corrcoef(sal_hike.YearsExperience,sal_hike.Salary)
#r=0.978
sal_hike.corr()


##################Linear Model######################
import statsmodels.formula.api as smf
lin_model=smf.ols('sal_hike.Salary~sal_hike.YearsExperience',data=sal_hike).fit()
lin_model.params
lin_model.summary()
#will check for p-value<0.05 and high R-squared value(0.957) to be a good model,if not go for transformation
lin_predict=lin_model.predict(sal_hike)
lin_predict
lin_Error=sal_hike.Salary-lin_predict
lin_Error
plt.scatter(sal_hike.YearsExperience,sal_hike.Salary,c='r');plt.plot(sal_hike.YearsExperience,lin_predict,c='b');plt.xlabel('years of experience');plt.ylabel('salary')
lin_model.conf_int()
np.corrcoef(sal_hike.Salary,lin_predict)
#r=0.978
from sklearn.metrics import mean_squared_error
from math import sqrt
lin_rmse=sqrt(mean_squared_error(sal_hike.Salary,lin_predict)) 
lin_rmse
#rmse=5592, R_sq=0.957

###################Log Model##################
import statsmodels.formula.api as smf
log_model=smf.ols('sal_hike.Salary~np.log(sal_hike.YearsExperience)',data=sal_hike).fit()
log_model.params
log_model.summary()
#will check for p-value<0.05 and high R-squared value(0.854) to be a good model,if not go for transformation
log_predict=log_model.predict(sal_hike)
log_predict
log_Error=sal_hike.Salary-log_predict
log_Error
plt.scatter(sal_hike.YearsExperience,sal_hike.Salary,c='r');plt.plot(sal_hike.YearsExperience,log_predict,c='b');plt.xlabel('years of experience');plt.ylabel('salary')
log_model.conf_int()
np.corrcoef(sal_hike.Salary,log_predict)
#r=0.924
from sklearn.metrics import mean_squared_error
from math import sqrt
log_rmse=sqrt(mean_squared_error(sal_hike.Salary,log_predict)) 
log_rmse
#rmse=10302.89, R_sq=0.854

###################Exponential Model##################
import statsmodels.formula.api as smf
Exp_model=smf.ols('np.log(sal_hike.Salary)~sal_hike.YearsExperience',data=sal_hike).fit()
Exp_model.params
Exp_model.summary()
#will check for p-value<0.05 and high R-squared value(0.932) to be a good model,if not go for transformation
pred=Exp_model.predict(sal_hike)
Exp_predict=np.exp(pred)
Exp_predict
Exp_Error=sal_hike.Salary-Exp_predict
Exp_Error
plt.scatter(sal_hike.YearsExperience,sal_hike.Salary,c='r');plt.plot(sal_hike.YearsExperience,Exp_predict,c='b');plt.xlabel('years of experience');plt.ylabel('salary')
Exp_model.conf_int()
np.corrcoef(sal_hike.Salary,Exp_predict)
#r=0.966
from sklearn.metrics import mean_squared_error
from math import sqrt
Exp_rmse=sqrt(mean_squared_error(sal_hike.Salary,Exp_predict)) 
Exp_rmse
#rmse=7213.23, R_sq=0.932

###################Quad Model##################
import statsmodels.formula.api as smf
#sal_hike['sq_exp']=sal_hike.YearsExperience*sal_hike.YearsExperience
#sal_hike.drop('sq_exp',axis=1,inplace=True)
#sal_hike
Quad_model=smf.ols('sal_hike.Salary~(sal_hike.YearsExperience*sal_hike.YearsExperience+sal_hike.YearsExperience)',data=sal_hike).fit()
#Quad_model=smf.ols('sal_hike.Salary~sal_hike.sq_exp+sal_hike.YearsExperience',data=sal_hike).fit()
Quad_model.params
Quad_model.summary()
#will check for p-value<0.05 and high R-squared value(0.957) to be a good model,if not go for transformation
Quad_predict=Quad_model.predict(sal_hike)
Quad_predict
Quad_Error=sal_hike.Salary-Quad_predict
Quad_Error
plt.scatter(sal_hike.YearsExperience,sal_hike.Salary,c='r');plt.plot(sal_hike.YearsExperience,Quad_predict,c='b');plt.xlabel('years of experience');plt.ylabel('salary')
Quad_model.conf_int()
np.corrcoef(sal_hike.Salary,Quad_predict)
#r=0.978
from sklearn.metrics import mean_squared_error
from math import sqrt
Quad_rmse=sqrt(mean_squared_error(sal_hike.Salary,Quad_predict)) 
Quad_rmse
#rmse=5592.04, R_sq=0.957

