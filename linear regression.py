# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:56:38 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


wc_at=pd.read_csv("wc-at.csv")
wc_at.corr()

#scatter plot

plt.scatter(wc_at['Waist'],wc_at['AT'],c='r');plt.xlabel('WAIST');plt.ylabel('AT')

#calculate correlation coefficient
wc_at.AT.corr(wc_at.Waist)
#(or)
np.corrcoef(wc_at.Waist,wc_at.AT)

#Linear Model
import statsmodels.formula.api as smf
model=smf.ols('AT~Waist',data=wc_at).fit()
model.params
model.summary()
#will check for p-value<0.05 and high R-squared value to be a good model,if not go for transformation
#confidence interval of parameters intercept and slope
model.conf_int(0.05)
#predicted values
pred=model.predict(wc_at)
pred
#Error
Error=wc_at['AT']-pred
Error
#Entire data
final_data=pd.concat([wc_at,pred,Error],axis=1)
final_data.columns
final_data.columns=['Waist','AT','predicted','Error']
final_data
# scatter plot with linear model
plt.scatter(final_data['Waist'],final_data['AT'],c='r');plt.plot(final_data['Waist'],final_data['predicted'],c='b');plt.xlabel('WAIST');plt.ylabel('AT');
#calculating root mean square error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(final_data.AT,final_data.predicted))
rmse
#rmse=32.76
np.corrcoef(wc_at.AT,pred)
#corr coeff=0.81


####################Log Transformation###################
model2=smf.ols('AT~np.log(Waist)',data=wc_at).fit()
model2.params
model.summary()
pred2=model2.predict(wc_at)
pred2
Error2=wc_at['AT']-pred2
Error2
plt.scatter(wc_at['Waist'],wc_at['AT'],c='r');plt.plot(wc_at['Waist'],pred2,c='g');plt.xlabel('WAIST');plt.ylabel('AT')
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse2=sqrt(mean_squared_error(wc_at.AT,pred2))
rmse2
#rmse=32.49
np.corrcoef(wc_at.AT,pred2)
#corr coeff = 0.821 

##########################Exponential Transformation###########################
model3=smf.ols('np.log(AT)~Waist',data=wc_at).fit()
model3.params
model3.summary()
pred3_log=model3.predict(wc_at)
pred3=np.exp(pred3_log)
pred3
Error3=wc_at.AT-pred3
Error3
plt.scatter(wc_at['Waist'],wc_at['AT'],c='g');plt.plot(wc_at['Waist'],pred3,c='b');plt.xlabel('WAIST');plt.ylabel('AT')
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse3=sqrt(mean_squared_error(wc_at['AT'],pred3))
rmse3
#rmse=38.52
np.corrcoef(wc_at.AT,pred3)
#corr coeff=0.76 
plt.scatter(pred3,wc_at.AT,c='y');plt.xlabel('predicted');plt.ylabel('Actual')


#################Quadratic Transformation########################
#let the Quadratic Eq be x*x+x
wc_at['Waist_sq']=wc_at['Waist']*wc_at['Waist']
model_quad=smf.ols('AT~Waist_sq+Waist',data=wc_at).fit()
model_quad.params
model_quad.summary()
pred4=model_quad.predict(wc_at)
pred4
Error4=wc_at.AT-pred4
Error4
plt.scatter(wc_at['Waist'],wc_at['AT'],c='y');plt.plot(wc_at['Waist'],pred4,c='r');plt.xlabel('WAIST');plt.ylabel('AT')
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse4=sqrt(mean_squared_error(wc_at.AT,pred4))
rmse4
#rmse=32.36
np.corrcoef(wc_at.AT,pred4)
#corr coeff=0.82
plt.scatter(pred4,wc_at.AT,c='y');plt.xlabel('predicted');plt.ylabel('Actual')
#Analysed_data=pd.concat([wc_at.Waist,wc_at.AT,pred,pred2,pred3,pred4,Error,Error2,Error3,Error4],axis=1)
#Analysed_data.columns
#Analysed_data.columns=['Waist','AT','Prediction','log prediction','exp prediction','Quad prediction','Error','log error','exp error','Quad error']


