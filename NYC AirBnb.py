# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 22:19:39 2019

@author: neelabh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

raw_df = pd.read_csv("AB_NYC_2019.csv")
new_df = raw_df.drop(["id","name","host_name","host_id","calculated_host_listings_count","last_review","reviews_per_month","latitude","longitude"], axis=1)
new_df['room_type'] = new_df['room_type'].map({'Shared room':1,'Private room':2,'Entire home/apt':3})
n_gp = list(new_df.neighbourhood_group.unique())
neighbrs_gp = new_df["neighbourhood_group"].str.get_dummies("EOL")
neighbrs = new_df["neighbourhood"].str.get_dummies("EOL")
new_df1 = pd.concat([new_df,neighbrs_gp,neighbrs],axis=1)
new_df1 = new_df1.drop(["neighbourhood_group", "neighbourhood"],axis=1)
new_df1["rate"] = new_df1["price"]/new_df1["minimum_nights"]
new_df["rate"] = new_df["price"]/new_df["minimum_nights"]

'''Data visualisation'''
#Which are the busiest areas?

crowd=[]
for i in (n_gp):
    crowd.append(new_df1[i].sum())
i=0
plt.ylim(0,25000)  
plt.bar(n_gp, crowd)
plt.xticks(rotation=45)
plt.title('Crowd movement in NYC')
plt.xlabel('Neighbourhood Groups') 
plt.ylabel('No. of visitors')
plt.show()

#Expensive neighbourhoods of NYC

costly = new_df.groupby('neighbourhood_group')['rate'].mean() 
plt.ylim(0,100)  
plt.bar(costly.index, costly)
plt.xticks(rotation=45)
plt.title('Mean rates of NYC areas')
plt.xlabel('Neighbourhood Groups') 
plt.ylabel('Mean rates')
plt.show()

#get nghbrhds

nghbr = list(new_df.neighbourhood.unique())

#Model training
model_df = new_df1.drop(['price','rate','minimum_nights','availability_365', 'number_of_reviews'], axis=1)
train_x, train_y, test_x, test_y = train_test_split(model_df, new_df1.loc[:, ('rate')], test_size=0.01, random_state=2)

linear_reg = LinearRegression()
X, Y = train_x.transpose(), train_y.transpose()
linear_reg.fit(X, Y)
pred_y = linear_reg.predict([test_x.transpose()])
rms = np.sqrt(mean_squared_error(test_y, pred_y.transpose()))
print('RMSE Value: ', rms)

