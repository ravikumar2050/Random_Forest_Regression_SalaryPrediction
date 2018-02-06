#Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Fitting the model
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#Predicting the value
y_pred=regressor.predict(6.5)

#Visualizing the plot
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.show()