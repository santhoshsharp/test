
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib


X = np.array([10, 20, 30, 40, 50, 10, 25, 30, 45, 50, 15, 20, 35, 40, 50, 50, 45, 30, 25, 20, 10])
Y = np.array([95, 185, 280, 370, 490, 100, 230, 290, 410, 500, 135, 200, 295, 395, 495, 480, 430, 305, 205, 175, 110])

data = pd.DataFrame({'Ads/Month':X, 'Paid/Month':Y})
data.head()
model = LinearRegression().fit(X.reshape(-1, 1), Y)
filename = "model.sav"
joblib.dump(model, filename)

loaded_model = joblib.load(filename)
loaded_model.predict([[20]])