import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import lognorm
from statsmodels.distributions.empirical_distribution import ECDF
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.initializers import he_normal

#earthquake data
quakes = pd.read_csv("earthquake.csv", sep=r'\s*,\s*', header=0, index_col=False, engine='python').fillna(0)
quakes = pd.DataFrame(quakes)
#print(quakes.columns.tolist())# knowing columns' names
quakes['Date'] = pd.to_datetime(quakes['Date'])
quakes['year'] = quakes['Date'].dt.year
quakes['t'] = quakes.age
epicenter = quakes.epicenter.drop_duplicates()
count1 = quakes['epicenter'].value_counts()

"""
#data visualization
#bar plot of years vs earthquake frequency
quakesperyear = pd.crosstab(index=quakes['year'], columns="count")
quakesperyear.plot(kind='barh',figsize=(10,5))
plt.xlabel("No of Earthquakes")
plt.ylabel("year")
plt.legend()
plt.title("Frequency of Earthquakes per Year")
plt.grid(False)
plt.show()

quakesperepicenter = pd.crosstab(index=quakes['epicenter'], columns="count")
quakesperepicenter.plot(kind='barh', figsize=(10,5))
plt.xlabel("No of Earthquakes")
plt.ylabel("Epicenter")
plt.legend()
plt.title("Frequency of Earthquakes per Epicenter")
plt.grid(False)
plt.show()

quakes.hist(column="magnitude", figsize=(10,5), color='black', range=(1,7))
plt.xlabel('Frequency of earthquake')
plt.ylabel('Magnitude')
plt.title('Histogram for Magnitude')
plt.grid(False)
plt.show()

quakes.boxplot(column="magnitude")
plt.title('Boxplot of Magnitude')
plt.grid(False)
plt.show()

plt.plot(quakes['magnitude'], color='black', marker='o')
plt.title('Distribution of Magnitude', fontsize=14)
plt.xlabel('Instances')
plt.ylabel('Earthquake Magnitude', fontsize=12)
plt.grid(False)
plt.show()
"""
"""
#checking for count of missing values
#quakes.loc[quakes["depth"] == 0.0, "depth"] = np.NAN
#print(quakes.isnull().sum()[0:7])
#stochastic regression imputation
X = quakes.iloc[:, 1].values.reshape(-1, 1)  # values converts it into a numpy array
Y = quakes.iloc[:, 2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
Y_pred = Y_pred.tolist()
print(Y_pred)
"""
"""
#descriptive statistics
#print(round(quakes.describe().transpose(), 4))
#skewness = quakes.skew(axis=0, skipna=True)#skewness
#kurtosis = quakes.kurt(axis=0, skipna=True)#Kurtosis
"""
"""
#Estimating lognormal distribution
t = np.array(quakes['age'])
y = ECDF(t)
#pars1 = lognorm.fit(t)
pars2 = lognorm.fit(t, floc=0)
sigma, loc, scale = lognorm.fit(t, floc=0)
#X = np.linspace(min(t), max(t), 76)
#plt.plot(y.x, y.y, 'ro')
#plt.plot(X, lognorm.cdf(X, pars1[0], pars1[1], pars1[2]), 'b-')
#plt.plot(X, lognorm.cdf(X, pars2[0], pars2[1], pars2[2]), 'g-')
#plt.show()
print(round(np.log(lognorm.pdf(t, sigma)).sum(), 4))
print(round(sigma, 4))
"""
#data pre-processing-data normalization i.e x has range [0,1]
quakes = quakes.drop(columns=["epicenter", "Date", "depth", "age", "year"], axis=1)#dropping unnecessary columns
quakes = quakes[['magnitude', 't', 'latitude', 'longitude']]#rearranging columns
col_names = list(quakes.columns)
#splitting data for training and testing
X = quakes.drop('magnitude', axis=1)
y = quakes['magnitude']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#data normalization
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))

#ANN plus He initialization
#defining the keras model
def build_and_compile_model(normalizer):
  model = keras.models.Sequential([
      normalizer,
      layers.Dense(20, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
      layers.Dense(10, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
      layers.Dense(1, activation='linear')

  ])

  model.compile(loss='mse', optimizer='sgd')
  return model

quakes_model = build_and_compile_model(normalizer)

history = quakes_model.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=32, shuffle=True,
                           verbose=1)

#predictions
yhat = quakes_model.predict(X_test).flatten()
r2 = r2_score(y_test, yhat)
mae = mean_absolute_error(y_test, yhat)#mean absolute error
mse = mean_squared_error(y_test, yhat)#mean square error
rmse = np.sqrt(mse)#root mean square error for model assessment
print('MAE: %.3f' % mae)
print('MSE: %.3f' % mse)
print('R squared: %.3f' %r2)
#saving the model and reloading it
#quakes_model.save('quakes_model')
#reloaded = tf.keras.models.load_model('quakes_model')

def plot_loss(history):
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.title('Loss Curve Plot')
  #plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.legend()
  plt.grid(False)
  plt.show()
plot_loss(history)#loss curve plot for model diagnostics
'''
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.title('Model Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    #plt.ylim([0, 10])
    plt.grid(False)
    #plt.show()
#plot_accuracy(history)#accuracy curve
'''
'''
a = plt.axes(aspect='equal')
plt.scatter(y_test, yhat, label='R squared: %.3f' %r2)
plt.title('Dataset 5')
plt.xlabel('Observed values')
plt.ylabel('Predicted values')
lims = [1, 8]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
_ = plt.plot(lims, lims)
plt.show()
'''
'''
error = yhat - y_test
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
_ = plt.ylabel('Count')
plt.show()
'''
