import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#data
X = np.linspace(0,1,1000)
y = 5 * X + 7 + np.random.randn(1000) 
print(y)

#Build the architecure
model = Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='linear'))

#compile
model.compile(optimizer='sgd', loss='mean_squared_error')

#Build the model
res = model.fit(X, y, epochs=10)

#Predict
y_pred = model.predict(X)

#Plot the results
plt.scatter(X, y, label='Actual Data')
plt.plot(X, y_pred, color='red', label='Prediction')
plt.plot(res.history['loss'], label='Loss')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with neural network')
plt.legend()
plt.show()





