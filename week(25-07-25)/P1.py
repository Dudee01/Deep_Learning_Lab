#To implement multi layer perceptron on MNIST dataset using keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.datasets import mnist

#Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape:", x_train.shape)

plt.imshow(x_train[0], cmap='gray')
plt.show()




