#To implement multi layer perceptron on MNIST dataset using keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical  


#Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#preprocess the data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Build the architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the input
model.add(Dense(units=10, activation='softmax'))  # Output layer with softmax activation

#compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Build the model
history = model.fit(x_train,y_train, epochs=10, batch_size=32,validation_data=(x_test, y_test))
print(history.history.keys())
print(history.history.items())

#Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

#visualization 
#loss curve on training and testing 