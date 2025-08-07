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
history = model.fit(x_train,y_train, epochs=3, batch_size=32,validation_data=(x_test, y_test))


#visualization 
#loss curve on training and testing 
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')  
plt.title('Loss Curve')
plt.xlabel('Epochs')        
plt.ylabel('Loss')
plt.legend()
plt.show()

#Accuracy curve on training and testing 
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')  
plt.title('Accuracy Curve')
plt.xlabel('Epochs')        
plt.ylabel('Accuracy')
plt.legend()
plt.show()
