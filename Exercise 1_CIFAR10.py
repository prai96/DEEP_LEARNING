# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XHrBAphQoKU5rePLf6Vhh9Vjl2wkEhBu
"""

#Final Importing libraries
# baseline model on the cifar10 dataset
#USING SIGMOID ACTIVATION FUNCTION
import numpy
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import np_utils
from keras.constraints import maxnorm

seed = 21
(X_train,y_train), (X_test,y_test) = cifar10.load_data()

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train=X_train / 255.0
X_test=X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

#defining model
def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
  
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
 

 model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Flatten())
 model.add(Dropout(0.2))
 model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))
 

 model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))



 model.add(Dense(class_num))
 model.add(Activation('sigmoid'))
  

 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 return model

model = define_model()

numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 10 , batch_size=64) 

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

#Final Importing libraries
# baseline model on the cifar10 dataset
#USING elu ACTIVATION FUNCTION
import numpy
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import np_utils
from keras.constraints import maxnorm

seed = 21
(X_train,y_train), (X_test,y_test) = cifar10.load_data()

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train=X_train / 255.0
X_test=X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

#defining model
def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
  
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
 

 model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Flatten())
 model.add(Dropout(0.2))
 model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))
 

 model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))



 model.add(Dense(class_num))
 model.add(Activation('elu'))
  

 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 return model

model = define_model()

numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 10 , batch_size=64) 

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

#Final Importing libraries
# baseline model on the cifar10 dataset
#USING SELU ACTIVATION FUNCTION
import numpy
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import np_utils
from keras.constraints import maxnorm

seed = 21
(X_train,y_train), (X_test,y_test) = cifar10.load_data()

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train=X_train / 255.0
X_test=X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

#defining model
def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
  
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
 

 model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Flatten())
 model.add(Dropout(0.2))
 model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))
 

 model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))



 model.add(Dense(class_num))
 model.add(Activation('selu'))
  

 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 return model

model = define_model()

numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 10 , batch_size=64) 

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

#Final Importing libraries
# baseline model on the cifar10 dataset
#USING RELU ACTIVATION FUNCTION
import numpy
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import np_utils
from keras.constraints import maxnorm

seed = 21
(X_train,y_train), (X_test,y_test) = cifar10.load_data()

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train=X_train / 255.0
X_test=X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

#defining model
def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
  
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
 

 model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Flatten())
 model.add(Dropout(0.2))
 model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))
 

 model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))



 model.add(Dense(class_num))
 model.add(Activation('relu'))
  

 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 return model

model = define_model()

numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 10 , batch_size=64) 

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

#Final Importing libraries
# baseline model on the cifar10 dataset
#USING SELU ACTIVATION FUNCTION
import numpy
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import np_utils
from keras.constraints import maxnorm

seed = 21
(X_train,y_train), (X_test,y_test) = cifar10.load_data()

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train=X_train / 255.0
X_test=X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

#defining model
def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
  
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Dropout(0.2))
 

 model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
 model.add(Dropout(0.2))
 
 model.add(Flatten())
 model.add(Dropout(0.2))
 model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))
 

 model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))



 model.add(Dense(class_num))
 model.add(Activation('tanh'))
  

 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 return model

model = define_model()

numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 10 , batch_size=64) 

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))