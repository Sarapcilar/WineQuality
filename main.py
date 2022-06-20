import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback 
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

red = pd.read_csv("data/winedata_red.csv", sep=',') # red wine data
white = pd.read_csv("data/winedata_white.csv", sep=',') # white wine data


red['type'] = 1
white['type'] = 0
wines = red.append(white, ignore_index = True)
X = wines.drop("quality", axis=1)
y = np.ravel(wines.quality)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34, random_state = 45)

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('mse')<0.01):
            self.model.stop_training = True
callbacks = myCallback()

from keras.models import Sequential
 

from keras.layers import Dense

model = Sequential()
 
# Add an input layer
model.add(Dense(12, activation ='relu', input_shape =(12, )))
 
# Add one hidden layer
model.add(Dense(24, activation ='relu'))
model.add(Dense(48, activation ='relu'))
model.add(Dense(96, activation ='relu'))
model.add(Dense(256, activation ='relu'))
model.add(Dense(128, activation ='relu'))
model.add(Dense(64, activation ='relu'))
# Add an output layer
model.add(Dense(1))
 

print(model.summary())
print(model.get_config())
print(model.get_weights())
model.compile(loss ='mse', optimizer ='adam', metrics =['mse'])


r = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 400, callbacks=[callbacks])


plt.plot(r.history['mse'], label='mse')
plt.plot(r.history['val_mse'], label='val_mse')
plt.legend()
plt.savefig('figures/graph.png')
plt.show()


y_pred = model.predict(X_test)
print((y_pred.round()))