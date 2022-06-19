import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import * 
from tensorflow.keras.models import Sequential

dataframe = pd.read_csv("data/winedata_red.csv")

x = dataframe.iloc[:, 0:11]
y = dataframe.iloc[:, 11]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.34, random_state = 45)

#val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
#train_dataframe = dataframe.drop(val_dataframe.index)

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>=0.975):
            self.model.stop_training = True
callbacks = myCallback()

#creating the model

model = Sequential()
model.add(Dense(11, activation ='relu', input_shape =(11, )))
model.add(Dense(24, activation ='relu'))
model.add(Dense(48, activation ='relu'))
model.add(Dense(12, activation ='relu'))
model.add(Dense(1, activation ='sigmoid'))


print(model.summary())

print(model.get_config())
 
print(model.get_weights())
model.compile(loss ='binary_crossentropy', optimizer ='adam', metrics =['acc'])

r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 30, callbacks=[callbacks],
           batch_size = 4, verbose = 1)


plt.plot(r.history['acc'], label='accuracy')
plt.plot(r.history['val_acc'], label='val_accuracy')
plt.legend()
plt.savefig('graph.png')
plt.show()