import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
ys = np.array([1,1.5,2,2.5,3,3.5,4])
xs = np.array([1,2,3,4,5,6,7])
model.fit(xs,ys,epochs=500)
print(model.predict([7.0]))