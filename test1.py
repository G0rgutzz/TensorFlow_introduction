import keras
import numpy as np
import tensorflow as tf

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
# loss function estimates how good initial guess is and gives it to the optimizer function
# function improves based on that
# sgd - stochastic gradient descent
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=5000)
# used to train model with xs and ys, epochs means that model will do it 500 times
print(model.predict([10.0]))
