import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):  # same possible for loss and accuracy
        if logs.get('loss') <= 0.4:  # Experiment with changing this value
            print("\nLoss is below 0,4 so cancelling training!")
            self.model.stop_training = True
            # Args:
            # epoch (integer) - index of epoch (required but unused in the function definition below)
            # logs (dict) - metric results from the training epoch


callback = myCallback()
# Load the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0  # same thing as testimages/255
np.set_printoptions(linewidth=320)
# any number between 0 and 59999
# index = 254
# Defining model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# Compile the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, callbacks=[callback])
