import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras import models


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):  # same possible for loss and accuracy
        if logs.get('loss') <= 0.4:  # Experiment with changing this value
            # print("\nReached 95% accuracy so cancelling training!")
            print("\nLoss is below 0,4 so cancelling training!")
            self.model.stop_training = True
    # Args:
    # epoch (integer) - index of epoch (required but unused in the function definition below)
    # logs (dict) - metric results from the training epoch


callback = myCallback()
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # 64 filters created, each shape 3x3, activation is relu - discards values < 0, input shape 28x28, 1 byte per color
    tf.keras.layers.MaxPooling2D(2, 2),
    # max size of pooling matrix
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # another set of convolutions on top of existing ones and again reduce the size
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.summary()  # allows to inspect layers of the model
# !!!! keep the eye on the output shape column !!!!
# conv2d shape is 26x26 because 1 pixel near the border is removed due to function being unable to perform there
# maxpooling makes 2x2 pixels which reduces the size of the image to 13x13
# conv2d again slices 1 pixels of each side - 11x11 matrix
# maxpooling makes 2x2 pixels - image is now 5x5, 64 filters means there is 64 images of 5x5 that are fed to the nn
# flatten layer has 1600 elements - 5x5x64
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)
print(test_loss)



