import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the data

# Get current working directory
current_dir = os.getcwd()
path = f"{os.getcwd()}/../tmp2/mnist.npz"
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):  # same possible for loss and accuracy
        if logs.get('accuracy') >= 99.5:  # Experiment with changing this value
            print("\nReached 99,5% accuracy so cancelling training!")
            # print("\nLoss is below 0,4 so cancelling training!")
            self.model.stop_training = True

def reshape_and_normalize(images):
    # Reshape the images to add an extra dimension
    images = images.reshape(60000, 28, 28, 1)
    # Normalize pixel values
    images = images/255.0

    return images


# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")
callback = myCallback()
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # 64 filters created, each shape 3x3, activation is relu - discards values < 0, input shape 28x28, 1 byte per color
    tf.keras.layers.MaxPooling2D(2, 2),
    # max size of pooling matrix
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # another set of convolutions on top of existing ones and again reduce the size
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
# test_loss = model.evaluate(test_images, test_labels)
# print(test_loss)