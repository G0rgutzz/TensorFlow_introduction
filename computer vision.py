import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
# Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 2 parts of the data, first to train model, second to test it against new stuff
np.set_printoptions(linewidth=320)
# any number between 0 and 59999
index = 254
# prints label and image
print(f"LABEL: {train_labels[index]}")
print(f"\nIMAGE PIXEL ARRAY:\n {train_images[index]}")
# visualize the image
plt.imshow(train_images[index], cmap="Greys")
# plt.show()
# normalizing pixel values of the train and test images
train_images = train_images/255.0
test_images = test_images/255.0
# building classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# sequential - defines number of layers in the neural network
# flatten - turns 28x28 pixels matrix and turns it into 1D array
# dense - adds layer of neurons, activation tells them what to do
'''relu works like this
if x > 0:
    return x
else: 
    return 0
    
softmax - softmax(x) = np.exp(x)/sum(np.exp(x))
basically gives probability of x point in the array'''
# declare sample inputs and convert it to a tensor
inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f"input to softmax function: {inputs.numpy()}")

# feeding the inputs to a softmax activation function
outputs = tf.keras.activations.softmax(inputs)
print(f"output of the softmax function: {outputs.numpy()}")

# sum of all the values after softmax
sum1 = tf.reduce_sum(outputs)
print(f"sum of outputs: {sum1}")

# index with highiest value
prediction = np.argmax(outputs)
print(f"class with highiest probability: {prediction}")
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, callbacks=[callback])

# evaluating model using test data
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)

print(classifications[0])  # prints list of numbers that are basically probability that the image is
# an object with a given label description
print(test_labels[0])
