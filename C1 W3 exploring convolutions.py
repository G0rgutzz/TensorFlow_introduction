import tensorflow as tf
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

# it's basically a filter and I can set it however I want it to
# loading ascent image
ascent_image = misc.ascent()

# visualize the image
plt.grid(False)
plt.gray()
# plt.axis('off')
plt.imshow(ascent_image)
plt.show()

# copy image to numpy array
image_transformed = np.copy(ascent_image)

# get the dimensions of the image
size_x = image_transformed.shape[0]  # x coordinate
size_y = image_transformed.shape[1]  # y coordinate

# Experiment with different values and see the effect
# filter1 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
filter1 = [[2, 1, 3], [1, -2, 1], [-10, 1, 3]]
# filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
'''
# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
'''
weight = 1

"""
Now you can create a convolution. You will iterate over the image, leaving a 1 pixel margin, 
and multiplying each of the neighbors of the current pixel by the value defined in the filter 
(i.e. the current pixel's neighbor above it and to the left will be multiplied by the top left 
item in the filter, etc.)
You'll then multiply the result by the weight, and then ensure the result is in the range 0-255.
Finally you'll load the new value into the transformed image.
"""
# Iterate over the image
for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        convolution = 0.0
        convolution += (ascent_image[x-1, y-1]*filter1[0][0])
        convolution += (ascent_image[x-1, y]*filter1[0][1])
        convolution += (ascent_image[x-1, y+1]*filter1[0][2])
        convolution += (ascent_image[x, y-1]*filter1[1][0])
        convolution += (ascent_image[x, y]*filter1[1][1])
        convolution += (ascent_image[x, y+1]*filter1[1][2])
        convolution += (ascent_image[x+1, y-1]*filter1[2][0])
        convolution += (ascent_image[x+1, y]*filter1[2][1])
        convolution += (ascent_image[x+1, y+1]*filter1[2][2])

        # Multiply by weight
        convolution *= weight

        # Check the boundaries of the pixel values
        if convolution < 0:
            convolution = 0
        if convolution > 255:
            convolution = 255

        # Load into the transformed image
        image_transformed[x, y] = convolution

# Plot the image. Note the size of the axes -- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(image_transformed)
plt.show()

"""The next cell will show a (2, 2) pooling. The idea here is to iterate over the image, 
and look at the pixel and it's immediate neighbors to the right, beneath, and right-beneath. 
It will take the largest of them and load it into the new image. Thus, the new image will be 1/4 
the size of the old -- with the dimensions on X and Y being halved by this process. 
You'll see that the features get maintained despite this compression!
"""
# Assign dimensions half the size of the original image
new_x = int(size_x/2)
new_y = int(size_y/2)

# Create blank image with reduced dimensions
newImage = np.zeros((new_x, new_y))

# Iterate over the image
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        # Store all the pixel values in the (2,2) pool
        pixels = []
        pixels.append(image_transformed[x, y])
        pixels.append(image_transformed[x+1, y])
        pixels.append(image_transformed[x, y+1])
        pixels.append(image_transformed[x+1, y+1])

        # Get only the largest value and assign to the reduced image
        newImage[int(x/2), int(y/2)] = max(pixels)

# Plot the image. Note the size of the axes -- it is now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.show()
# test 1
