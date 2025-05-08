import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
mnist_idx = 100
print('[label]')
print('number label = ', train_labels[mnist_idx])
print('\n')
print('[image]')
for row in train_images[mnist_idx]:
    for col in row:
        print("%10f" % col, end = "")
    print('\n')
plt.figure(figsize=(5,5))
image = train_images[mnist_idx]
plt.imshow(image)
plt.show()