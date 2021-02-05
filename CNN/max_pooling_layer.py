import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from elice_utils import EliceUtils

elice_utils = EliceUtils()


def Visualize(image, x, y):
    plt.imshow(image.reshape(x, y), cmap='Greys')
    plt.savefig('plot.png')
    elice_utils.send_image("plot.png")


# 임의의 3 x 3 x 1 영상을 하나 만들어줍니다.
image = tf.constant([[[[1], [2], [3], [4]],
                      [[4], [5], [6], [7]],
                      [[7], [8], [9], [10]],
                      [[3], [5], [7], [9]]]], dtype=np.float32)

'''
지시사항 1번
Max Pooling Layer를 선언하세요.
'''
pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(image)

print(pool.shape)
print(pool.numpy())

# 원본 영상과 Max Pooling 후 영상을 출력합니다..
Visualize(image.numpy(), 4, 4)
Visualize(pool.numpy(), 2, 2)
