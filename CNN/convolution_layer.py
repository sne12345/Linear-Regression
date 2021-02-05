import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from elice_utils import EliceUtils

elice_utils = EliceUtils()


def Visualize(image, x, y):
    plt.imshow(image.reshape(x, y), cmap='Greys')
    plt.savefig('plot.png')
    elice_utils.send_image("plot.png")


# 임의의 3 x 3 x 1 영상을 하나 만들어줍니다.
image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)

# 합성곱 연산을 위해 임의의 2 x 2 x 1 커널을 하나 만들어줍니다.
kernel = np.array([[[[1.]], [[1.]]],
                   [[[1.]], [[1.]]]])

# 이미지 Shape 출력 : (num of image, width, height, channel)
print('Image shape : ', image.shape)
# 커널 Shape 출력 : (width, height, channel, num of kernel)
print('Kernel shape : ', kernel.shape)
# tf.nn.conv2d에 넣기 위해 이미지와 커널의 Shape을 위와 같이 만들었습니다.


# Gray 이미지 출력
Visualize(image, 3, 3)

kernel_init = tf.constant_initializer(kernel)
# Convolution Layer 선언
'''
지시사항1번 
   keras.layers.Conv2D()를 완성하세요.
'''
conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding='VALID', kernel_initializer=kernel_init)(image)
Visualize(conv2d.numpy(), 2, 2)
