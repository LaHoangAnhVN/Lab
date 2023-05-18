from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.signal import convolve2d
from skimage.util import random_noise


def add_object(img, _object, count=6):
    img_copy = img.copy()
    h, w = img.shape
    H, W = _object.shape

    for i in range(count):
        ty, tx = random.randrange(h - H), random.randrange(w - W)
        img_copy[ty:ty+H, tx:tx+W] += _object
    return img_copy


def correlation_algorithm(img, _object):
    mask = np.sqrt(1/(_object**2).sum()) * _object
    return convolve2d(img, np.flip(mask))


def threshold(img, t):
    return np.where(img >= t, 255, 0)


object_1 = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]])
object_2 = np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]])
noise = random_noise(np.zeros((64, 64)), var=0.05, clip=False)

# Image 1 no object
plt.subplot(241)
plt.title('Фон - Белый шум')
plt.imshow(noise, cmap='gray')

plt.subplot(242)
plt.title('Корреляционной алгоритм с маском объекта 1')
plt.imshow(correlation_algorithm(noise, object_1), cmap='gray')

plt.subplot(243)
plt.title('Корреляционной алгоритм с маском объекта 2')
plt.imshow(correlation_algorithm(noise, object_2), cmap='gray')

plt.subplot(244)
plt.title('Гистограмма корреляции с маском объекта 1')
plt.hist(correlation_algorithm(noise, object_1).flatten(), 256, histtype='step')

plt.subplot(245)
plt.title('Пороговая с маском объекта 1')
plt.imshow(threshold(correlation_algorithm(noise, object_1), 2), cmap='gray')

plt.subplot(246)
plt.title('Пороговая с маском объекта 2')
plt.imshow(threshold(correlation_algorithm(noise, object_2), 2), cmap='gray')

plt.subplot(247)
plt.title('Гистограмма корреляции с маском объекта 2')
plt.hist(correlation_algorithm(noise, object_2).flatten(), 256, histtype='step')

plt.show()


# Image 1 with object 1
img1 = add_object(noise, object_1)
plt.subplot(241)
plt.title('Фон - Белый шум')
plt.imshow(noise, cmap='gray')

plt.subplot(242)
plt.title('Фон с 1 объектами')
plt.imshow(img1, cmap='gray')

plt.subplot(243)
plt.title('Корреляционной алгоритм с маском объекта 1')
plt.imshow(correlation_algorithm(img1, object_1), cmap='gray')

plt.subplot(244)
plt.title('Корреляционной алгоритм с маском объекта 2')
plt.imshow(correlation_algorithm(img1, object_2), cmap='gray')

plt.subplot(245)
plt.title('Пороговая с маском объекта 1')
plt.imshow(threshold(correlation_algorithm(img1, object_1), 1.9), cmap='gray')

plt.subplot(246)
plt.title('Пороговая с маском объекта 2')
plt.imshow(threshold(correlation_algorithm(img1, object_2), 1.6), cmap='gray')

plt.subplot(247)
plt.title('Гистограмма корреляции с маском объекта 1')
plt.hist(correlation_algorithm(img1, object_1).flatten(), 256, histtype='step')

plt.subplot(248)
plt.title('Гистограмма корреляции с маском объекта 2')
plt.hist(correlation_algorithm(img1, object_2).flatten(), 256, histtype='step')

plt.show()


# Image 1 with object 1 and 2
img2 = add_object(img1, object_2)

plt.subplot(241)
plt.title('Фон - Белый шум')
plt.imshow(noise, cmap='gray')

plt.subplot(242)
plt.title('Фон с двуми объектами')
plt.imshow(img2, cmap='gray')

plt.subplot(243)
plt.title('Корреляционной алгоритм с маском объекта 1')
plt.imshow(correlation_algorithm(img2, object_1), cmap='gray')

plt.subplot(244)
plt.title('Корреляционной алгоритм с маском объекта 2')
plt.imshow(correlation_algorithm(img2, object_2), cmap='gray')

plt.subplot(245)
plt.title('Пороговая с маском объекта 1')
plt.imshow(threshold(correlation_algorithm(img2, object_1), 1.9), cmap='gray')

plt.subplot(246)
plt.title('Пороговая с маском объекта 2')
plt.imshow(threshold(correlation_algorithm(img2, object_2), 2), cmap='gray')

plt.subplot(247)
plt.title('Гистограмма корреляции с маском объекта 1')
plt.hist(correlation_algorithm(img2, object_1).flatten(), 256, histtype='step')

plt.subplot(248)
plt.title('Гистограмма корреляции с маском объекта 2')
plt.hist(correlation_algorithm(img2, object_2).flatten(), 256, histtype='step')

plt.show()
