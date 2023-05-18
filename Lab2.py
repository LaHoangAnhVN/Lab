import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


def linear_contract(img_input, g_min=0, g_max=255):
    f_min = img_input.min()
    f_max = img_input.max()

    a = (g_max - g_min) / (f_max - f_min)
    b = g_max - a * f_max
    return np.round(a * img_input + b).clip(0, 255)


def conv(img_input, mask_input):
    return convolve2d(img_input, mask_input, boundary="symm", mode="same")


img = plt.imread("E:/Метод обработки и анализа изображений/Lab 1/ImgTif/09_lena2.tif")


# No1
s1 = conv(img, [[1], [-1]]) # окно 2 * 1 / f(m, n) - f(m-1, n)
s2 = conv(img, [[1, -1]]) # окно 1 * 2 / f(m, n) - f(m, n-1)
grad = np.sqrt(np.square(s1) + np.square(s2))
cont_1 = np.where(grad > 20, 255, 0)

plt.subplot(231)
plt.title('Исходное изображение')
plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(232)
plt.title('Результат выделения контура')
plt.imshow(cont_1, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(233)
plt.title('Результат градиентного метода')
plt.imshow(linear_contract(grad), cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(234)
plt.title('Гистограмма результата градиентного метода')
hist, bins = np.histogram(grad.flatten(), 256, [0, 256])
plt.xlim(-10, 257)
plt.xlabel('Яркость')
plt.ylabel('Количество')
plt.plot(np.arange(0, 256), hist)

plt.subplot(235)
plt.title('s1')
plt.imshow(linear_contract(abs(s1)), cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(236)
plt.title('s2')
plt.imshow(linear_contract(abs(s2)), cmap=plt.cm.gray, vmin=0, vmax=255)

plt.show()


# No2
mask1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
mask2 = 1/2 * np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]])
mask3 = 1/3 * np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
#
plt.subplot(221)
plt.title('Исходное изображение')
plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(222)
plt.title('Результат маска 1')
plt.imshow(linear_contract(abs(conv(img, mask1))), cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(223)
plt.title('Контур')
plt.imshow(np.where(conv(img, mask1) > 50, 255, 0), cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(224)
plt.title('Гистограмма')
hist, bins = np.histogram(conv(img, mask1).flatten(), 256, [0, 256])
plt.xlim(-10, 257)
plt.xlabel('Яркость')
plt.ylabel('Количество')
plt.plot(np.arange(0, 256), hist)

plt.show()

plt.subplot(221)
plt.title('Исходное изображение')
plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(222)
plt.title('Результат маска 2')
plt.imshow(linear_contract(abs(conv(img, mask2))), cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(223)
plt.title('Контур')
plt.imshow(np.where(conv(img, mask2) > 50, 255, 0), cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(224)
plt.title('Гистограмма')
hist, bins = np.histogram(conv(img, mask2).flatten(), 256, [0, 256])
plt.xlim(-10, 257)
plt.xlabel('Яркость')
plt.ylabel('Количество')
plt.plot(np.arange(0, 256), hist)

plt.show()

plt.subplot(221)
plt.title('Исходное изображение')
plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(222)
plt.title('Результат маска 3')
plt.imshow(linear_contract(abs(conv(img, mask3))), cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(223)
plt.title('Контур')
plt.imshow(np.where(conv(img, mask3) > 50, 255, 0), cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(224)
plt.title('Гистограмма')
hist, bins = np.histogram(conv(img, mask3).flatten(), 256, [0, 256])
plt.xlim(-10, 257)
plt.xlabel('Яркость')
plt.ylabel('Количество')
plt.plot(np.arange(0, 256), hist)

plt.show()


# No3
# mask1 = 1/6 * np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
# mask2 = 1/6 * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
#
s1 = conv(img, mask1)
s2 = conv(img, mask2)
result = np.sqrt(np.square(s1) + np.square(s2))
cont_3 = np.where(result > 15, 255, 0)
#
# plt.subplot(221)
# plt.title('Исходное изображение')
# plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(222)
# plt.title('Результат метода оператора Прюитт')
# plt.imshow(linear_contract(result), cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(223)
# plt.title('Контур')
# plt.imshow(cont, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(224)
# plt.title('Гистограмма')
# hist, bins = np.histogram(result.flatten(), 256, [0, 256])
# plt.xlim(-10, 257)
# plt.xlabel('Яркость')
# plt.ylabel('Количество')
# plt.plot(np.arange(0, 256), hist)
#
# plt.show()

# No 4
mask = 1/3 * np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]])
result = conv(img, mask)
cont_4 = np.where(result > 30, 255, 0)
#
# plt.subplot(221)
# plt.title('Исходное изображение')
# plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(222)
# plt.title('Результат метода оператора Прюитт')
# plt.imshow(linear_contract(abs(result)), cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(223)
# plt.title('Контур')
# plt.imshow(cont, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(224)
# plt.title('Гистограмма')
# hist, bins = np.histogram(result.flatten(), 256, [0, 256])
# plt.xlim(-10, 257)
# plt.xlabel('Яркость')
# plt.ylabel('Количество')
# plt.plot(np.arange(0, 256), hist)


# plt.subplot(221)
# plt.title('No1')
# plt.imshow(cont_1, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(222)
# plt.title('No3')
# plt.imshow(cont_3, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.show()

# plt.subplot(221)
# plt.title('No2 mask 1')
# plt.imshow(np.where(conv(img, mask1) > 20, 255, 0), cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(222)
# plt.title('No2 mask 2')
# plt.imshow(np.where(conv(img, mask2) > 20, 255, 0), cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(223)
# plt.title('No2 mask 3')
# plt.imshow(np.where(conv(img, mask3) > 50, 255, 0), cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(224)
# plt.title('No4')
# plt.imshow(cont_4, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.show()
