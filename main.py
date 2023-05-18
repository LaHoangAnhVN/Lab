import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

img = plt.imread('E:\\Метод обработки и анализа изображений\\Lab 1\\ImgTif\\08_goldhill2.tif')

# No1:Пороговая обработка.
# const_gray = 75
# img_1 = (img > const_gray)*255
# plt.subplot(321)
# plt.title('Исходное изображение')
# plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
# plt.subplot(322)
# plt.title('Изображение после пороговой обработки')
# plt.imshow(img_1, cmap=plt.cm.gray, vmin=0, vmax=255)
# plt.subplot(323)
# hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# plt.title('Исходная гистограмма')
# plt.xlim(-10, 257)
# plt.xlabel('Яркость')
# plt.ylabel('Количество')
# plt.plot(np.arange(0, 256), hist)
# plt.subplot(324)
# plt.title('Гистограмма после пороговой обработки ')
# hist, bins = np.histogram(img_1.flatten(), 256, [0, 256])
# plt.xlim(-10, 257)
# plt.xlabel('Яркость')
# plt.ylabel('Количество')
# plt.plot(np.arange(0, 256), hist)
# plt.subplot(325)
# plt.title('График функции поэлементного преобразования')
# plt.plot(np.arange(0, 256), (np.arange(0, 256) > const_gray) * 255)
# plt.show()


# No2: Контрастирование.

f_min = img.min()
f_max = img.max()

a = (255 - 0) / (f_max - f_min)
b = 255 - a * f_max

plt.subplot(321)
plt.title('Исходное изображение')
plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)

img_2 = np.round(a * img + b).clip(0, 255)
plt.subplot(322)
plt.title('Изображение после пороговой обработки')
plt.imshow(img_2, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(323)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
plt.title('Исходная гистограмма')
plt.xlim(-10, 257)
plt.xlabel('Яркость')
plt.ylabel('Количество')
plt.plot(np.arange(0, 256), hist)

plt.subplot(324)
hist, bins = np.histogram(img_2.flatten(), 256, [0, 256])
plt.title('Гистограмма после пороговой обработки')
plt.xlim(-10, 257)
plt.xlabel('Яркость')
plt.ylabel('Количество')
plt.plot(np.arange(0, 256), hist)

plt.subplot(325)
plt.title('График функции поэлементного преобразования')
plt.plot(np.arange(0, 256), (a * np.arange(0, 256) + b).clip(0, 255))

plt.show()


# No3: Эквализация

# hist, bins_org = np.histogram(img.flatten(), 256, [0, 256])
# F = hist.cumsum() / (512 * 512)
#
# g_min = 0
# g_max = 255
#
# g = (g_max - g_min)*F[img] + g_min
# g_hist, bins = np.histogram(g.flatten(), 256, [0, 256])
#
# plt.subplot(231)
# plt.title('Исходное изображение')
# plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(232)
# plt.title('Изображение после самописной эквализации')
# plt.imshow(g, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(233)
# plt.title('Изображение после стандратной эквализации')
# plt.imshow(exposure.equalize_hist(img)*255, cmap=plt.cm.gray, vmin=0, vmax=255)
#
# plt.subplot(234)
# plt.title('Гистограмма исходного изображения')
# plt.xlim(-10, 257)
# plt.xlabel('Яркость')
# plt.ylabel('Количество')
# plt.plot(np.arange(0, 256), hist)
#
# plt.subplot(235)
# plt.title('Изображение после самописной эквализации')
# plt.xlim(-10, 257)
# plt.xlabel('Яркость')
# plt.ylabel('Количество')
# plt.plot(np.arange(0, 256), g_hist)
#
# plt.subplot(236)
# plt.title('Изображение после стандратной эквализации')
# plt.xlim(-10, 257)
# plt.xlabel('Яркость')
# plt.ylabel('Количество')
# std_hist, bins = np.histogram(exposure.equalize_hist(img)*255, 256, [0, 256])
# plt.plot(np.arange(0, 256), std_hist)
#
# plt.show()
#
# plt.subplot(221)
# plt.title('График интегральной функции распределения яркости - ДО')
# plt.plot(np.arange(0, 256), F)
#
# plt.subplot(222)
# plt.title('График интегральной функции распределения яркости - ПОСЛЕ')
# plt.plot(np.arange(0, 256), std_hist.cumsum()/(512*512))
#
# plt.subplot(223)
# plt.title('График функции поэлементного преобразования')
# plt.plot(np.arange(0, 256), (g_max - g_min)*F+g_min)
#
# plt.show()
#
# print(hist)
# print(bins_org) # 16-235
