import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter


def generate_chessboard(size_board, size_square, v_min, v_max):
    board = np.full((size_board, size_board), v_min, dtype=np.uint8)

    for i in range(len(board)):
        if (i // size_square) % 2 == 1:
            for j in range(len(board[i])):
                if (j // size_square) % 2 == 0:
                    board[i][j] = v_max
        if (i // size_square) % 2 == 0:
            for j in range(len(board[i])):
                if (j // size_square) % 2 == 1:
                    board[i][j] = v_max

    return board, board.var()


def generate_image_with_white_sum(image, d):
    image_with_noise = random_noise(image, var=(image.var())/d/255**2)*255
    noise = image_with_noise - image + 128
    return image_with_noise, noise


def generate_image_with_impulse_noise(image, p):
    noise = random_noise(np.full(image.shape, 0.5), mode='s&p', amount=p) * 255
    image_with_noise = image.copy()
    image_with_noise[noise == 0] = 0
    image_with_noise[noise == 255] = 255
    return image_with_noise, noise


def linear_filter(image):
    mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    return convolve2d(image, mask, mode='same', boundary='symm')


def variance_error(original_image, filtered_image):
    return ((filtered_image - original_image)**2).mean()


def coeff_noise_supp(original_image, filtered_image, noise):
    delta = filtered_image - original_image
    delta_ = delta**2
    noise_ = noise**2
    return delta_.mean()/noise_.mean()


chessboard, D = generate_chessboard(128, 16, 96, 160)

image_with_white_noise_d_1, noise_d_1 = generate_image_with_white_sum(chessboard, 1)
image_with_white_noise_d_10, noise_d_10 = generate_image_with_white_sum(chessboard, 10)

image_with_impulse_noise_p_01, noise_p_01 = generate_image_with_impulse_noise(chessboard, 0.1)
image_with_impulse_noise_p_03, noise_p_03 = generate_image_with_impulse_noise(chessboard, 0.3)

# No1
print('Белый шум с d = 1:')
plt.subplot(231)
plt.title('Исходное изображение (D = ' + str(D) + ')')
plt.imshow(chessboard, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(232)
plt.title('Белый шум с d = 1 (D = ' + str(noise_d_1.var()) + ')')
plt.imshow(noise_d_1, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(233)
plt.title('Зашумленное изображение')
plt.imshow(image_with_white_noise_d_1, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(234)
median_filtered_image = median_filter(image_with_white_noise_d_1, size=3)
plt.title('Медианый фильтр')
plt.imshow(median_filtered_image, cmap=plt.cm.gray, vmin=0, vmax=255)
print('\tДисперсия ошибок при медианым фильтре: ', variance_error(chessboard, median_filtered_image))
print('\tКоэффициент подавления шума: ', variance_error(chessboard, median_filtered_image)/(noise_d_1**2).mean())

plt.subplot(235)
linear_filtered_image = linear_filter(image_with_white_noise_d_1)
plt.title('Линейный фильтр')
plt.imshow(linear_filtered_image, cmap=plt.cm.gray, vmin=0, vmax=255)
print('\n\tДисперсия ошибок при линейном фильтре: ', variance_error(chessboard, linear_filtered_image))
print('\tКоэффициент подавления шума: ', variance_error(chessboard, linear_filtered_image)/(noise_d_1**2).mean())

plt.show()

print('\nБелый шум с d = 10:')
plt.subplot(231)
plt.title('Исходное изображение (D = ' + str(D) + ')')
plt.imshow(chessboard, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(232)
plt.title('Белый шум с d = 10 (D = ' + str(noise_d_10.var()) + ')')
plt.imshow(noise_d_10, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(233)
plt.title('Зашумленное изображение')
plt.imshow(image_with_white_noise_d_10, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(234)
median_filtered_image = median_filter(image_with_white_noise_d_10, size=3)
plt.title('Медианый фильтр')
plt.imshow(median_filtered_image, cmap=plt.cm.gray, vmin=0, vmax=255)
print('\tДисперсия ошибок при медианым фильтре: ', variance_error(chessboard, median_filtered_image))
print('\tКоэффициент подавления шума: ', coeff_noise_supp(chessboard, median_filtered_image, noise_d_10))

plt.subplot(235)
linear_filtered_image = linear_filter(image_with_white_noise_d_10)
plt.title('Линейный фильтр')
plt.imshow(linear_filtered_image, cmap=plt.cm.gray, vmin=0, vmax=255)
print('\n\tДисперсия ошибок при линейном фильтре: ', variance_error(chessboard, linear_filtered_image))
print('\tКоэффициент подавления шума: ', coeff_noise_supp(chessboard, linear_filtered_image, noise_d_10))

plt.show()

# No2
print('\nИмпульсный шум с p = 0.1:')
plt.subplot(231)
plt.title('Исходное изображение (D = ' + str(D) + ')')
plt.imshow(chessboard, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(232)
plt.title('Импульсный шум с p = 0.1 (D = ' + str(noise_p_01.var()) + ')')
plt.imshow(noise_p_01, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(233)
plt.title('Зашумленное изображение с p = 0.1')
plt.imshow(image_with_impulse_noise_p_01, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(234)
median_filtered_image = median_filter(image_with_impulse_noise_p_01, size=3)
plt.title('Медианый фильтр')
plt.imshow(median_filtered_image, cmap=plt.cm.gray, vmin=0, vmax=255)
print('\tДисперсия ошибок при медианым фильтре: ', variance_error(chessboard, median_filtered_image))
print('\tКоэффициент подавления шума: ', coeff_noise_supp(chessboard, median_filtered_image, noise_p_01))

plt.subplot(235)
linear_filtered_image = linear_filter(image_with_impulse_noise_p_01)
plt.title('Линейный фильтр')
plt.imshow(linear_filtered_image, cmap=plt.cm.gray, vmin=0, vmax=255)
print('\n\tДисперсия ошибок при линейном фильтре: ', variance_error(chessboard, linear_filtered_image))
print('\tКоэффициент подавления шума: ', coeff_noise_supp(chessboard, linear_filtered_image, noise_p_01))

plt.show()

print('\nИмпульсный шум с p = 0.3')
plt.subplot(231)
plt.title('Исходное изображение (D = ' + str(D) + ')')
plt.imshow(chessboard, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(232)
plt.title('Импульсный шум с p = 0.1 (D = ' + str(noise_p_03.var()) + ')')
plt.imshow(noise_p_03, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(233)
plt.title('Зашумленное изображение с p = 0.1')
plt.imshow(image_with_impulse_noise_p_03, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.subplot(234)
median_filtered_image = median_filter(image_with_impulse_noise_p_03, size=3)
plt.title('Медианый фильтр')
plt.imshow(median_filtered_image, cmap=plt.cm.gray, vmin=0, vmax=255)
print('\tДисперсия ошибок при медианым фильтре: ', variance_error(chessboard, median_filtered_image))
print('\tКоэффициент подавления шума: ', coeff_noise_supp(chessboard, median_filtered_image, noise_p_03))

plt.subplot(235)
linear_filtered_image = linear_filter(image_with_impulse_noise_p_03)
plt.title('Линейный фильтр')
plt.imshow(linear_filtered_image, cmap=plt.cm.gray, vmin=0, vmax=255)
print('\n\tДисперсия ошибок при линейном фильтре: ', variance_error(chessboard, linear_filtered_image))
print('\tКоэффициент подавления шума: ', coeff_noise_supp(chessboard, linear_filtered_image, noise_p_03))

plt.show()
