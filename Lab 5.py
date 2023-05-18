import matplotlib.pyplot as plt
import numpy as np
import skimage.io


def pred1(i, j, y):
    if i == 0 and j == 0:
        return 0
    elif i != 0 and j == 0:
        return y[i - 1][-1]
    else:
        return y[i][j - 1]


def pred2(i, j, y):
    if i == 0 or j == 0:
        return 0
    # elif i == 0 and j != 0:
    #     return 0.5 * y[i][j - 1]
    # elif i != 0 and j == 0:
    #     return 0.5 * y[i - 1][j]
    else:
        return int(0.5 * (y[i][j - 1] + y[i - 1][j]))


def myDifCode(_x, e, r):
    y = np.zeros(_x.shape)
    f = np.zeros(_x.shape)
    q = np.zeros(_x.shape)

    if r == 1:
        pred = pred1
    if r == 2:
        pred = pred2

    for i in range(_x.shape[0]):
        for j in range(_x.shape[1]):
            p = pred(i, j, y)
            f[i][j] = _x[i][j] - p
            q[i][j] = np.sign(f[i][j]) * ((abs(f[i][j]) + e) // (2 * e + 1))
            y[i][j] = p + q[i][j] * (2 * e + 1)
            assert abs(y[i][j] - _x[i][j]) <= e
    return q, f


def myDifDecode(q, e, r):
    y = np.zeros(q.shape)

    if r == 1:
        pred = pred1
    if r == 2:
        pred = pred2
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            p = pred(i, j, y)
            y[i][j] = p + q[i][j] * (2 * e + 1)
    return y


def entropy(arr):
    _, counts = np.unique(arr, return_counts=True)
    prob_counts = counts/sum(counts)
    return - (prob_counts * np.log2(prob_counts)).sum()


image = plt.imread("E:/Метод обработки и анализа изображений/Lab 1/ImgTif/09_lena2.tif")

# No 3
graph1 = [entropy(myDifCode(image, e, 1)[0]) for e in range(0, 51, 5)]
graph2 = [entropy(myDifCode(image, e, 2)[0]) for e in range(0, 51, 5)]
x = range(0, 51, 5)
plt.title('График зависимости энтропии массива q от максимальной погрешности е = 0..50')
plt.plot(x, graph1, color='red', label='предсказатель 1')
plt.plot(x, graph2, color='green', label='предсказатель 2')
plt.legend()
plt.show()

# No 5
e = np.array([5, 10, 20, 40])
for i in range(len(e)):
    Q, _ = myDifCode(image, e[i], 1)
    Y = myDifDecode(Q, e[i], 1)
    plt.subplot(4, 2, 2*(i+1)-1)
    plt.title('Исходное изображение')
    plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.subplot(4, 2, 2*(i+1))
    plt.title(f'Декомпрессированных с e = {e[i]}')
    plt.imshow(Y, cmap='gray')
plt.show()

# No 6
Q, F = myDifCode(image, 0, 1)
plt.imshow(F, cmap="gray")
plt.title("Разностный сигнал при е = 0")
plt.show()

# No 7
e = [0, 5, 10]
for i in range(len(e)):
    Q, _ = myDifCode(image, e[i], 1)
    plt.subplot(1, 3, i+1)
    plt.title(f"Квантованный разностный сигнал для е = {e[i]}")
    plt.imshow(Q, cmap="gray")
plt.show()


