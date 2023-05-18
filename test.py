import numpy as np
import matplotlib.pyplot as plt

a = np.ones((64,64))*255

plt.imshow(a, cmap='gray', vmin=0, vmax=255)
plt.show()
