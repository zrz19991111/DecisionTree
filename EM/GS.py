import numpy as np
import matplotlib.pyplot as plt


def func_gaosi(x, miu, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - miu) ** 2 / 2 / sigma ** 2)


x = np.linspace(150, 200, 100)
y = func_gaosi(x, 175,15)
plt.plot(x, y)
plt.xlabel('男生身高')
plt.ylabel('概率密度')
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.show()
