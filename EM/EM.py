import numpy as np
from sklearn.utils import shuffle
import math
import matplotlib
import matplotlib.pyplot as plt


# 数据生成与采样
def sampling():
    u1 = 165
    u2 = 175
    sigma1 = 10
    sigma2 = 15
    sample1 = 200
    sample2 = 400
    y_girl = np.zeros([1, sample1])
    y_boy = np.zeros([1, sample2])
    girl = np.random.normal(u1, sigma1, sample1)
    boy = np.random.normal(u2, sigma2, sample2)
    x = np.hstack((girl, boy))
    x = x.reshape(1, 600)
    y = np.hstack((y_girl, y_boy))
    x_data, y_data = shuffle(x, y)
    return x_data, y_data


# 正态分布
def norm_func(u, sigma, x):
    if sigma == 0:
        sigma = 0.01
    temp = -1 * ((x-u)**2) / (sigma**2) / 2
    coe = 1 / (2*math.pi)**(1/2) / sigma
    return coe * np.exp(temp)


# EM算法部分
def expectation(x, u1, u2, sigma1, sigma2, p):
    # expectation部分
    norm1 = norm_func(u1, sigma1, x)
    norm2 = norm_func(u2, sigma2, x)
    ux = p*norm1 / (p*norm1 + (1-p)*norm2)

    # 防止出现0分母
    num1 = np.sum(ux)
    num2 = np.sum(1-ux)
    den1 = np.sum(ux*x)
    den2 = np.sum((1-ux)*x)
    if den1 == 0:
        den1 = 0.01
    if den2 == 0:
        den2 = 0.01
    if num1 == 0:
        num1 = 0.01
    if num2 == 0:
        num2 = 0.01

    # maximization部分
    next_u1 = den1 / num1
    next_u2 = den2 / num2
    next_p = num1 / 600

    den3 = np.sum(ux*((x-next_u1)**2))
    den4 = np.sum((1-ux)*((x-next_u2)**2))
    if den3 == 0:
        den3 = 0.01
    if den4 == 0:
        den4 = 0.01

    next_sigma1 = (den3 / num1) ** 0.5
    next_sigma2 = (den4 / num2) ** 0.5

    return next_u1, next_u2, next_sigma1, next_sigma2, next_p


# 最大似然函数
def maximum(x, u1, u2, sigma1, sigma2, p):
    norm1 = norm_func(u1, sigma1, x)
    norm2 = norm_func(u2, sigma2, x)
    temp = np.log(p * norm1 + (1 - p) * norm2)
    likelihood = np.sum(temp)
    return likelihood


# 拟合结束判断
def end_decision(l_last, l):
    if abs(l - l_last) < 0.001:
        return 0
    else:
        return 1

# 迭代训练 返回likely_list, u1_list, u2_list, sigma1_list, sigma2_list, p_list表示迭代过程数据
def train(x_train, y_train):
    # 随机初始化
    u1 = np.random.randint(155, 175, 1)
    u2 = np.random.randint(165, 185, 1)
    sigma1 = np.random.randint(0, 20, 1)
    sigma2 = np.random.randint(0, 30, 1)
    p = np.random.rand()

    l_last = -10
    l = 0
    likely_list = []
    u1_list = []
    u2_list = []
    sigma1_list = []
    sigma2_list = []
    p_list = []
    i = 0
    while end_decision(l_last, l):
        u1, u2, sigma1, sigma2, p = expectation(x_train, u1, u2, sigma1, sigma2, p)
        likely = maximum(x_train, u1, u2, sigma1, sigma2, p)
        likely_list.append(likely)
        u1_list.append(u1)
        u2_list.append(u2)
        sigma1_list.append(sigma1)
        sigma2_list.append(sigma2)
        p_list.append(p)
        print("第{}组数据".format(str(i)))
        print("女生平均身高：{}".format(str(u1)))
        print("女生平均身高方差：{}".format(str(sigma1)))
        print("男生平均身高：{}".format(str(u2)))
        print("男生平均身高方差：{}".format(str(sigma2)))
        print("女生数量含量：{}".format(str(p)))
        print("最大似然估计值：{}".format(likely))
        i = i + 1
        l_last = l
        l = likely
    return likely_list, u1_list, u2_list, sigma1_list, sigma2_list, p_list


if __name__ == "__main__":
    x_train, y_train = sampling()
    likely_list, u1_list, u2_list, sigma1_list, sigma2_list, p_list = train(x_train, y_train)
    plt.figure()
    plt.plot(likely_list)
    plt.rcParams['axes.unicode_minus']=False
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    plt.xlabel("迭代次数")
    plt.ylabel("最大似然估计值")
    plt.show()