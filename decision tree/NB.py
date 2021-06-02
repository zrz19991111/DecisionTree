from pylab import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33333333, random_state=0)
    # 将数据集按照 训练集比测试集为8：2的比例随机拆分数据集
    permutation = np.random.permutation(X.shape[0])
    shuffled_dataset = X[permutation, :]
    shuffled_labels = y[permutation]

    train_data = shuffled_dataset[:100, :]
    train_label = shuffled_labels[:100]
    test_data = shuffled_dataset[100:150:]
    test_label = shuffled_labels[100:150]

    clf = GaussianNB(var_smoothing=1e-8)
    clf.fit(train_data, train_label)  # 带入训练集训练模型
    num_test = len(test_label)
    # 预测
    y_test_pre = clf.predict(test_data)  # 利用拟合的贝叶斯进行预测
    acc = sum(y_test_pre == test_label) / num_test
    print('正确率为:{}'.format(acc))  # 显示预测准确率
