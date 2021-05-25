import math
import pandas as pd
import treePlotter
from sklearn.datasets import load_iris
from pylab import *
from collections import Counter
from matplotlib import pyplot as plt

class decision:
    def __init__(self, dimension=None, comparison=None, results=None, NH=None, lessb=None, moreb=None, max_label=None):
        self.dimension = dimension   #数据的维度
        self.comparison = comparison  #二分时的比较值，每次将样本集分为2类
        self.results = results  #已经分出的叶节点代表的类别，类别分别为0，1，2
        self.NH = NH  #存储各节点的样本量与熵的乘积，便于剪枝时使用
        self.lessb = lessb  #决策节点,样本在d维的数据小于比较值时，树上相对于当前节点的子树上的节点
        self.moreb = moreb  #决策节点,样本在d维的数据大于比较值时，树上相对于当前节点的子树上的节点
        self.max_label = max_label  #记录当前节点包含的label中同类最多的label

def entropy(y):     #计算信息熵
    if y.size > 1:
        cate = list(set(y))
    else:
        cate = [y.item()]
        y = [y.item()]
    entropy = 0
    for label in cate:
        p = len([label_ for label_ in y if label_ == label]) / len(y)
        #print(p)
        entropy += -p * math.log(p, 2)  #信息熵
        # print(entropy)
    return entropy


def GainEnt_max(X, y, dimension):   #计算选择每个属性中产生信息增益最大的一个数据
    ent_X = entropy(y)
    X_attribute = list(set(X[:, dimension]))    #取出第d维的数据
    X_attribute = sorted(X_attribute)
    Gain ,comparison  = 0, 0

    for i in range(len(X_attribute) - 1):#迭代计算在第d维数据中产生信息熵最大的分类值
        comparison_temp = (X_attribute[i] + X_attribute[i + 1]) / 2
        y_s_index = [i_arg for i_arg in range(
            len(X[:, dimension])) if X[i_arg, dimension] <= comparison_temp]
        y_s = y[y_s_index]
        y_b_index = [i_arg for i_arg in range(
            len(X[:, dimension])) if X[i_arg, dimension] > comparison_temp]
        y_b = y[y_b_index]

        Gain_temp = ent_X - (len(y_s) / len(y)) * \
                    entropy(y_s) - (len(y_b) / len(y)) * entropy(y_b)
        # print(Gain_temp)
        if Gain < Gain_temp:
            Gain ,comparison = Gain_temp ,comparison_temp
            #print(comparison)
    return Gain, comparison

def attribute_max_GainEnt(X, y):   #选取分类时产生最大信息熵的的维度进行先分类
    Dimension = np.arange(len(X[0]))
    Gain_max ,comparison_ , dimension_= 0, 0, 0
    for dimension in Dimension:
        Gain, comparison = GainEnt_max(X, y, dimension)
        if Gain_max < Gain:
            Gain_max = Gain
            comparison_ = comparison
            dimension_ = dimension
         #print(Gain_max,comparison_,dimension_)
    return Gain_max, comparison_, dimension_

def devide_group(X, y, comparison, dimension):  #对测试数据按照第d维的分类值comparisond进行分类
    x_s_index = [i_arg for i_arg in range(
        len(X[:, dimension])) if X[i_arg, dimension] <= comparison]
    x_b_index = [i_arg for i_arg in range(
        len(X[:, dimension])) if X[i_arg, dimension] > comparison]
    X_s = X[x_s_index]
    y_s = y[x_s_index]
    X_b = X[x_b_index]
    y_b = y[x_b_index]
    return X_s, y_s, X_b, y_b

def NH(y):    #计算信息熵与节点样本量的乘积
    ententropy = entropy(y)
    #print(ententropy, len(y), ententropy * len(y))
    return ententropy * len(y)

def maxlabel(y):
    label_ = Counter(y).most_common(1)
    return label_[0][0]

def buildtree(X, y, method='GainEnt'):  #构造决策树
    if y.size > 1:
        if method == 'GainEnt':
            Gain_max, comparison, dimension = attribute_max_GainEnt(X, y)
        if (Gain_max > 0 and method == 'GainEnt'):
            X_s, y_s, X_b, y_b = devide_group(X, y, comparison, dimension)
            left_branch = buildtree(X_s, y_s, method=method)
            right_branch = buildtree(X_b, y_b, method=method)
            nh = NH(y)
            max_label = maxlabel(y)
            return decision(dimension=dimension, comparison=comparison, NH=nh, lessb=left_branch, moreb=right_branch, max_label=max_label)
        else:
            nh = NH(y)
            max_label = maxlabel(y)
            return decision(results=y[0], NH=nh, max_label=max_label)
    else:
        nh = NH(y)
        max_label = maxlabel(y)
        return decision(results=y.item(), NH=nh, max_label=max_label)

def printtree(tree, indent='-', direction='L'):    #绘制剪枝前的决策树
    if tree.results != None:
        print(tree.results)

        dict_tree = {direction: str(tree.results)}
    else:
        print("分类维度为：{}".format(str(tree.dimension)),"分类值为：{}".format(str(tree.comparison)))
        print(indent + "Left->",)
        leftleaves = printtree(tree.lessb, indent=indent + "-", direction='Left')
        ll = leftleaves.copy()
        print(indent + "Right->",)
        rightleaves = printtree(tree.moreb, indent=indent + "-", direction='Right')
        rl = rightleaves.copy()
        ll.update(rl)
        stri = str(tree.dimension) + ":" + str(tree.comparison) + "?"
        if indent != '-':
            dict_tree = {direction: {stri: ll}}
        else:
            dict_tree = {stri: ll}
    return dict_tree

def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.dimension]
        if v > tree.comparison:
            branch = tree.moreb
        else:
            branch = tree.lessb
        return classify(observation, branch)

def pruning(tree, alpha=0.1):   #对决策树进行剪枝
    if tree.lessb.results == None:
        pruning(tree.lessb, alpha)
    if tree.moreb.results == None:
        pruning(tree.moreb, alpha)
    if tree.lessb.results != None and tree.moreb.results != None:
        before_pruning = tree.lessb.NH + tree.moreb.NH + 2 * alpha
        after_pruning = tree.NH + alpha
        print('before_pruning={},after_pruning={}'.format(
            before_pruning, after_pruning))
        if after_pruning <= before_pruning:
            print('剪枝维数：{}-剪枝比较值:{}'.format(tree.dimension, tree.comparison))
            tree.lessb, tree.moreb = None, None
            tree.results = tree.max_label

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    permutation = np.random.permutation(X.shape[0])
    shuffled_dataset = X[permutation, :]
    shuffled_labels = y[permutation]

    train_data = shuffled_dataset[:100, :]
    train_label = shuffled_labels[:100]
    test_data = shuffled_dataset[100:150:]
    test_label = shuffled_labels[100:150]

    tree = buildtree(train_data, train_label, method='GainEnt')
    b = printtree(tree=tree)
    true_count1 = 0
    for i in range(len(test_label)):
        predict = classify(test_data[i], tree)
        if predict == test_label[i]:
            true_count1 += 1
    print("正确分类数量:{}".format(true_count1))

    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

    treePlotter.createPlot(b, 2)
    pruning(tree=tree, alpha=4)
    b = printtree(tree=tree)

    true_count2 = 0
    for i in range(len(test_label)):
        predict = classify(test_data[i], tree)
        if predict == test_label[i]:
            true_count2 += 1
    print("剪枝后正确分类数量:{}".format(true_count2))

    treePlotter.createPlot(b, 4)
    plt.show()



