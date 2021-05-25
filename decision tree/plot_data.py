import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('white',{'font.sans-serif':['simhei','Arial']})

iris=datasets.load_iris()
iris_data= pd.DataFrame(iris.data,columns=iris.feature_names)
iris_data['species']=iris.target_names[iris.target]
iris_data.head(3).append(iris_data.tail(3))
iris_data.rename(columns={"sepal length (cm)":"萼片长",
                     "sepal width (cm)":"萼片宽",
                     "petal length (cm)":"花瓣长",
                     "petal width (cm)":"花瓣宽",
                     "species":"种类"},inplace=True)
kind_dict = {
    "setosa":"山鸢尾",
    "versicolor":"杂色鸢尾",
    "virginica":"维吉尼亚鸢尾"
}
iris_data["种类"] = iris_data["种类"].map(kind_dict)

plt.figure()
sns.pairplot(iris_data,hue='种类')
plt.show()