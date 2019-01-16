#数据降维
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(data)
#特征向量
print(pca.components_)
#贡献率
print(pca.explained_variance_ratio_)

#建立决策树模型，基于信息熵
from sklearn.tree import DecisionTreeClassifier as DTC
dtc=DTC(criterion='entropy')
#训练模型
dtc.fit(x,y)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
