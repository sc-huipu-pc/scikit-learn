########机器学习###########
#KNN分类与KNN回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  neighbors,datasets
from sklearn.model_selection import train_test_split

####KNN分类
#KNN三要素：k值选择（k=1时又称最邻近算法），距离度量（欧氏距离，哈曼顿距离等），分类决策规则（多数表决，加权投票）
def load_classification_data():
    digits=datasets.load_digits()#scikit-learn自带的手写识别数据集
    X_train=digits.data
    y_train=digits.target
    return train_test_split(X_train,y_train,test_size=0.25,random_state=0,stratify=y_train)
def test_KNeighborsClassifier(*data):
    X_train,X_test,y_train,y_test=data
    clf=neighbors.KNeighborsClassifier()#用于实现k邻近法分类模型
    clf.fit(X_train,y_train)
    print("train score:%f"%clf.score(X_train,y_train))
    print("Test score:%f"%clf.score(X_test,y_test))
    print(X_test.shape)
    print(clf.predict(X_test[:20]))#取前20个进行预测
X_train,X_test,y_train,y_test=load_classification_data()
test_KNeighborsClassifier(X_train,X_test,y_train,y_test)
###K邻近法K值及分类决策对性能的影响
def test_KNeighborsClassifier_k_w(*data):
    X_train,X_test,y_train,y_test=data
    Ks=np.linspace(1,y_train.size,num=100,endpoint=False,dtype="int")
    weights=["uniform","distance"]
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for weight in weights:
        training_scores=[]
        testing_scores=[]
        for K in Ks:
            clf=neighbors.KNeighborsClassifier(weights=weight,n_neighbors=K)
            clf.fit(X_train,y_train)
            testing_scores.append(clf.score(X_test,y_test))
            training_scores.append(clf.score(X_train,y_train))
        ax.plot(Ks,testing_scores,label="testing score:weight=%s"%weight)
        ax.plot(Ks,training_scores,label="training score:weight=%s"%weight)
    ax.legend(loc="best")
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()
# X_train,X_test,y_train,y_test=load_classification_data()
test_KNeighborsClassifier_k_w(X_train,X_test,y_train,y_test)




###KNN回归，将待预测样本点最邻近的K个训练样本点的平均值作为带预测样本点的值
def creat_regression_data(n):
    X=5*np.random.rand(n,1)
    y=np.sin(X).ravel()
    y[::5]+=1*(0.5-np.random.rand(int(n/5)))
    return train_test_split(X,y,test_size=0.25,random_state=0)
def test_KNeighborsRegressor(*data):
    X_train,X_test,y_train,y_test=data
    regr=neighbors.KNeighborsRegressor()
    regr.fit(X_train,y_train)
    print("train score:%f"%regr.score(X_train,y_train))
    print("Test score:%f"%regr.score(X_test,y_test))
#距离度量形式对预测性能的影响
def test_KNeighborsRegressor_k_p(*data):
    X_train,X_test,y_train,y_test=data
    Ks=np.linspace(1,y_train.size,endpoint=False,dtype="int")
    Ps=[1,2,10]

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for P in Ps:
        training_scores=[]
        testing_scores=[]
        for K in Ks:
            regr=neighbors.KNeighborsRegressor(p=P,n_neighbors=K)
            regr.fit(X_train,y_train)
            testing_scores.append(regr.score(X_test,y_test))
            training_scores.append(regr.score(X_train,y_train))
        ax.plot(Ks,testing_scores,label="testing score:p=%d"%P)
        ax.plot(Ks,training_scores,label="training score:p=%d"%P)
    ax.legend(loc="best")
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNeighborRegressor")
    plt.show()
X_train,X_test,y_train,y_test=creat_regression_data(1000)
test_KNeighborsRegressor_k_p(X_train,X_test,y_train,y_test)
