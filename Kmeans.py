import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn import mixture

#creat_data返回一个元组，第一个元素为样本点，第二个元素为样本点的真实簇分类标记，该函数产生的是分割的高斯分布的簇聚类
def creat_data(centers,num=100,std=0.7):
    X,labels_true=make_blobs(n_samples=num,centers=centers,cluster_std=std)
    return X,labels_true
def plot_data(*data):
    X,labels_true=data
    labels=np.unique(labels_true)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors="rgbyckm"
    for i,label in enumerate(labels):
        position=labels_true==label
        ax.scatter(X[position,0],X[position,1],label="cluster %d"%label,color=colors[i%len(colors)])
    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title("data")
    plt.show()
X,labels_true=creat_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
plot_data(X,labels_true)

##K均值算法测试
def test_Kmeans(*data):
    X,labels_true=data
    clst=cluster.KMeans(n_clusters=4)##K均值算法聚类模型
    print(clst.get_params())
    clst.fit(X)
    predicted_labels=clst.predict(X)
    print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))#ARI取值范围为[−1,1]，值越大意味着聚类结果与真实情况越吻合。从广义的角度来讲，ARI衡量的是两个数据分布的吻合程度。
    print("Sum center distance %s"%clst.inertia_)
X,labels_true=creat_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
test_Kmeans(X,labels_true)

#簇数量对模型的影响
def test_Kmeans_nclusters(*data):
    X,labels_true=data
    nums=range(1,50)
    ARIs=[]
    Distances=[]
    for num in nums:
        clst=cluster.KMeans(n_clusters=num)
        clst.fit(X)
        predicted_labels=clst.predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        Distances.append(clst.inertia_)
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs,marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances,marker="o")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("inertia_")
    fig.suptitle("KMeans")
    plt.show()
X,labels_true=creat_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
test_Kmeans(X,labels_true)