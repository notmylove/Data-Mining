from sklearn.cluster import KMeans
import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame
import random
from sklearn import preprocessing
import time
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt


# 计算每个样本的相异度
def xyd(vec1, vec2, vec3):
    return np.sum(np.abs(vec1-vec2)/vec3)


# 构造相异度矩阵
def xyd_mat(x):
    n_sample, n_features = x.shape
    mat1 = np.zeros((n_sample, n_sample), dtype=np.float64)
    vec3 = np.array([max(x[:, ii]) - min(x[:, ii]) for ii in range(n_features)], dtype=x.dtype)
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            mat1[i, j] = xyd(x[i], x[j], vec3)
            mat1[j, i] = xyd(x[i], x[j], vec3)
    return mat1


# 计算每个样本点的相异度参数
def xyd_parameter(x):
    mat1 = xyd_mat(x)
    # 样本的平均相异度的计算
    mean_r = np.sum(np.sum(mat1)) / np.power(x.shape[0], 2)
    return np.sum(mat1 < mean_r, axis=1) - 1, mat1, mean_r


# 基于相异度矩阵的初始聚类中心选取
def init_centers(x, k):
    center_s = np.array(np.ones((k, x.shape[1])), dtype=x.dtype)
    xyd_para, mat1, mean_r = xyd_parameter(x)
    xyd_para.astype(np.int32)
    # 出现相异度最大值不唯一的情况次数
    uniq_num = 0
    for i in range(k):
        # 如果相异度参数最大值唯一
        if np.argwhere(xyd_para == np.max(xyd_para)).shape[0] == 1:
            center_s[i] = x[np.argmax(xyd_para)]
            # 删除数组元素
            pd2 = DataFrame(x)
            index = np.nonzero(mat1[np.argmax(xyd_para)] < mean_r)[0]
            x = np.array(pd2.drop(index, axis=0))
            mat1 = np.delete(mat1, index, axis=0)
            mat1 = np.delete(mat1, index, axis=1)
            mean_r = np.sum(np.sum(mat1)) / np.power(x.shape[0], 2)
            xyd_para = np.sum(mat1 < mean_r, axis=1) - 1
        #  相异度参数最大值不唯一
        else:
            arr1 = np.argwhere(xyd_para == np.max(xyd_para))[:, 0]
            sum1 = [0] * len(arr1)
            for ii, jj in enumerate(arr1):
                sum1[ii] = np.sum(mat1[jj][mat1[jj] < mean_r])
            center_s[i] = x[arr1[np.argmax(sum1)]]
            # print(sum1)
            # 删除数组元素
            pd2 = DataFrame(x)
            index = np.nonzero(mat1[np.argmax(xyd_para)] < mean_r)[0]
            x = np.array(pd2.drop(index, axis=0))
            mat1 = np.delete(mat1, index, axis=0)
            mat1 = np.delete(mat1, index, axis=1)
            mean_r = np.sum(np.sum(mat1)) / np.power(x.shape[0], 2)
            xyd_para = np.sum(mat1 < mean_r, axis=1) - 1
            uniq_num += 1
    print("出现相异度参数最大值不唯一的次数：", uniq_num)
    return center_s


if __name__ == '__main__':
    data_name = input("请输入测试数据集的名称：")
    if data_name == 'iris':
        # 输入iris测试数据集
        pd1 = pd.read_csv('iris.txt', names=['a', 'b', 'c', 'd', 'e'], index_col='e')
        # K 为聚类数目
        K = 3
        result_stand = np.select([np.array(pd1.index) == pd1.index.unique()[0], np.array(pd1.index) ==
                                  pd1.index.unique()[1], np.array(pd1.index) == pd1.index.unique()[2]], [1, 2, 3])
    elif data_name == 'wine':
        # 输入wine测试数据集
        pd1 = pd.read_csv("wine.txt", header=None)
        pd1 = pd1.set_index([0])
        # K 为聚类数目
        K = 3
        result_stand = np.select([np.array(pd1.index) == pd1.index.unique()[0], np.array(pd1.index) ==
                                  pd1.index.unique()[1], np.array(pd1.index) == pd1.index.unique()[2]], [1, 2, 3])
    elif data_name == 'haberman':
        # 输入 haberman 数据集
        pd1 = pd.read_csv("haberman.txt", header=None)
        pd1 = pd1.set_index([3])
        # K 为聚类数目
        K = 2
        result_stand = np.select([np.array(pd1.index) == pd1.index.unique()[0], np.array(pd1.index) ==
                                  pd1.index.unique()[1]], [1, 2])
    elif data_name == 'diabetes':
        # 输入 diabetes测试数据集
        xls_file = pd.ExcelFile('diabetes.xls')
        table = xls_file.parse('Sheet1')
        pd1 = table.set_index(['Diabetes'])
        pd1 = pd1.drop('id', axis=1)
        # K 为聚类数目
        K = 2
        result_stand = np.select([np.array(pd1.index) == pd1.index.unique()[0], np.array(pd1.index) ==
                                  pd1.index.unique()[1]], [1, 2])
    elif data_name == 'spiral':
        # 输入spiral测试数据集
        pd1 = pd.read_csv("spiral.txt", header=None, sep='\s+')
        K = 3
    elif data_name == 'pasepathed':
        # 输入pasepathed测试数据集
        pd1 = pd.read_csv("pasepathed.txt", header=None, sep='\s+')
        pd1 = pd1.set_index([2])
        K = 3
        result_stand = np.array(pd1.index)
    elif data_name == 'seed':
        # 输入seed测试数据集
        pd1 = pd.read_csv("seed.txt", header=None, sep='\s+')
        pd1 = pd1.set_index([7])
        K = 3
        result_stand = np.select([np.array(pd1.index) == pd1.index.unique()[0], np.array(pd1.index) ==
                                  pd1.index.unique()[1], np.array(pd1.index) == pd1.index.unique()[2]], [1, 2, 3])
        pd1 = pd1.set_index([1])
    elif data_name == 'ionosphere':
        # 输入libras测试数据集
        pd1 = pd.read_csv("ionosphere.txt", header=None, sep=',')
        pd1 = pd1.set_index([34])
        K = 2
        result_stand = np.select([np.array(pd1.index) == pd1.index.unique()[0], np.array(pd1.index) ==
                                  pd1.index.unique()[1]], [1, 2])
        pd1 = pd1.set_index([1])
    elif data_name == 'aggregation':
        # 输入aggregation测试数据集
        pd1 = pd.read_csv("aggregation.txt",header=None, sep='\s+')
        K = 7
    elif data_name == 'flame':
        # 输入flame测试数据集
        pd1 = pd.read_csv("flame.txt", header=None, sep='\s+')
        K = 2
    process = input("请输入预处理数据方法：")
    if process == "标准化":
        X = preprocessing.scale(np.array(pd1, dtype=np.float64))
    elif process == "最大最小":
        X = preprocessing.MinMaxScaler().fit_transform(np.array(pd1, dtype=np.float64))
    elif process == "归一化":
        X = preprocessing.normalize(np.array(pd1, dtype=np.float64), norm='l2')
    else:
        X = np.array(pd1)

    # 根据肘部法则确定K
    SSE = np.nonzero(20)
    for i in range(1, 21):
        centers = init_centers(X, i)
        kmeans = KMeans(n_clusters=i, init=centers, n_init=1, max_iter=300, tol=0.0001).fit(X)
        SSE[i-1] = kmeans.inertia_
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(range(1, 21), SSE, 'b*-')

    K = input("请输入类簇个数：")
    centers = init_centers(X, K)
    print("初始聚类中心为; \n", centers)
    kmeans = KMeans(n_clusters=K, init=centers, n_init=1, max_iter=300, tol=0.0001).fit(X)
    result = kmeans.labels_
    sse = kmeans.inertia_
    # compute rate
    if data_name == 'spiral' or data_name == 'flame' or data_name == 'aggregation' or data_name == 'pasepathed':
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        rgb = np.random.random((20, 3))
        color = ['b', 'r', 'k', 'y', 'c', 'g', 'm']
        num = len(np.unique(result))
        marker = ['o', 'v', '1', '8', 's', 'p', '*', '+', 'x', 'd', '<', 'D', '>']
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        for j, i in enumerate(np.unique(result)):
            ax2.scatter(X[np.nonzero(result == i)[0], 0], X[np.nonzero(result == i)[0], 1], marker=marker[i], c=rgb[i],
                        s=18)
            plt.tick_params(labelsize=15)
        plt.show()
    else:
        result = result + 1
        arr1 = np.unique(result)
        if len(arr1) == 3:
            rate_list = [0]*6
            rate_list[0] = metrics.accuracy_score(result_stand, np.select([result == arr1[0], result == arr1[1],
                                                                           result == arr1[2]],
                                                                          [arr1[0], arr1[1], arr1[2]]))
            rate_list[1] = metrics.accuracy_score(result_stand, np.select([result == arr1[0], result == arr1[1],
                                                                           result == arr1[2]],
                                                                          [arr1[0], arr1[2], arr1[1]]))
            rate_list[2] = metrics.accuracy_score(result_stand, np.select([result == arr1[0], result == arr1[1],
                                                                           result == arr1[2]],
                                                                          [arr1[1], arr1[0], arr1[2]]))
            rate_list[3] = metrics.accuracy_score(result_stand, np.select([result == arr1[0], result == arr1[1],
                                                                           result == arr1[2]],
                                                                          [arr1[1], arr1[2], arr1[0]]))
            rate_list[4] = metrics.accuracy_score(result_stand, np.select([result == arr1[0], result == arr1[1],
                                                                           result == arr1[2]],
                                                                          [arr1[2], arr1[0], arr1[1]]))
            rate_list[5] = metrics.accuracy_score(result_stand, np.select([result == arr1[0], result == arr1[1],
                                                                           result == arr1[2]],
                                                                          [arr1[2], arr1[1], arr1[0]]))
            rate = np.max(rate_list)
        elif len(arr1) == 2:
            rate_list = [0] * 2
            rate_list[0] = metrics.accuracy_score(result_stand, np.select([result == arr1[0], result == arr1[1]],
                                                                          [arr1[0], arr1[1]]))
            rate_list[1] = metrics.accuracy_score(result_stand, np.select([result == arr1[0], result == arr1[1]],
                                                                          [arr1[1], arr1[0]]))
            rate = np.max(rate_list)
        arr1 = np.unique(result)
        rate_list = []
        if K == 2:
            if len(arr1) >= K:
                for i in itertools.permutations(arr1, K):
                    rate_list.append(
                        metrics.accuracy_score(np.select([result_stand == 1, result_stand == 2], np.array(i)), result))
                    rate = np.max(rate_list)
            else:
                for i in itertools.permutations(np.unique(result_stand), len(arr1)):
                    rate_list.append(metrics.accuracy_score(result_stand, np.select([result == 1], np.array(i))))
                    rate = np.max(rate_list)

        elif K == 3:
            if len(arr1) >= K:
                for i in itertools.permutations(arr1, K):
                    rate_list.append(metrics.accuracy_score(
                        np.select([result_stand == 1, result_stand == 2, result_stand == 3], np.array(i)), result))
                rate = np.max(rate_list)
            else:
                if len(arr1) == 2:
                    for i in itertools.permutations(np.unique(result_stand), len(arr1)):
                        rate_list.append(
                            metrics.accuracy_score(result_stand, np.select([result == 1, result == 2], np.array(i))))
                    rate = np.max(rate_list)
                else:
                    for i in itertools.permutations(np.unique(result_stand), len(arr1)):
                        rate_list.append(metrics.accuracy_score(result_stand, np.select([result == 1], np.array(i))))
                    rate = np.max(rate_list)
        # print(list_rate)
        AMI = metrics.v_measure_score(result_stand, result)
        ARI = metrics.adjusted_rand_score(result_stand, result)
        print("准确率为: %.4f" % rate,'\n', "AMI: ", AMI, '\n', "ARI: ", ARI, '\n', ''"迭代次数为: ",  kmeans.n_iter_, '\n'
              , "误差平方和为: %.4f" % sse)

