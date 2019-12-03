from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn import metrics
import itertools


def xyd(vec1, vec2, vec3):
    return np.sum(np.abs(vec1 - vec2) / vec3)


def xyd_mat(x):
    n_sample, n_features = x.shape
    mat1 = np.zeros((n_sample, n_sample), dtype=np.float64)
    vec3 = np.array([max(x[:, ii]) - min(x[:, ii]) for ii in range(n_features)], dtype=x.dtype)
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            mat1[i, j] = xyd(x[i], x[j], vec3)
            mat1[j, i] = xyd(x[i], x[j], vec3)
    return mat1


# 计算每个样本点的相异度参数
def xyd_parameter(x):
    mat1 = xyd_mat(x)
    # k-nearest
    K = 7
    k_dist = np.sort(mat1, axis=1)[:, K]
    k_near = np.sort(mat1, axis=1)[:, 1:K + 1]
    k_near_weighs = k_near / np.sum(k_near, axis=1).reshape(k_near.shape[0], 1)
    xyd_para = np.sum(np.exp(-k_near) * k_near_weighs, axis=1)
    mat_knn = np.argsort(mat1, axis=1)
    return xyd_para, mat1, k_dist, mat_knn, K


# 基于相异度矩阵的初始聚类中心选取
def init_centers(x):
    xyd_para, mat1, k_dist, mat_knn, K = xyd_parameter(x)
    delt = np.zeros(x.shape[0], dtype=np.float64)
    # xyd_para最大值不唯一
    arr1 = np.argsort(xyd_para)[::-1]
    for i, j in enumerate(arr1):
        if i == 0:
            delt[j] = np.max(mat1[j, :])
        else:
            delt[j] = np.min(mat1[j, arr1[:i]])
    # 根据决策图选取初始聚类中心
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.scatter(xyd_para, delt, marker='*', c='b', s=20)
    plt.tick_params(labelsize=15)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.scatter(np.linspace(0, len(delt), len(delt)), np.sort(xyd_para * delt)[::-1], marker='o', c='y', s=20)
    plt.tick_params(labelsize=15)
    plt.show()
    number = input("请输入密度峰值点：")
    center_choice = (xyd_para*delt).argsort()[::-1][0:number]
    result = np.zeros(x.shape[0], dtype=np.int)
    k = len(center_choice)
    center_s = np.array(np.ones((k, x.shape[1])), dtype=x.dtype)
    center_s[:] = x[center_choice, :]
    kerner_point = []
    for i in range(len(k_dist)):
        if k_dist[i] < np.mean(k_dist[mat_knn[i, 1:K + 1]]):
            kerner_point.append(i)
    kerner_point = np.array(kerner_point, dtype=np.int)
    kerner_point = np.hstack((kerner_point, center_choice))
    kerner_point = np.unique(kerner_point)
    print(kerner_point, center_choice)
    num = len(kerner_point)
    A = np.zeros((num, num), dtype=np.int)
    for i in range(num):
        for j in range(i + 1, num):
            if mat1[kerner_point[i], kerner_point[j]] <= np.min([k_dist[kerner_point[i]], k_dist[kerner_point[j]]]):
                A[i, j] = 1
                A[j, i] = 1

    for i in range(num):
        if len(np.nonzero(A[:, i] == 1)[0]) != 0:
            for j in np.nonzero(A[:, i] == 1)[0]:
                A[j, :] = A[j, :] | A[i, :]

    kerner_1 = []
    for i, j in enumerate(center_choice):
        if len(np.argwhere(kerner_point == j)) != 0:
            kerner_1.append(kerner_point[np.nonzero(A[np.argwhere(kerner_point == j)[0][0], :])[0]])

    print(k, kerner_1)
    for j, i in enumerate(kerner_1):
        result[np.unique(mat_knn[i, 0:K + 1])] = j + 1
    for j, i in enumerate(np.unique(result)):
        result = np.select([result == i], [j], result)

    for i in np.where(result != 0)[0]:
        part = np.bincount(result[mat_knn[i, 1:K + 1]])[1:]
        if len(part) == 0:
            result[i] = 0
        elif len(np.where(part == np.max(part))[0]) == 1:
            result[i] = np.argmax(np.bincount(result[mat_knn[i, 1:K + 1]])[1:]) + 1

    # 分配策略
    instance_zero = np.nonzero(result == 0)[0]
    mat_knn_instance = mat_knn[instance_zero, 1:K + 1]
    P = np.zeros((len(np.nonzero(result == 0)[0]), len(np.unique(result)) - 1))
    for i, j in enumerate(instance_zero):
        for ii in range(len(np.unique(result)) - 1):
            if len(np.bincount(result[mat_knn[j, 1:K + 1]])) == 2 + ii:
                P[i, 0:ii + 1] = np.bincount(result[mat_knn[j, 1:K + 1]])[1:ii + 2]
                break

    while 1:
        num1 = np.sum(result == 0)
        if np.max(P) == K:
            a = np.where(P == np.max(P))
            result[instance_zero[a[0]]] = a[1] + 1
            P[a[0], :] = 0
            mat_knn_instance[a[0], :] = 10000
            for ii, i in enumerate(instance_zero[a[0]]):
                if len(np.where(mat_knn_instance == i)[0]) != 0:
                    for j in np.where(mat_knn_instance == i)[0]:
                        P[j, a[1][ii]] = P[j, a[1][ii]] + 1
        elif 0 < np.max(P) < K:
            a = np.where(P == np.max(P))
            ins_index = instance_zero[a[0]][np.argmax(xyd_para[instance_zero[a[0]]])]
            b = np.where(instance_zero[a[0]] == ins_index)[0]
            if len(b) == 1:
                result[ins_index] = a[1][b[0]] + 1
                P[a[0][b[0]], :] = 0
                mat_knn_instance[a[0][b[0]], :] = 10000
                if len(np.where(mat_knn_instance == ins_index)[0]) != 0:
                    for j in np.where(mat_knn_instance == ins_index)[0]:
                        P[j, a[1][b[0]]] = P[j, a[1][b[0]]] + 1
            else:
                for i in mat_knn[ins_index, 1:K + 1]:
                    if result[i] != 0:
                        result[ins_index] = result[i]
                        P[a[0][b[0]], :] = 0
                        mat_knn_instance[a[0][b[0]], :] = 10000
                        ii = i
                        break
                if len(np.where(mat_knn_instance == ins_index)[0]) != 0:
                    for j in np.where(mat_knn_instance == ins_index)[0]:
                        P[j, result[ii] - 1] = P[j, result[ii] - 1] + 1
        num2 = np.sum(result == 0)
        if num1 == num2:
            break
    print(len(np.nonzero(result == 0)[0]))

    # 与密度比它大最近的点所属的类一样
    for i, j in enumerate(arr1):
        if result[j] == 0:
            result[j] = result[arr1[np.argmin(mat1[j, arr1[0:i]])]]

    for j, i in enumerate(np.unique(result)):
        result = np.select([result == i], [j + 1], result)
    return center_s, xyd_para, delt, result, kerner_point, A, kerner_1, P


if __name__ == '__main__':
    data_name = input("请输入测试数据集的名称：")
    process = input("请输入预处理数据方法：")
    if data_name == 'iris':
        # 输入iris测试数据集
        pd1 = pd.read_csv('iris.txt', names=['a', 'b', 'c', 'd', 'e'], index_col='e')
        # K 为聚类数目
        K = 3
        result_stand = np.select([np.array(pd1.index) == pd1.index.unique()[0], np.array(pd1.index) ==
                                  pd1.index.unique()[1], np.array(pd1.index) == pd1.index.unique()[2]], [1, 2, 3])

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
    elif data_name == 'libras':
        # 输入libras测试数据集
        pd1 = pd.read_csv("libras.txt", header=None, sep=',')
        pd1 = pd1.set_index([90])
        K = 15
        result_stand = np.array(pd1.index)
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
        pd1 = pd.read_csv("aggregation.txt", header=None, sep='\s+')
        K = 7
    elif data_name == 'flame':
        # 输入flame测试数据集
        pd1 = pd.read_csv("flame.txt", header=None, sep='\s+')
        K = 2
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
    # process = input("请输入预处理数据方法：")
    if process == "标准化":
        X = preprocessing.scale(np.array(pd1, dtype=np.float64))
        # X = TSNE(n_components=2, init='pca').fit_transform(X)
    elif process == "最大最小化":
        X = preprocessing.MinMaxScaler().fit_transform(np.array(pd1, dtype=np.float64))
    elif process == "归一化":
        X = preprocessing.normalize(np.array(pd1, dtype=np.float64), norm='l2')
    else:
        X = np.array(pd1)
    centers, xyd_para, delt, result, kerner_point, A, kerner_1, P = init_centers(X)
    if data_name == 'spiral' or data_name == 'flame' or data_name == 'aggregation' or data_name == 'pasepathed':
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        rgb = np.random.random((20, 3))
        color = ['b', 'r', 'k', 'y', 'c', 'g', 'm']
        num = len(np.unique(result))
        marker = ['o', 'v', '1', '8', 's', 'p', '*', '+', 'x', 'd', '<', 'D', '>']
        ax1.scatter(X[:, 0], X[:, 1], marker=marker[0], c=rgb[1], s=17)
        ax1.scatter(X[kerner_point, 0], X[kerner_point, 1], marker=marker[6], c=rgb[6], s=30)
        plt.tick_params(labelsize=15)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        for j, i in enumerate(np.unique(result)):
            ax2.scatter(X[np.nonzero(result == i)[0], 0], X[np.nonzero(result == i)[0], 1], marker=marker[i], c=rgb[i],
                        s=18)
            plt.tick_params(labelsize=15)
        for j in range(centers.shape[0]):
            ax2.scatter(centers[j][0], centers[j][1], marker=marker[0], c='r', s=28)
        plt.show()

    else:
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
        AMI = metrics.v_measure_score(result_stand, result)
        ARI = metrics.adjusted_rand_score(result_stand, result)
        print('rate: ', rate, '\n', 'AMI: ', AMI, '\n', 'ARI: ', ARI)
