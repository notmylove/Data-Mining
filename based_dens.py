import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics



def xyd(vec1, vec2, vec3):
    return np.sum(np.abs(vec1-vec2)/vec3)


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
    K = 7
    k_dist = np.sort(mat1, axis=1)[:, K]
    k_near = np.sort(mat1, axis=1)[:, 1:K + 1]
    k_near_weighs = k_near/np.sum(k_near, axis=1).reshape(k_near.shape[0], 1)
    xyd_para = np.sum(np.exp(-k_near) * k_near_weighs, axis=1)
    mat_knn = np.argsort(mat1, axis=1)
    return xyd_para, mat1, k_dist, mat_knn, K


def dis_cluster(arr1, arr2, mat1, d_c):
    for i in arr1:
        if np.min(mat1[i, arr2]) < d_c:
            print('merge')
            a = np.min(mat1[i, arr2])
            print(a, d_c)
            return 1
            break
    return 0


# 基于相异度矩阵的自动确定聚类中心的密度峰值算法
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
    # Ec = np.nonzero(delt >= (np.mean(delt)+1.2*np.std(delt)))[0]
    # 归一化
    # delt = (delt - np.min(delt) / (np.max(delt) - np.min(delt)))
    # xyd_para = (xyd_para - np.min(xyd_para) / (np.max(xyd_para) - np.min(xyd_para)))

    garmar = xyd_para * delt
    Ec = np.nonzero(garmar >= (np.mean(garmar) + 1.5 * np.std(garmar)))[0]
    Lc = np.nonzero(xyd_para >= np.median(xyd_para))[0]
    center_choice = []

    for i in Ec:
        if np.any(Lc == i):
            center_choice.append(i)
    # center_choice = (xyd_para*delt).argsort()[::-1][0:3]
    result = np.zeros(x.shape[0], dtype=np.int)
    k = len(center_choice)
    center_s = np.array(np.ones((k, x.shape[1])), dtype=x.dtype)
    center_s[:] = x[center_choice, :]

    kerner_point = []
    for i in range(len(k_dist)):
        if k_dist[i] < np.mean(k_dist[mat_knn[i, 1:K+1]]):
            kerner_point.append(i)
    kerner_point = np.array(kerner_point, dtype=np.int)
    kerner_point = np.hstack((kerner_point, center_choice))
    kerner_point = np.unique(kerner_point)
    print(kerner_point, center_choice)
    num = len(kerner_point)
    A = np.zeros((num, num), dtype=np.int)
    # k_dist2 = np.sort(k_dist[kerner_point])[::-1]
    for i in range(num):
        for j in range(i+1, num):
            if mat1[kerner_point[i], kerner_point[j]] <= np.min([k_dist[kerner_point[i]], k_dist[kerner_point[j]]]):
                # np.max([k_dist[kerner_point[i]], k_dist[kerner_point[j]]])
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
        result[np.unique(mat_knn[i, 0:K+1])] = j+1
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
                P[i, 0:ii+1] = np.bincount(result[mat_knn[j, 1:K + 1]])[1:ii+2]
                break

    while 1:
        num1 = np.sum(result==0)
        # instance_zero_1 = np.nonzero(result == 0)[0]
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
                        P[j, result[ii]-1] = P[j, result[ii]-1] + 1
        num2 = np.sum(result == 0)
        if num1 == num2:
            break
    print(len(np.nonzero(result == 0)[0]))
    # 与密度比它大最近的点所属的类一样
    for i, j in enumerate(arr1):
        if result[j] == 0:
            print(i)
            result[j] = result[arr1[np.argmin(mat1[j, arr1[0:i]])]]


    result_origin = result.copy()
    # merge
    k = len(np.unique(result))
    cent = [0] * k
    for i in range(k):
        cent[i] = Lc[[np.any(np.nonzero(result == i+1)[0] == j) for j in Lc]]
        # cent[i] = kerner_point[[np.any(cent[i] == j) for j in kerner_point]]
        # cent[i] = kerner_point[[np.any(np.nonzero(result == i+1)[0] == j) for j in kerner_point]]
    # k_dist3 = k_dist[Lc[[np.any(kerner_point == j) for j in Lc]]]
    # print(kerner_point, Lc[[np.any(kerner_point == j) for j in Lc]])

    cut_distance = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            ker1 = kerner_point[[np.any(np.nonzero(result == i + 1)[0] == jj) for jj in kerner_point]]
            ker2 = kerner_point[[np.any(np.nonzero(result == j + 1)[0] == jj) for jj in kerner_point]]
            cut_distance[i, j] = (np.min(k_dist[Lc[[np.any(ker1 == jj) for jj in Lc]]]) + np.min(k_dist[Lc[[np.any(ker2 == jj) for jj in Lc]]]))/2

    for i in range(k):
        for j in range(i+1, k):
            if dis_cluster(cent[i], cent[j], mat1, cut_distance[i, j]) == 1:
                # print(i+1, j+1)
                result = np.select([result == j+1], [i+1], result)

    for j, i in enumerate(np.unique(result)):
        result = np.select([result == i], [j+1], result)
    return center_s, xyd_para, delt, result, kerner_point, A, kerner_1, P, result_origin


if __name__ == '__main__':
    data_name = input("请输入测试数据集的名称：")
    if data_name == 'glass':
        # 输入iris测试数据集
        pd1 = pd.read_csv("glass.txt", header=None)
        # K 为聚类数目
        K = 3
        pd1 = pd1.set_index([pd1.shape[1] - 1])
        result_stand = np.array(pd1.index)
    elif data_name == 'lympho':
        # 输入wine测试数据集
        pd1 = pd.read_csv("lympho.txt", header=None)
        # K 为聚类数目
        K = 3
        pd1 = pd1.set_index([pd1.shape[1] - 1])
        result_stand = np.array(pd1.index)
    elif data_name == 'wbc':
        # 输入 haberman 数据集
        pd1 = pd.read_csv("wbc.txt", header=None)
        # K 为聚类数目
        K = 3
        pd1 = pd1.set_index([pd1.shape[1] - 1])
        result_stand = np.array(pd1.index)
    elif data_name == 'ionosphere':
        # 输入libras测试数据集
        pd1 = pd.read_csv("ionosphere.txt", header=None)
        K = 2
        pd1 = pd1.set_index([pd1.shape[1] - 1])
        result_stand = np.array(pd1.index)
    elif data_name == 'pima':
        # 输入libras测试数据集
        pd1 = pd.read_csv("pima.txt", header=None)
        K = 3
        pd1 = pd1.set_index([pd1.shape[1] - 1])
        result_stand = np.array(pd1.index)
    process = input("请输入预处理数据方法：")
    if process == "标准化":
        X = preprocessing.scale(np.array(pd1, dtype=np.float64))
    elif process == "最大最小":
        X = preprocessing.MinMaxScaler().fit_transform(np.array(pd1, dtype=np.float64))
    elif process == "归一化":
        X = preprocessing.normalize(np.array(pd1, dtype=np.float64), norm='l2')
    else:
        X = np.array(pd1)
    centers, xyd_para, delt, result, kerner_point, A, kerner_1, P, result_origin = init_centers(X)
    outlier_result = np.zeros(len(result))
    for i in np.unique(result):
        arr_1 = np.nonzero(result == i)[0]
        x = X[arr_1]
        n_sample, n_features = x.shape
        arr = np.zeros(n_sample, dtype=np.float64)
        vec3 = np.array([max(x[:, ii]) - min(x[:, ii]) for ii in range(n_features)], dtype=x.dtype)
        cen_1 = np.mean(x, axis=0)
        for j in range(n_sample):
            arr[j] = xyd(x[j], cen_1, vec3)
        norm_arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
        result_1 = np.select([norm_arr >= 0.6, norm_arr < 0.6], [1, 0])
        outlier_result[arr_1] = result_1
    AUC = metrics.roc_auc_score(result_stand, outlier_result)
    print("AUC为: ", AUC)
