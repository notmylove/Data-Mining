from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor



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
        K = 3
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
    centers = init_centers(X, K)
    print("初始聚类中心为; \n", centers)
    kmeans = KMeans(n_clusters=K, init=centers, n_init=1, max_iter=300, tol=0.0001).fit(X)
    result = kmeans.labels_
    sse = kmeans.inertia_
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
    lof = LocalOutlierFactor(n_neighbors=10)
    y = lof.fit_predict(np.array(pd1, dtype=np.float64))
    lof_result = np.select([y == -1, y == 1], [1, 0])
    AUC_1 = metrics.roc_auc_score(result_stand, lof_result)
    print("AUC_`为: ", AUC_1)




