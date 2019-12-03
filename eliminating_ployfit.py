import numpy as np
from array import array
from scipy import interpolate
import matplotlib.pyplot as plt


class Eliminating:
    def __init__(self, arr1, arr2, arr1_index):
        self.arr1 = arr1
        self.arr2 = arr2
        self.arr1_index = arr1_index

    def spline_smoothing(self):
        # 样条平滑方法,平滑系数s取默认值
        spline1 = interpolate.UnivariateSpline(self.arr1, self.arr2, k=3)
        arr3 = spline1(self.arr1)
        arr4 = self.arr2 - arr3
        # 残差的标准差估计值
        sigma = arr4.std(ddof=1)
        # 判断门限
        self.lamda = 2.5 * sigma

    def zhenglian(self, arr1, arr2, arr1_index):
        # 连续四点检验,构建三阶差分模型
        for i in range(0, len(arr1) - 3):
            if np.abs(np.diff(arr2[i:i + 4], n=3)) < self.lamda:
                self.k = i + 3 + arr1_index[0]
                break
            elif i == int(len(arr1) - 4):
                print('模型失效')
                self.k = i + 3 + arr1_index[0]
                break
        return self.k

    def zhengjian_test(self, arr1, arr2, arr1_index):
        k_1 = self.zhenglian(arr1, arr2, arr1_index)
        k = k_1 - arr1_index[0]
        result1 = np.ones(len(arr1))
        list1 = array('d', [])
        # 这部分未被检测，默认为正常
        if k != 3:
            result1[arr1_index[0]:k-3] = 3
        # 正向递推
        # 4个连续的递推序列
        csth = np.copy(arr1[k - 3:k + 1]).tolist()
        csnh = np.copy(arr2[k - 3:k + 1]).tolist()
        for i in range(k + 1, len(arr1), 1):
            if np.abs(np.polyval(np.polyfit(csth, csnh, 3), arr1[i]) - arr2[i]) < self.lamda:
                result1[i] = 1
                del csth[0]
                del csnh[0]
                csth.append(arr1[i])
                csnh.append(arr2[i])
            elif np.abs(np.polyval(np.polyfit(csth, csnh, 2), arr1[i]) - arr2[i]) < self.lamda:
                result1[i] = 1
                del csth[0]
                del csnh[0]
                csth.append(arr1[i])
                csnh.append(arr2[i])
            elif np.abs(np.polyval(np.polyfit(csth, csnh, 1), arr1[i]) - arr2[i]) < self.lamda:
                result1[i] = 1
                del csth[0]
                del csnh[0]
                csth.append(arr1[i])
                csnh.append(arr2[i])
            else:
                result1[i] = 0
                list1.append(i)
                if len(list1) >= 10:
                    # 连续异常数据超限(这里设置的是最大连续异常为10)，模型失效，需要数据分段
                    if np.sum(np.diff(list1[-10:])) == 9:
                        f = int(list1[-10]) + arr1_index[0]
                        print("连续异常超限")
                        return f
                        break
        return 0

    def zhengjian(self, num2):
        if len(num2) == 0:
            k_1 = self.zhenglian(self.arr1, self.arr2, self.arr1_index)
            k = k_1 - self.arr1_index[0]
            result1 = np.ones(len(self.arr1))
            # 这部分未被检测，默认为正常
            if k != 3:
                result1[0:k-3] = 3
            # 正向递推
            csth = np.copy(self.arr1[k - 3:k + 1]).tolist()
            csnh = np.copy(self.arr2[k - 3:k + 1]).tolist()
            for i in range(k + 1, len(self.arr1), 1):
                if np.abs(np.polyval(np.polyfit(csth, csnh, 3), self.arr1[i]) - self.arr2[i]) < self.lamda:
                    result1[i] = 1
                    del csth[0]
                    del csnh[0]
                    csth.append(self.arr1[i])
                    csnh.append(self.arr2[i])
                elif np.abs(np.polyval(np.polyfit(csth, csnh, 2), self.arr1[i]) - self.arr2[i]) < self.lamda:
                    result1[i] = 1
                    del csth[0]
                    del csnh[0]
                    csth.append(self.arr1[i])
                    csnh.append(self.arr2[i])
                elif np.abs(np.polyval(np.polyfit(csth, csnh, 1), self.arr1[i]) - self.arr2[i]) < self.lamda:
                    result1[i] = 1
                    del csth[0]
                    del csnh[0]
                    csth.append(self.arr1[i])
                    csnh.append(self.arr2[i])
                else:
                    result1[i] = 0
            return result1, k_1
        else:
            # 分段处理
            list_k = []
            result_1 = np.array([])
            for i in range(len(num2)-1):
                # 这里是对arr1数据进行分段处理得到新的arr_1,对arr_1数据进行处理的结果
                arr_1 = self.arr1[num2[i]:num2[i+1]]
                arr_2 = self.arr2[num2[i]:num2[i+1]]
                arr_1_index = self.arr1_index[num2[i]:num2[i + 1]]
                k_1 = self.zhenglian(arr_1, arr_2, arr_1_index)
                k = k_1 - arr_1_index[0]
                list_k.append(k_1)
                result1 = np.ones(len(arr_1))
                # 这部分未被检测，默认为正常
                if k != 3:
                    result1[0:k - 3] = 3
                # 正向递推
                csth = np.copy(arr_1[k - 3:k + 1]).tolist()
                csnh = np.copy(arr_2[k - 3:k + 1]).tolist()
                for j in range(k + 1, len(arr_1), 1):
                    if np.abs(np.polyval(np.polyfit(csth, csnh, 3), arr_1[j]) - arr_2[j]) < self.lamda:
                        result1[j] = 1
                        del csth[0]
                        del csnh[0]
                        csth.append(arr_1[j])
                        csnh.append(arr_2[j])
                    elif np.abs(np.polyval(np.polyfit(csth, csnh, 2), arr_1[j]) - arr_2[j]) < self.lamda:
                        result1[j] = 1
                        del csth[0]
                        del csnh[0]
                        csth.append(arr_1[j])
                        csnh.append(arr_2[j])
                    elif np.abs(np.polyval(np.polyfit(csth, csnh, 1), arr_1[j]) - arr_2[j]) < self.lamda:
                        result1[j] = 1
                        del csth[0]
                        del csnh[0]
                        csth.append(arr_1[j])
                        csnh.append(arr_2[j])
                    else:
                        result1[j] = 0
                # 对分段处理结果进行连接
                result_1 = np.hstack((result_1, result1))
            return result_1, list_k

    def nilian(self, arr1, arr2, arr1_index):
        for i in range(len(arr1)-1, 2, -1):
            if np.abs(np.diff(arr2[i - 3:i + 1], n=3)) < self.lamda:
                k1 = i - 3 + arr1_index[0]
                break
            elif i == 3:
                print('模型失效')
                k1 = i - 3 + arr1_index[0]
                break
        return k1

    # 逆向递推
    def nijian(self, num2):
        if len(num2) == 0:
            k1 = self.nilian(self.arr1, self.arr2, self.arr1_index)
            result2 = np.ones(len(self.arr1))
            # 这部分未被检测，默认为正常
            if k1 != len(self.arr1)-4:
                result2[k1+4:] = 3
            # 逆向递推
            csth = np.copy(self.arr1[k1:k1 + 4]).tolist()
            csnh = np.copy(self.arr2[k1:k1 + 4]).tolist()
            for i in range(k1 - 1, -1, -1):
                if np.abs(np.polyval(np.polyfit(csth, csnh, 3), self.arr1[i]) - self.arr2[i]) < self.lamda:
                    result2[i] = 1
                    del csth[-1]
                    del csnh[-1]
                    csth.insert(0, self.arr1[i])
                    csnh.insert(0, self.arr2[i])
                elif np.abs(np.polyval(np.polyfit(csth, csnh, 2), self.arr1[i]) - self.arr2[i]) < self.lamda:
                    result2[i] = 1
                    del csth[0]
                    del csnh[0]
                    csth.insert(0, self.arr1[i])
                    csnh.insert(0, self.arr2[i])
                elif np.abs(np.polyval(np.polyfit(csth, csnh, 1), self.arr1[i]) - self.arr2[i]) < self.lamda:
                    result2[i] = 1
                    del csth[0]
                    del csnh[0]
                    csth.insert(0, self.arr1[i])
                    csnh.insert(0, self.arr2[i])
                else:
                    result2[i] = 0
            return result2, k1
        else:
            list_k = []
            result_2 = np.array([])
            for i in range(len(num2)-1):
                # 这里是对arr1数据进行分段处理得到新的arr_1,对arr_1数据进行处理的结果,同上一样
                arr_1 = self.arr1[num2[i]:num2[i + 1]]
                arr_2 = self.arr2[num2[i]:num2[i + 1]]
                arr_1_index = self.arr1_index[num2[i]:num2[i + 1]]
                k1 = self.nilian(arr_1, arr_2, arr_1_index)
                list_k.append(k1)
                k = k1 - arr_1_index[0]
                result2 = np.ones(len(arr_1))
                # 这部分未被检测，默认为正常
                if k != len(arr_1)-4:
                    result2[k+4:] = 3
                # 逆向递推
                csth = np.copy(arr_1[k:k + 4]).tolist()
                csnh = np.copy(arr_2[k:k + 4]).tolist()
                for i in range(k - 1, -1, -1):
                    if np.abs(np.polyval(np.polyfit(csth, csnh, 3), arr_1[i]) - arr_2[i]) < self.lamda:
                        result2[i] = 1
                        del csth[-1]
                        del csnh[-1]
                        csth.insert(0, arr_1[i])
                        csnh.insert(0, arr_2[i])
                    elif np.abs(np.polyval(np.polyfit(csth, csnh, 2), arr_1[i]) - arr_2[i]) < self.lamda:
                        result2[i] = 1
                        del csth[0]
                        del csnh[0]
                        csth.insert(0, arr_1[i])
                        csnh.insert(0, arr_2[i])
                    elif np.abs(np.polyval(np.polyfit(csth, csnh, 1), arr_1[i]) - arr_2[i]) < self.lamda:
                        result2[i] = 1
                        del csth[0]
                        del csnh[0]
                        csth.insert(0, arr_1[i])
                        csnh.insert(0, arr_2[i])
                    else:
                        result2[i] = 0
                # 对分段处理的结果横向连接起来
                result_2 = np.hstack((result_2, result2))
            return result_2, list_k

    # 进一步检验
    def pro_test(self, result3):
        idx = np.nonzero(result3 == 2)
        for i in idx[0]:
            list3 = array('d', [])
            list4 = array('d', [])
            for k in range(i - 1, 0, -1):
                if result3[k] == 1:
                    list3.insert(0, self.arr1[k])
                    list4.insert(0, self.arr2[k])
                    if len(list3) == 2:
                        break

            for j in range(i + 1, len(self.arr1), 1):
                if result3[j] == 1:
                    list3.append(self.arr1[j])
                    list4.append(self.arr2[j])
                    if len(list3) == 4:
                        break

            if np.abs(np.polyval(np.polyfit(list3, list4, 3), self.arr1[i]) - self.arr2[i]) < self.lamda:
                result3[i] = 1
            elif np.abs(np.polyval(np.polyfit(list3, list4, 2), self.arr1[i]) - self.arr2[i]) < self.lamda:
                result3[i] = 1
            elif np.abs(np.polyval(np.polyfit(list3, list4, 1), self.arr1[i]) - self.arr2[i]) < self.lamda:
                result3[i] = 1
            else:
                result3[i] = 0
        return result3


def main():
    data = np.loadtxt("test_data.txt")
    # 时间序列（有缺失）
    arr1 = data[0, :]
    arr2 = data[1, :]
    # 索引值
    arr1_index = np.linspace(0, len(arr1), len(arr1), endpoint=False).astype(np.int)
    eliminating = Eliminating(arr1, arr2, arr1_index)
    eliminating.spline_smoothing()
    num1 = []
    # 首先进行检测是否需要数据分段处理,并得到数据分段位置
    f = eliminating.zhengjian_test(arr1, arr2, arr1_index)
    if f != 0:
        num1.append(f)
        for i in range(10):
            f = eliminating.zhengjian_test(arr1[f:], arr2[f:], arr1_index[f:])
            if f != 0:
                num1.append(f)
            else:
                break
    # 没有数据分段处理的结果
    if len(num1) == 0:
        result1, k = eliminating.zhengjian(num1)
        result2, k1 = eliminating.nijian(num1)
        # 由于是采用双向检验，对于双向检验的结果都判定为非异常时，则该数据正常，
        # 对于双向检验结果都判定为异常时，则该数据为异常值，
        # 对于双向检验结果不同时，则需要进一步检验
        result3 = result1 + result2
        result3 = np.select([result3 == 2, result3 == 1], [1, 2], result3)
        result3[0:k - 3] = result2[0:k - 3]
        result3[k1 + 4:] = result1[k1 + 4:]
        result3[k - 3:k + 1] = 1
        result3[k1:k1 + 4] = 1
    # 数据分段之后处理的结果
    else:
        num2 = sorted(num1 + [0, len(arr1)])
        result1, list_k1 = eliminating.zhengjian(num2)
        result2, list_k2 = eliminating.nijian(num2)
        # 由于是采用双向检验，对于双向检验的结果都判定为非异常时，则该数据正常，
        # 对于双向检验结果都判定为异常时，则该数据为异常值，
        # 对于双向检验结果不同时，则需要进一步检验
        result3 = result1 + result2
        result3 = np.select([result3 == 2, result3 == 1], [1, 2], result3)
        list_k1 = sorted(list_k1)
        list_k2 = sorted(list_k2)
        for i in range(len(num2) - 1):
            result3[num2[i]:list_k1[i] - 3] = result2[num2[i]:list_k1[i] - 3]
            result3[list_k2[i] + 4:num2[i + 1]] = result1[list_k2[i] + 4:num2[i + 1]]
            result3[list_k1[i] - 3:list_k1[i] + 1] = 1
            result3[list_k2[i]:list_k2[i] + 4] = 1
    # 进一步检验
    result = eliminating.pro_test(result3)
    print(np.sum(result == 0))
    # 异常数据剔除后的效果图
    arr_1 = arr1[np.nonzero(result)[0]]
    arr_2 = arr2[np.nonzero(result)[0]]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(arr_1, arr_2, 'r-')
    plt.tick_params(labelsize=14)
    ax1.set_xlabel('t/s')
    ax1.set_ylabel(r"$V_x/(m/s)$")
    # 异常数据剔除后的残差图
    spline1 = interpolate.UnivariateSpline(arr_1, arr_2, k=3)
    arr3 = spline1(arr_1)
    arr4 = arr_2 - arr3
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(arr_1, arr4, 'r-')
    ax2.set_xlabel('t/s')
    ax2.set_ylabel(r"$\sigma(V_x)/(m/s)$")
    plt.yticks([-20, 0, 20])
    plt.tick_params(labelsize=14)
    plt.show()


if __name__ == "__main__":
    main()
