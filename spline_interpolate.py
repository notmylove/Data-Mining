import numpy as np
from array import array
from scipy import interpolate
import matplotlib.pyplot as plt


def zhenglian(arr1, arr2, lamda):
    for i in range(len(arr1)-3):
        if np.abs(np.diff(arr2[i:i+4], n=3)) < lamda :
            k = i+3
            break
        elif i == 500:
            print('模型失效')
            k = None
            break
    return k


def zhengjian(arr1, arr2, lamda, k):
    result1 = np.ones(len(arr1))
    # 这部分未被检测，默认为正常
    result1[0:k+1] = 3
    # 正向递推
    list1 = array('d', [])
    csth = np.copy(arr1[k-3:k+1]).tolist()
    csnh = np.copy(arr2[k-3:k+1]).tolist()
    for i in range(k+1, len(arr1), 1):
        if np.abs(interpolate.UnivariateSpline(csth, csnh, k=3, s=0)(arr1[i]) - arr2[i]) < lamda:
            result1[i] = 1
            del csth[0]
            del csnh[0]
            csth.append(arr1[i])
            csnh.append(arr2[i])
        elif np.abs(interpolate.UnivariateSpline(csth, csnh, k=2, s=0)(arr1[i]) - arr2[i]) < lamda:
            result1[i] = 1
            del csth[0]
            del csnh[0]
            csth.append(arr1[i])
            csnh.append(arr2[i])
        elif np.abs(interpolate.UnivariateSpline(csth, csnh, k=1, s=1)(arr1[i]) - arr2[i]) < lamda:
            result1[i] = 1
            del csth[0]
            del csnh[0]
            csth.append(arr1[i])
            csnh.append(arr2[i])
        else:
            result1[i] = 0
            list1.append(i)
            if len(list1) > 5:
                    #连续异常数据超限，模型失效，需要数据分段
                    for i in range(0, len(list1)-4):
                        if sum(np.abs(np.diff(list1[i:i+5]))==1)>=4:
                            f = int(list1[i])
                            # print("连续异常超限")
                            k1 = zhenglian(arr1[0:f], arr2[0:f], lamda)
                            k2 = zhenglian(arr1[f:], arr2[f:], lamda)
                            return np.hstack((zhengjian(arr1[0:f], arr2[0:f], lamda, k1), zhengjian(arr1[f:], arr2[f:], lamda, k2)))
            if ('f' in vars()):
                    break
    return result1


def nilian(arr1, arr2, lamda):
    for i in range(len(arr1)-1, 2, -1):
        if np.abs(np.diff(arr2[i-3:i+1], n=3)) < lamda :
            k1 = i-3
            break
        elif i == 500:
            print('模型失效')
            k = None
            break
    return k1


#逆向递推
def nijian(arr1, arr2, lamda, k1):
    result2 = np.ones(len(arr1))
    result2[k1:] = 3
    #逆向递推
    list2 = array('d', [])
    csth = np.copy(arr1[k1:k1+4]).tolist()
    csnh = np.copy(arr2[k1:k1+4]).tolist()
    for i in range(k1-1, -1, -1):
        if np.abs(interpolate.UnivariateSpline(csth, csnh, k=3, s=0)(arr1[i]) - arr2[i]) < lamda:
            result2[i] = 1
            del csth[-1]
            del csnh[-1]
            csth.insert(0, arr1[i])
            csnh.insert(0, arr2[i])
        elif np.abs(interpolate.UnivariateSpline(csth, csnh, k=2, s=0)(arr1[i]) - arr2[i]) < lamda:
            result2[i] = 1
            del csth[0]
            del csnh[0]
            csth.insert(0, arr1[i])
            csnh.insert(0, arr2[i])
        elif np.abs(interpolate.UnivariateSpline(csth, csnh, k=1, s=1)(arr1[i]) - arr2[i]) < lamda:
            result2[i] = 1
            del csth[0]
            del csnh[0]
            csth.insert(0, arr1[i])
            csnh.insert(0, arr2[i])
        
        else:
            result2[i] = 0
            list2.append(i)
            if len(list2) > 5:
                #连续异常数据超限，模型失效，需要数据分段
                for i in range(0, len(list2)-4):
                    if sum(np.abs(np.diff(list2[i:i+5]))==1)>=4:
                        f = int(list2[i])
                        # print("连续异常超限")
                        k3 = nilian(arr1[0:f+1], arr2[0:f+1], lamda)
                        k4 = nilian(arr1[f+1:], arr2[f+1:], lamda)
                        return np.hstack((nijian(arr1[0:f+1], arr2[0:f+1], lamda, k3), nijian(arr1[f+1:], arr2[f+1:], lamda, k4)))
            if ('f' in vars()):
                break
    return result2


if __name__ == '__main__':
    data = np.loadtxt("test_data.txt")
    # 时间序列（有缺失）
    arr1 = data[0, :]
    arr2 = data[1, :]
    # 样条平滑方法
    spline1 = interpolate.UnivariateSpline(arr1, arr2, k=3)
    arr3 = spline1(arr1)
    arr4 = arr2 - arr3
    #残差的标准差估计值
    sigma = arr4.std(ddof=1)
    #判断门限
    lamda = 3*sigma
    #初始参数的设定
    n_r = 5
    n_l = 5
    k = zhenglian(arr1, arr2, lamda)
    result1 = zhengjian(arr1, arr2, lamda, k)
    
    k1 = nilian(arr1, arr2, lamda)
    result2 = nijian(arr1, arr2, lamda, k1)
    
    result3 = result1 + result2
    result3 = np.select([result3==2, result3==1], [1, 2], result3)
    result3[0:k+1] = result2[0:k+1]
    result3[k1:] = result1[k1:]
    #进一步检验
    idx = np.nonzero(result3 == 2)
    for i in idx[0]:
        list3 = array('d', [])
        list4 = array('d', [])
        for k in range(i-1, 0, -1):
            if result3[k] == 1:
                list3.insert(0, arr1[k])
                list4.insert(0, arr2[k])
                if len(list3) == 2:
                    break
                      
        for j in range(i+1, len(arr1), 1):
            if result3[j] == 1:
                list3.append(arr1[j])
                list4.append(arr2[j])
                if len(list3) == 4:
                    break
        if np.abs(interpolate.UnivariateSpline(list3, list4, k=3, s=0)(arr1[i]) - arr2[i]) < lamda:
            result3[i] = 1
        elif np.abs(interpolate.UnivariateSpline(list3, list4, k=2, s=0)(arr1[i]) - arr2[i]) < lamda:
            result3[i] = 1 
        elif np.abs(interpolate.UnivariateSpline(list3, list4, k=1, s=1)(arr1[i]) - arr2[i]) < lamda:
            result3[i] = 1
        
        else:
            result3[i] = 0
    print(np.sum(result3 == 0))
    arr_1 = arr1[np.nonzero(result3)[0]]
    arr_2 = arr2[np.nonzero(result3)[0]]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(arr_1, arr_2, 'r-')
    ax1.set_xlabel('t/s', {'size': 14})
    ax1.set_ylabel('Vx/(m/s)', {'size': 14})
    plt.tick_params(labelsize=14)
    spline1 = interpolate.UnivariateSpline(arr_1, arr_2, k=3)
    arr3 = spline1(arr_1)
    arr4 = arr_2 - arr3
    sigma_1 = arr4.std(ddof=1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(arr_1, arr4, 'r-')
    ax2.set_xlabel('t/s', {'size': 14})
    ax2.set_ylabel(r"$\sigma(Vx)/(m/s)$", {'size': 14})
    plt.tick_params(labelsize=14)
    plt.yticks([-20,0,20])
    plt.show()
    

