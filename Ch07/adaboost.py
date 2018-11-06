'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Schro
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


'''通过阈值比较数据进行分类：只分两类，1和-1；threshVal是阈值的角色，每一个threshVal会进行 ['lt', 'gt']两种分类'''
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

'''核心函数，构建最佳单层决策树;最佳单层决策树的目标是获取在某个特征下给定特定的阈值后取小的一边或者大的一边获得到的正确数据最多！！
   D为权重向量;
   bestStump存储的三个数据分别是用来划分的特征、划分的阈值、以及取大于阈值还是小于阈值
'''
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {}; #用来存储单层决策树
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity
    for i in range(n):  # loop over all dimensions
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps     #步长:第i列特征的最大值减去最小值除去10作为步长
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than，定义大于或者小于的取值情况
                threshVal = (rangeMin + float(j) * stepSize)  # 每次阈值增加的大小,直到最大值，float(j) * stepSize为变化的步长
                # 对同一个threshVal  'lt', 'gt'都分一下，来区别分到按照特征i是大于阈值的占多数还是小于阈值的占多数
                predictedVals = stumpClassify(dataMatrix, i, threshVal,inequal)  # call stump classify with i, j, lessThan，
                #使用和label的比较来calculate分类的错误率，在通过错误率来修订权重的大小
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # calc total error multiplied by D,分类正确的越多，errArr就包含越多的0，weightedError就越小
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                #取得错误率最低的权重值，并记录划分的方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

if __name__ == '__main__':
    D = mat(ones((5,1))/5)
    print (D)
    a,b = loadSimpData()
    print (buildStump( a,b ,D))