# -*- encoding=utf-8 -*-
'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *
import math
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#===============局部加权线性回归函数==================
# 对某一点计算估计值
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat    = mat(xArr)
    yMat    = mat(yArr).T
    m       = shape(xMat)[0]    #xMat的行数
    weights = mat(eye((m)))     #创建对角阵,eye生成对角矩阵,m*m
    for j in range(m):          #对每个点的计算都用上了整个数据集
        diffMat = testPoint - xMat[j,:]#计算测试点坐标和所有数据坐标的差值
        # 计算权值 w =exp（(-（xi-X）^2)/(2*k*k)）
        weights[j, j] = math.exp(diffMat*diffMat.T/(-2.0*k**2))#高斯核，对角阵，随样本点与待预测点距离的递增，权重值大小以指数级衰减 #为何要乘以矩阵的转置呢？
    xTx = xMat.T *(weights*xMat)        #对x值进行加权计算          # 奇异矩阵不能计算
    if linalg.det(xTx)==0.0:
        print("this matrix is singular, cannot do inverse")
        ws = linalg.pinv(xTx)*(xMat.T*(weights*yMat))

    ws =xTx.I*(xMat.T*(weights*yMat))                   #回归参数
    return testPoint*ws

def lwlrTest(testArr, xArr, yArr, k=1.0): #当对testArr中所有点的估计，testArr=xArr时，即对所以点的全部估计
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

if __name__ == '__main__':
    xArr, yArr = loadDataSet("ex0.txt")
    print("原始值：", yArr[0])
    print("估计值,k为1.0：", lwlr(xArr[0], xArr, yArr, 1.0))
    print("估计值：k为0.001", lwlr(xArr[0], xArr, yArr, 0.001))

    # =========对所有点进行估计,绘图========#排序是如何排的？？？？？？
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = mat(xArr)


    srtInd = xMat[:, 1].argsort(0) #返回数据第二列从小到大的索引值；注意返回的是排序后的索引值
    xSort = xMat[srtInd][:, 0, :]  ##返回第一列和第二列??，根据上面索引顺序取出

    print(xSort)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c="red")
    plt.show()

# def rssError(yArr,yHatArr):
#     return((yArr-yHatArr)**2).sum()
#
# if __name__ == '__main__':
#     abX, abY = loadDataSet("abalone.txt")
#     yHat01 = lwlrTest(abX[0:99],abX[0:99], abY[0:99], 0.1)
#     yHat1 = lwlrTest(abX[0:99],abX[0:99], abY[0:99], 1)
#     yHat10 = lwlrTest(abX[0:99],abX[0:99], abY[0:99], 10)
#
#     print(rssError(abY[0:99],yHat01.T))
#     print(rssError(abY[0:99],yHat1.T))
    print(rssError(abY[0:99],yHat10.T))

