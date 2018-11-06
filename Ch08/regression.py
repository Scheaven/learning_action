'''
Created on Jan 8, 2018

@author: Schro
'''
from numpy import *
import matplotlib.pyplot as plt

# Get dataMat list and label list
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of features fields
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


#用于计算最佳拟合线
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  #为numpy中的线性代数库，计算行列式linalg.det(xTx)
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


def showScatter(xMat,yMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0],
               yMat.T[:, 0].flatten().A[0])
    return ax

def showLineValue(xMat,ws,ax):
    xCopy = xMat.copy()
    yHat = xCopy * ws   ##预测值
    print("++++++预测值+++++++")
    print(yHat.T)
    ax.plot(xCopy[:, 1], yHat)
    # plt.show()


# if __name__ == '__main__':
#     abX, abY     = loadDataSet("abalone.txt")
#     ridgeWeights = standRegres(abX, abY)
#     fig = plt.figure()
#     ax  = fig.add_subplot(111)
#     ax.plot(ridgeWeights)       #ridgeWeights,为30*8的矩阵，对矩阵画图，则以每列为一个根线，为纵坐标，横坐标为range(shape(ridgeWeights)[0])也即从0到29,第一行的横坐标为0,最后一行的行坐标为29
#     plt.show()
# #
if __name__ == '__main__':
    xArr, yArr = loadDataSet("ex0.txt")
    xMat = mat(xArr)
    yMat = mat(yArr)    #真实值
    print("========真实值======")
    print (yMat)
    ax = showScatter(xMat,yMat) #绘制数据集散点图

    ws = standRegres(xArr, yArr)  # 回归系数
    showLineValue(xMat,ws,ax)
    plt.show()  # 调用显示






