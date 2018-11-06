from numpy import *
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

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam #eye生成单位矩阵
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)   #按照行取平均，每行平均
    # regularize X's:使得所有的特征影响相同，方法是每个特征减去自己的平均值后除以自身方差
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

if __name__ == '__main__':
    abX, abY     = loadDataSet("abalone.txt")
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(ridgeWeights)       #ridgeWeights,为30*8的矩阵，对矩阵画图，则以每列为一个根线，为纵坐标，横坐标为range(shape(ridgeWeights)[0])也即从0到29,第一行的横坐标为0,最后一行的行坐标为29
    plt.show()