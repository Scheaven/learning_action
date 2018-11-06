from numpy import *

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

#用于计算最佳拟合线
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  #为numpy中的线性代数库，计算行列式linalg.det(xTx)
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsMax = ws.copy()
    for i in range(numIt):
        print (ws.T)
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign  #增大或者缩小特征
                yTest = xMat*wsTest   #获得新的权重
                rssE = rssError(yMat.A,yTest.A) #计算新的误差值
                if rssE < lowestError: #如果误差值变小则采用新的权重值
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat


if __name__ == '__main__':
    xArr,yArr = loadDataSet("abalone.txt")
    print ("步长为0.01,迭代次数为200：\n",stageWise(xArr, yArr, 0.01, 200))
    print ("步长为0.001,迭代次数为5000：\n",stageWise(xArr, yArr, 0.001, 5000))

    xMat = mat(xArr)
    yMat = mat(yArr).T
    xMat = regularize(xMat)
    yM   = mean(yMat, 0)
    yMat = yMat- yM
    weights = standRegres(xMat, yMat.T)
    print (weights.T)
