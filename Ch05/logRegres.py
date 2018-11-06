# coding=utf-8
from numpy import *
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#sigmoid函数
def sigmoid(inX):
    return longfloat(1.0/(1+exp(-inX)))  #用于防止计算溢出

'''
核心函数，使用梯度上升计算Logistic回归算法的权重值
@param dataMatIn 添加了第0列的特征值；classLabels确定的目标标签
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))

    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

'''
核心函数,是循环遍历的改进算法，对每个点（每次仅仅一个点）计算误差，然后调整整个权重；
'''
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights
'''
核心函数,是循环遍历的改进算法，对每个点（每次仅仅一个点）计算误差，然后调整整个权重；
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # del(dataIndex[randIndex])
            del (list(dataIndex)[randIndex])
    return weights

def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green') #绘制散点图
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)   #指定xy坐标，显示样式
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

#在整个数据集上运行200次
def stocGradAscent0_200(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix) # 100x3
    alpha = 0.01
    weights = np.ones(n) # 1x3 p.s.np.ones((3,1))为3x1
    weights_all = []
    for times in range(200):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i]*weights)) # 标量
            error = classLabels[i] - h
            weights = weights + alpha*error*dataMatrix[i]
            #print(weights)
            weights_all.append(weights)
    return weights,weights_all

# 改进的随机梯度上升算法
def stocGradAscent2(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    weights_all = []
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01 # 会随着迭代次数不断减少，当j<<max(i),alpha就不是严格下降的（类似于模拟退火算法）
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            weights_all.append(weights)
            del(dataIndex[randIndex])
    return weights, weights_all
def showX1():
    dataArr, labelMat = loadDataSet()
    weights, weights_all = stocGradAscent0_200(np.array(dataArr), labelMat)
    x0 = [];
    x1 = [];
    x2 = []
    for i in range(len(weights_all)):
        x0.append(float(list(weights_all[i])[0]))
        x1.append(float(list(weights_all[i])[1]))
        x2.append(float(list(weights_all[i])[2]))
    X = np.linspace(0, 200, len(weights_all), endpoint=True)
    fig = pp.subplot(311)
    pp.plot(X, x0)
    fig = pp.subplot(312)
    pp.plot(X, x1)
    fig = pp.subplot(313)
    pp.plot(X, x2)
    plt.show()
def showX2():
    dataArr, labelMat = loadDataSet()
    weights, weights_all = stocGradAscent2(array(dataArr), labelMat)
    x0 = [];
    x1 = [];
    x2 = []
    for i in range(len(weights_all)):
        x0.append(float(list(weights_all[i])[0]))
        x1.append(float(list(weights_all[i])[1]))
        x2.append(float(list(weights_all[i])[2]))
    '''在指定的间隔内返回均匀间隔的数字。
    返回num均匀分布的样本，在[start, stop]。
    这个区间的端点可以任意的被排除在外。
    '''
    X = np.linspace(0, 200, len(weights_all), endpoint=True)
    fig = pp.subplot(311)
    pp.plot(X, x0)
    fig = pp.subplot(312)
    pp.plot(X, x1)
    fig = pp.subplot(313)
    pp.plot(X, x2)
    plt.show()
if __name__ == '__main__':
    a,b = loadDataSet()
    '''
    也就是说矩阵通过这个getA()这个方法可以将自身返回成一个n维数组对象
    为什么要这样做呢？
    因为plotBestFit()函数中有计算散点x,y坐标的部分，其中计算y的时候用到了weights，
    如果weights是矩阵的话，weights[1]就是[[0.48007329]]（注意这里有中括号！），
    就不是一个数了，最终你会发现y的计算结果的len()只有1，而x的len()则是60
    '''
    # weights = gradAscent(a,b)
    # plotBestFit(weights.getA())

    # weights = stocGradAscent0(array(a),b)
    # plotBestFit(weights)

    # weights = stocGradAscent1(array(a),b)
    # plotBestFit(weights)


    # print(colicTest())

    showX1()
    showX2()


