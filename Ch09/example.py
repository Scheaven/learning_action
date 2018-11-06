from numpy import shape,zeros,power,mat,mean,var,inf,ones,linalg,nonzero,corrcoef
from matplotlib import pyplot as  plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # python3不适用：fltLine = map(float,curLine) #map all elements to float() ,map（func,a）使得a参数全部使用func处理后返回list
        fltLine = list(map(float, curLine))  # 将每行映射成浮点数，python3返回值改变，所以需要使用list处理
        dataMat.append(fltLine)
    return dataMat

#使用特征切分数据集
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    #下面原书代码报错 index 0 is out of bounds,使用上面两行代码
    #mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    #mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0,mat1

#生成叶结点，在回归树中是目标变量特征的均值
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1]) #最后一列的均值

#误差计算函数：回归误差
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0] #计算目标的平方误差（均方误差*总样本数）

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 切分特征的参数阈值，用户初始设置好
    tolS = ops[0] #允许的误差下降值
    tolN = ops[1] #切分的最小样本数
    #if all the target variables are the same value: quit and return value #若所有特征值都相同，停止切分
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)   ## 找不到好的切分特征，调用regLeaf直接生成叶结点
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        # for splitVal in set(dataSet[:,featIndex]): python3报错修改为下面
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):#遍历每个特征里不同的特征值.A是返回自身的nparray()对象
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val #满足停止条件时返回叶结点值
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


#模型树
def linearSolve(dataSet):   #将数据集格式化为X Y
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0: #X Y用于简单线性回归，需要判断矩阵可逆
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#不需要切分时生成模型树叶节点
    ws,X,Y = linearSolve(dataSet)
    return ws #返回回归系数

def modelErr(dataSet):#用来计算误差找到最佳切分
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws

    return sum(power(Y - yHat,2)) #返回Y和yHat的平均误差

#判断输入是否为一棵树
def isTree(obj):
    return (type(obj).__name__=='dict') #判断为字典类型返回true

#用树回归进行预测
#1-回归树
def regTreeEval(model, inDat):
    return float(model)

#2-模型树
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)

#对于输入的单个数据点，treeForeCast返回一个预测值。
def treeForeCast(tree, inData, modelEval=regTreeEval):#指定树类型
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):#有左子树 递归进入子树
            return treeForeCast(tree['left'], inData, modelEval)
        else:#不存在子树 返回叶节点
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

#对数据进行树结构建模,就是将构建的决策树里边的预测值都给按照结构获取出来
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat



if __name__ == '__main__':

    #预测值和真是值相比较，使用pearson相关系数计算相似度
    trainMat = mat(loadDataSet("bikeSpeedVsIq_train.txt"))
    testMat = mat(loadDataSet("bikeSpeedVsIq_test.txt"))
    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    print("回归树的皮尔逊相关系数：", corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])  #将获取到的结构化数据和测试数据的标签值做比较

    myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print("模型树的皮尔逊相关系数：", corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    ws, X, Y = linearSolve(trainMat)
    print("线性回归系数：", ws)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print("线性回归模型的皮尔逊相关系数：", corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
