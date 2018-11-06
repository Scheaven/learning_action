# -*-coding:utf-8 -*-
from math import log
import operator
from matchine_learning_test.learning_action.Ch03 import treePlotter
''' @description：决策树分类算法
    @author schro
    @time 2017_10_26
'''

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]#最后一列yes和no为结果标签
    labels = ['no surfacing','flippers'] #labels是特征的名称
    #change to discrete values
    return dataSet, labels

#根据香农公式计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelFeat = {}
    for featVec in dataSet:  #the the number of unique elements and their occurance
        key = featVec[-1]
        if key not in labelFeat.keys():labelFeat[key] = 0
        labelFeat[key]+=1
    shannonent = 0.0
    for key in labelFeat.keys():
        labelProb = float(labelFeat[key]/numEntries) #Calculate the probability
        shannonent -=labelProb*log(labelProb,2) #log base 2  #Calculate the comentropy
    return shannonent

'''划分数据集：按照某特征划分数据集，将符合要求的数据元素抽取出来，
   即将返回所有axis位置值等于value的数据，并且去除第axis位置上的数字；
   注意：append两个集合，会将两个集合直接添加到一起，形成多维集合
 extend 则是将两个集合合并成一个集合'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        #抽取符合要求的数据
        if featVec[axis] == value:
            #下边两句其实是删除符合要求的axis位置的的数（具体例子可以看课本38页），然后拼接到一起
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的特征，等价于后边两个函数调用
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)   #整个数据集的信息熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]  #create a list of all the examples of this feature，第i个特征的所有值的列表
        uniqueVals = set(featList)       #get a set of unique values，某个特征出现的所有不重复的值，因为根据这些值将被分为不同的分类
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)    #去除第i个特征，并将第i个特征的值为value的数据从数据集中划分出来
            prob = len(subDataSet)/float(len(dataSet))      #该特征值占整个数据的比例
            '''该特征去掉后的信息熵：方法是计算所有根据该特征分类后计算特征对应不同值所对应的信息熵，
              calcShannonEnt(subDataSet) 仅仅是该特征的一个值的信息熵
          '''
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy，信息增益
        #计算每个数据的信息增益，并根据每个数据的信息增益选择信息增益最大的特征值
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

# 根据信息增益选择最好的特征
def chooseBestFeature(dataSet):
    baseEntropy = calcShannonEnt(dataSet)
    featureNum = len(dataSet[0])-1  #获取特征的个数，the last column is used for the labels
    bestFeature = 0 ;bestInfoGain = 0.0 #定义最好的的特征和信息增益
    for featureAxis in range(featureNum):    #iterate over all the features
        axisFeatureGain = calcInfoGain(dataSet,baseEntropy,featureAxis)    #计算按照axis位置分类后信息增益
        if bestInfoGain>axisFeatureGain:
            bestInfoGain = axisFeatureGain
            bestFeature = featureAxis
    return bestFeature

# 计算信息增益，根据特征i分类后的信息熵变化
def calcInfoGain(dataSet,baseEntropy,axis):
    # 分为了两步：首先是建立一个list的列表，之后使用set获取特征值
    # create a list of all the examples of this feature，第i个特征的所有值的列表.
    # get a set of unique values，某个特征出现的所有不重复的值，因为根据这些值将被分为不同的分类,
    featureSet = set([single[axis] for single in dataSet])
    axisEntropy = 0.0
    #计算每个value分类对应的信息熵
    for value in featureSet:
        subDataSet = splitDataSet(dataSet,axis,value)   #去除第i个特征，并将第i个特征的值为value的数据从数据集中划分出来
        subProb = float(len(subDataSet)/len(dataSet))    #该特征值占整个数据的比例
        '''该特征去掉后的信息熵：方法是计算所有根据该特征分类后计算特征对应不同值所对应的信息熵，
                     calcShannonEnt(subDataSet) 仅仅是该特征的一个值的信息熵                 '''
        axisEntropy += subProb * calcShannonEnt(subDataSet)
    return baseEntropy - axisEntropy


#作用是：当决策树分类时无法返回确定的类别，特征已经使用完毕，但仍然无法划分完毕，就返回出现次数最多的标签类别
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] #返回数量最多的值

#使用递归方式创建决策树
def createTree(dataSet,labels):
    classList = [singleSet[-1] for singleSet in dataSet] #分类的标签列表
    if classList.count(classList[0]) == len(classList): #标签全部一样，即处在第一个位置的标签数量和整个待分数据的标签数量相等，返回标签
        return classList[0]
    if len(dataSet[0]) == 1: #只剩下标签栏了，只能选择标签最多的作为这个分类标签
        return majorityCnt(classList)
    bestFeature = chooseBestFeature(dataSet)
    bestFeaterName = labels[bestFeature] #按照最好的特征值进行分类
    myTree = {bestFeaterName:{}}#创建分类树的形式。，标签为key,其余为value
    del(labels[bestFeature])
    # 根据选出的最好的特征标签标签进行分类，
    featureSet = set([singleSet[bestFeature] for singleSet in dataSet])
    for value in featureSet:    #一个特征标签可能对应多个值，根据值的不同，化为为不同的分类
        #下边两行其实也可以不要，labels已经将分类过的特征名称删除，第二句可写成  myTree[bestFeaterName] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
        subLabels = labels[:]
        myTree[bestFeaterName][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
    return myTree

#Gets the depth of the tree
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex =  featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict': #判断节点类型是不是为字典
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel

#Use the pickle storage the decision tree
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},{'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

if __name__ == '__main__':
    myDat,labels = createDataSet()

    print(calcShannonEnt(myDat)) #信息熵

    # print(myDat)
    print(splitDataSet(myDat,0,1)) #extrat the 0 position is equal to 1.and delete 0 position date

    # Get the best partitioning feature subscript.
    print(chooseBestFeatureToSplit(myDat))

    #创建决策树，其他函数在决策树中调用
    print(createTree(myDat,labels))



    print(classify(retrieveTree(0),labels,[1,0]))
    print(classify(retrieveTree(0),labels,[1,1]))