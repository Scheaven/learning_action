import matplotlib.pyplot as plt

#定义文本框和箭头格式:# boxstyle是文本框类型 fc是边框粗细 sawtooth是锯齿形、构建了一个1*1的模块
decisionNode = dict(boxstyle='sawtooth',fc = '0.8')
leafNode = dict(boxstyle='round4',fc = '0.8')
arrow_args = dict(arrowstyle="<-")

class treePlotter():
    '''annotate 注释的意思
       nodeTxt用于记录nodeTxt，即节点的文本信息。centerPt表示那个节点框的位置。 parentPt表示那个箭头的起始位置。nodeType表示的是节点的类型，也就会用我们之前定义的全局变量。

    '''
    def __init__(self,inTree):
        fig = plt.figure(1,facecolor='white')   # 新建一个画布，背景设置为白色的
        fig.clf()    # 将画图清空
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)  # 设置一个多图展示，构建了一个1*1的模块
        self.totalW = float(self.getNumLeafs(inTree))  # Store the width of the tree.
        self.totalD = float(self.getTreeDepth(inTree))  # Store the depth of the tree
        # plotTree.xOff,plotTree.yOff: Trace the location of the drawing node
        self.xOff = -0.5 / self.totalW
        self.yOff = 1.0

    def plotNode(self,nodeTxt,centerPt,parentPt,nodeType):
        #annotate是注释的意思，也就是作为原来那个框的注释，也是添加一些新的东西
        self.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
                                va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

    #Gets the number of leaf nodes
    def getNumLeafs(self,myTree):
        numLeafs = 0
        firstStr = list(myTree.keys())[0] #这是由于python3改变了dict.keys,返回的是dict_keys对象,支持iterable 但不支持indexable，我们可以将其明确的转化成list：
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict':
                numLeafs += self.getNumLeafs(secondDict[key])
            else:
                numLeafs += 1
        return numLeafs



    #Gets the depth of the tree
    def getTreeDepth(self,myTree):
        maxDepth = 0
        thisDepth = 1
        # print(myTree.keys())
        firstStr = list(myTree.keys())[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict': #判断节点类型是不是为字典
                maxDepth = 1 + self.getTreeDepth(secondDict[key])
            if thisDepth>maxDepth :maxDepth = thisDepth
        return maxDepth

    #Add text to the parent node.
    def plotMidText(self,cntrPt,parentPt,txtString):
        #Calculate intermediate position
        xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
        self.ax1.text(xMid,yMid,txtString, va="center", ha="center", rotation=30)


    def plotTree(self,myTree,parentPt,nodeTxt):
        numLeafs = self.getNumLeafs(myTree)
        depth = self.getTreeDepth(myTree)
        firstStr = list(myTree.keys())[0]
        cntrPt = (self.xOff + (1.0 + float(numLeafs))/2.0/self.totalW, self.yOff)
        self.plotMidText(cntrPt, parentPt, nodeTxt) #The attribute value of the tag of child node
        self.plotNode(firstStr,cntrPt,parentPt,decisionNode)
        secondDict = myTree[firstStr]
        self.yOff = self.yOff - 1.0/self.totalD
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict': #判断节点类型是不是为字典
                self.plotTree(secondDict[key], cntrPt, str(key))
            else:
                self.xOff = self.xOff + 1.0/self.totalW
                self.plotNode(secondDict[key],(self.xOff,self.yOff),cntrPt,leafNode)
                self.plotMidText((self.xOff,self.yOff),cntrPt,str(key))
        self.yOff = self.yOff + 1.0/self.totalD

    def createPlot(self,inTree):
        #创建createPlot.ax1:意思是这个只是一个新框。
        self.plotTree(inTree,(0.5,1.0),'')
        plt.show()

if __name__ == '__main__':
    def retrieveTree(i):
        listOfTrees=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},\
                     {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
        return listOfTrees[i]

    # createPlot()
    plotter = treePlotter(retrieveTree(1))
    print(retrieveTree(1))
    # print(plotter.getNumLeafs(retrieveTree(1)))
    # print(plotter.getTreeDepth(retrieveTree(1)))

    plotter.createPlot(retrieveTree(1))
