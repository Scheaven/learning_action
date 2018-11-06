import _tkinter
import tkinter
from tkinter import *

# root=Tk() #创建一个空的tk窗口，注意弹出后不要关闭，然后继续输入下一行
# myLabel=Label(root,text="hello,Tkinter!")
# myLabel.grid()#输入以上两行，框内显示文字
# root.mainloop()


# !/usr/bin/env python
# coding=utf-8
# 用于构建树管理器界面的Tkinter小部件
from numpy import *
from tkinter import *
from matchine_learning_test.learning_action.Ch09 import regTrees


def reDraw(tolS, tolN):
    pass


def drawNewTree():
    pass


root = Tk()

Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)  # 设置文本，第0行，距0的行值为3,

Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)  # Entry为允许单行文本输入的文本框，设置文本框，再定位置第1行第1列，再插入数值
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, "10")

Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, "1.0")

Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)  # Botton按钮，设置第1行第2列，列值为3
chkBtnVar = IntVar()  # IntVar为按钮整数值小部件
chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(regTrees.loadDataSet("sine.txt"))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)
root.mainloop()