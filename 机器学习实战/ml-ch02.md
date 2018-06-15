---
title: 机器学习实战 K-近邻算法
date: 2018-06-06 15:56:45
tags: "机器学习"
categories: "技术"
---

### K近邻分类器算法 预测约会网站配对

<!--more-->

```
# -*- coding: UTF-8 -*-
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

# k 近邻算法第一个分类器
def craetaDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A','A','B','B']
    return group,labels

# k近邻分类器   inX 分类输入的向量  dataSet 输入的训练样本集  labels标签  k选择临近的数目
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]   # numpy函数shape[0]返回dataSet的行数
    diffMat = tile(inX,(dataSetSize,1)) - dataSet   # 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    sqDiffMat = diffMat**2  #二维特征相减后平方
    sqDistances = sqDiffMat.sum(axis=1)  #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    distances = sqDistances**0.5  #开方，计算出距离
    sortedDistIndices = distances.argsort()  #返回distances中元素从小到大排序后的索引值
    classCount = {}  #定一个记录类别次数的字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]  #取出前k个元素的类别
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #计算类别次数
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]  #返回次数最多的类别,即所要分类的类别

# 打开数据集 转化成矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()   # 删除首尾空白符
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector


# 数据归一化   newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]   # m为 dataSet 的行数
    normDataSet = dataSet - tile(minVals,(m,1))  # 原始值减去最小值
    normDataSet = normDataSet/tile(ranges,(m,1))
    return  normDataSet , ranges , minVals

# 测试分类器错误率
def datingclassTest():
    hoRatio = 0.10  #取所有数据的前百分之10 进行测试
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normalMat, ranges, minVals = autoNorm(datingDataMat)
    m = normalMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normalMat[i,:], normalMat[numTestVecs:m,:],datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))


def classifyperson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(raw_input("每年获得的飞行常客里程数:"))
    iceCream = float(raw_input("每周消费的冰激淋公升数:"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normalMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normalMat,datingLabels,3)
    print "You will probably like this person:  %s"  %(resultList[classifierResult-1])

if __name__ == '__main__':
    # group,labels = craetaDataSet()
    # print(group)
    # print(labels)
    # k = classify0([0,0],group,labels,3)
    # print  k

    #datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    # plt.show()
    # print datingLabels
    #
    # normalMat,ranges , minVals = autoNorm(datingDataMat)
    # print(datingDataMat)
    # print normalMat

    #datingclassTest()
    classifyperson()
```

### 手写识别系统

```
# -*- coding: UTF-8 -*-
from os import listdir  # 列出给定目录的文件名
from numpy import *
from kNN import classify0


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 因为值为0,1 所以不用归一化
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('G:\机器学习\机器学习实战\机器学习实战（中文版+英文版+源代码）\machinelearninginaction\Ch02\\trainingDigits')
    m = len(trainingFileList)
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i, :] = img2vector(
            'G:\机器学习\机器学习实战\机器学习实战（中文版+英文版+源代码）\machinelearninginaction\Ch02\\trainingDigits\\%s' % fileNameStr)
    testFileList = listdir('G:\机器学习\机器学习实战\机器学习实战（中文版+英文版+源代码）\machinelearninginaction\Ch02\\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(
            'G:\机器学习\机器学习实战\机器学习实战（中文版+英文版+源代码）\machinelearninginaction\Ch02\\testDigits\\%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainMat, hwLabels, 3)
        print('分类结果为:%d'  '真实结果为:%d' % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print('\n 错误个数为:%d' % errorCount)
    print('\n 错误比例为:%f' % (errorCount / float(mTest)))


if __name__ == '__main__':
    # tesstVector = img2vector('G:\机器学习\机器学习实战\机器学习实战（中文版+英文版+源代码）\machinelearninginaction\Ch02\\trainingDigits\\0_13.txt')
    # print(tesstVector)
    handwritingClassTest()
```

#### 总结

K-近邻算法 简单有效 但是执行效率低，占用大量存储空间，而且必须对数据集中的每个数据计算距离值，耗时。