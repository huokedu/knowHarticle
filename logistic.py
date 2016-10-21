#coding=utf-8
from numpy import *
from operator import *
# 丛文件testSet中读取数据
def loadDataSet():
	dataMat = []
	labelMat = []
	f = open("testSet.txt")
	for line in f.readlines():
		lineArr = line.strip().split()
		# 将数据和结果填充到数组
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat


# sigmoid函数
def sigmoid(z):
	return 1.0 / (1 + exp(-z))

# 核心函：梯度上升算法
def gradAscent(dataMat,labelMat):
	dataMat = mat(dataMat)
	labelMat = mat(labelMat).transpose()
	m,n = shape(dataMat)
	alpha = 0.001
	maxCycles = 1000
	weights = ones((n,1))
	for i in range(maxCycles):
		h = sigmoid(dataMat * weights)
		error = (labelMat - h)
		# 这一行是啥意思？
		weights = weights + alpha * dataMat.transpose() * error
	return weights



# 把图像画出来
def plotBest(weights,dataMat,labelMat):
	import matplotlib.pyplot as plt
	dataArr = array(dataMat)
	# 行数n
	n = shape(dataArr)[0]
	xcord1 = []
	xcord2 = []
	ycord1 = []
	ycord2 = []
	for i in range(n):
		if(int(labelMat[i]) == 1):
			# 如果是第一类。下一行代码写法等同于xcord1.append(dataArr[i][1])
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	# 准备画图
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# c=red红色，marker=s方块，没有就是圆圈
	ax.scatter(xcord1,ycord1, s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2, s=30,c='green')
	# arange和range基本一样，只是返回的不是list而是array对象而已，－3到3，步长为1
	x = arange(-3.0,3.0,1)
	# 画直线，这直线我不太懂啊
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x,y)
	plt.xlabel("X1");plt.ylabel("Y1")
	plt.show()
	return

dataMat,labelMat = loadDataSet()
weights = gradAscent(dataMat,labelMat)
plotBest(weights.getA(),dataMat,labelMat)