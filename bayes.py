#coding=utf-8
from numpy import *
import os
from operator import *
# 获取训练样本和样本的分类
def loadDataSet():
	postList = [['my','dog','has','flea','problems','help','me'],
				['maybe','not','take','him','to','dog','park','stupid'],
				['my','dalmation','is','so','cute','i','love','him'],
				['stop','posting','stupid','worthless','garbage'],
				['mr','licks','ate','my','steak','how','to','stop','him'],
				['quit','buying','worthless','dog','food','stupid']]
	# 1代表侮辱性，0代表正常文章
	classVec=[0,1,0,1,0,1]
	return postList,classVec

# 创建文本训练集向量
def createDocVec(dataSet):
	vec = set([])
	for doc in dataSet:
		vec = vec | set(doc)
	return list(vec)


# 为每一个训练样本生成一个向量，词集模型
def wordToVec(vecList,doc):
	returnVec = [0] * len(vecList)
	for word in doc:
		# 如果
		if(word in vecList):
			returnVec[vecList.index(word)] = 1
		else:
			print("word is not contained:  " + word)
	return returnVec

# 为每一个训练样本生成一个向量，词袋模型
def wordToBag(vecList,doc):
	returnVec = [0] * len(vecList)
	for word in doc:
		# 如果
		if(word in vecList):
			returnVec[vecList.index(word)] += 1
		else:
			print("word is not contained:  " + word)
	return returnVec

# 训练朴素贝叶斯函数
# trainDoc，训练样本
# trainCategory，一维数组表示trainDoc中的训练样本的分类，每个元素对应一个样本
def trainNB(trainMat,trainCategory):
	docNum = len(trainMat)
	wordNum = len(trainMat[0])
	# 坏的样本占总体的比例
	pBad = sum(trainCategory) / float(docNum)
	# 原代码
	# p0Num = zeros(wordNum)
	# p1Num = zeros(wordNum)
	# p0Denom = 0.0
	# p1Denom = 0.0
	# 新代码
	p0Num = ones(wordNum)
	p1Num = ones(wordNum)
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(docNum):
		# 如果是坏的样本
		if(trainCategory[i] == 1):
			# 两个向量相加
			p1Num += trainMat[i]
			p1Denom += sum(trainMat[i])
		else:
			# 如果是好的样本
			p0Num += trainMat[i]
			p0Denom += sum(trainMat[i])
	# p1Vect = p1Num / p1Denom;
	# p0Vect = p0Num / p0Denom;
	p1Vect = log(p1Num / p1Denom);
	p0Vect = log(p0Num / p0Denom);
	# 顺序：类别为0，为1，坏样本占总体的概率
	return p0Vect,p1Vect,pBad


# 根据训练结果，对新的数据分类
def classifyNB(vec2Classify,p0V,p1V,pBad):
	p1 = sum(vec2Classify * p1V) + log(pBad)
	p0 = sum(vec2Classify * p0V) + log(1 - pBad)
	if(p1 > p0):
		# bad
		return 1
	else:
		# good
		return 0


# -----开始函数调用----- #




# # 获取训练的文章，和每个文章的类别
# listPosts,listClass = loadDataSet()
# # 将所有单词生成一个向量，无重复单词
# vecList = createDocVec(listPosts)

# # 生成训练样本
# trainMat = []
# for document in listPosts:
# 	trainMat.append(wordToVec(vecList,document))
# p0V,p1V,pBad = trainNB(trainMat,listClass)

# testDoc = ['worthless','my','stupid']
# testVec = array(wordToVec(vecList,testDoc))

# if(classifyNB(testVec,p0V,p1V,pBad) == 1):
# 	print("文章测试：具有侮辱性")
# else:
# 	print("文章测试：正常")

# normal file list
nFileList = os.listdir("normalResource/")
# bad file list
bFileList = os.listdir("badResource/")
# 生成所有文件的类别标签
fileClasses = []
for i in bFileList:
	fileClasses.append(1)
for i in nFileList:
	fileClasses.append(0)

# 是这样，utf-8下，每个中文文字会占用3个数组元素，比如“我说”就是['\xe6', '\x88', '\x91', '\xe8', '\xaf', '\xb4']
# 我们知道，中文的符号比如：也是占三个数组元素的
# 但是难免会有一些英文符号，比如“>”在数组中就是：['>']
# 所以我决定用这个函数，抛弃所有英文符号，再将中文符号三个一组组成一个新的数组，这样才能做训练样本
def utfChinese(str):
	return str
# 读文件，生成训练样本postList
postList = []
for file in bFileList:
	path = 'badResource/' + file
	f = open(path)
	str = []
	for line in f.readlines():
		line = unicode(line,"utf-8")
		for word in line:
			str.append(word)
	f.close()
	postList.append(str)

for file in nFileList:
	path = 'normalResource/' + file
	f = open(path)
	str = []
	for line in f.readlines():
		line = unicode(line,"utf-8")
		for word in line:
			str.append(word)
	f.close()
	postList.append(str)

# 将所有单词生成一个向量，无重复单词
vecList = createDocVec(postList)
# 生成训练样本
trainMat = []
for document in postList:
	trainMat.append(wordToBag(vecList,document))
# 训练后产出三个关键概率
p0V,p1V,pBad = trainNB(trainMat,fileClasses)




# 获取测试文章
f = open('test.txt')
str = []
for line in f.readlines():
	line = unicode(line,"utf-8")
	for word in line:
		str.append(word)
f.close()
testDoc = str
testVec = array(wordToBag(vecList,testDoc))

if(classifyNB(testVec,p0V,p1V,pBad) == 1):
	print("这是一篇小黄文，你真污")
else:
	print("这是简朴的文章而已")
