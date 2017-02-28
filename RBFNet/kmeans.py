#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import copy


class KMeans(object):
    '''the origin formal KMeans
    '''

    def __init__(self, k, max_iter=300):
        '''
        k: 聚类的数目
        max_iter: 最大迭代次数
        '''
        self._k = k
        self._max_iter = max_iter

    def _calDist(self, x, y):
        '''
        计算两个数据点的距离
        '''
        if not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            raise TypeError('np.ndarray is necessary')
        return np.math.sqrt(np.sum(np.square(x-y)))

    def _randomInitCenter(self, dataX, k):
        '''
        随机选择k个节点作为中心
        '''
        Num, dim = dataX.shape
        # 初始化k*dim的矩阵存储中心
        centers = np.empty((k, dim))
        for i in range(k):
            ind = int(np.random.uniform(0, Num))
            centers[i,:] = dataX[ind,:]

        return centers

    def train(self, dataX):
        '''
        训练数据
        '''
        if not isinstance(dataX, np.ndarray):
            try:
                dataX = np.asarray(dataX)
            except:
                raise TypeError('np.ndarray is necessary')

        Num, dim = dataX.shape
        self._centers = self._randomInitCenter(dataX, self._k)
        self._init_centers= copy.deepcopy(self._centers)

        # 第一列存储当前数据属于的簇
        # 第二列存储当前数据与中心的距离(err)
        self._clusterAssment = np.empty((Num,2))

        for _ in xrange(self._max_iter):
            isConverge = True
            # step1: 更新簇内元素
            for i in range(Num):
                MinDist = np.inf
                ind = 0
                for j in range(self._k):
                    # 寻找最近的中心
                    dist = self._calDist(dataX[i,:], self._centers[j,:])
                    if dist < MinDist:
                        MinDist = dist
                        ind = j
                if self._clusterAssment[i,0] != ind:
                    isConverge = False
                    self._clusterAssment[i,:] = ind, MinDist

            # 所有数据均未修改所属簇，已经收敛
            if isConverge:
                break

            # step2: 更新质心
            for i in range(self._k):
                allindex = self._clusterAssment[:,0]
                cur_index = np.nonzero(allindex == i)[0] # 返回allindex中为i的下标
                if len(dataX[cur_index]) == 0:
                    continue
                self._centers[i,:] = np.mean(dataX[cur_index], axis=0)

        self._label = self._clusterAssment[:, 0]
        self._sse = sum(self._clusterAssment[:,1])

    def predict(self, X):
        '''
        根据train结果进行预测
        '''
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError('np.ndarray is necessary')

        num = X.shape[0]
        result = np.empty((num,))

        for i in range(num):
            MinDist = np.inf
            for j in range(self._k):
                dist = self._calDist(X[i,:], self._centers[j,:])
                if MinDist > dist:
                    MinDist = dist
                    index = j

            result[i] = index

        return result
