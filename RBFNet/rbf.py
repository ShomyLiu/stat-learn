#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from kmeans import KMeans

def getData(n=100, d=5):
    '''
    生成数据 y = 2x, 多维
    '''
    X = np.random.uniform(1., 3.0,(n,d))
    y = np.sum(X, axis=1)
    return X,y


class RBFNet(object):
    '''RBF Network
    '''
    def __init__(self, k=10, delta=0.1):
        '''
        delta: 高斯函数中的扩展参数
        beta: 隐层到输出层的权重
        k: 中心的个数
        '''
        self._delta = delta
        self._beta = None
        self._hidden_num = k
        self.kms = KMeans(k)
        pass

    def _calRBF(self,x,c):
        '''
        计算RBF函数的输出，这里使用高斯函数
        '''
        return np.exp(-self._delta* np.sqrt(np.sum(np.square(x-c))))

    def _calG(self, X):
        '''
        输入层到隐层的特征转换
        G相当于公式中的大写的Z=[z1,z2,z3...zN], N为数据样本量
        G维度：N * hidden
        '''
        num, dim = X.shape
        G = np.empty((num, self._hidden_num))
        for i in range(num):
            for j in range(self._hidden_num):
                # 计算每一个数据与所有的重心的RBF输出，作为隐层神经元的输出
                G[i,j] = self._calRBF(X[i,:], self._centers[j])

        return G

    def _calPseudoInvese(self,x):
        '''
        计算矩阵伪逆
        '''
        return np.linalg.pinv(x)

    def fit(self, train_x, train_y):
        '''
        训练函数
        '''

        num, dim = train_x.shape

        # 使用KMeans无监督确定中心
        self.kms.train(train_x)
        self._centers = self.kms._centers
        # 计算Z
        self.G = self._calG(train_x)

        # 计算权重矩阵,其中包含一个求伪逆的过程
        self._beta = self._calPseudoInvese(np.dot(np.transpose(self.G), self.G))
        self._beta = np.dot(self._beta, np.transpose(self.G))
        self._beta = np.dot(self._beta, train_y)

    def predict(self, test_x):
        '''
        预测
        test_x: 可以是多个x
        '''

        if not isinstance(test_x, np.ndarray):
            try:
                test_x = np.asarray(test_x)
            except:
                raise TypeError('np.ndarray is necessary')
        if len(test_x.shape) == 1:
            test_x = test_x.reshape(1, test_x.shape[0])

        # 计算输入x的隐层的神经元的值
        # 相当于公式中\phi(X)
        G = self._calG(test_x)

        #计算最终输出
        Y = np.dot(G, self._beta)
        return Y

def main():
    data = getData(100,5)
    rbf = RBFNet()
    rbf.fit(*data)

    test_data = getData(5, 5)
    print test_data[0]
    print 'result',test_data[1]
    print 'prediction',rbf.predict(test_data[0])

if __name__ == "__main__":
    main()
