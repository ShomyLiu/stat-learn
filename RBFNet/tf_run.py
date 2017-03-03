#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from tf_rbf import TFRBFNet as RBFNet


def getData(n=100, d=5, method='sum'):
    '''
    数据生成模式
    sum: 表示y为x的和
    mul: 表示 y = 2*x
    '''
    X = np.random.uniform(1., 3.0,(n,d)).astype(np.float32)
    if method == 'sum':
        y = np.sum(X, axis=1)
    else:
        y = 2*X
    if len(y.shape) < 2:
        y = np.reshape(y,(y.shape[0], 1))
    return X,y

def main():

    numSample = 30000
    dim = 5

    method = 'sum'
    data = getData(numSample,dim, method=method)

    rbf = RBFNet(k=dim * 2)
    rbf.fit(*data)

    testNum = 10
    test_data = getData(testNum, dim, method=method)
    prediction = rbf.predict(test_data[0])
    for i in range(testNum):
        print 'X:', test_data[0][i]
        print 'real      :',test_data[1][i]
        print 'prediction:',prediction[i]
        print '*' * 30

if __name__ == "__main__":
    main()
