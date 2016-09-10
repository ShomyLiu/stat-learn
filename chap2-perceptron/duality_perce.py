#!/usr/bin/env python
# encoding: utf-8

import numpy as np

class Perce(object):
    '''
    感知机模型的对偶形式
    基本使用np包来实现
    '''
    def __init__(self, train_set, a=None, b=0, p=1):
        '''
        初始化
        '''
        self.train_set = train_set
        self.x = np.zeros((len(train_set), len(train_set[0][0])), np.float)
        for i in range(len(train_set)):
            self.x[i] = train_set[i][0]
        self.y = train_set[:, 1]
        self.a = a or np.zeros(len(train_set), np.float)
        self.b = b
        self.p = p
        self.current_update_item = None
        self.iter_time = 0
        self.Gram = self.getGram()

    def update(self, i):
        '''
        更新参数 a,b
        '''
        self.iter_time += 1
        self. a[i] += 1
        self.b = self.b + self.p * self.current_update_item[1]
        print '第', self.iter_time, '次迭代:', 'a =', self.a, 'b =', self.b

    def getGram(self):
        '''
        计算Gram矩阵
        '''
        Gram = np.zeros((len(self.x), len(self.x)))
        for i in range(Gram.shape[0]):
            for j in range(Gram.shape[1]):
                Gram[i][j] = np.dot(self.x[i], self.x[j])
        return Gram

    def isClassifyRight(self, item):
        '''
        item是否分类正确
        '''
        res = np.dot(self.a * self.y, self.Gram[item])
        res = self.y[item] * (res + self.b)
        if res <= 0:
            return False
        return True

    def check(self):
        '''
        检测误分类点
        '''
        flag = True
        for i, j in enumerate(self.train_set):
            if not self.isClassifyRight(i):
                flag = False
                print j, '误分类'
                self.current_update_item = j
                self.update(i)
        if flag:
            w = np.dot(self.a * self.y, self.x)
            print '*' * 30
            print 'The Final Result: w =', w, 'b =', self.b

        return flag

if __name__ == "__main__":
    train_set = np.array([[[3, 3], 1], [[4, 3], 1], [[1, 1], -1]])
    per = Perce(train_set=train_set)
    for i in range(100):
        if per.check():
            break
