#!/usr/bin/env python
# encoding: utf-8


class Perce(object):
    '''
    感知机模型的原始形式

    '''

    def __init__(self, train_set, w=None, b=0, p=1):
        '''
        初始化训练数据集
        :para train_set: [(), ]
        '''
        self.train_set = train_set
        self.w = w or [0 for i in range(len(train_set[0][0]))]
        self.b = b
        self.p = p
        self.current_update_item = None
        self.iter_time = 0

    def update(self):
        '''
        更新参数 w, b
        '''
        self.iter_time += 1
        assert self.w==0 or len(self.w) == len(self.current_update_item[0])
        x, y = self.current_update_item
        self.w = map(lambda x: x[0]+x[1] * y * self.p, zip(self.w, x))
        self.b = self.b + self.p * y
        print '第', self.iter_time, '迭代:', '更新: w=', self.w, 'b=', self.b

    def isClassifyRight(self, item):
        '''
        判断是否分类正确
        '''
        loss = 0
        for i in range(len(self.w)):
            loss += self.w[i] * item[0][i]
        loss = (loss + self.b) * item[1]
        if loss <=0 :
            return False
        return True

    def check(self):
        '''
        检测误分类点
        '''
        flag = True
        for item in self.train_set:
            if not self.isClassifyRight(item):
                flag = False
                print item, '误分类'
                self.current_update_item = item[:]
                self.update()
        if flag:
            print 'The Final Result: w=', self.w, 'b=', self.b
        return flag

if __name__ == "__main__":
    train_set = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
    # train_set = [[[3, 3], 1], [[4, 3], 1], [[1, 1], -1], [[5, 2], -1]]
    per = Perce(train_set=train_set)
    for i in range(100): # 这里其实严格来说应该判断是否收敛，不过由于题目简单，就check最多100次
        if per.check():
            break
