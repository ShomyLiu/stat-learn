## 线性回归模型
[机器学习笔记-线性回归](http://shomy.top/2016/11/12/focus-linear-regression/)
## 实现


线性回归模型练习


```python
import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
%matplotlib inline
```


```python
path = os.getcwd() + '/../data/ex1data1.txt'
data = pd.read_csv(path, header=None, names=["Population", "Profit"]) # 手动设置 names, 这样可以填充第一行
```


```python
data.head(5) # 显示前5行
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.1101</td>
      <td>17.5920</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.5277</td>
      <td>9.1302</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.5186</td>
      <td>13.6620</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.0032</td>
      <td>11.8540</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.8598</td>
      <td>6.8233</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe() # 对数据进行简单预览
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>97.000000</td>
      <td>97.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.159800</td>
      <td>5.839135</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.869884</td>
      <td>5.510262</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.026900</td>
      <td>-2.680700</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.707700</td>
      <td>1.986900</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.589400</td>
      <td>4.562300</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.578100</td>
      <td>7.046700</td>
    </tr>
    <tr>
      <th>max</th>
      <td>22.203000</td>
      <td>24.147000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.plot(kind="Scatter",x="Population", y="Profit") #这就是Pandas的强大之处
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8c6083bad0>




![png](http://7xotye.com1.z0.glb.clouddn.com/output_5_1.png)


现在对数据进行线性回归建模，流程就是，数据处理-> 计算$w_{lin} = (X^TX)^{-1}X^Ty$->测试


```python
# 首先分离train_data:X 和lable: y
# 需要注意的是，x=[1,value],而非只有value, 因为 y=wx+b, 这个1对应W的第一项b.
data.insert(0, 'b', 1)
data.shape
```




    (97, 3)




```python
x = data.iloc[:,0:2]
y = data.iloc[:,2:3]
```


```python
x.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6.1101</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5.5277</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>8.5186</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>7.0032</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5.8598</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = np.matrix(x.values) # 形成X矩阵
Y = np.matrix(y.values) # 形成Y 矩阵
```


```python
X.shape,Y.shape
```




    ((97, 2), (97, 1))




```python
W = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
W
```




    matrix([[-3.89578088],
            [ 1.19303364]])



这里计算出来了W,可以知道了 该方程为y = -3.89578088x + 1.19303364


```python
# 绘制图像
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = W[0][0] + (W[1][0] * x)[0]
```


```python
f=np.array(f[0])[0]
```


```python
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
```




    <matplotlib.text.Text at 0x7f8c601f6b10>




![png](http://7xotye.com1.z0.glb.clouddn.com/output_16_1.png)

