---
layout: post
title:  "模式识别（一）非参数估计：Parzen窗估计和Kn近邻估计"
image: ''
date:   2016-11-5 15:40:00
tags:
- computer science
- pattern recognition
- parzen windows
- knn
description: '北师大模式识别课程实验一：Parzen窗估计和Kn近邻估计'
categories:
- Pattern Recognition
twitter_text: '北师大模式识别课程实验一：Parzen窗估计和Kn近邻估计'
---
## Parzen窗估计

### 代码

``` matlab
function [p] = parzen(data, range, h, f)
  
  if isempty(f)
    f = @(u)(1 / sqrt(2 * pi)) * exp(-0.5 * u.^2);
  end;
  
  N = size(data, 2);
  h = h / sqrt(N);
  [X Xi] = meshgrid(range, data);
  p = sum(f((X - Xi) / h)) / N / h;
```

### 解释

#### 参数

 - data为样本数据（行向量）
 - range为待估点（行向量）
 - h为窗宽参数（程序内会做预处理：除以样本个数的平方根）
 - f为核函数，为空默认为正态函数
 
#### 预处理

 - N为样本个数
 - h除以样本个数的平方根
 - [X Xi]为待估点和样本数据的笛卡儿积

#### 结果

$$
p(x) = \frac{1}{Nh^D} \sum_{n=1}^N K\big(\frac{x - x_i}{h})
$$

### 示例

#### 代码

``` matlab
S = 1000;
xi = [rand(1, S) * 0.5 - 2.5, rand(1, S) * 2 + 1];
x = linspace(-3, 4, 400);
p = parzen(xi, x, 22, @(u)(1 * (abs(u) < 0.5)));
plot(x, p);
figure();
p = parzen(xi, x, 5,[]);
plot(x, p);
```

#### 解释

 - S为各部分数据量
 - xi为符合一定分布的样本数据
 - x为待估点
 - p为通过Parzen窗估计得出的概率密度函数值

#### 结果

** 正态核函数 **

![f1]()

** 方核函数 **

![f2]()

## Kn近邻估计

### 代码

``` matlab
function [p] = knn(data, range, kn)
  N = size(data, 2);
  [X Xi] = meshgrid(range, data);
  Dis = abs(X - Xi);
  for i = 1:kn-1
    [~, ind] = min(Dis);
    ind=sub2ind(size(Dis), ind, [1:size(Dis, 2)]);
    Dis(ind) = Inf;
  end
  Dis = min(Dis);
  V = 2 * Dis;
  p = kn * ones(1, size(range, 2)) ./ V / N;
```

### 解释

#### 参数

 - data为样本数据（行向量）
 - range为待估点（行向量）
 - kn为限定的窗内样本数
 
#### 预处理

 - N为样本个数
 - [X Xi]为待估点和样本数据的笛卡儿积
 - Dis为待估点与样本数据的距离
 
#### 寻找Kn近距离

通过$$K_n$$次`min`运算，且每次将得到的最小值改为正无穷，得到第$$K_n$$近的距离。
最终Dis为各待估点的第$$K_n$$近距离。

#### 窗体积

$$
V = c_D R_k^D(x)
$$

其中

$$R_k^D(x)$$是待估点与$$K_n$$近邻点的距离   
$$c_D = \frac{2\pi^{\frac{D}{2}}}{D\cdot\Gamma\big(\frac{D}{2}\)$$

即

$$c_1=2, c_2=\pi, c_3=\frac{4\pi}{3}$$

所以仅考虑一维情况

$$ V = 2 \cdot Dis $$

#### 结果

$$
p(x) \cong \frac{k}{NV}
$$

### 示例

#### 代码

``` matlab
S = 5000;
xi = [randn(1, S) * 0.43 + 1.5, rand(1, S) + 3];
x = linspace(0, 5, 100);
p = knn(xi, x, sqrt(S) * 20);
figure();
plot(x, p);
```

#### 解释

 - S为各部分数据量
 - xi为根据要求的分布生成的样本数据
 - x为待估点
 - p为通过Kn近邻估计得出的概率密度函数
 
#### 结果

![f3]()
