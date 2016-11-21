---
published: true
layout: post
date: 2016-11-21 8:53:00
tags:
  - computer science
  - machine learning
  - neural networks
  - backpropagation
description: 机器学习Coursera学习笔记：第十部分 神经网络反向传播算法
categories:
  - Machine Learning
twitter_text: 机器学习Coursera学习笔记：第十部分 神经网络反向传播算法
title: 机器学习（十）神经网络反向传播算法
---
## 简述

前面介绍了人工神经网络的表示，了解了如何利用神经网络预测结果。但我们还不知道如何训练神经网络，确定轴突的权值。

即将介绍的反向传播（backpropagation）算法，就是实现这个目的的。

## 符号约定

$$
\begin{align*}
z_i^{(j)} =& \text{第$j$层的第$i$个节点（神经元）的“计算值”} \newline
a_i^{(j)} =& \text{第$j$层的第$i$个节点（神经元）的“激活值”} \newline
\Theta^{(l)}_{i,j} =& \text{映射第$l$层到第$l+1$层的权值矩阵的第$i$行第$j$列的分量} \newline
L =& \text{神经网络总层数（包括输入层、隐层和输出层）} \newline
s_l =& \text{第$l$层节点（神经元）个数，不包括偏移量节点。} \newline 
K =& \text{输出节点个数} \newline
h_{\theta}(x)_k =& \text{第$k$个预测输出结果} \newline
x^{(i)} =& \text{第$i$个样本特征向量} \newline
x^{(i)}_k =& \text{第$i$个样本的第$k$个特征值} \newline
y^{(i)} =& \text{第$i$个样本实际结果向量} \newline
y^{(i)}_k =& \text{第$i$个样本结果向量的第$k$个分量} \newline
\end{align*}
$$

## 代价函数

回顾一下正规化的逻辑回归的代价函数：

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
$$

在神经网络中，代价是相对输出层的全部节点而言的，因此代价函数更复杂一些：

$$
\begin{gather*}\large J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}
$$

可以看到，正规化的部分也更加复杂，遍历了全部权值（除去偏移量）。

## 反向传播算法

### 目标

求 $$ \min_\Theta J(\Theta) $$

### 思路

类似梯度下降法，给定一个初值后，计算出所有节点的计算值和激活值，然后根据代价函数的变化不断调整参数值（权值），最终不断逼近最优结果，使代价函数值最小。

### 推导

为了实现上述思路，我们必须首先计算代价函数的偏导数：

$$
\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)
$$

这个偏导并不好求，为了方便推导，我们假设只有一个样本（$$m=1$$，可忽略代价函数中的外部求和），并舍弃正规化部分，然后分为两种情况来求。

#### 情况1 隐层$$\rightarrow$$输出层

我们知道：

$$
h_\Theta(x) = a^{(j+1)} = g(z^{(j+1)})
$$

$$
z^{(j)} = \Theta^{(j-1)}a^{(j-1)}
$$

另外，输出层即第$$L$$层。

所以：

$$
\dfrac{\partial}{\partial \Theta_{i,j}^{(L)}}J(\Theta)
= \dfrac{\partial J(\Theta)}{\partial h_{\Theta}(x)_i} \dfrac{\partial h_{\Theta}(x)_i}{\partial z_i^{(L)}} \dfrac{\partial z_i^{(L)}}{\partial  \Theta_{i,j}^{(L)}}
= \dfrac{\partial J(\Theta)}{\partial a_i^{(L)}} \dfrac{\partial a_i^{(L)}}{\partial z_i^{(L)}} \dfrac{\partial z_i^{(L)}}{\partial \Theta_{i,j}^{(L)}}
$$

其中：

$$
\dfrac{\partial J(\Theta)}{\partial a_i^{(L)}} = \dfrac{a_i^{(L)} - y_i}{(1 - a_i^{(L)})a_i^{(L)}}
$$

$$
\dfrac{\partial a_i^{(L)}}{\partial z_i^{(L)}} = \dfrac{\partial g(z_i^{(L)})}{\partial z_i^{(L)}} = \dfrac{e^{z_i^{(L)}}}{(e^{z_i^{(L)}}+1)^2} = a_i^{(L)} (1 - a_i^{(L)})
$$

$$
\dfrac{\partial z_i^{(L)}}{\partial \Theta_{i,j}^{(L)}} = \dfrac{\partial ( \sum_{k=0}^{s_{L-1}} \Theta_{i,k}^{(L)} a_k^{(L-1)})}{\partial  \Theta_{i,j}^{(L)}} = a_j^{(L-1)} 
$$

综上：

$$
\dfrac{\partial}{\partial \Theta_{i,j}^{(L)}}J(\Theta)
= \dfrac{\partial J(\Theta)}{\partial a_i^{(L)}} \dfrac{\partial a_i^{(L)}}{\partial z_i^{(L)}} \dfrac{\partial z_i^{(L)}}{\partial \Theta_{i,j}^{(L)}}
= \dfrac{a_i^{(L)} - y_i}{(1 - a_i^{(L)})a_i^{(L)}} a_i^{(L)} (1 - a_i^{(L)}) a_j^{(L-1)} = (a_i^{(L)} - y_i)a_j^{(L-1)}
$$

#### 情况2 隐层/输入层$$\rightarrow$$隐层

因为$$ a^{(1)}=x $$，所以可以将输入层和隐层同样对待。

$$
\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)
=\dfrac{\partial J(\Theta)}{\partial a_i^{(l)}} \dfrac{\partial a_i^{(l)}}{\partial z_i^{(l)}} \dfrac{\partial z_i^{(l)}}{\partial \Theta_{i,j}^{(l)}}\ (l = 1, 2, ..., L-1)
$$

其中后两部分偏导很容易根据前面所得类推出来：

$$
\dfrac{\partial a_i^{(l)}}{\partial z_i^{(l)}} = \dfrac{e^{z_i^{(l)}}}{(e^{z_i^{(l)}}+1)^2} = a_i^{(l)} (1 - a_i^{(l)})
$$

$$
\dfrac{\partial z_i^{(l)})}{\partial \Theta_{i,j}^{(l)}} = a_j^{(l-1)} 
$$

第一部分偏导是不好求解的，或者说是没法直接求解的，我们可以得到一个递推式：

$$
\dfrac{\partial J(\Theta)}{\partial a_i^{(l)}} = 
\sum_{k=1}^{s_{l+1}} \dfrac{\partial J(\Theta)}{\partial a_k^{(l+1)}} \dfrac{\partial a_k^{(l+1)}}{\partial z_k^{(l+1)}} \dfrac{\partial z_k^{(l+1)}}{\partial a_i^{(l)}}
$$

 > 因为该层的激活值与下一层各节点都有关，链式法则求导时需一一求导，所以有上式中的求和。

递推式中第一部分是递推项，后两部分同样易求：

$$
\dfrac{\partial a_k^{(l+1)}}{\partial z_{k}^{(l+1)}} = \dfrac{e^{z_{k}^{(l+1)}}}{(e^{z_{k}^{(l+1)}}+1)^2} = a_k^{(l+1)} (1 - a_k^{(l+1)})
$$

$$
\dfrac{\partial z_k^{(l+1)}}{\partial a_i^{(l)}} = \dfrac{\partial ( \sum_{j=0}^{s_l} \Theta_{k,j}^{(l+1)} a_j^{(l)})}{\partial a_i^{(l)}} = \Theta_{k,i}^{(l+1)}
$$
