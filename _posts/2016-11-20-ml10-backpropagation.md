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
L =& 神经网络总层数（包括输入层、隐层和输出层） \newline
s_l =& 第l层节点（神经元）个数，不包括偏移量节点。 \newline 
K =& 输出节点个数 \newline
h_{\theta}(x)_k =& 第k个预测输出结果 \newline
x^{(i)} =& 第i个样本特征向量 \newline
x^{(i)}_k =& 第i个样本的第k个特征值 \newline
y^{(i)} =& 第i个样本实际结果向量 \newline
y^{(i)}_k =& 第i个样本结果向量的第k个分量 \newline
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

类似梯度下降法，给定一个初值后，不断根据代价函数的变化调整参数值（权值），最终不断逼近最优结果（代价函数值最小）。

### 推导

为了实现上述思路，我们必须首先计算代价函数的偏导数：

$$
\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)
$$

这个偏导并不好求，我们假设只有一个样本（$$m=1$$，可忽略代价函数中的外部求和），并分为两种情况来求。

#### 隐层$$\rightarrow$$输出层

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
= \dfrac{\partial J(\Theta)}{\partial h_{\Theta}(x)_i} \dfrac{\partial h_{\Theta}(x)_i}{\partial g(z_i^{(L)})} \dfrac{\partial g(z_i^{(L)})}{\partial  \Theta_{i,j}^{(L)}}
= \dfrac{\partial J(\Theta)}{\partial a_i^{(L)}} \dfrac{\partial a_i^{(L)}}{\partial g(z_i^{(L)})} \dfrac{\partial g(z_i^{(L)})}{\partial  \Theta_{i,j}^{(L)}}
$$
