大家好，这篇是有关Learning from data第七章习题的详解，这一章主要介绍了神经网络。

我的github地址：  
https://github.com/Doraemonzzz

个人主页：  
http://doraemonzzz.com/

参考资料:  
https://blog.csdn.net/a1015553840/article/details/51085129  
http://www.vynguyen.net/category/study/machine-learning/page/6/  
http://book.caltech.edu/bookforum/index.php  
http://beader.me/mlnotebook/



## Chapter7 Neural Networks

### Part 1: Exercise

#### Exercise 7.1  (Page 2)

Consider a target function $f$ whose ‘+’ and ‘-’ regions are illustrated below.  

![](https://github.com/Doraemonzzz/Learning-from-data/blob/master/photo/Chapter7/Exercise%207.1a.png?raw=true)

The target $f$ has three perceptron components $h_1, h_2, h_3$:  

![](https://github.com/Doraemonzzz/Learning-from-data/blob/master/photo/Chapter7/Exercise%207.1b.png?raw=true)

Show that 
$$
f = \overline h_1h_2h_3 + h_1\overline h_2h_3 + h_1h_2\overline h_3
$$
Is there a systematic way of going from a target which is a decomposition of perceptrons to a Boolean formula like this? [Hint: consider only the regions of $f$ which are ‘+’ and use the disjunctive normal form (or of ands).]  

这个比较简单，直接验证即可，$f$有三个区域为+，上方的区域对应$\overline h_1h_2h_3$，左边区域对应$h_1h_2\overline h_3$，右边区域对应$h_1\overline h_2h_3$，这三者合在一起即为
$$
f = \overline h_1h_2h_3 + h_1\overline h_2h_3 + h_1h_2\overline h_3
$$



#### Exercise 7.2 (Page 3)

(a) The Boolean or and and of two inputs can be extended to more than two inputs: $\text{OR}(x_1, . . . , x_M) = +1​$ if any one of the $M​$ inputs is $+1​$; $\text{AND}(x_1, . . . , x_M) = +1​$ if all the inputs equal $+1​$. Give graph representations of $\text{OR}(x_1, . . . , x_M)​$ and $\text{AND}(x_1, . . . , x_M)​$. 

(b) Give the graph representation of the perceptron: $h(x) = \text{sign} (w^Tx)​$. 

(c) Give the graph representation of $\text{OR}(x_1, \overline  x_2, x_3)$. 

(a)$\text{AND}$表示每个$x_i$都为$+1 $时结果才为$+1 $，可以如下构造
$$
\text{AND}(x_1, . . . , x_M) = \text{sign}(-M+\frac 1 2+\sum_{i=1}^Mx_i)
$$
$\text{OR}​$表示至少存在一个$x_i​$为$+1 ​$时结果为$+1 ​$，可以如下构造
$$
\text{OR}(x_1, . . . , x_M) = \text{sign}(M-\frac 1 2+\sum_{i=1}^Mx_i)
$$
(b)感知机的图像为一个超平面，二维情形为一条直线。

(c)图像为三维空间中的一个平面。



#### Exercise 7.3 (Page 4)

Use the graph representation to get an explicit formula for $f$ and show that:    
$$
f(x) = \text{sign}\Big[\text{sign}(h_1(x)-h_2(x)-\frac 3 2) -\text{sign}(h_1(x)-h_2(x)+\frac 3 2)+\frac 3 2\Big]
$$
where $h_1(x) = \text{sign}(w_1^Tx)​$ and $h_2(x) = \text{sign}(w_2^Tx)  ​$  

这个只是把下图用公式表达出来。

![](https://github.com/Doraemonzzz/Learning-from-data/blob/master/photo/Chapter7/Exercise%207.3.png?raw=true)



#### Exercise 7.4 (Page 5)

For the target function in Exercise 7.1, give the MLP in graphical form, as well as the explicit algebraic form.

先回顾$f$的形式
$$
f = \overline h_1h_2h_3 + h_1\overline h_2h_3 + h_1h_2\overline h_3
$$
图像形式电脑比较难画，这里略过，只给出代数形式，之前讨论过$\text{OR}$以及$\text{AND}$，所以这个不是很难，先考虑$\overline h_1h_2h_3$，这个是$\text{AND}$的形式，注意$\overline h_1$对应$-h_1$
$$
f_1=\text{sign}(-2.5-h_1+h_2+h_3)
$$
同理$h_1\overline h_2h_3 ,h_1h_2\overline h_3$分别对应
$$
f_2=\text{sign}(-2.5+h_1-h_2+h_3),f_3=\text{sign}(-2.5+h_1+h_2-h_3)
$$
接下来处理$\text{OR}​$，由之前讨论可知
$$
f=\text{sign}(2.5 + f_1+f_2+f_3)
$$
所以$f​$可以表达为如下形式
$$
\begin{aligned}
f&=\text{sign}(2.5 + f_1+f_2+f_3)\\
f_1&=\text{sign}(-2.5-h_1+h_2+h_3)\\
f_2&=\text{sign}(-2.5+h_1-h_2+h_3)\\
f_3&=\text{sign}(-2.5+h_1+h_2-h_3)
\end{aligned}
$$



#### Exercise 7.5 (Page 6) 

Given $w_1​$ and $\epsilon > 0​$, find $w_2​$ such that $|\text{sign}(w_1^Tx_n) - \text{tanh}(w_2^Tx_n)| ≤ \epsilon ​$ for $x_n ∈ D​$. [Hint: For large enough $α​$, $\text{sign}(x) ≈ \text{tanh}(αx)​$.]    

先回顾$\text{tanh}(x)​$的定义
$$
\text{tanh}(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}
$$
注意当$x\to +\infty​$时，$\text{tanh}(x)\to 1​$，当$x\to -\infty​$时，$\text{tanh}(x)\to -1​$，所以如果$x<0​$，$\text{sign}(x)=-1​$，当$\alpha ​$充分大时，那么$ \text{tanh}(αx)\to -1​$；如果$x>0​$，$\text{sign}(x)=1​$，当$\alpha ​$充分大时，那么$ \text{tanh}(αx)\to 1​$，从而
$$
对于充分大的\alpha，\text{sign}(x) ≈ \text{tanh}(αx)
$$
现在的目标是对与固定的$w_1​$，找到$w_2​$使得$|\text{sign}(w_1^Tx_n) - \text{tanh}(w_2^Tx_n)| ≤ \epsilon ​$，从之前论述可以知道，只要取$w_2=kw_1​$，$k​$是一个充分大的正数即可。



#### Exercise 7.6 (Page 10)

Let $V$ and $Q$ be the number of nodes and weights in the neural network, 
$$
V = \sum_{ℓ =0 }^Ld^{(ℓ)}, Q =\sum_{ℓ =0 }^L d^{(ℓ)}(d^{(ℓ-1)} + 1)
$$
In terms of $V$ and $Q$, how many computations are made in forward propagation (additions, multiplications and evaluations of $θ$). 

[Answer: $O(Q)$ multiplications and additions, and $O(V )$ $θ$-evaluations.]    

先验证下这两个等式的正确性。我们知道第$ℓ$层有$d^{(ℓ)}$个节点，所以节点数量为
$$
V = \sum_{ℓ =0 }^Ld^{(ℓ)}
$$
第$ℓ-1$层到第$ℓ+1$层的权重数量为$(d^{(ℓ-1)}+1)d^{(ℓ)}​$，所以权重数量一共有
$$
Q =\sum_{ℓ =0 }^L d^{(ℓ)}(d^{(ℓ-1)} + 1)
$$
首先看下用使用多少次$\theta$函数，由神经网络的定义我们知道，第$0$层之后每层都要使用$\theta$函数，所以使用$\theta$函数的数量为
$$
V = \sum_{ℓ =1 }^Ld^{(ℓ)}
$$
忽略第$0$层，这个数量可以近似为$O(V)$

接着看使用了多少次加法以及乘法，加法以及乘法发生的情形在如下计算中
$$
s^{(ℓ)}=(W^{(ℓ)})^Tx^{(ℓ−1)},W^{(ℓ)}\in R^{(d^{(ℓ-1)} + 1)\times d^{(ℓ)}}
$$
由矩阵乘法的定义可知$(W^{(ℓ)})^Tx^{(ℓ−1)}$一共发生了$(d^{(ℓ-1)} + 1)\times d^{(ℓ)}$次乘法，所以加法以及乘法的数量为$O(Q)$ 



#### Exercise 7.7 (Page 11) 

For the sigmoidal perceptron, $h(x) = \text{tanh}(w^Tx)​$, let the in-sample error be $E_{\text{in}}(w) = \frac{1}{N}\sum_{n=1}^N(\text{tanh}(w^Tx_n) - y_n)^2​$. Show that 
$$
∇E_{\text{in}}(w) = \frac{2} {N} \sum_{n=1}^N(\text{tanh}(w^Tx_n) - y_n)(1 - \text{tanh}^2(w^Tx_n))x_n
$$
If $w → ∞$, what happens to the gradient; how this is related to why it is hard to optimize the perceptron.

这题就是对$\text{tanh}(x)$求偏导，回顾$\text{tanh}(x)$的定义
$$
\text{tanh}(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}=2\frac{e^x}{e^x+e^{-x}}-1=2\frac{1}{1+e^{-2x}}-1
$$
回忆$\theta(x) = \frac{1}{1+e^{-x}},\theta^{'}(x)=\theta(x) (1-\theta(x) )​$，所以上式可以化为
$$
\text{tanh}(x)=2\theta(2x)-1\\
\frac{d\text{tanh}(x)}{dx}=4\theta^{'}(2x)=4\theta(2x)(1-\theta(2x))=1-(2\theta(2x)-1)^2=1-\text{tanh}^2(x)
$$
由此可以计算梯度
$$
\begin{aligned}
∇E_{\text{in}}(w) 
& = ∇ \frac{1}{N}\sum_{n=1}^N(\text{tanh}(w^Tx_n) - y_n)^2\\
&=\frac{2} {N}\sum_{n=1}^N(\text{tanh}(w^Tx_n) - y_n)∇\text{tanh}(w^Tx_n)\\
&=\frac{2} {N}\sum_{n=1}^N(\text{tanh}(w^Tx_n) - y_n)(1 - \text{tanh}^2(w^Tx_n))∇(w^Tx_n)\\
&= \frac{2} {N} \sum_{n=1}^N(\text{tanh}(w^Tx_n) - y_n)(1 - \text{tanh}^2(w^Tx_n))x_n
\end{aligned}
$$



#### Exercise 7.8 (Page 15)

Repeat the computations in Example 7.1 for the case when the output transformation is the identity. You should compute $s^{(ℓ)}, x^{(ℓ)}, δ^{(ℓ)}$ and $\frac{∂e}{∂W^{(ℓ)}}​$ 

![](https://github.com/Doraemonzzz/Learning-from-data/blob/master/photo/Chapter7/Exercise%207.8.png?raw=true)

这个是课本上对应的图，我们这里需要求的是激活函数为$h(x)=x$的情形，先计算$s^{(ℓ)},x^{(ℓ)}$。
$$
\begin{aligned}
s^{(1)}&= \left[
 \begin{matrix}
  0.1 &0.3 \\
  0.2&0.4  
  \end{matrix}
  \right]
   \left[
 \begin{matrix}
1 \\
2 
  \end{matrix}
  \right]=   \left[
  \begin{matrix}
0.7 \\
1
  \end{matrix}
  \right],
  x^{(1)}=
    \left[
  \begin{matrix}
  1\\
\tanh(0.7) \\
\tanh(1)
  \end{matrix}
  \right]=\left[
  \begin{matrix}
  1\\
0.60 \\
0.76
  \end{matrix}
  \right]\\
  
  s^{(2)}&= \left[ \begin{matrix}0.2&1&-3\end{matrix}\right] 
 \left[
  \begin{matrix}
  1\\
0.60 \\
0.76
  \end{matrix}
  \right]=-1.48,x^{(2)}=\left[
  \begin{matrix}
  1\\
\tanh(-1.48)
  \end{matrix}
  \right]=\left[
  \begin{matrix}
  1\\
-0.90
  \end{matrix}
  \right]\\
  
  s^{(3)}&=\left[ \begin{matrix}1&2\end{matrix}\right] \left[
  \begin{matrix}
  1\\
-0.90
  \end{matrix}
  \right]=-0.8,x^{(3)}=-0.8
  \end{aligned}
$$
接着计算$δ^{(ℓ)}​$，回顾更新公式
$$
\begin{aligned}
\delta_j ^{{(L)}}&=2(x_j^{(L)}-y_j)\theta^{'}(s^{L})\\
\delta _j^{{(ℓ)}}&=\theta^{'}(s_j^{(ℓ)})\sum_{k=1}^{d^{(ℓ+1)}}w_{jk}^{(ℓ+1)}\delta _k^{{(ℓ+1)}}(ℓ<L)
\end{aligned}
$$
这里$\theta^{'}(s_j^{(ℓ)})=1$，所以更新公式简化为
$$
\begin{aligned}
\delta_j ^{{(L)}}&=2(x_j^{(L)}-y_j)\\
\delta _j^{{(ℓ)}}&=\sum_{k=1}^{d^{(ℓ+1)}}w_{jk}^{(ℓ+1)}\delta _k^{{(ℓ+1)}}
\end{aligned}
$$
现在来计算$δ^{(ℓ)}$
$$
\begin{aligned}
δ^{(3)}&=2(-0.8-1)=-3.6\\
δ^{(2)}&=W^{(3)} δ^{(3)}|_1^1=2\times(-3.6)=-7.2\\
δ^{(1)}&=W^{(2)} δ^{(2)}|_1^2=\left[
  \begin{matrix}
  1\times (-7.2) \\
-3 \times (-7.2)
  \end{matrix}
  \right]=\left[
  \begin{matrix}
  -7.2 \\
21.6
  \end{matrix}
  \right]
  \end{aligned}
$$
最后计算$\frac{∂e}{∂W^{(ℓ)}}$ 
$$
\begin{aligned}

\frac{∂e}{∂W^{(1)}} &=x^{(0)}(δ^{(1)})^T=
\left[
  \begin{matrix}
  0.3\\
0.4
  \end{matrix}
  \right]
\left[\begin{matrix} -7.2& 21.6\end{matrix}\right]=
\left[
  \begin{matrix}
  -2.16&6.48\\
-2.88&8.64
  \end{matrix}
  \right]\\
  
  \frac{∂e}{∂W^{(2)}} &=x^{(1)}(δ^{(2)})^T=-7.2\times \left[
  \begin{matrix}
  1\\
0.60 \\
0.76
  \end{matrix}
  \right]=\left[
  \begin{matrix}
  -7.2\\
-4.32 \\
 -5.472
  \end{matrix}
  \right]\\
  
   \frac{∂e}{∂W^{(3)}} &=x^{(2)}(δ^{(3)})^T=-3.6\times\left[
  \begin{matrix}
  1\\
-0.90
  \end{matrix}
  \right]=\left[
  \begin{matrix}
  -3.6\\
3.24
  \end{matrix}
  \right]
  \end{aligned}
$$



#### Exercise 7.9 (Page 18)

What can go wrong if you just initialize all the weights to exactly zero?

首先来看更新公式
$$
\begin{aligned}
s^{(ℓ)} &= (W^{(ℓ)})^Tx^{(ℓ−1)}\\
x^{(ℓ)}&=\theta(s^{(ℓ)})
\end{aligned}
$$
如果权重都为$0$，那么$s^{(ℓ)} =0$，对于$\theta(x)=\text{tanh}(x),\theta(0)=0$，所以$x^{(ℓ)}=\theta(s^{(ℓ)})=0$，这说明除了第$0$层的节点，每一个节点大小均为$0​$。再看反向传播的公式
$$
\begin{aligned}
\frac{∂ e}{ ∂w_{ij}^{(ℓ)}}&=x_i^{^{(ℓ-1)}}\delta _j^{{(ℓ)}}\\
\delta _j^{{(ℓ)}}&=\theta^{'}(s_j^{(ℓ)})\sum_{k=1}^{d^{(ℓ+1)}}w_{jk}^{(ℓ+1)}\delta _k^{{(ℓ+1)}}
\end{aligned}
$$
由于权重均为$0$，所以$\frac{∂ e}{∂ w_{ij}^{(ℓ)}}=0$。结合以上两点可得，如果初始权重均为$0$，那么梯度均为$0$，从而更新量为$0$，从而更新之后每个节点依旧为$0$，这样就无法训练数据了，所以每个节点的初始值不能都取$0$。



#### Exercise 7.10 (Page 20)

It is no surprise that adding nodes in the hidden layer gives the neural network more approximation ability, because you are adding more parameters. How many weight parameters are there in a neural network with architecture specified by $d = [d^{(0)}, d^{(1)}, . . . , d^{(L)}]​$, a vector giving the number of nodes in each layer? Evaluate your formula for a $2​$ hidden layer network with $10​$ hidden nodes in each hidden layer.    

这题就是第$6​$题，这里直接给出公式
$$
Q =\sum_{ℓ =0 }^L d^{(ℓ)}(d^{(ℓ-1)} + 1)
$$



#### Exercise 7.11 (Page 24)

For weight elimination, show that $\frac{∂{E_{\text{aug}}}} {∂w^{(ℓ)}_{ij}} = \frac{∂{E_{{\text{in}}}}} {∂w^{(ℓ)}_{ij}} + 2\frac{λ}{N} \frac{w^{(ℓ)}_{ij}} {(1 + (w^{(ℓ)}_{ij})^2)^2}$ . Argue that weight elimination shrinks small weights faster than large ones.    

首先回顾$E_{\text{aug}}$
$$
E_{\text{aug}}(w, λ) = E_{\text{{in}}}(w) + \frac{λ}{N}\sum_{ℓ,i,j} \frac{(w^{(ℓ)}_{ij})^2} {1 + (w^{(ℓ)}_{ij})^2}
$$
对这个式子稍作变形
$$
E_{\text{aug}}(w, λ) = E_{\text{{in}}}(w) + \frac{λ}{N}\sum_{ℓ,i,j} \frac{(w^{(ℓ)}_{ij})^2} {1 + (w^{(ℓ)}_{ij})^2}= E_{\text{{in}}}(w)+ \frac{λ}{N}\sum_{ℓ,i,j}\Big (1-\frac{1} {1 + (w^{(ℓ)}_{ij})^2}\Big)
$$
求偏导可得
$$
\frac{∂{E_{\text{aug}}}} {∂w^{(ℓ)}_{ij}} = \frac{∂{E_{{\text{in}}}}} {∂w^{(ℓ)}_{ij}} + \frac{λ}{N} \frac{\partial \sum_{ℓ,i,j}\Big (1-\frac{1} {1 + (w^{(ℓ)}_{ij})^2}\Big)}{\partial w^{(ℓ)}_{ij}}=\frac{∂{E_{{\text{in}}}}} {∂w^{(ℓ)}_{ij}} + 2\frac{λ}{N} \frac{w^{(ℓ)}_{ij}} {(1 + (w^{(ℓ)}_{ij})^2)^2}
$$
对于较大的$w^{(ℓ)}_{ij}​$，$ \frac{w^{(ℓ)}_{ij}} {(1 + (w^{(ℓ)}_{ij})^2)^2}\to 0​$，而对于较小的$w^{(ℓ)}_{ij}​$，$ \frac{w^{(ℓ)}_{ij}} {(1 + (w^{(ℓ)}_{ij})^2)^2}\to w^{(ℓ)}_{ij}​$，所以较小的$w^{(ℓ)}_{ij}​$对应的梯度更大一些。



#### Exercise 7.12 (Page 27)

Why does outputting $w_{t^∗}$ rather than training with all the data for $t^∗$ iterations not go against the wisdom that learning with more data is better. 

[Hint: “More data is better” applies to a fixed model $(\mathcal H, A)​$. Early stopping is model selection on a nested hypothesis sets $\mathcal H_1 ⊂ \mathcal H_2 ⊂ · · ·​$ determined by $\mathcal D_{\text{train}}​$. What happens if you were to use the full data $\mathcal D​$?]    

这里我们是选模型，所以和数据多少没有关系，只有模型固定的时候，数据越多，效果才越好。



#### Exercise 7.13 (Page 27) 

Suppose you run gradient descent for $1000​$ iterations. You have $500​$ examples in $\mathcal D​$, and you use $450​$ for $\mathcal D_{\text{train}}​$ and $50​$ for $\mathcal D_{\text{val}}​$. You output the weight from iteration $50​$, with $E_{\text{val}}(w_{50}) = 0.05​$ and $E_{\text{train}}(w_{50}) = 0.04​$. 

(a) Is $E_{\text{val}}(w_{50}) = 0.05$ an unbiased estimate of $E_{\text{out}}(w_{50})$? 

(b) Use the Hoeffding bound to get a bound for $E_{\text{out}}$ using $E_\text{val}$ plus an error bar. Your bound should hold with probability at least $0.1$. 

(c) Can you bound $E_\text{out}$ using $E_{\text{train}}$ or do you need more information?

(a)根据前面的知识可知
$$
\mathbb E(E_{\text{val}} )=E_{\text{out}}
$$
所以$E_{\text{val}}(w_{50}) = 0.05$是$E_{\text{out}}(w_{50})$的无偏估计。

(b)由Hoeffding不等式可知
$$
\mathbb P\Big(E_{\text{out}}\le E_\text{val}+\sqrt{\frac{1}{2N}\text{ln}\frac{2M}{\delta}}\Big)\ge 1-\delta
$$
$M$为模型的数量，$N$为数据的数量，对于此题来说$1-\delta =0.1,\delta =0.9,N=50$，因为迭代了$1000$次，所以$M=1000$，从而
$$
\sqrt{\frac{1}{2N}\text{ln}\frac{2M}{\delta}}=\sqrt{\frac{1}{2\times 50}\text{ln}\frac{2\times 1000}{0.9}}\approx0.2776 \\
E_{\text{out}}\le E_\text{val}+\sqrt{\frac{1}{2N}\text{ln}\frac{2M}{\delta}}\approx 0.05+0.2776=0.3276
$$
(c)需要更多的信息，因为使用Hoeffding不等式的前提为
$$
\mathbb E(E_\text{train})=E_{\text{out}}
$$
但这个是不成立的。



#### Exercise 7.14 (Page 29) 

Consider the error function $E(w) = (w - w^∗)^TQ(w - w^∗)$, where $Q$ is an arbitrary positive definite matrix. Set $w = 0$. Show that the gradient $∇E(w) = -2Qw^∗$. What weights minimize $E(w)$. Does gradient descent move you in the direction of these optimal weights? Reconcile your answer with the claim in Chapter 3 that the gradient is the best direction in which to take a step. [Hint: How big was the step?]

原题中$∇E(w) = -Qw^∗$，但感觉这里应该是$∇E(w) = -2Qw^∗$

由梯度计算公式可知
$$
∇E(w) = 2Q(w-w^∗)
$$
当$w=0$时
$$
∇E(w) = -2Qw^∗
$$
由于$Q$为正定矩阵，所以$E(w) = (w - w^∗)^TQ(w - w^∗)\ge0$，并且当$w=w^{*}$时取最小值。

如果我们的初始值为$w=0$,那么负梯度的方向为$-∇E(w) = 2Qw^∗$，但这个方向并不是最优解$w^{*}$的方向，因为最优解的方向为$w^*$的方向。



#### Exercise 7.15 (Page 32) 

Show that $|η_3 - η_1|​$ decreases exponentially in the bisection algorithm. [Hint: show that two iterations at least halve the interval size.]    

如果$E(\overline \eta )>E(\eta_2)$，那么$\{\overline \eta,\eta_2,\eta_3\}$为新的U-arrangement，从而新的区间长度为
$$
\Big|\eta_3-\overline \eta\Big|=\Big|\eta_3-\frac{\eta_1+\eta_3}{2}\Big|=\frac 1 2\Big|{\eta_1- \eta_3}\Big|
$$
如果$E(\overline \eta )<E(\eta_2)$，那么$\{\eta_1,\overline \eta,\eta_2\}$为新的U-arrangement，如果再做一次更新，那么新的中点为$\eta^*=\frac{\eta_1+\eta_2}{2}$，所以第二次的U-arrangement或者为$\{\eta_1,\overline \eta,\eta^*\}$，此时第二次的区间长度为
$$
\Big|\eta_1-\eta ^*\Big|=\Big|\eta_1-\frac{\eta_1+\eta_2}{2}\Big|=\frac 1 2\Big|{\eta_1- \eta_2}\Big|<\frac 1 2\Big|{\eta_1- \eta_3}\Big|
$$
或者为$\{\overline \eta,\eta^*,\eta_3\}$，此时第二次的区间长度为
$$
\Big|\eta_3-\overline \eta\Big|=\Big|\eta_3-\frac{\eta_1+\eta_3}{2}\Big|=\frac 1 2\Big|{\eta_1- \eta_3}\Big|
$$
所以，每两次更新之后，区间长度至少缩短为原来的$\frac 1 2$



#### Exercise 7.16 (Page 34) 

Why does the new search direction pass through the optimal weights?

![](https://github.com/Doraemonzzz/Learning-from-data/blob/master/photo/Chapter7/Exercise%207.16.png?raw=true)

如图考虑二维情形，共轭梯度法的意思是，假设已经找到一个局部最优方向$w(t+1)$，那么某个方向的梯度为$0$，现在只要使得与其正交的方向的梯度变为$0$，所以要沿着这个正交方向减少梯度即可。我们现在新的搜索方向就是保证每处的梯度在减少正交方向的梯度，最终可以达到梯度完全为$0$的点，得到最优权重。



#### Exercise 7.17 (Page 37) 

The basic shape $\phi_3$ is in both the ‘$1$’ and the ‘$5$’. What other digits do you expect to contain each basic shape $\phi_1 · · · \phi_6$. How would you select additional basic shapes if you wanted to distinguish between all the digits. (What properties should useful basic shapes satisfy?)

这题就是根据数字的形状选特征，略过。



#### Exercise 7.18 (Page 38) 

Since the input $x​$ is an image it is convenient to represent it as a matrix $[x_{ij}]​$ of its pixels which are black ($x_{ij} = 1​$) or white ($x_{ij} = 0​$). The basic shape $\phi_k​$ identifies a set of these pixels which are black. 

(a) Show that feature $\phi_k$ can be computed by the neural network node 
$$
\phi_k(x) =\text{ tanh}\Big(w_0 + \sum_{ij} w_{ij}x_{ij} \Big)
$$
(b) What are the inputs to the neural network node? 

(c) What do you choose as values for the weights? [Hint: consider separately the weights of the pixels for those $x_{ij} ∈ \phi_k​$ and those $x_{ij} \not∈ \phi_k​$.] 

(d) How would you choose $w_0$? (Not all digits are written identically, and so a basic shape may not always be exactly represented in the image.) 

(e) Draw the final network, filling in as many details as you can.    

(a)因为$w_{ij}$为参数，所以只要构造特殊的$w$，必然能满足$\phi_k(x)$为指定的结果。

(b)输入数据为图像对应的$0,1$向量

(c)如果输出值为$1$，那么只要按如下方式赋值即可
$$
w_{ij}=1,x_{ij}\in \phi_k,w_{ij}=0,x_{ij}\notin \phi_k
$$
如果输出值为$-1$，那么只要按如下方式赋值即可
$$
w_{ij}=-1,x_{ij}\in \phi_k,w_{ij}=0,x_{ij}\notin \phi_k
$$
(d)我的理解是如果输入为一张空白的图片，那么应该不做任何判断，输出为$0$，从而
$$
\phi_k(0) =\text{tanh}(w_0)=0\\
w_0=0
$$
(e)略过



#### Exercise 7.19  (Page 41) 

Previously, for our digit problem, we used symmetry and intensity. How do these features relate to deep networks? Do we still need them?

由深度学习的讨论可知，我们之前选择的特征相当于auto-encoder，可以利用auto-encoder自动学习这些特征。



### Part 2: Problems

#### Problem 7.1 (Page 43)    

Implement the decision function below using a 3-layer perceptron.

![](https://raw.githubusercontent.com/Doraemonzzz/Learning-from-data/master/photo/Chapter7/Problem%207.1.png)

第一层的神经元为感知机，效果是在边界处画一条直线
$$
水平方向：s^{(1)}_1=\text{sign}(y-2),s^{(1)}_2=\text{sign}(y-1),s^{(1)}_3=\text{sign}(y+1)\\
竖直方向：s^{(1)}_4=\text{sign}(x-2),s^{(1)}_5=\text{sign}(x-1),s^{(1)}_6=\text{sign}(x+1),,s^{(1)}_7=\text{sign}(x+2)
$$
第二层的神经元是产生上下两个矩形的效果，例如产生$x\in [-2,+2],y\in [-1,+1]$这样的区间，根据之前讨论的$\text {AND}$表达形式可得
$$
x\in [-2,+2],y\in [-1,+1]: s^{(2)}_1=\text{sign}(-3.5-s^{(1)}_4+s^{(1)}_7-s^{(1)}_2+s^{(1)}_3)\\
x\in [-1,+1],y\in [+1,+2]:s^{(2)}_2=\text{sign}(-3.5-s^{(1)}_5+s^{(1)}_6-s^{(1)}_1+s^{(1)}_2)
$$
第三层把两个区域合并为一个区域，利用$\text{OR}$对应的公式可得
$$
z=\text{sign}(1.5-s^{(2)}_1-s^{(2)}_2)
$$



#### Problem 7.2 (Page 43) 

A set of $M$ hyperplanes will generally divide the space into some number of regions. Every point in $\mathbb R^d$ can be labeled with an $M$ dimensional vector that determines which side of each plane it is on. Thus, for example if $M = 3$, then a point with a vector $(-1, +1, +1)$ is on the $-1$ side of the first hyperplane, and on the $+1$ side of the second and third hyperplanes. A region is defined as the set of points with the same label.

(a) Prove that the regions with the same label are convex. 

(b) Prove that $M$ hyperplanes can create at most $2^M$ distinct regions. 

(c) [hard] What is the maximum number of regions created by $M$ hyperplanes in d dimensions? 

[Answer: $\sum_{i=0}^d\binom{M}{i}$]

[Hint: Use induction and let $B(M, d)$ be the number of regions created by $M$ $(d - 1)$-dimensional hyperplanes in $d$-space. Now consider adding the $(M + 1)$th hyperplane. Show that this hyperplane intersects at most $B(M, d - 1)$ of the $B(M, d)$ regions. For each region it intersects, it adds exactly one region, and so $B(M + 1, d)$ ≤ $B(M, d)$ + $B(M, d - 1)$. (Is this recurrence familiar?) Evaluate the boundary conditions: $B(M, 1)$ and $B(1, d)$, and proceed from there. To see that the $M +1$th hyperplane only intersects $B(M, d - 1)$ regions, argue as follows. Treat the $M + 1$th hyperplane as a $(d - 1)$-dimensional space, and project the initial $M$ hy perplanes into this space to get $M$ hyperplanes in a $(d - 1)$-dimensional space. These $M$ hyperplanes can create at most $B(M, d - 1)$ regions in this space. Argue that this means that the $M + 1$th hyperplane is only intersecting at most $B(M, d - 1)$ of the regions created by the $M$ hyperplanes in $d$-space.]    

(a)只要利用以下这个简单的事实即可：
$$
如果\text{sign}(w^Tx_1)=\text{sign}(w^Tx_2)=i，那么\text{sign}(w^T(\lambda x_1+(1-\lambda )x_2))=i\\
其中\lambda \in (0,1)
$$
这里证明一下这个结论。

不妨设$i=1;i=-1$时同理，因为
$$
\text{sign}(w^Tx_1)=\text{sign}(w^Tx_2)=1>0
$$
所以
$$
w^Tx_1> 0,w^Tx_2>0
$$
注意
$$
\lambda \in (0,1)
$$
所以
$$
w^T(\lambda x_1+(1-\lambda )x_2)=\lambda w^Tx_1+(1-\lambda )w^Tx_2>0
$$
因此
$$
\text{sign}(w^T(\lambda x_1+(1-\lambda )x_2))=1=\text{sign}(w^Tx_1)=\text{sign}(w^Tx_2)
$$
现在证明题目中的结论，假设区域内有两个点$z_1,z_2\in \mathbb R^d$，那么对于每个超平面对应的法向量$w_i$，
$$
\text{sign}(w_i^Tz_1)=\text{sign}(w_i^Tz_2)
$$
所以对于$\lambda \in (0,1)​$
$$
\text{sign}(w_i^T(\lambda z_1+(1-\lambda )z_2))=\text{sign}(w_i^Tz_1)=\text{sign}(w_i^Tz_2)
$$
因为每个$w_i$都满足这个条件，所以$\lambda z_1+(1-\lambda )z_2$对应的$M$维向量与$z_1,z_2$对应的$M$维向量都相等，从而$M$维向量对应的区域都是凸的。

(b)$M$个超平面对应一个$M$维向量，每个分量属于$\{-1,+1 \}$，所以最多有$2^M$个独立的区域

(c)令$B(M, d)$为$d$维空间中$M$个$d-1$维超平面划分的区域，考虑$B(M+1, d)$与$B(M, d)$的递推关系。假设有$M$个$d-1$维超平面，那么它最多划分$B(M, d)$个区域，现在增加一个$d-1$维超平面，那么原有的区域数量不变，考虑这个平面带来的增量，将原有的$M$个$d-1$维超平面投影到第$M+1$个超平面上，那么在这个$d-1$维超平面上最多新增$B(M, d-1)$个区域，每个$d-1$维区域对应着一个$d$维的区域，所以原区域中最多增加$B(M, d-1)$个区域。综上，以下递推关系成立：
$$
B(M+1, d)\le B(M, d)+B(M, d-1)
$$
接着证明$B(M, d)=\sum_{i=0}^d\binom{M}{i}$，先考虑两类特殊情形：$d=1$或者$M=1$
$$
B(M,1)相当于1维空间对M个点中间切一刀，一侧全为-1,另一侧全为+1,一共有2M种\\
B(1, d)表示一个平面，只能表示(+1,-1),(-1,+1)两种情况，所以B(1, d)=2
$$
所以结论对于$M=1$以及$d=1$成立，假设结论对于$M\le x,d\le y$成立，我们计算$B(x+1,y ),B(x,y+1 ),B(x+1,y+1 )$

先考虑$B(x+1,y )$
$$
\begin{aligned}
B(x+1,y )
&\le B(x,y )+B(x,y-1 )\\
&\le \sum_{i=0}^y\binom{x}{i}+\sum_{i=0}^{y-1}\binom{x}{i}\\
&=1+\sum_{i=1}^{y}\binom{x}{i}+\sum_{i=0}^{y-1}\binom{x}{i}\\
&=1+\sum_{i=0}^{y-1}\binom{x}{i+1}+\sum_{i=0}^{y-1}\binom{x}{i}\\
&=1+\sum_{i=0}^{y-1}\binom{x+1}{i+1}\\
&= \sum_{i=0}^y\binom{x+1}{i}
\end{aligned}
$$
所以结论对于$B(x+1,y )$成立。接着考虑$B(x+1,y )$
$$
\begin{aligned}
B(x,y+1 )-B(x-1,y+1 )&\le B(x-1,y )\\
B(x-1,y+1 )-B(x-2,y+1 )&\le B(x-2,y )\\
...\\
B(2,y+1 )-B(1,y+1 )&\le B(1,y )
\end{aligned}

$$

累加可得
$$
B(x,y+1 )-B(1,y+1 )\le \sum_{i=1}^{x-1}B(i,y )
$$
接着对$\sum_{i=1}^{x-1}B(i,y )$进行处理
$$
\begin{aligned}
\sum_{i=1}^{x-1}B(i,y )
&\le \sum_{i=1}^{x-1}\sum_{j=0}^y\binom{i}{j}\\
&=\sum_{j=0}^y \sum_{i=1}^{x-1} \binom{i}{j}\\
&=\sum_{j=0}^y \sum_{i=j}^{x-1} \binom{i}{j}-\binom{0}{0}\\
&=\sum_{j=0}^y \binom{x}{j+1}-1\\
&=\sum_{i=1}^{y+1} \binom{x}{i}-1
\end{aligned}
$$
因此
$$
\begin{aligned}

B(x,y+1 )&\le B(1,y+1 )+ \sum_{i=1}^{x-1}B(i,y )\\
&\le 2+  \sum_{i=1}^{y+1} \binom{x}{i}-1\\
&= \sum_{i=0}^{y+1} \binom{x}{i}
\end{aligned}
$$
所以结论对于$B(x,y+1 )$成立。最后考虑$B(x+1,y+1 )$
$$
B(x+1,y+1 )  \le B(x,y+1 )+B(x,y )
$$
由于$B(x,y+1 ),B(x,y )​$均满足结论，所以接下来的步骤同证明$B(x+1,y )​$的步骤，从而结论对于$B(x+1,y+1 )​$也成立，由数学归纳法可知，该结论成立。



#### Problem 7.3 (Page 44)

Suppose that a target function $f​$ (for classification) is represented by a number of hyperplanes, where the different regions defined by the hyperplanes (see Problem 7.2) could be either classified $+1​$ or $-1​$, as with the 2-dimensional examples we considered in the text. Let the hyperplanes be $h_1, h_2, . . . , h_M​$, where $h_m(x) = \text{sign}(w_m. x)​$. Consider all the regions that are classified $+1​$, and let one such region be $r^+​$. Let $c = (c_1, c_2, . . . , c_M)​$ be the label of any point in the region (all points in a given region have the same label); the label $c_m = ±1​$ tells which side of $h_m​$ the point is on. Define the AND-term corresponding to region $r​$ by 
$$
t_r = h^{c_1}_1h^{c_2}_2 . . . h^{c_M}_M , \text{where } 
h^{c_m}_m =\begin{cases} 
h_m & \text{if } c_m=+1\\
\overline h_m & \text{if } c_m=-1\\
\end{cases}
$$
Show that $f = t_{r_1} +t_{r_2} +· · ·+t_{r_k}$ , where $r_1, . . . , r_k$ are all the positive regions. (We use multiplication for the AND and addition for the OR operators.) 

题目的意思是将某个用于二分类的目标函数$f$利用$M$个超平面表示出来，超平面为$h_1, h_2, . . . , h_M$，其中$h_m(x) = \text{sign}(w_m. x)$，这样就产生了一个$M$维向量$(h_1(x), h_2(x), . . . , h_M(x))$，$M$维向量一致的点构成了区域，接着将部分$M$维向量映射到$+1$，其余的映射到$-1$，二元分类函数$f$就完成了。举一个具体例子，现在有两个超平面$h_1,h_2$，这样会产生$(+1,+1),(+1,-1),(-1,+1),(-1,-1)$四个$2$维向量，现在将$(+1,+1),(+1,-1),(-1,+1)$对应的分类记为$+1$，$(-1,-1)$对应的分类记为$-1​$，这样分类就完成了。

现在考虑分类结果为$+1​$的某个区域，假设这个区域内的点对应的$M​$维向量均为$c = (c_1, c_2, . . . , c_M)​$，一个点$x​$属于该区域等价于
$$
(h_1(x), h_2(x), . . . , h_M(x))=(c_1, c_2, . . . , c_M)
$$
这里假设
$$
h_m(x)= c_m
$$
我们考虑$h^{c_m}_m(x)$，当$c_m=+1$时，那么$h_m(x)=+1=c_m$，因此$h^{c_m}_m(x)=h_m(x)=+1$；当$c_m=-1$时，那么$h_m(x)=-1=c_m$，因此$h^{c_m}_m(x)=\overline h_m(x)=+1$。

所以
$$
h_m(x)=c_m \Leftrightarrow h^{c_m}_m(x)=+1
$$
从而
$$
\begin{aligned}
&(h_1(x), h_2(x), . . . , h_M(x))=(c_1, c_2, . . . , c_M)  \\
\Leftrightarrow& h^{c_m}_m(x)=+1 (m=1,...,M) \\
\Leftrightarrow& \Pi _{m=1}^M h^{c_m}_m(x)=+1\\
\Leftrightarrow &t_r(x)=+1
\end{aligned}
$$
这里是讨论属于某一个结果为$+1​$的某个区域，如果考虑全部结果为$+1​$的某个区域$r_1, . . . , r_k​$，那么关系应该为或，所以目标函数为
$$
f = t_{r_1} +t_{r_2} +· · ·+t_{r_k}
$$



#### Problem 7.4 (Page 44)

Referring to Problem 7.3, any target function which can be decomposed into hyperplanes $h_1, h_2, . . . , h_M​$ can be represented by $f = t_{r_1} +t_{r_2} +· · ·+t_{r_k}​$ , where there are $k​$ positive regions. What is the structure of the 3-layer perceptron (number of hidden units in each layer) that will implement this function, proving the following theorem: 

Theorem. Any decision function whose $±1$ regions are defined in terms of the regions created by a set of hyperplanes can be implemented by a 3-layer perceptron.

回顾上题：
$$
t_r = h^{c_1}_1h^{c_2}_2 . . . h^{c_M}_M,r=1,...,k
$$
所以第一层构造$kM$个神经元，用来表示每个
$$
h_m^{c_{r,m}}
$$
其中$c_{r,m}$表示$t_r$中$h_m $对应的$c_r$：
$$
s^{(1)}_{r,m}= \text{sign}(c_{r,m}w_m. x)(r=1,...k,m=1,...,M)
$$
下面验证
$$
s^{(1)}_{r,m}= \text{sign}(c_{r,m}w_m. x)= h_m^{c_{r,m}} \tag 1
$$
由上一题可得
$$
h_m(x)= \text{sign}(w_m. x)=c_{r,m} \Leftrightarrow h^{c_{r,m}}_m(x)=+1
$$
因此
$$
\text{sign}(c_{r,m}w_m. x) =1 \Leftrightarrow h^{c_{r,m}}_m(x)=+1
$$
所以(1)成立。

第二层用来表示如下整体
$$
t_r = h^{c_1}_1h^{c_2}_2 . . . h^{c_M}_M,r=1,...,k
$$
构造$k$个神经元，每个神经元表示$t_r $
$$
s^{(2)}_r= \text{sign}(-k +\frac 1 2+\sum_{m=1}^M s^{(1)}_{r,m})(r=1,...,k)
$$
最后一层表示
$$
f = t_{r_1} +t_{r_2} +· · ·+t_{r_k}
$$
所以
$$
f=\text{sign}(k-\frac 1 2 -\sum_{r=1}^{k}s^{(2)}_r)
$$



#### Problem 7.5 (Page 44) 

[Hard] State and prove a version of a Universal Approximation Theorem: 

Theorem. Any target function $f$ (for classification) defined on $[0, 1]^d$, whose classification boundary surfaces are smooth, can arbitrarily closely be approximated by a $3$-layer perceptron. 

[Hint: Decompose the unit hypercube into $\epsilon $-hypercubes ( $\frac 1 {\epsilon ^d} $ of them); The volume of these $\epsilon $-hypercubes which intersects the classification boundaries must tend to zero (why? – use smoothness). Thus, the function which takes on the value of $f$ on any $\epsilon $-hypercube that does not intersect the boundary and an arbitrary value on these boundary $\epsilon $-hypercubes will approximate $f$ arbitrarily closely, as $\epsilon → 0 $ . ]

我们知道分类问题其实是将某些区域的点映射为$+1$，其余区域的点映射为$-1​$，由Problem 7.3,7.4我们知道，对于超平面划分出来的区域，我们可以利用三层感知机来表示这个分类函数，而对于一般区域，由于分类边界光滑，所以由数学分析中的知识可知只要超平面足够多，这些超平面划分的区域可以与目标区域无限接近，从而只要超平面足够多，我们就可以利用三层感知机表达任意的分类问题。

注：这里没有严格证明，只简单解释了思路。



#### Problem 7.6 (Page 44)

The finite difference approximation to obtaining the gradient is based on the following formula from calculus: 
$$
\frac{∂h} {∂w^{(ℓ)}_{ij}} =\frac{ h(w^{(ℓ)}_{ij}+ \epsilon) - h(w^{(ℓ)}_{ij} -  \epsilon)} {2\epsilon} + O(\epsilon^2)
$$
where $h(w^{(ℓ)}_{ij}+\epsilon)$ to denotes the function value when all weights are held at their values in $w$ except for the weight $w^{(ℓ)}_{ij}$, which is perturbed by $\epsilon$. To get the gradient, we need the partial derivative with respect to each weight. 

Show that the computational complexity of obtaining all these partial derivatives $O(W^2)​$. [Hint: you have to do two forward propagations for each weight.]    

题目中没有交代$W$是什么，我推测为权重的数量，公式同Exercise 7.6的$W=Q =\sum_{ℓ =0 }^L d^{(ℓ)}(d^{(ℓ-1)} + 1)$

这里要计算
$$
h(w^{(ℓ)}_{ij}\pm \epsilon)
$$
注意到如果修改了第$\ell $层的某个权重，由计算过程我们知道$l$层之前的节点无需重新计算，$l$层需要重新计算一次，之后的每一层的节点都要重新计算，所以修改第$l$层的某个权重计算的复杂度为
$$
1+\sum_{ℓ =l+1 }^L d^{(ℓ)}(d^{(ℓ-1)} + 1)
$$
第$l​$层一共有$d^{(l)}(d^{(l-1)} + 1)​$个权重，所以修改第$l​$层的全部权重的计算复杂度为
$$
d^{(l)}(d^{(l-1)} + 1)\times (1+\sum_{ℓ =l+1 }^L d^{(ℓ)}(d^{(ℓ-1)} + 1))
$$
从而修改每个权重的计算复杂度为上式关于$l$累加，可得
$$
\begin{aligned}
\sum_{l=0}^L d^{(l)}(d^{(l-1)} + 1)\times (1+\sum_{ℓ =l+1 }^L d^{(ℓ)}(d^{(ℓ-1)} + 1)) 
&\le  \sum_{l=0}^L d^{(l)}(d^{(l-1)} + 1)\times (1+\sum_{ℓ =0 }^L d^{(ℓ)}(d^{(ℓ-1)} + 1)) 
\\=W(1+W)
\end{aligned}
$$

所以计算复杂度为$O(W^2)$



#### Problem 7.7 (Page 45)

Consider the $2$-layer network below, with output vector $\hat y$. This is the two layer network used for the greedy deep network algorithm. 

![](https://github.com/Doraemonzzz/Learning-from-data/blob/master/photo/Chapter7/Problem%207.7.png?raw=true)

Collect the input vectors $x_n$ (together with a column of ones) as rows in the input data matrix $X$, and similarly form $Z$ from $z_n$. The target matrices $Y$ and $\hat Y$ are formed from $y_n$ and $\hat y_n$ respectively. Assume a linear output node and the hidden layer activation is $θ(·)$.

(a) Show that the in-samp 
$$
E_{\text{in}} = \frac 1 N \text{trace}\Big ((Y - \hat Y)(Y - \hat Y) ^ T\big)
$$
 where 
$$
\hat Y = ZV\\ 
Z = [1, θ(XW)]\\
X \text{ is } N × (d + 1) \\
W \text{ is } (d + 1) × d^{(1)}\\
Z  \text{ is } N × (d^{(1)} + 1) \\
V =  \left[ \begin{matrix}V_0 \\V_1\end{matrix} \right]  \text{ is }  (d^{(1)} + 1) × \text{dim}(y) \\
Y, \hat Y \text{ are } N × \text{dim}(y)
$$
(It is convenient to decompose $V$ into its first row $V_0$ corresponding to the biases and its remaining rows $V_1$; $ 1$ is the $N × 1$ vector of ones.) 

(b) derive the gradient matrices: 
$$
\begin{aligned}
\frac{∂E_{\text{in}}}{∂V}& = \frac  1 N(2Z^TZV - 2Z^TY)\\
\frac{∂E_{\text{in}}} {∂W}& = \frac  2 N X^T [θ′(XW) ⊗ (θ(XW)V_1V_1^T + 1V_0V_1^T - YV_1^T)] 
\end{aligned}
$$
 where $⊗​$ denotes element-wise multiplication. Some of the matrix deriva tives of functions involving the trace from the appendix may be useful.    

(a)首先写出$E_{\text{in}}$的定义
$$
E_{\text{in}}=\frac 1 N \sum_{i=1}^N||y_i-\hat y_i||^2
$$
考虑$(Y - \hat Y)(Y - \hat Y) ^ T$的第$i,j$个元素
$$
[(Y - \hat Y)(Y - \hat Y) ^ T]_{ij}=(y_i-\hat y_i)(y_j-\hat y_j)^T
$$
所以
$$
E_{\text{in}} = \frac 1 N \text{trace}\Big ((Y - \hat Y)(Y - \hat Y) ^ T\big)
$$
(b)将$\hat Y = ZV$带入，展开化简
$$
\begin{aligned}
E_{\text{in}} 
&= \frac 1 N \text{trace}\Big ((Y - ZV)(Y -ZV) ^ T\big)\\
&= \frac 1 N \text{trace}\Big (YY^T-ZVY^T-YV^TZ^T+ZVV^TZ^T\big)\\
&= \frac 1 N \text{trace}\Big (YY^T-2ZVY^T+ZVV^TZ^T\big) &\text{trace X}= \text{trace }X^T
\end{aligned}
$$
接着利用如下两个性质
$$
\begin{aligned}
\frac{\partial  (\text{trace}(AXB))}{\partial X}&=A^TB^T\\
\frac{\partial  (\text{trace}(AXX^TB))}{\partial X}&=A^TB^TX+BAX
\end{aligned}
$$
计算$\frac{\partial E_{\text{in}}}{\partial V}​$
$$
\begin{aligned}
\frac{\partial E_{\text{in}}}{\partial V}
&=\frac 1N \big( -2 \frac{ \partial(\text{trace}(ZVY^T))}{\partial V}+\frac{\partial( \text{trace}(ZVV^TZ^T))}{\partial V}\big)\\
&=\frac 1 N \big( -2Z^TY + Z^TZV+Z^TZV\big)\\
&=\frac 1 N \big( 2Z^TZV-2Z^TY\big)
\end{aligned}
$$
为了计算$\frac{∂E_{\text{in}}} {∂W} $，将$Z=[1, θ(XW)],V =  \left[ \begin{matrix}V_0 \\V_1\end{matrix} \right] $带入
$$
\begin{aligned}
E_{\text{in}} 
&= \frac 1 N \text{trace}\Big ((Y - ZV)(Y -ZV) ^ T\big)\\
&= \frac 1 N \text{trace}\Big (YY^T-2ZVY^T+ZVV^TZ^T\big)\\
&= \frac 1 N \text{trace}\Big (YY^T-2(V_0+θ(XW)V_1)Y^T+(V_0+θ(XW)V_1)(V_0+θ(XW)V_1)^T\big)\\
&= \frac 1 N \text{trace}\Big (YY^T-2V_0Y^T-2θ(XW)V_1Y^T+V_0V_0^T+θ(XW)V_1V_0^T+V_0V_1^Tθ(XW)^T + θ(XW)V_1V_1^Tθ(XW)^T\big)\\
&= \frac 1 N \text{trace}\Big (YY^T-2V_0Y^T-2θ(XW)V_1Y^T+V_0V_0^T+2θ(XW)V_1V_0^T+ V_1V_1^Tθ(XW)^Tθ(XW)\big)
\end{aligned}\\
最后一步是因为\text{trace X}= \text{trace }X^T以及\text{trace AB}=\text{trace BA}
$$
接着利用如下两个性质
$$
\begin{aligned}
\frac{\partial  (\text{trace}(\theta(BX)A))}{\partial X}&=B^T(\theta^{'}(BX)⊗A^T)\\
\frac{\partial  (\text{trace}(A\theta(BX)^T\theta(BX)))}{\partial X}&=B^T(\theta^{'}(BX)⊗[\theta(BX)(A+A^T)])
\end{aligned}
$$

计算$\frac{\partial E_{\text{in}}}{\partial W}​$
$$
\begin{aligned}
\frac{\partial E_{\text{in}}}{\partial W}
&=\frac 1 N \Big(-2\frac {\partial (\text{trace}(θ(XW)V_1Y^T))}{\partial  W} +
2 \frac{\partial (\text{trace}(θ(XW)V_1V_0^T)}{\partial  W}+\frac{\partial (\text{trace}( V_1V_1^Tθ(XW)^Tθ(XW)))}{\partial  W}\Big)\\
&=\frac 1 N  \Big( -2X^T (\theta ^{'}(XW)⊗ YV_1^T)+2X^T (\theta ^{'}(XW)⊗V_0V_1^T)+X^T(\theta ^{'}(XW) ⊗[\theta(XW)2V_1V_1^T])
\Big)\\
&=\frac 1  N(2 X^T[\theta ^{'}(XW) ⊗(\theta(XW)V_1V_1^T)+1V_0V_1^T-YV_1^T)])
\end{aligned}
$$
所以结论成立。

（备注，上述梯度公式请参考附录）



#### Problem 7.8 (Page 46) 

Quadratic Interpolation for Line Search Assume that a U-arrangement has been found, as illustrated below. 

![](https://github.com/Doraemonzzz/Learning-from-data/blob/master/photo/Chapter7/Problem%207.8.png?raw=true)

Instead of using bisection to construct the point $\overline η$, quadratic interpolation fits a quadratic curve $E(η) = aη^2 +bη +c$ to the three points and uses the minimum of this quadratic interpolation as $\overline η$. 

(a) Show that the minimum of the quadratic interpolant for a U-arrangement is within the interval $[η_1, η_3]$. 

(b) Let $e_1 = E(η_1), e_2 = E(η_2), e_3 = E(η_3)$. Obtain the quadratic function that interpolates the three points $\{(η_1, e_1), (η_2, e_2), (η_3, e_3)\}$. Show that the minimum of this quadratic interpolant is given by: 
$$
\overline η =\frac 1 2  \Big[\frac{(e_1- e_2)(\eta_1^2-\eta_3^2)-(e_1- e_3)(\eta_1^2-\eta_2^2)}{(e_1- e_2)(\eta_1-\eta_3)-(e_1- e_3)(\eta_1-\eta_2)}\Big]
$$
[Hint: $e_1 = aη_1^2 + bη_1 + c, e_2 = aη_2^2 + bη_2 + c, e_3 = aη_3^2 + bη_3 + c$. Solve for $a, b, c$ and the minimum of the quadratic is given by $\overline η = -b/2a$. ] 

(c) Depending on whether $E(\overline η)$ is less than $E(η_2)$, and on whether $\overline η$ is to the left or right of $η_2$, there are $4$ cases. In each case, what is the smaller U-arrangement?

(d) What if $\overline η = η_2$, a degenerate case? 

Note: in general the quadratic interpolations converge very rapidly to a locally optimal η. In practice, $4$ iterations are more than sufficient. 

(a)由课本之前论述可知
$$
E(\eta_2)< \text{min}\{E(\eta_1),E(\eta_3)\}
$$
$\overline \eta​$为过$\{(η_1, E(\eta_1)), (η_2, E(\eta_2)), (η_3, E(\eta_3))\}​$三点的二次函数的对称轴，如果$\overline \eta <\eta_1​$，那么$E(\eta_1)<E(\eta_2)<E(\eta_3)​$或$E(\eta_1)>E(\eta_2)>E(\eta_3)​$，与$E(\eta_2)< \text{min}\{E(\eta_1),E(\eta_3)\}​$矛盾，所以$\overline \eta \ge \eta_1​$，同理$\overline \eta \le \eta_3​$，所以
$$
\overline \eta \in [η_1, η_3]
$$
(b)$\eta =-\frac{b}{2a}​$，所以只要计算$a,b​$即可，将$e_i​$的定义带入
$$
e_1 = aη_1^2 + bη_1 + c, e_2 = aη_2^2 + bη_2 + c, e_3 = aη_3^2 + bη_3 + c
$$
计算$e_1-e_2,e_1-e_3$
$$
a(\eta_1^2-\eta_2^2)+b(\eta_1-\eta_2)=e_1-e_2\\
a(\eta_1^2-\eta_3^2)+b(\eta_1-\eta_3)=e_1-e_3
$$
解这个二元一次方程组可得
$$
\begin{aligned}
b&= \frac{-(e_1- e_2)(\eta_1^2-\eta_3^2)+(e_1- e_3)(\eta_1^2-\eta_2^2)}{(\eta_1^2-\eta_2^2)(\eta_1-\eta_3)-(\eta_1^2-\eta_3^2)(\eta_1-\eta_2)}\\\
a &= \frac{(e_1- e_2)(\eta_1-\eta_3)-(e_1- e_3)(\eta_1-\eta_2)}{(\eta_1^2-\eta_2^2)(\eta_1-\eta_3)-(\eta_1^2-\eta_3^2)(\eta_1-\eta_2)}\\
\overline \eta &= -\frac {b}{2a}=\frac 1 2  \Big[\frac{(e_1- e_2)(\eta_1^2-\eta_3^2)-(e_1- e_3)(\eta_1^2-\eta_2^2)}{(e_1- e_2)(\eta_1-\eta_3)-(e_1- e_3)(\eta_1-\eta_2)}\Big]
\end{aligned}
$$
(c)如果$E(\overline \eta)<E(\eta_2)​$

- 当$\overline \eta<\eta_2$时，U-arrangement为$(\eta_1,\overline \eta,\eta_2)$
- 当$\overline \eta>\eta_2$时，U-arrangement为$(\eta_2,\overline \eta,\eta_3)$

如果$E(\overline \eta)>E(\eta_2)$

- 当$\overline \eta<\eta_2$时，U-arrangement为$(\overline \eta,\eta_2,\eta_3)$
- 当$\overline \eta>\eta_2$时，U-arrangement为$(\eta_1,\eta_2,\overline \eta)$

(d)如果$\overline \eta = \eta_2​$，那么可以在$\eta_2​$的邻域内再找一个点$\eta_2^{'}​$，使得$\overline \eta \ne \eta_2​$且$E(\eta_2^{'})< \text{min}\{E(\eta_1),E(\eta_3)\}​$，然后重复上述迭代步骤即可。



#### Problem 7.9 (Page 46)

[Convergence of Monte-Carlo Minimization] Suppose the global minimum $w^∗$ is in the unit cube and the error surface is  quadratic near $w^∗$. So, near $w^∗$,
$$
E(w) = E(w^∗) + \frac 12
(w − w^∗)^T H(w − w^∗)
$$
where the Hessian $H$ is a positive definite and symmetric. 

(a) If you uniformly sample $w$ in the unit cube, show that 
$$
P [E ≤ E(w^∗) +\epsilon] = \underset{x^THx\le 2\epsilon}{\int} {d^dx} = \frac{S_d(2\epsilon)}{\sqrt{\text{det}H}}
$$
where $S_d(r)$ is the volume of the $d$-dimensional sphere of radius $r$, 
$$
S_d(r) = π^{d/2}r^d/Γ( \frac d2 + 1)
$$
[Hints: $P [E ≤ E(w^∗) + \epsilon] = P [\frac 12(w - w^∗)^T H(w - w^∗) ≤ \epsilon]$. Suppose the orthogonal matrix $A$ diagonalizes $H$: $A^THA =\text{diag}[λ^2_1, . . . , λ^2_d]$. Change variables to $u = A^Tx$ and use $\text {det} H = λ^2_1λ^2_2 . . . λ^2_d$.]        

(b) Suppose you sample $N$ times and choose the weights with minimum error, $w_{\min}$. Show that 
$$
P [E(w_{\min}) > E(w^∗) + \epsilon] ≈\Big (1 - \frac{1} {\sqrt{πd}}\Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d\Big)^N
$$
where $\mu ≈ \sqrt{8eπ/\overline λ}$ and $\overline λ$ is the geometric mean of the eigenvalues of $H$. (You may use $Γ(x + 1) ≈ x^xe^{-x}\sqrt{2πx}$.)    

(c) Show that if $N ∼ ( \frac{\sqrt d} {\mu \epsilon} )^d \text{log} \frac 1 η$ , then with probability at least $1 - η$, $E(w_{min}) ≤ E(w^∗) + \epsilon$. (You may use $\text{log}(1 - a) ≈ -a$ for small $a$ and $(πd)^{1/d} ≈ 1$.)    

题目中(b)的结论有点问题，我对其做了修改。

(a)因为$H$半正定，所以存在正交矩阵$H$，$A^THA =\text{diag}[λ^2_1, . . . , λ^2_d]= S$，所以$H=ASA^T$，可以得到以下变形
$$
\begin{aligned}
P [E ≤ E(w^∗) +\epsilon] 
&= \underset{x^THx\le 2\epsilon}{\int} {d^dx} \\
&= \underset{x^TASA^Tx\le 2\epsilon}{\int} {d^dx}\\
&\overset {y=A^Tx} =\underset{y^TSy\le 2\epsilon}{\int} |\text{det}\frac{\partial x}{\partial y}|{d^dy}\\
&= \underset{y^TSy\le 2\epsilon}{\int} {d^dy}&A是正交矩阵
\end{aligned}
$$
注意$y^TSy=\sum_{i=1}^d λ^2_iy_i^2​$ ，所以令$z_i=\lambda_i y_i​$，注意$\text {det} H = λ^2_1λ^2_2 . . . λ^2_d​$，那么$|\text{det}\frac{\partial y}{\partial z}|=\frac 1 {\prod _{i=1}^d\lambda_i}=\frac  1 {\sqrt{\text{det}H}}​$，带入上式可得
$$
\begin{aligned}
P [E ≤ E(w^∗) +\epsilon] 
&= \underset{y^TSy\le 2\epsilon}{\int} {d^dy}\\
& =\underset{z^TSz\le 2\epsilon}{\int}|\text{det}\frac{\partial y}{\partial z}| {d^dz}\\
&=\frac{\underset{z^TSz\le 2\epsilon}{\int} {d^dz}}{{\sqrt{\text{det}H}}}\\
&=\frac{S_d(2\epsilon)}{\sqrt{\text{det}H}}
\end{aligned}
$$
(b)
$$
\begin{aligned}
P [E(w_{\min}) > E(w^∗) + \epsilon]
&=\prod_{i=1}^N P [E(w) > E(w^∗) + \epsilon]\\
&=\Big (P [E(w) > E(w^∗) + \epsilon]\Big)^N\\
&=\Big (1-P [E(w) \le E(w^∗) + \epsilon]\Big)^N\\
&=\Big (1-\frac{S_d(2\epsilon)}{\sqrt{\text{det}H}}\Big)^N
\end{aligned}
$$
接着计算$S_d(2\epsilon)​$，利用$S_d(r) = π^{d/2}r^d/Γ( \frac d2 + 1)​$以及$Γ(x + 1) ≈ x^xe^{-x}\sqrt{2πx}​$
$$
\begin{aligned}
S_d(2\epsilon)&=π^{d/2}(2\epsilon)^d/Γ( \frac d2 + 1)\\
&\approx \frac{π^{d/2}(2\epsilon)^d}{(\frac d2)^{\frac d2}e^{-\frac d2}\sqrt{2 \pi \frac d2}}\\
&=\frac {{(8e\pi) }^{\frac d 2}\epsilon ^d}{\sqrt{\pi d}d^{\frac d 2 }}
\end{aligned}
$$
注意$\overline λ$为$H$特征值的几何平均数，所以
$$
{\overline λ}^d =\text{det}H
$$
注意$\mu \approx \sqrt{8eπ/\overline λ}​$，将这些带入可得
$$
\begin{aligned}
P [E(w_{\min}) > E(w^∗) + \epsilon]
&=\Big (1-\frac{S_d(2\epsilon)}{\sqrt{\text{det}H}}\Big)^N\\
&\approx \Big (1-\frac {{(8e\pi) }^{\frac d 2}\epsilon ^d}{\sqrt{\pi d}d^{\frac d 2 }{\overline λ}^{\frac d2}}\Big)^N\\
&=\Big (1-\frac {\mu^d \epsilon ^d}{\sqrt{\pi d}d^{\frac d 2 }}\Big)^N\\
&=\Big (1-\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d\Big)^N
\end{aligned}
$$
(c)感觉这题可能有点问题，先按照题目中的条件对其化简
$$
\begin{aligned}
P [E(w_{\min}) > E(w^∗) + \epsilon]
&\approx \Big (1-\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d\Big)^N\\
&=e^{N\text {ln} (1-\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d)}\\
&\approx e^{-N\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d}
\end{aligned}
$$
利用$N ∼ ( \frac{\sqrt d} {\mu \epsilon} )^d \text{log} \frac 1 η​$对$-N\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d​$进行化简
$$
\begin{aligned}
-N\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d 
&\approx -( \frac{\sqrt d} {\mu \epsilon} )^d \text{log} \frac 1 η\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d\\
&= \frac 1{\sqrt{\pi d}} \text{log}η
\end{aligned}
$$
从而
$$
\begin{aligned}
P [E(w_{\min}) > E(w^∗) + \epsilon]

&\approx e^{-N\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d}\\
&\approx e^{ \frac 1{\sqrt{\pi d}} \text{log}η}\\
&= \eta^{ \frac 1{\sqrt{\pi d}}}
\end{aligned}
$$
于是做到这一步就没法继续了，但是如果对条件加以修改：$N ∼ ( \frac{\sqrt d} {\mu \epsilon} )^d(\pi d)^{\frac 1 2+\frac  1d} \text{log} \frac 1 η​$，那么
$$
\begin{aligned}
-N\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d 
&\approx -( \frac{\sqrt d} {\mu \epsilon} )^d(\pi d)^{\frac 1 2+\frac  1d} \text{log} \frac 1 η\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d\\
&=(\pi d)^{\frac 1 d} \text{log}η
\end{aligned}
$$

$$
\begin{aligned}
P [E(w_{\min}) > E(w^∗) + \epsilon]

&\approx e^{-N\frac 1{\sqrt{\pi d}} \Big(\mu\frac{\epsilon}{\sqrt d}\Big)^d}\\
&\approx e^{ (\pi d)^{\frac 1 d} \text{log}η}\\
&\approx e^{\text{log}η}\\
&=\eta
\end{aligned}
$$




#### Problem 7.10 (Page 47)

For a neural network with at least $1$ hidden layer and $\text{tanh}(·)$ transformations in each non-input node, what is the gradient (with respect to the weights) if all the weights are set to zero. Is it a good idea to initialize the weights to zero?    

同Exercise 7.9。



#### Problem 7.11 (Page 47) 

[Optimal Learning Rate] Suppose that we are in the vicinity of a local minimum, $w^∗​$, of the error surface, or that the error surface is quadratic. The expression for the error function is then given by 
$$
E(w_t) = E(w^∗) +\frac  1 2 (w_t - w^∗)H(w_t - w^∗) \tag {7.8 }
$$
from which it is easy to see that the gradient is given by $g_t = H(w_t-w^∗)​$. The weight updates are then given by $w_{t+1} = w_t - ηH(w_t - w^∗)​$, and subtracting $w^∗​$ from both sides, we see that 
$$
\epsilon _{t+1} = (I - ηH)\epsilon _{t}  \tag{7.9}
$$
Since $H​$ is symmetric, one can form an orthonormal basis with its eigenvectors. Projecting $\epsilon _{t}​$ and $\epsilon _{t+1}​$ onto this basis, we see that in this basis, each component decouples from the others, and letting $\epsilon (α)​$ be the $α^{th}​$ component in this basis, we see that 
$$
\epsilon _{t+1}(α) = (1 - ηλ_α)\epsilon _{t}(α)\tag {7.10}
$$
so we see that each component exhibits linear convergence with its own coefficient of convergence $k_α = 1 - ηλ_α$. The worst component will dominate the convergence so we are interested in choosing $η$ so that the $k_α$ with largest magnitude is minimized. Since $H$ is positive definite, all the $λ_α$’s are positive, so it is easy to see that one should choose $η$ so that $1 - ηλ_{\min} = 1 - \Delta $ and $1 - ηλ_{\max} = 1+  \Delta$, or one should choose. Solving for the optimal $η​$, one finds that 
$$
η_{opt} =\frac {2}{λ_{\min} + λ_{\max}},
 k_{opt} = \frac{1-c}{1+c}\\ \tag {7.11}
$$
where $c = λ_{\min}/λ_{\max}$ is the condition number of $H$, and is an important measure of the stability of $H$. When $c ≈ 0$, one usually says that $H$ is illconditioned. Among other things, this affects the one’s ability to numerically compute the inverse of $H​$. 

先对题目中的式子简单解释下，因为$H$为半正定矩阵，所以它的特征向量可以构成正交基，将$\epsilon _{t+1} = (I - ηH)\epsilon _{t}$投影在每个特征向量上，记第$\alpha$个特征向量对应的特征值为$\lambda_{\alpha}$，那么可得
$$
\epsilon _{t+1}(α) = (1 - ηH)\epsilon _{t}(α)= (1 - ηλ_α)\epsilon _{t}(α)
$$
所以收敛速度与$|1 - ηλ_α|$有关，我们要求得最优$η_{opt}$就是解决以下问题
$$
\text{min }\text{max }|1 - ηλ_α|
$$
因为$\lambda_{\alpha}\ge 0$，所以由绝对值函数的性质可知
$$
\text{max }|1 - ηλ_α|=\text{max}\{|1-\etaλ_{\max} |,|1-\eta λ_{\min}|\}
=\begin{cases}
1-\etaλ_{\max} &(\eta \le 0)\\
1-\eta λ_{\min} &(0 < \eta<\frac {2}{λ_{\min} + λ_{\max}} )\\
\etaλ_{\max}-1 &(\eta\ge \frac {2}{λ_{\min} + λ_{\max}})
\end{cases}
$$
所以当$\eta  = \frac {2}{λ_{\min} + λ_{\max}}$时，$\text{max }|1 - ηλ_α|​$取最小值，从而
$$
\begin{aligned}
η_{opt}& =\frac {2}{λ_{\min} + λ_{\max}}\\
 k_{opt} &=1-η_{opt}λ_{\min}\\
 &=1-\frac {2λ_{\min}}{λ_{\min} + λ_{\max}}\\
 &=\frac{ λ_{\max}-λ_{\min}}{λ_{\min} + λ_{\max}}\\
 &= \frac{1-c}{1+c}\\
 &其中c= \frac{λ_{\min}}{λ_{\max}}
 \end{aligned}
$$



#### Problem 7.12 (Page 48) 

[Hard] With a variable learning rate, suppose that $η_t → 0$ satisfying $\sum_t η_t = +∞$ and $\sum_t η_t^2 < ∞$, for example one could choose $η_t = 1/(t + 1)$. Show that gradient descent will converge to a local minimum.    

接着采取上一题的思路，更新公式为
$$
\epsilon _{t+1} = (I - η_tH)\epsilon _{t}
$$
将其投影到特征向量方向可得
$$
\epsilon _{t+1}(α) = (1 - η_tλ_α)\epsilon _{t}(α)
$$
现在考虑其收敛性，将上述式子递推可得
$$
\epsilon _{t+1}(α) =\epsilon _{1}(α)\prod_{i=1}^t (1 - η_iλ_α)
$$
只要考虑$\prod_{i=1}^t (1 - η_iλ_α)$，这是个无穷乘积的问题。因为$η_t → 0$，所以当$t$充分大时，$1 - η_tλ_α>0$。因为有限项不影响无穷乘积的收敛性，所以这里可以假设$1 - η_tλ_α>0$恒成立，从而可以对上式进行变形
$$
\begin{aligned}
\prod_{i=1}^t (1 - η_iλ_α)
&=\prod_{i=1}^t e^{\text{ln}(1 - η_iλ_α)}\\
&=e^{\sum_{i=1}^t\text{ln}(1 - η_iλ_α)}\\
&\approx e^{\sum_{i=1}^t(-\eta_iλ_α+\frac 1 2 λ_α^2\eta_i^2)}\\
&=e^{-λ_α\sum_{i=1}^t \eta_i+\frac 1 2 λ_α^2 \sum_{i=1}^t \eta_i^2}
\end{aligned}
$$
因为$\sum_t η_t = +∞,\sum_t η_t^2 < ∞​$，所以
$$
e^{-λ_α\sum_{i=1}^t \eta_i}\to 0\\
e^{\frac 1 2 λ_α^2 \sum_{i=1}^t \eta_i^2}\to c
$$
从而
$$
\prod_{i=1}^t (1 - η_iλ_α)\approx e^{-λ_α\sum_{i=1}^t \eta_i+\frac 1 2 λ_α^2 \sum_{i=1}^t \eta_i^2}\to 0\\
\epsilon _{t+1}(α) =\epsilon _{1}(α)\prod_{i=1}^t (1 - η_iλ_α)\to 0
$$
即$\epsilon$在每个特征方向上的投影都会收敛到$0$，所以$\epsilon$收敛到$0$，因此梯度下降法会达到局部最小值。



#### Problem 7.13  (Page 48)

[Finite Difference Approximation to Hessian] 

(a) Consider the function $E(w_1, w_2)$. Show that the finite difference approx imation to the second order partial derivatives are given by
$$
\begin{aligned}
\frac{∂^2E} {∂w^2_1} &= \frac{E(w_1+2h,w_2)+E(w_1-2h,w_2)-2E(w_1,w_2)} {4h^2}\\
  \frac{∂^2E} {∂w^2_2}& = \frac{E(w_1,w_2+2h)+E(w_1,w_2-2h)-2E(w_1,w_2)} {4h^2}\\
 \frac{∂^2E} {∂w_2∂w_1}& = \frac{E(w_1+h,w_2+h)+E(w_1-h,w_2-h)-E(w_1+h,w_2-h)-E(w_1-h,w_2+h)} {4h^2}
 \end{aligned}
$$
(b) Give an algorithm to compute the finite difference approximation to the Hessian matrix for $E_{\text{in}}(w)​$, the in-sample error for a multilayer neural network with weights $w = [W^{(1)}, . . . , W^{(ℓ)}]​$.

(c) Compute the asymptotic running time of your algorithm in terms of the number of weights in your network and then number of data points.    

(a)利用如下公式计算偏导数
$$
对于f(x,y),\frac{\partial f}{\partial x}\approx \frac{f(x+h,y)-f(x-h,y)}{2h}
$$
二阶偏导数可以用同样的方法计算
$$


\begin{aligned}
f_1(w_1,w_2)&=\frac{∂E} {∂w_1} = \frac{E(w_1+h,w_2)-E(w_1-h,w_2)} {2h}\\
f_2(w_1,w_2)&=  \frac{∂E} {∂w_2} = \frac{E(w_1,w_2+h)-E(w_1,w_2-h)} {2h}\\
\frac{∂^2E} {∂w^2_1} &=\frac{f_1(w_1+h,w_2)+f_1(w_1-h,w_2)} {2h}\\
&=\frac{\frac{E(w_1+2h,w_2)-E(w_1,w_2)}{2h}-\frac{E(w_1,w_2)-E(w_1-2h,w_2)}{2h}}{2h}\\
&= \frac{E(w_1+2h,w_2)+E(w_1-2h,w_2)-2E(w_1,w_2)} {4h^2}\\
由对称性可得 \frac{∂^2E} {∂w^2_2} &= \frac{E(w_1,w_2+2h)+E(w_1,w_2-2h)-2E(w_1,w_2)} {4h^2}\\

 \frac{∂^2E} {∂w_2∂w_1} &=\frac{f_1(w_1,w_2+h)+f_1(w_1,w_2-h)} {2h}\\
&=\frac{\frac{E(w_1+h,w_2+h)-E(w_1-h,w_2+h)}{2h}-\frac{E(w_1+h,w_2-h)-E(w_1-h,w_2-h)}{2h}}{2h}\\
&= \frac{E(w_1+h,w_2+h)+E(w_1-h,w_2-h)-E(w_1+h,w_2-h)-E(w_1-h,w_2+h)} {4h^2}
\end{aligned}
$$

(b)(c)使用Exercise 7.6的记号
$$
V = \sum_{i=0 }^ℓd^{(i)}, Q =\sum_{ℓ =0 }^ℓ d^{(i)}(d^{(i-1)} + 1)
$$
$V$为节点数量，$Q$为权重的数量，记$Q_i=d^{(i)}(d^{(i-1)} + 1)$，即第$i$层有$Q_i$个权重，那么$Q$可以改写为
$$
Q =\sum_{i =0 }^ℓ Q_i
$$
首先看计算第$i$层的Hessian矩阵需要计算哪些。

首先是对角线上的元素$\frac{∂^2E} {∂{w^{(i)}_{jk}}^2}$,需要计算$E(w^{(i)}_{jk}\pm h)$，同Problem 7.6的讨论可知一共需要的计算量为
$$
2Q_i(\sum_{j =i }^ℓ Q_j)
$$
接着需要计算非对角线元素$\frac{∂^2E} {∂{w^{(i)}_{j_1k_1}}∂{w^{(i)}_{j_2k_2}}}$，，需要计算$E(w^{(i)}_{j_1k_1}\pm h, w^{(i)}_{j_2k_2}\pm h)$一共需要的计算量为
$$
2Q_i(Q_i-1)(\sum_{j =i }^ℓ Q_j)
$$
从而计算第$i​$层的Hessian矩阵需要的次数为
$$
2Q_i(\sum_{j =i }^ℓ Q_j)+2Q_i(Q_i-1)(\sum_{j =i }^ℓ Q_j)=2Q_i^2\sum_{j=i}^ℓ Q_j
$$
计算全部的Hessian矩阵需要的次数为
$$
\sum_{i=0}^ ℓ2Q_i^2\sum_{j=i}^ℓ Q_j \le Q \sum_{i=0}^ ℓ2Q_i^2\le 2 Q^3
$$
所以计算次数为
$$
O( Q^3)
$$



####  Problem 7.14 (Page 48)

Suppose we take a fixed step in some direction, we ask what the optimal direction for this fixed step assuming that the quadratic model for the error surface is accurate:
$$
E_{\text{in}}(w_t + δw) = E_{\text{in}}(w_t) + g_t^T\Delta w + \frac 12\Delta w ^TH_t\Delta w
$$
So we want to minimize $E_{\text{in}}(\Delta w)$ with respect to $\Delta w$ subject to the constraint that the step size is $η$, i.e., that $\Delta w^T\Delta w = η^2$.    

(a) Show that the Lagrangian for this constrained minimization problem is: 
$$
\mathcal L = E_{\text{in}}(w_t) + g_t^T\Delta w + \frac 12\Delta w ^T(H_t+2αI)\Delta w-\alpha \eta ^2  \tag {7.12}
$$
where $α$ is the Lagrange multiplier. 

(b) Solve for $\Delta w$ and $α$ and show that they satisfy the two equations: 
$$
\begin{aligned}
\Delta w &= -(H_t + 2αI)^{-1}g_t, \\
\Delta w^T\Delta w& = η^2
\end{aligned}
$$
(c) Show that α satisfies the implicit equation: 
$$
α = - \frac {1} {2η^2} (\Delta w^Tg_t +\Delta w^TH_t \Delta w).
$$
Argue that the second term is $θ(1)$ and the first is $O(∼ ||g_t||/η)$. So, $α$ is large for a small step size $η$.

(d) Assume that $α$ is large. Show that, To leading order in $\frac 1 η$ 
$$
α = \frac {||g_t||} {2η}
$$
Therefore $α$ is large, consistent with expanding $\Delta w$ to leading order in $\frac 1 α$ . [Hint: expand $\Delta w$ to leading order in $\frac 1 \alpha$ ] 

(e) Using (d), show that $\Delta w = - (H_t +  \frac {||g_t||} {η}I)^{-1} g_t$

这里对(a)的题目进行了修改，感觉原问题可能不对。

(a)利用拉格朗日乘子法，可以把上述条件极值问题转化为无条件极值问题
$$
\begin{aligned}
\mathcal L &=E_{\text{in}}(w_t) + g_t^T\Delta w + \frac 12\Delta w ^TH_t\Delta w + α(\Delta w^T\Delta w- \eta ^2)\\
&= E_{\text{in}}(w_t) + g_t^T\Delta w + \frac 12\Delta w ^T(H_t+2αI)\Delta w-\alpha \eta ^2
\end{aligned}
$$
(b)关于$\Delta w ​$求梯度
$$
\begin{aligned}
\nabla \mathcal L&= g_t+(H_t+2αI)\Delta w=0\\
\Delta w &= -(H_t + 2αI)^{-1}g_t
\end{aligned}
$$
关于$\Delta \alpha ​$求梯度
$$
\begin{aligned}
\nabla \mathcal L&= \Delta w^T\Delta w- \eta ^2=0\\
\Delta w^T\Delta w &= η^2
\end{aligned}
$$
(c)对$\Delta w = -(H_t + 2αI)^{-1}g_t​$两边左乘$H_t + 2αI​$
$$
\begin{aligned}
(H_t + 2αI) \Delta w &=-g_t\\
 \Delta w^T(H_t + 2αI) \Delta w&=- \Delta w^Tg_t\\
  \Delta w^TH_t\Delta w+2α η^2&=- \Delta w^Tg_t\\
  α &= - \frac {1} {2η^2} (\Delta w^Tg_t +\Delta w^TH_t \Delta w)
  \end{aligned}
$$
(d)考虑第一项$- \frac {1} {2η^2} \Delta w^Tg_t ​$，求其模长
$$
||- \frac {1} {2η^2} \Delta w^Tg_t ||\le  \frac 1 {2η^2}||\Delta w||||g_t ||=\frac{||g_t ||}{2η}
$$
所以第一项为$O(∼ ||g_t||/η)$

接着考虑第二项$ - \frac {1} {2η^2} \Delta w^TH_t \Delta w$，求其模长，记$H_t$中元素绝对值的最大值为$k_1$，最小值为$k_2$
$$
 ||- \frac {1} {2η^2} \Delta w^TH_t \Delta w||\le  \frac {1} {2η^2} k_1 || \Delta w||^2=\frac {k_1}{2}\\
  ||- \frac {1} {2η^2} \Delta w^TH_t \Delta w||\ge   \frac {1} {2η^2} k_2 || \Delta w||^2=\frac {k_2}{2}
$$
所以第二项为$θ(1)$

(e)对$\Delta w= -(H_t + 2αI)^{-1}g_t$进行变形
$$
\begin{aligned}
\Delta w& = -(H_t + 2αI)^{-1}g_t\\
&=-\frac{1}\alpha (\frac 1 \alpha H_t + 2I)^{-1}g_t
\end{aligned}
$$
因为$\alpha$很大，所以$\frac 1 \alpha H_t$可以忽略，从而
$$
\Delta w \approx -\frac{1}\alpha (2I)^{-1}g_t= -\frac{g_t}{2\alpha}
$$
带入$\Delta w^T\Delta w = η^2$可得
$$
\begin{aligned}
\frac{||g_t||^2}{4\alpha^2}& \approx η^2\\
α &\approx  \frac {||g_t||} {2η}
\end{aligned}
$$
将$α \approx  \frac {||g_t||} {2η}​$代入$\Delta w \approx  -(H_t + 2αI)^{-1}g_t​$可得
$$
\Delta w \approx  - (H_t +  \frac {||g_t||} {η}I)^{-1} g_t
$$



#### Problem 7.15 (Page 49)

The outer-product Hessian approximation is $H = \sum ^N _{n=1} g_ng_n^T$ . Let $H_k = \sum ^k _{n=1} g_ng_n^T$ be the partial sum to $k$, and let $H_ k^{-1}$ be its inverse. 

(a) Show that $H_{k+1}^{-1} = H_k^{-1} - \frac{H_k^{-1}g_{k+1}g_{k+1}^TH_k^{-1}}{1+g_{k+1}^TH_k^{-1}g_{k+1}}g_{k+1}$. [Hints: $H_{k+1} = H_k +g_{k+1}g_{k+1}^T$; and, $(A+zz^T)^{-1} = A^{-1} - \frac{A^{-1}zz^TA^{-1}}{1+z^TA^{-1}z}$ .] 

(b) Use part (a) to give an $O(NW^2)$ algorithm to compute $H_t^{-1}$, the same time it takes to compute $H$. ($W$ is the number of dimensions in $g$). 

Note: typically, this algorithm is initialized with $H_0 = \epsilon I$ for some small $\epsilon$. So the algorithm actually computes $(H + \epsilon I)^{-1}$; the results are not very sensitive to the choice of $\epsilon$, as long as $\epsilon$ is small.   

(a)利用$(A+zz^T)^{-1} = A^{-1} - \frac{A^{-1}zz^TA^{-1}}{1+z^TA^{-1}z}$以及$H_{k+1} = H_k +g_{k+1}g_{k+1}^T$可得
$$
H_{k+1}^{-1} = H_k^{-1} - \frac{H_k^{-1}g_{k+1}g_{k+1}^TH_k^{-1}}{1+g_{k+1}^TH_k^{-1}g_{k+1}}
$$
(b)题目的意思应该是设计一个算法可以在$O(NW^2)​$时间内可以计算全部的$H_t^{-1}​$，来分析下上述公式，假设$H_k^{-1}​$已知，首先看分子
$$
H_k^{-1}g_{k+1}g_{k+1}^TH_k^{-1}=(g_{k+1}^TH_k^{-1})^T(g_{k+1}^TH_k^{-1})
$$
计算$g_{k+1}^TH_k^{-1}$需要$O(W^2)$的时间复杂度，所以计算$H_k^{-1}g_{k+1}g_{k+1}^TH_k^{-1}=(g_{k+1}^TH_k^{-1})^T(g_{k+1}^TH_k^{-1})$需要$O(W^2)+O(W)=O(W^2)$。接着看分母
$$
1+g_{k+1}^TH_k^{-1}g_{k+1}
$$
计算该分母需要$O(W^2)$的时间复杂度，其余就是向量加减法，需要的复杂度为$O(W)$，所以从$H_k^{-1}$计算$H_{k+1}^{-1}$需要的时间复杂度为$O(W^2)$。注意$H_0^{-1}=\frac 1 \epsilon I$，是已知的量，所以可以用上述方法递推地求出$H_k^{-1}$，每一项的时间复杂度为$O(W^2)$，因为一共有$N$项，所以一共的时间复杂度为
$$
O(NW^2)
$$



#### Problem 7.16 (Page 50)

In the text, we computed an upper bound on the $VC​$ dimension of the $2​$-layer perceptron is $d_{\text{vc}} = O(md \text {log}(md))​$ where $m​$ is the number of hidden units in the hidden layer. Prove that this bound is essentially tight by showing that $d_{\text{vc}} = Ω(md)​$. To do this, show that it is possible to find $md​$ points that can be shattered when $m​$ is even as follows. Consider any set of $N​$ points $x_1, . . . , x_N​$ in general position with $N = md​$. $N​$ points in $d​$ dimensions are in general position if no subset of $d + 1​$ points lies on a $d - 1​$ dimensional hyperplane. Now, consider any dichotomy on these points with $r​$ of the points classified $+1​$. Without loss of generality, relabel the points so that $x_1, . . . , x_r​$ are $+1​$.    

(a) Show that without loss of generality, you can assume that $r ≤ N/2​$. For the rest of the problem you may therefore assume that $r ≤ N/2​$. 

(b) Partition these $r$ positive points into groups of size $d$. The last group may have fewer than $d$ points. Show that the number of groups is at most $\frac N 2$ . Label these groups $\mathcal D_i$ for $i = 1 . . . q ≤ N/2$. 

(c) Show that for any subset of $k$ points with $k ≤ d$, there is a hyperplane containing those points and no others.    

(d) By the previous part, let $w_i, b_i$ be the hyperplane through the points in group $\mathcal D_i$, and containing no others. So 
$$
w^T_i x_n + b_i = 0
$$
if and only if $x_n ∈ \mathcal D_i​$. Show that it is possible to find $h​$ small enough so that for $x_n ∈ \mathcal D_i​$, 
$$
|w^T_i x_n + b_i| <h
$$
and for $x_n \not ∈  \mathcal D_i$
$$
|w^T_i x_n + b_i| >h
$$
(e) Show that for $x_n ∈ \mathcal D_i$, 
$$
\text{sign}(w_i^Tx_n + b_i + h) + \text{sign}(-w_i^Tx_n - b_i + h)= 2
$$
$x_n \not ∈  \mathcal D_i$
$$
\text{sign}(w_i^Tx_n + b_i + h) + \text{sign}(-w_i^Tx_n - b_i + h)= 0
$$
(f) Use the results so far to construct a $2$-layer MLP with $2r$ hidden units which implements the dichotomy (which was arbitrary). Complete the argument to show that $d_{\text{vc}}  ≥ md$.    

(a)因为一共有$N$个点，每个点不是$+1$就是$-1$，所以至少有一类的数量$\le \frac N2$，又由$+1,-1$的对称性，所以不妨假设$+1$类的数量$\le \frac N2 $，即
$$
r \le \frac N2
$$
(b)组的数量为$\frac  r d$，注意$d\ge 1,r \le \frac N2$，从而
$$
\frac  r d \le \frac N 2
$$
(c)一个$d$维点所在的平面需要$d+1$个参数来确定，如下
$$
\begin{aligned}
w^T x + b &= 0\\
\sum_{i=1}^d w_ix_i+b&=0
\end{aligned}
$$
参数为$(b,w_1,...,w_d)​$，要确定这个$d+1​$个参数，至少需要$d+1​$个点，所以对于$k\le d​$个点，必然存在无数个超平面过这$k​$个点。接下来就要找到经过这$k​$点，但是不经过其他点的超平面，这就要使用题目中的条件：任意$d+1​$个点都不在一个$d-1​$维的超平面上，分两种情况讨论：

- $k=d$，这种情形直接利用题目中的条件即可（$d+1$个点不共面）。

- $k<d$，假设这$k$个点为$x_1,...,x_k$，补充$d-k$个点$x_{k+1},...,x_{d}$，使得
  $$
  \begin{aligned}
  
  w^T x_i + b &= 0(i=1,...,k)\\
  w^T x_i + b &= 1(i=k+1,...,d)
  \end{aligned}
  $$
  从而$w^Tx+b=0​$是一个特殊的平面，过$x_1,...,x_k​$，但是不过$x_{k+1},...,x_{d}​$，由题目的假设可知，任意其他的点都不在这个平面上，这说明我们构造了一个只经过这$k​$个点的平面。

结合上述两点可知，存在只经过这$k$个点的平面。

(d)将(c)的结论用式子写出来
$$
存在w_i,b_i，使得\\
对于每个属于\mathcal D_i的x_n，w^T_i x_n + b_i = 0\\
对于每个不属于\mathcal D_i的x_n，w^T_i x_n + b_i \neq 0
$$
对于$x_n \not ∈  \mathcal D_i$，记$h_1=\text{min} |w^T_i x_n + b_i|,h=\frac {h_1} 2 >0$，所以
$$
每个不属于\mathcal D_i的x_n，|w^T_i x_n + b_i| \ge h_1 >\frac {h_1} 2 = h>0\\
每个属于\mathcal D_i的x_n，|w^T_i x_n + b_i| =0 <h
$$
从而可以找到$h$满足条件。

(e)对于$x_n ∈ \mathcal D_i$
$$
|w^T_i x_n + b_i| <h \\
w^T_i x_n + b_i <h,-w^T_i x_n - b_i <h\\
-w_i^Tx_n - b_i + h>0,w_i^Tx_n + b_i + h>0\\
\text{sign}(w_i^Tx_n + b_i + h) + \text{sign}(-w_i^Tx_n - b_i + h)= 2
$$
对于$x_n \not ∈  \mathcal D_i​$
$$
|w^T_i x_n + b_i| >h \\
w^T_i x_n + b_i <-h或w^T_i x_n + b_i >h\\
w_i^Tx_n + b_i + h<0或-w_i^Tx_n - b_i + h<0
$$
如果
$$
w_i^Tx_n + b_i + h<0
$$
那么
$$
-w_i^Tx_n - b_i + h>2h>0
$$
所以
$$
\text{sign}(w_i^Tx_n + b_i + h) + \text{sign}(-w_i^Tx_n - b_i + h)= 0
$$
如果
$$
-w_i^Tx_n - b_i + h<0
$$
那么
$$
w_i^Tx_n +b_i +h>2h>0
$$
所以
$$
\text{sign}(w_i^Tx_n + b_i + h) + \text{sign}(-w_i^Tx_n - b_i + h)= 0
$$
从而结论成立。

(f)现在构造一个$2$层的神经网络，假设输入为$x$，第一层为
$$
\text{sign}(w_i^Tx_n + b_i + h) (i=1,...,r)\\
\text{sign}(-w_i^Tx_n - b_i + h)(i=1,...,r)\\
其中(w_i,b_i)为之前讨论的每一组点对应的权重。
$$
这样第一层（隐藏层）有$2r$个神经元，记为$y_i(i=1,...,2r)$，第二层的神经元用如下方法构造
$$
\text{sign}(\sum_{i=1}^{2r}y_i -1)
$$
我们来分析下这个式子，如果$x_n ∈ \mathcal D_k​$，那么
$$
\text{sign}(w_i^Tx_n + b_i + h) + \text{sign}(-w_i^Tx_n - b_i + h)
=\begin{cases}
2, &i=k\\
0, & i\neq k
\end{cases}\\
\sum_{i=1}^{2r}y_i -1 =2-1=1\\
\text{sign}(\sum_{i=1}^{2r}y_i -1)=1
$$
如果$x_n \not ∈ \mathcal D_k$，那么
$$
\text{sign}(w_i^Tx_n + b_i + h) + \text{sign}(-w_i^Tx_n - b_i + h)=0\\
\sum_{i=1}^{2r}y_i -1 =0-1=-1\\
\text{sign}(\sum_{i=1}^{2r}y_i -1)=-1
$$
所以可以组合出$x_1,...,x_N$的任何dichotomy，从而
$$
d_{\text{vc}}\ge N=md
$$

（备注，本题乍一看这里似乎没有使用$N=md$的条件，实际上(b)(c)(d)都利用到了该条件）