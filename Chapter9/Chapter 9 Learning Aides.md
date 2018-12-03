#### Exercise 9.1 (Page 1)

The Bank of Learning (BoL) gave Mr. Good and Mr. Bad credit cards based on their (Age, Income) input vector.

|                                          | Mr. Good | Mr. Bad |
| ---------------------------------------- | -------- | ------- |
| (Age in years, Income in thousands of $) | (47,35)  | (22,40) |

Mr. Good paid off his credit card bill, but Mr. Bad defaulted. Mr. Unknown who has ‘coordinates’ (21yrs,$36K) applies for credit. Should the BoL give him credit, according to the nearest neighbor algorithm? If income is measured in dollars instead of in “K” (thousands of dollars), what is your answer? 

分别计算距离
$$
d_1 = \sqrt{(47 -21)^2 + (36-35)^2} = \sqrt{677} \\
d_2 = \sqrt{(22 -21)^2 + (40-35)^2} =\sqrt{26}\\
d_1 > d_2
$$
根据KNN算法可得，应该不批准。

如果收入的单位换为K，那么
$$
d_1 = \sqrt{(47 -21)^2 + (36-35)^2 \times 1000^2} = \sqrt{1000676} \\
d_2 = \sqrt{(22 -21)^2 + (40-35)^2 \times 1000^2} =\sqrt{25000001} \\
d_2 > d_1
$$
根据KNN算法可得，应该不批准。

这个例子告诉我们，使用KNN之前应该把数据归一化，这也是本章介绍的内容。



#### Exercise 9.2 (Page 3)

Define the matrix $γ = \mathbb I - \frac 1 N 11^T$. Show that $Z = γX$. ($γ$ is called the centering operator (see Appendix B.3) which projects onto the space orthogonal to $1$.)
$$
Z = X − 1\overline x^T, \overline x =\frac 1 N X^T 1\\
Z = (\mathbb I - \frac 1 N 1 1^T) X = \gamma X
$$


#### Exercise 9.3 (Page 3)

Consider the data matrix $X$ and the transformed data matrix $Z$. Show that $Z = XD$ and $Z^TZ = DX^TXD$.  
根据课本我们有
$$
z_n = D x_n,D=\text{diag}\{\frac 1 {\sigma_1},..., \frac 1 {\sigma_d}\}
$$

所以
$$
z_n^T = x_n ^TD^T =x_n^T D
$$
注意
$$
Z =  \left[
 \begin{matrix}
z_1^T \\
  ... \\
z_n^T \\
  \end{matrix}
  \right] ,
  X= \left[
 \begin{matrix}
x_1^T \\
  ... \\
x_n^T \\
  \end{matrix}
  \right]
$$
所以
$$
Z=\left[
 \begin{matrix}
x_1^TD \\
  ... \\
x_n^TD \\
  \end{matrix}
  \right] =XD
$$
因此
$$
Z^TZ=D^TX^T XD = DX^T XD
$$



#### Exercise 9.4 (Page 4)

Let $\hat x_1$ and $\hat x_2$ be independent with zero mean and unit variance. You measure inputs $x_1 = \hat x_1$ and $x_2 =\sqrt{1- \epsilon^2 }\hat x_1 + \epsilon \hat x_2$.
(a) What are variance($x_1$), variance($x_2$) and covariance($x_1, x_2$)?
(b) Suppose $f(\hat x) = \hat w_1 \hat x_1 + \hat w_2 \hat x_2$ (linear in the independent variables).Show that $f$ is linear in the correlated inputs, $f(x) = w_1x_1 + w_2x_2$.(Obtain $w_1, w_2$ as functions of $\hat w_1, \hat w_2$.)
(c) Consider the ‘simple’ target function $f(\hat x) = \hat x_1 + \hat x_2$. If you perform regression with the correlated inputs $x$ and regularization constraint $w_1^2 +w_2^2 ≤ C$, what is the maximum amount of regularization you can use (minimum value of $C$) and still be able to implement the target?
(d) What happens to the minimum $C$ as the correlation increases ($\epsilon → 0$).
(e) Assuming that there is significant noise in the data, discuss your results in the context of bias and var. 

(a)
$$
\text{Var}(x_1) = 1,\mathbb E (x_1) = 0\\
\text{Var}(x_2) = \text{Var}(\sqrt{1- \epsilon^2 }\hat x_1 + \epsilon \hat x_2)
= 1- \epsilon^2 + \epsilon^2 = 1
,\mathbb E (x_2) = 0 \\
\begin{aligned}
\text{Cov}(x_1,x_2) 
&= \mathbb E(x_1x_2) - \mathbb E (x_1)\mathbb E (x_2) \\
&=\mathbb E[(\sqrt{1- \epsilon^2 }\hat x_1 + \epsilon \hat x_2)\hat x_1] 
-\mathbb E(\sqrt{1- \epsilon^2 }\hat x_1 + \epsilon \hat x_2) \mathbb E(x_1)\\
&= \sqrt {1- \epsilon^2} \mathbb E(\hat x_1^2)+ \epsilon \mathbb E (\hat x_1 \hat x_2)\\
&=\sqrt {1- \epsilon^2}
\end{aligned}
$$
(b)解出$x_1, x_2$
$$
\hat x_1 =  x_1 \\
\hat x_2 = \frac{x_2 -\sqrt{1- \epsilon^2 }\hat x_1}{\epsilon} =\frac{x_2 -\sqrt{1- \epsilon^2 } x_1}{\epsilon}
$$
带入得
$$
\begin{aligned}
f(\hat x)& = \hat w_1 \hat x_1 + \hat w_2 \hat x_2 \\
&=  \hat w_1 x_1 + \hat w_2 \frac{x_2 -\sqrt{1- \epsilon^2 } x_1}{\epsilon} \\
&= (\hat w_1 -\frac{\sqrt{1- \epsilon^2 }}{\epsilon}\hat w_2 )x_1 +
\frac{\hat w_2}{\epsilon} x_2
\end{aligned}
$$
因此
$$
w_1 = \hat w_1 -\frac{\sqrt{1- \epsilon^2 }}{\epsilon}\hat w_2 \\
w_2 = \frac{\hat w_2}{\epsilon}
$$
(c)

将$\hat w_1 = 1,\hat w_2$带入上式可得
$$
w_1= 1 - \frac{\sqrt{1- \epsilon^2 }}{\epsilon} \\
w_2 = \frac{1}{\epsilon}
$$
因此
$$
\begin{aligned}
w_1^2 + w_2 ^2 
&= ( 1 - \frac{\sqrt{1- \epsilon^2 }}{\epsilon} )^2 + \frac{1}{\epsilon^2} \\
&=1 +  \frac{{1- \epsilon^2 }}{\epsilon^2} -\frac{2\sqrt{1- \epsilon^2 }}{\epsilon} +\frac{1}{\epsilon^2} \\
&=\frac{2}{\epsilon^2} -\frac{2\sqrt{1- \epsilon^2 }}{\epsilon}
\end{aligned}
$$
令$\epsilon = \cos {\theta},\theta \in [0, \frac \pi 2]$，带入可得
$$
\begin{aligned}
w_1^2 + w_2 ^2 
&= 2\sec^2 \theta  - 2 \tan \theta \\
&= 2(1 +\tan^2 \theta)  -2 \tan \theta\\
&=2(\tan \theta -\frac 1 2 )^2  +\frac 3 2\\
&\ge  \frac 3 2
\end{aligned}
$$
所以$C$的最小值为$\frac 3 2$

(d)当$\epsilon \to 0$时，因为$\epsilon = \cos {\theta}$，所以$\theta \to \frac \pi 2 $，从而$\tan \theta \to \infty $，因此
$$
w_1^2 + w_2 ^2 =2(\tan \theta -\frac 1 2 )^2  +\frac 3 2 \to \infty
$$
这说明当$\epsilon \to 0$时，$C$的最小值会趋于无穷，从而正则化没有效果

(e)如果数据中有噪音，那么偏差方差都会变大。




#### Exercise 9.5 (Page 6)
Consider a data set with two examples,
$$
(x^T_1 = [-1, a_1, . . . , a_d], y_1 = +1); (x^T_2 = [1, b_1, . . . , b_d], y_2 = -1),
$$
where $a_i, b_i$ are independent random $±1$ variables. Let $x^T_{\text{test}} =[-1, -1, . . . , -1]$. Assume that only the first component of $x$ is relevant to $f$. However, the actual measured $x$ has additional random components in the additional $d$ dimensions. If the nearest neighbor rule is used, show, either mathematically or with an experiment, that the probability of classifying $x_{\text{test}}$ correctly is $\frac 1 2 +O( \frac{1}{\sqrt d} )$ ($d$ is the number of irrelevant dimensions).
What happens if there is a third data point ($x^T_3 = [1, c_1, . . . , c_d], y_3 = -1$)?

由nearest neighbor方法可知，分类正确当且仅当
$$
d(x_1, x_{\text {test}}) < d(x_2,  x_{\text {test}}) \\
\sum_{i=1}^d(a_i +1)^2 < 4+  \sum_{i=1}^d(b_i +1)^2
$$
因为$a_i, b_i \in \{1,-1\}$，所以$(a_i+1)^2, (b_i+1)^2 \in \{0,4\}$，所以上式成立当且仅当$b_i$中取$1$的个数大于等于$a_i$中取$1$的个数，因为取$1,-1$的概论相同，所以该事件发生的概率为：
$$
\sum_{j=0}^d\sum_{i=0}^j C_{d}^i \frac 1 {2^d} C_d^j \frac 1 {2^d} 
= \frac{1}{4^d}\sum_{j=0}^d\sum_{i=0}^j C_{d}^i  C_d^j
$$
考虑如下$(d+1)\times (d+1)$矩阵，第$i,j$个元素为$C_{d}^{i-1}  C_d^{j-1}$
$$
\left[
 \begin{matrix}
   C_{d}^0  C_d^0 &  C_{d}^0  C_d^1 & ...& C_{d}^0  C_d^d \\
   ... & ... & ...& ... \\
      C_{d}^d  C_d^0 &  C_{d}^d  C_d^1 & ...& C_{d}^d  C_d^d
  \end{matrix}
  \right]
$$
注意该矩阵所有元素的和为
$$
\sum_{i=1}^{d+1}\sum_{j=1}^{d+1}C_{d}^{i-1}  C_d^{j-1} 
= \sum_{i=1}^{d+1}C_{d}^{i-1}  2^d = 2^{2d}
$$
我们要计算的是对角线及其上方元素的和，由对称性可知，除去对角线下方以及对角线上方的元素和相等，而对角线上元素和为
$$
\sum_{i=1}^{d+1}C_{d}^{i-1}  C_d^{i-1}  
= \sum_{i=1}^{d+1}C_{d}^{i-1}  C_d^{d-i+1} 
=C_{2d}^d
$$
该公式只要下式两边$x^d$系数即可
$$
(1+x)^d (1+x)^d  = (1+x)^{2d}
$$
从而对角线上方元素的和为
$$
\frac{2^{2d} - C_{2d}^d}{2}
$$
从而$\sum_{j=0}^d\sum_{i=0}^j C_{d}^i  C_d^j$为
$$
\sum_{j=0}^d\sum_{i=0}^j C_{d}^i  C_d^j 
=\frac{2^{2d} - C_{2d}^d}{2} +   C_{2d}^d = \frac {4^d} 2 +\frac{C_{2d}^d}{2}
$$
概率为
$$
P= \frac 1 {4^d}(\frac {4^d} 2 +\frac{C_{2d}^d}{2}) =\frac 1 2 + \frac{{C_{2d}^d}}{2\times4^d }
$$
由斯特林公式$n!\approx \sqrt{2\pi n}(\frac{n}{e})^n$
$$
{C_{2d}^d} =\frac{(2d)!}{d! d!} \approx\frac{\sqrt{2\pi \times2d} \times(\frac{2d}{e})^{2d}}
{2\pi d \times(\frac{d}{e})^{2d}}= \frac{4^d  }{\sqrt{\pi d}}
$$

从而
$$
P\approx  \frac{1} 2  + \frac{1}{2\sqrt{\pi d}} = \frac 1 2 +O( \frac{1}{\sqrt d} )
$$
如果还有$x_3$，则分类正确当且仅当
$$
\sum_{i=1} ^d (a_i+1)^2 < 8 +\sum_{i=1} ^d (b_i+1)^2 +\sum_{i=1} ^d (a_i+1)^2
$$
假设$a_i$中有$l$个$1$，$b_i$中有$m$个$1$，$c_i$中有$k$个$1$，上式成立当且仅当
$$
m+k +1 \ge  l
$$
同之前的分析可知，概率为
$$
\begin{aligned}
P &= \frac{1}{2^{3d}} \sum_{l=0}^{\min \{m+k+1,d\}}\sum_{m=0}^d\sum_{k=0}^{d}
C_d^lC_d^mC_d^k \\
&= \frac{1}{2^{3d}}  
\end{aligned}
$$
这个式子比较难计算



#### Exercise 9.6 (Page 8)

Try to build some intuition for what the rotation is doing by using the illustrations in Figure 9.1 to qualitatively answer these questions.
(a) If there is a large offset (or bias) in both measured variables, how will this affect the ‘natural axes’, the ones to which the data will be rotated? Should you perform input centering before doing PCA?

![](https://github.com/Doraemonzzz/Learning-from-data/blob/master/photo/Chapter9/Exercise%209.6.png?raw=true)

(b) If one dimension (say $x_1$) is inflated disproportionately (e.g., income is measured in dollars instead of thousands of dollars). How will this affect the ‘natural axes’, the ones to which the data should be rotated?Should you perform input normalization before doing PCA?
(c) If you do input whitening, what will the ‘natural axes’ for the inputs be? Should you perform input whitening before doing PCA? 

(a)PCA相当于旋转坐标轴，如果有较大的偏差，中心就不在原点，则旋转坐标轴之后无法使得数据落在坐标轴上，没有达到PCA的目的

(b)如果数据不成比例的膨胀，例如$x_1$非常大，那么$x_1$就会占主导成分，PCA无法萃取有效的信息，所以PCA之前要进行input whitening

(c)如果做了input whitening，那么每个变量的方差近似一样，这样PCA就无法区分哪些变量是主要变量，算法失效

注：

(b)和(c)看起来矛盾，我的理解是(b)想表达的是不要人为让数据的数量级差很多，(c)表达的意思是，如果数量级接近，我们不要使用input whitening，因为这样无法判断出哪个分量重要




#### Exercise 9.7 (Page 10)
(a) Show that $z$ is a linear transformation of $x$, $z = V^Tx$. What are the dimensions of the matrix $V$ and what are its columns?
(b) Show that the transformed data matrix is $Z = XV$.
(c) Show that $\sum_{ i=1}^d z_i^2 = \sum_{ i=1}^d x_i^2$  and hence that $||z|| ≤ ||x||$.

(a)
$$
z_i = x^T v_i = v_i^T x,V=[v_1...v_k], v_i,x \in \mathbb R^d\\
z =  V^T x,V\in \mathbb R ^{d\times k}
$$
注意$v_i $是正交向量，所以
$$
\text{rank} (V) =  k
$$
(b)
$$
Z =  \left[
 \begin{matrix}
Z_1^T \\
  ... \\
Z_n^T \\
  \end{matrix}
  \right] ,
  X= \left[
 \begin{matrix}
x_1^T \\
  ... \\
x_n^T \\
  \end{matrix}
  \right]
$$
注意
$$
Z_i = V^T x_i, Z_i^T =x_i^T V
$$
则
$$
Z =  \left[
 \begin{matrix}
Z_1^T \\
  ... \\
Z_n^T \\
  \end{matrix}
  \right] =
  \left[
 \begin{matrix}
x_1^TV \\
  ... \\
x_n^TV \\
  \end{matrix}
  \right] =
    \left[
 \begin{matrix}
x_1^T \\
  ... \\
x_n^T \\
  \end{matrix}
  \right]V= XV
$$
(c)

注意$z_i= x^Tv_i$
$$
\sum_{ i=1}^d z_i^2 = \sum_{ i=1}^d  z_iz_i^T =x^T \sum_{ i=1}^d (v_i v_i^T) x\\
\sum_{ i=1}^d x_i^2 = ||x||^2= x^T x\\
$$
因为$v_i$为正交向量，所以
$$
(v_1,...,v_d) (v_1,...,v_d) ^T = I_{d}
$$
从而
$$
\sum_{ i=1}^d z_i^2 =x^T x =\sum_{ i=1}^d x_i^2
$$
因为$k\le d$，所以
$$
||z||^2 =\sum_{ i=1}^k z_i^2 \le \sum_{ i=1}^d z_i^2 =x^T x=||x||^2 \\
||z||\le ||x||
$$






#### Exercise 9.8 (Page 11)

Show $U^TX = ΓV^T$ and $XV = UΓ$, and hence $X^Tu_i = γ_iv_i$ and $Xv_i = γ_iu_i$.
(The $i$th singular vectors and singular value $(u_i, v_i, γ_i)$ play a similar role to eigenvector-eigenvalue pairs.) 

因为$X= U\Gamma V^T$，$U,V$都为正交矩阵，左乘$U^T$，右乘$V$可得
$$
U^T X= \Gamma V^T\\
XV = U\Gamma
$$
比较第二个式子左右两边第$i$列可得
$$
Xv_i=  \gamma_i u_i
$$
对第一个式子取转置可得
$$
X^T U = V \Gamma^T
$$
比较这个式子左右两边第$i$列可得
$$
X^T u_i =  \gamma_i  v_i
$$


#### Exercise 9.9 (Page 12)
Consider an arbitrary matrix $A$, and any matrices $U$,$V$ with orthonormal columns ($U^TU = I$ and $V^TV = I$).
(a) Show that $||A||^2_F = \text{trace}(AA^T) = \text{trace}(A^TA)$.
(b) Show that $||UAV^T||^2_F = ||A||^2_F$ (assume all matrix products exist).
[Hint: Use part (a).]

(a)由定义可知
$$
||A||^2_F = \sum_{i=1}^n \sum_{j=1}^n  a_{ij}^2
$$
 由trace的性质可知$ \text{trace}(AA^T) = \text{trace}(A^TA)$，所以只计算第一项，$AA^T$第$(i,i)$个元素为：
$$
\sum_{j=1}^n a_{ij}^2
$$
所以
$$
\text{trace}(AA^T) = \sum_{i=1}^n \sum_{j=1}^n  a_{ij}^2
$$


(b)由(a)可知
$$
||UAV^T||^2_F =  \text{trace}(UAV^TVA^T U^T) = \text{trace}(UAA^T U^T)
$$
接着利用trace的性质$\text{trace}(AB) = \text{trace}(BA)$可得
$$
||UAV^T||^2_F  = \text{trace}(UAA^T U^T) = \text{trace}(U^TUAA^T ) = \text{trace}(AA^T)
$$


