#### Problem 6.27 (Page 50)

(a)如果$K=1$，此时$w_k=1$，那么似然函数为
$$
L = \prod_{i=1}^n \frac{1}{2\pi \sigma_1^2} \exp\Big(-\frac{(x_i-\mu_1)^2}{2\sigma_1^2}\Big)
=\frac{1}{(2\pi \sigma_1^2)^n}\exp\Big(-\sum_{i=1}^n\frac{(x_i-\mu_1)^2}{2\sigma_1^2}\Big)
$$
从而对数似然函数$l=\ln(L)$为
$$
l = -n\ln(2\pi) - n\ln \sigma_1^2  -\sum_{i=1}^n\frac{(x_i-\mu_1)^2}{2\sigma_1^2}
$$
求偏导可得
$$
\frac{\partial l}{\partial \sigma_1^2} = -\frac n {\sigma_1^2} +
\sum_{i=1}^n\frac{(x_i-\mu_1)^2}{2\sigma_1^4}\\
\frac{\partial l}{\partial \mu_1} = -\sum_{i=1}^n\frac{(\mu_1-x_i)}{\sigma_1^2}
$$
令偏导数等于$0$可得
$$
\hat \sigma_1^2 =  \frac 1 n \sum_{i=1}^n(x_i-\mu_1)^2 \\
\hat \mu_1  =\frac 1 n \sum_{i=1}^n  x_i
$$


(b)当$K>1$时，似然函数为
$$
L = \prod_{i=1}^n\sum_{k=1}^K \frac{w_k}{2\pi \sigma_k^2} \exp\Big(-\frac{(x_i-\mu_k)^2}{2\sigma_k^2}\Big)
$$
取$\mu_1 =x_1$，那么有如下不等式
$$
L\ge \prod_{i=1}^n \frac{w_1}{2\pi \sigma_1^2}
$$
如果$\sigma_1 \to 0$，右边$\to \infty$，这说明$L\to \infty$，所以$L$没有最大值

(c)这题不是特别确定，给出一些自己的理解。

回顾上题不难发现，只有存在$\mu_k = x_i$，且$\sigma_i^2 \to 0$两个条件同时成立时，$L$才会趋于正无穷，回顾40页的估计式
$$
\mu_j= \frac 1 {N_j}\sum_{n=1}^N \gamma_{nj} x_n \\
\sigma^2_j = \frac{1}{N_j} \sum_{n=1}^N \gamma_{nj} x_n x_n^T- \mu_j \mu_j^T
$$
只要右边的估计式不产生上述结果即可。

(d)

(i)题目有误，正确的应该是
$$
\mathbb P(x_n \in B_n) =\sum_{k=1}^K w_k 
\Big(F_{\mathcal N}\Big(\frac{x_n +\epsilon -\mu_k}{\sigma_k}\Big)-
F_{\mathcal N}\Big(\frac{x_n -\epsilon -\mu_k}{\sigma_k}\Big) \Big)
$$
我们来计算左边的概率，利用全概率公式
$$
\begin{aligned}
\mathbb P(x_n \in B_n) 
&=  \mathbb P(x_n -\epsilon \le x \le x_n +\epsilon) \\
&=  \sum_{n=1}^K\mathbb P(x_n -\epsilon \le x \le x_n +\epsilon|
x \sim \mathcal N(\mu_k, \sigma_k^2))w_k \\
&=  \sum_{n=1}^K\mathbb P(x_n -\epsilon \le x \le x_n +\epsilon|
x \sim \mathcal N(\mu_k, \sigma_k^2))w_k \\
&=  \sum_{n=1}^K\mathbb 
P(\frac{x_n -\epsilon -\mu_k}{\sigma_k} \le \frac{x-\mu_k}{\sigma_k} \le \frac{x_n +\epsilon -\mu_k}{\sigma_k}
|x \sim \mathcal N(\mu_k, \sigma_k^2))w_k \\
&=  \sum_{k=1}^K w_k
\int_{\frac{x_n -\epsilon -\mu_k}{\sigma_k}}^{\frac{x_n +\epsilon -\mu_k}{\sigma_k}} 
\frac 1 {\sqrt{2\pi}} e^{-\frac {t^2} {2}} dt\\
&=  \sum_{k=1}^K w_k 
\Big(F_{\mathcal N}\Big(\frac{x_n +\epsilon -\mu_k}{\sigma_k}\Big)-
F_{\mathcal N}\Big(\frac{x_n -\epsilon -\mu_k}{\sigma_k}\Big) \Big)
\end{aligned}
$$
(ii)注意上面倒数第二个式子，我们有
$$
\int_{\frac{x_n -\epsilon -\mu_k}{\sigma_k}}^{\frac{x_n +\epsilon -\mu_k}{\sigma_k}} 
\frac 1 {\sqrt{2\pi}} e^{-\frac {t^2} {2}} dt \le 1 
$$
从而
$$
\mathbb P(x_n \in B_n)  \le \sum_{k=1}^K w_k  =1
$$
似然函数如下
$$
L = \prod_{n=1}^ N \mathbb P(x_n \in B_n)  \le 1
$$
这说明似然函数是良定义的，如果$\epsilon \to 0$，那么
$$
F_{\mathcal N}\Big(\frac{x_n +\epsilon -\mu_k}{\sigma_k}\Big)-
F_{\mathcal N}\Big(\frac{x_n -\epsilon -\mu_k}{\sigma_k}\Big) \to 0
$$
此时$L \to 0$

(iii)这里给出启发式的算法，定义课本40页一样的参数，除了$\gamma$的更新公式以外保持不变，$\gamma$的更新公式修改为
$$
\gamma _{nj}(t+1)= \mathbb P[j|x_n] =\frac{w_k 
\Big(F_{\mathcal N}\Big(\frac{x_n +\epsilon -\mu_k}{\sigma_k}\Big)-
F_{\mathcal N}\Big(\frac{x_n -\epsilon -\mu_k}{\sigma_k}\Big) \Big)}{P(x_n)}
$$



#### Problem 6.28 (Page 51)

(a)将$\mathbb E_{P(x,y)}$记为$\mathbb E$
$$
\begin{aligned}
E_{out}(h) &=\mathbb E[(h(x)-y)^2] \\
&=\mathbb E[(h(x)-\mathbb E[y|x]+\mathbb E[y|x]-y)^2] \\
&=\mathbb E[(h(x)-\mathbb E[y|x])^2]+ \mathbb E[(\mathbb E[y|x]-y)^2]
+2\mathbb E[(h(x)-\mathbb E[y|x])(\mathbb E[y|x]-y)]\\
&=\mathbb E[(h(x)-\mathbb E[y|x])^2]+ \mathbb E[(\mathbb E[y|x]-y)^2]
+2\mathbb E[\mathbb E [(h(x)-\mathbb E[y|x])(\mathbb E[y|x]-y)]|x]\\
&=\mathbb E[(h(x)-\mathbb E[y|x])^2]+ \mathbb E[(\mathbb E[y|x]-y)^2]
+2\mathbb E[(h(x)-\mathbb E[y|x]) \mathbb E [(\mathbb E[y|x]-y)]|x]\\
&=\mathbb E[(h(x)-\mathbb E[y|x])^2]+ \mathbb E[(\mathbb E[y|x]-y)^2]
+2\mathbb E[(h(x)-\mathbb E[y|x]) \mathbb (\mathbb E[y|x] -\mathbb E[y|x])]\\
&=\mathbb E[(h(x)-\mathbb E[y|x])^2]+ \mathbb E[(\mathbb E[y|x]-y)^2]\\
&\ge \mathbb E[(h(x)-\mathbb E[y|x])^2]
\end{aligned}
$$
当且仅当$y=\mathbb E[y|x]$时等式成立。

(b)首先证明一个引理：
$$
X= \left(
 \begin{matrix}
   X_1 \\
   X_2
  \end{matrix}
  \right) 
  \sim N_n 
  \left(
  \left(
 \begin{matrix}
   \mu_1 \\
   \mu_2
  \end{matrix}
  \right) ,
  \left(
 \begin{matrix}
   \sum_{11} &   \sum_{12} \\
   \sum_{21} &\sum_{22} 
  \end{matrix}
  \right)
   \right) \\
   X_1 ,\mu _1 \in \mathbb R^k ,{\sum}_{11}为k阶方阵， 
   {\sum}_{11}，{\sum}_{21}{\sum}_{22}相应矩阵，|{\sum}_{22}| \neq 0\\
   
  那么(1)X_1 \sim  N_k (\mu_1, {\sum}_{11})\\
  (2)在X_1= x_1条件下，X_2的条件分布是\\
  N_{n-k} 
  \left(
  \mu_2+{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1),
 {\sum}_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12}
   \right)
$$
(1)证明：

令
$$
B = \left(
 \begin{matrix}
   I_k & 0 \\
   -{\sum}_{21}{\sum}_{11}^{-1} &  I_{n-k}  
  \end{matrix}
  \right) (该矩阵为分块初等矩阵)
$$
那么
$$
\begin{aligned}
B\sum B'&=\left(
 \begin{matrix}
   I_k & 0 \\
   -{\sum}_{21}{\sum}_{11}^{-1} &  I_{n-k}  
  \end{matrix}
  \right)
    \left(
 \begin{matrix}
   \sum_{11} &   \sum_{12} \\
   \sum_{21} &\sum_{22} 
  \end{matrix}
  \right)
  
  \left(
 \begin{matrix}
   I_k &  -{\sum}_{11}^{-1}{\sum}_{12} \\
   0&  I_{n-k}  
  \end{matrix}
  \right) \\
  &=
  \left(
 \begin{matrix}
   \sum_{11} &   \sum_{12} \\
   0 &\sum_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12}
  \end{matrix}
  \right)
   \left(
 \begin{matrix}
   I_k &  -{\sum}_{11}^{-1}{\sum}_{12} \\
   0&  I_{n-k}  
  \end{matrix}
  \right)\\
  &=\left(
 \begin{matrix}
   \sum_{11} &  0 \\
   0 &\sum_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12}
  \end{matrix}
  \right)
\end{aligned}\\
B\left(
 \begin{matrix}
   \mu_1 \\
   \mu_2
  \end{matrix}
  \right)  = \left(
 \begin{matrix}
   I_k & 0 \\
   -{\sum}_{21}{\sum}_{11}^{-1} &  I_{n-k}  
  \end{matrix}
  \right) \left(
 \begin{matrix}
   \mu_1 \\
   \mu_2
  \end{matrix}
  \right)
  = \left(
 \begin{matrix}
   \mu_1 \\
   \mu_2 -{\sum}_{21}{\sum}_{11}^{-1} \mu_1
  \end{matrix}
  \right)
$$
设$Y = BX$，那么
$$
Y= \left(
 \begin{matrix}
   Y_1 \\
   Y_2
  \end{matrix}
  \right) =BX =\left(
 \begin{matrix}
   I_k & 0 \\
   -{\sum}_{21}{\sum}_{11}^{-1} &  I_{n-k}  
  \end{matrix}
  \right)
  \left(
 \begin{matrix}
   X_1 \\
   X_2
  \end{matrix}
  \right)  = 
  \left(
 \begin{matrix}
   X_1 \\
   X_2-{\sum}_{21}{\sum}_{11}^{-1}X_1
  \end{matrix}
  \right) \\
   B(X-\mu)=\left(
 \begin{matrix}
   X_1-\mu_1 \\
   X_2-\mu_2-{\sum}_{21}{\sum}_{11}^{-1}(X_1 -\mu_1)
  \end{matrix}
  \right)
$$
因此
$$
Y= \left(
 \begin{matrix}
   Y_1 \\
   Y_2
  \end{matrix}
  \right) 
  \sim N_n 
  \left(
   \left(
 \begin{matrix}
   \mu_1 \\
   \mu_2 -{\sum}_{21}{\sum}_{11}^{-1} \mu_1
  \end{matrix}
  \right) ,
 \left(
 \begin{matrix}
    \sum_{11} &  0 \\
   0 &\sum_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12}
  \end{matrix}
  \right)
   \right)
$$
不难看出$Y_1,Y_2$独立，从而
$$
Y_1=X_1 \sim N_k (\mu_1, {\sum}_{11})
$$
(2)证明：按定义将$X,X_1$的分布写出来
$$
f_X(x) = \frac{1}{(2\pi)^{\frac n 2 } |\sum|^{\frac 1 2 }} 
\exp(-\frac 1 2 (x-\mu)^T{\sum}^{-1} (x-\mu)) \\
f_{X_1}(x_1) = \frac{1}{(2\pi)^{\frac k 2 } |\sum_{11}|^{\frac 1 2 }} 
\exp(-\frac 1 2 (x_1-\mu_1)^T{\sum}_{11}^{-1} (x_1-\mu_1))
$$
对$(x-\mu)'{\sum}^{-1} (x-\mu)$进行处理，利用上一题的$B$，不难看出$B$为正交矩阵，即$B^TB=BB^T =I$，所以
$$
\begin{aligned}
(x-\mu)^T{\sum}^{-1} (x-\mu) 
&= (x-\mu)^TB^T B {\sum}^{-1} B^T B (x-\mu) \\
&= (B(x-\mu))^T (B{\sum} B^T)^{-1} (B (x-\mu)) \\
&= \left(
 \begin{matrix}
   x_1-\mu_1 \\
   x_2-\mu_2-{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1)
  \end{matrix}
  \right)^T\left(
 \begin{matrix}
   \sum_{11} &  0 \\
   0 &\sum_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12}
  \end{matrix}
  \right)\left(
 \begin{matrix}
   x_1-\mu_1 \\
   x_2-\mu_2-{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1)
  \end{matrix}
  \right) \\
  &=( x_1-\mu_1 )^T \sum_{11}( x_1-\mu_1 )+
  (x_2-\mu_2-{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1))^T
 ( \sum_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12})
 (x_2-\mu_2-{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1))
\end{aligned}
$$
注意到
$$
|\sum| =\det (\left(
 \begin{matrix}
   \sum_{11} &  0 \\
   0 &\sum_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12}
  \end{matrix}
  \right)) =|\sum_{11}||\sum_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12}|
$$
所以条件概率为
$$
\begin{aligned}
f_{X_2|X_1}(x|x_1)&=
\frac{\frac{1}{(2\pi)^{\frac n 2 } |\sum|^{\frac 1 2 }} 
\exp(( x_1-\mu_1 )^T \sum_{11}( x_1-\mu_1 )+
  (x_2-\mu_2-{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1))^T
 ( \sum_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12})
 (x_2-\mu_2-{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1)))}
{\frac{1}{(2\pi)^{\frac k 2 } |\sum_{11}|^{\frac 1 2 }} 
\exp(-\frac 1 2 (x_1-\mu_1)^T{\sum}_{11}^{-1} (x_1-\mu_1))}\\
&= \frac{1}{(2\pi)^{\frac {n-k} 2 }|\sum_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12}|^{\frac 1 2 }}
\exp((x_2-\mu_2-{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1))^T
 ( {\sum}_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12})
 (x_2-\mu_2-{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1)))
\end{aligned}
$$
即
$$
f_{X_2|X_1}(x|x_1) \sim N_{n-k} 
  \left(
  \mu_2+{\sum}_{21}{\sum}_{11}^{-1}(x_1 -\mu_1),
 {\sum}_{22} -{\sum}_{21}{\sum}_{11}^{-1}{\sum}_{12}
   \right)
$$




回到原题，记$Z_k \sim N(\mu_k, S_k^{-1})$，对应的$X,Y$分别记为$X_k,Y_k$那么
$$
f_Z(z) = \sum_{k=1}^K \frac{w_k |S_k|^{\frac 1 2 }}{(2\pi)^{\frac{d+1}{2}}}
\exp(-\frac 1 2 (z-\mu_k)^T S_k (z-\mu_k))=\sum_{k=1}^K 
w_k f_{Z_k}(z)
$$
由引理的第一部分可知
$$
f_X(x)  = \sum_{k=1}^K \frac{w_k |A_k|^{\frac 1 2 }}{(2\pi)^{\frac{d}{2}}}
\exp(-\frac 1 2 (x-\alpha_k)^T A_k (x-\alpha_k))
= \sum_{k=1}^K w_k f_{X_k}(x)
$$
所以
$$
f_{Y|X}(y|x) =\frac{\sum_{k=1}^K w_k f_{X_k}(x)f_{Y_k|X_k}(y|x)}{f_X(x)}
$$
由引理的第二部分可知
$$
f_{Y_k|X_k}(y|x) \sim N(\beta_k+\frac 1 {c_k}b_k^T(x-\mu_k) , S_k^* )\\
S_k^*可由引理计算出来
$$
对上式取期望可得
$$
\mathbb E_{Y_k|X_k}[y|x] = \beta_k+\frac 1 {c_k}b_k^T(x-\mu_k)
$$
对整体取期望可得
$$
\begin{aligned}
g(x) &=\mathbb E[y|x]\\
&= \frac{\sum_{k=1}^K \frac{w_k |A_k|^{\frac 1 2 }}{(2\pi)^{\frac{d}{2}}}
\exp(-\frac 1 2 (x-\alpha_k)^T A_k (x-\alpha_k))(\beta_k+\frac 1 {c_k}b_k^T(x-\mu_k))}{\sum_{k=1}^K \frac{w_k |A_k|^{\frac 1 2 }}{(2\pi)^{\frac{d}{2}}}
\exp(-\frac 1 2 (x-\alpha_k)^T A_k (x-\alpha_k)} \\
&=\frac{\sum_{k=1}^K {w_k |A_k|^{\frac 1 2 }}
\exp(-\frac 1 2 (x-\alpha_k)^T A_k (x-\alpha_k))(\beta_k+\frac 1 {c_k}b_k^T(x-\mu_k))}{\sum_{k=1}^K {w_k |A_k|^{\frac 1 2 }}
\exp(-\frac 1 2 (x-\alpha_k)^T A_k (x-\alpha_k))}
\end{aligned}
$$


(c)此时为(b)的特殊情形
$$
K= N,w_k =\frac 1 K ,\alpha_k= x_k,\beta_k =y_k,S_k =r^2 I
$$
带入上式可得
$$
\begin{aligned}
g(x) &=\mathbb E[y|x]\\
&=\frac{\sum_{n=1}^N \exp({-\frac 1 {2r^2}||x-x_n||^2  })y_n}{\sum_{n=1}^N \exp({-\frac 1 {2r^2}||x-x_n||^2  })}
\end{aligned}
$$
