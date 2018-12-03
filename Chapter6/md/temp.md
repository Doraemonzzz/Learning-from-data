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
