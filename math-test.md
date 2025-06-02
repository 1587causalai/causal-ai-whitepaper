# 数学公式测试

## 行内公式测试

这是一个简单的行内公式：$E = mc^2$。

概率论中的贝叶斯定理：$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$。

## 块级公式测试

### 基本数学公式

$$
f(x) = \int_{-\infty}^{\infty} \hat{f}(\xi) e^{2\pi i \xi x} d\xi
$$

### 矩阵表示

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
=
\begin{bmatrix}
ax + by \\
cx + dy
\end{bmatrix}
$$

### 因果推断相关公式

因果图中的后门准则：

$$
P(Y | do(X = x)) = \sum_{z} P(Y | X = x, Z = z) P(Z = z)
$$

干预分布与观察分布的关系：

$$
P(y | do(x)) = \int P(y | x, z) P(z) dz
$$

### 机器学习损失函数

交叉熵损失：

$$
L = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(p_{ic})
$$

其中 $y_{ic}$ 是真实标签，$p_{ic}$ 是预测概率。

### 复杂数学表达式

$$
\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}
$$

$$
\oint_{\partial \Omega} \vec{F} \cdot d\vec{A} = \int_{\Omega} (\nabla \cdot \vec{F}) dV
$$

### 分段函数

$$
f(x) = \begin{cases}
x^2 & \text{if } x \geq 0 \\
-x^2 & \text{if } x < 0
\end{cases}
$$

### 求和与积分

$$
\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}
$$

$$
\int_0^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

## 复杂公式组合

### 神经网络反向传播

前向传播：
$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = \sigma(z^{(l)})
$$

反向传播中的梯度：
$$
\frac{\partial C}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
$$

$$
\frac{\partial C}{\partial b^{(l)}} = \delta^{(l)}
$$

### 变分自编码器（VAE）

VAE 的损失函数：
$$
\mathcal{L}(\theta, \phi; x) = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + D_{KL}(q_\phi(z|x) || p(z))
$$

其中 KL 散度为：
$$
D_{KL}(q_\phi(z|x) || p(z)) = \int q_\phi(z|x) \log \frac{q_\phi(z|x)}{p(z)} dz
$$

## 希腊字母测试

常用希腊字母：$\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \eta, \theta, \iota, \kappa, \lambda, \mu, \nu, \xi, \pi, \rho, \sigma, \tau, \phi, \chi, \psi, \omega$

大写希腊字母：$\Gamma, \Delta, \Theta, \Lambda, \Xi, \Pi, \Sigma, \Phi, \Psi, \Omega$

## 特殊符号测试

集合论：$\in, \notin, \subset, \subseteq, \cup, \cap, \emptyset, \mathbb{R}, \mathbb{N}, \mathbb{Z}, \mathbb{Q}, \mathbb{C}$

逻辑符号：$\forall, \exists, \neg, \land, \lor, \implies, \iff$

关系符号：$\leq, \geq, \neq, \approx, \equiv, \sim, \propto$

## 测试结果

如果你能看到上面所有的数学公式都正确渲染，那么数学公式支持就配置成功了！ 