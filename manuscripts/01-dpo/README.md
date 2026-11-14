# DPO

> **既然观测到的是偏好，而最终想学的是策略，
> 能不能直接写出「这个策略预测人会怎么选」的概率？**

## 要解决的问题

**将传统的 RLHF 中“先训练奖励模型，再用PPO等强化学习算法微调策略模型”的两阶段过程，压缩为一个单一的解析损失函数，直接在偏好数据上用监督学习的方式优化语言模型**

## 我的理解

**用策略概率表示奖励差，再对偏好标签做最大似然估计**

**直接用策略拟合偏好, 而不是用奖励拟合偏好再用奖励优化策略**

**DPO 则把策略直接放进偏好模型里，对人的选择做最大似然估计**

我们手里的数据是「同一个问题下，人更喜欢哪条回答」

传统 RLHF 先用 Bradley–Terry 模型把奖励差转成偏好概率，对偏好标签做最大似然估计，学出一个奖励模型

然后固定奖励模型，用 PPO 近似求解带 KL 惩罚的奖励最大化问题，得到生成策略

DPO 利用的是这个优化问题的解析解

固定参考策略和 $`\beta`$ 后，每个奖励 $`r`$ 都有一个对应的最优策略 $`\pi_r^*`$

反过来，可以用策略表示奖励：

```math
r(x,y)=\beta\log\frac{\pi_r^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+c(x).
```

对同一个问题，$`c(x)`$ 对所有回答都相同，在计算奖励差时会消掉

因此，与策略一一对应的是**相差一个仅依赖 $`x`$ 的常数的奖励等价类**

而 BT 恰好只需要奖励差，所以可以把 BT 里的奖励差直接写成策略概率的函数

这样，DPO 仍然是用最大似然拟合**人选择哪条回答的概率**，但待训练的参数已经是**语言模型生成回答的策略**

传统 RLHF 在奖励学习阶段做最大似然，之后再用 PPO 优化策略

DPO 把策略直接放进偏好似然里，一步完成训练

## 公式

**DPO 优化的是正负例之间的差距**

> **DPO 不是简单地“提高正例概率、降低负例概率”，而是提高正例相对于负例的概率优势，并且以参考策略为基准**

偏好目标可以写成:

```math
\log \pi_\theta(y_w\mid x) - \log \pi_\theta(y_l\mid x)
```

其实是:

```math
\log \pi_\theta(y_w\mid x) - \log \pi_\theta(y_l\mid x) = \log \frac{\pi_\theta(y_w\mid x)} {\pi_\theta(y_l\mid x)}
```

先不看 $`\log`$, 可以看到里面的一项如果很大, 那就是我们想要的结果: 让生成好答案的概率相对生成坏答案的概率变大

取对数可以方便计算

## 参考模型

如果只不断拉大正负例概率比，模型可能偏离原来的语言分布太远，损害原有能力("灾难性遗忘")

所以 DPO 引入一个**冻结的参考模型**，通常是训练前的 SFT 模型

当前模型的偏好优势:

```math
m_\theta = \log\frac{\pi_\theta(y_w\mid x)} {\pi_\theta(y_l\mid x)},
```

参考模型的偏好优势:

```math
m_{\mathrm{ref}} = \log\frac{\pi_{\mathrm{ref}}(y_w\mid x)} {\pi_{\mathrm{ref}}(y_l\mid x)}.
```

DPO 的真正目标是, 相对于参考模型, 要更能提高好答案的概率

两个相对: 当前模型相对参考模型, 好答案相对坏答案

```math
\boxed{m_\theta-m_{\mathrm{ref}}}
```

**相比训练前，当前模型对正例的相对偏好增强了多少？**

## 损失函数

定义相对优势:

```math
z= \beta \left[ \log\frac{\pi_\theta(y_w\mid x)} {\pi_\theta(y_l\mid x)} - \log\frac{\pi_{\mathrm{ref}}(y_w\mid x)} {\pi_{\mathrm{ref}}(y_l\mid x)} \right],
```

损失函数为:

```math
\boxed{ \mathcal L_{\mathrm{DPO}} = -\mathbb E_{(x,y_w,y_l)} \left[\log\sigma(z)\right] }
```

其中:

```math
\sigma(z)=\frac{1}{1+e^{-z}}
```

为 sigmoid 函数

可以把 $`\sigma(z)`$ 理解为:

> 当前参数化策略对“这条偏好数据应该成立”的置信程度

- $`z`$ 越大，损失越小；
- $`z=0`$，$`\sigma(z)=0.5`$，损失为 $`\log 2`$；
- $`z\lt 0`$，说明相对参考模型，当前模型反而更偏向负例，损失较大。

在每轮训练刚开始时，通常 $`\pi_\theta=\pi_{\mathrm{ref}}`$，所以：

```math
m_\theta-m_{\mathrm{ref}}=0 \quad\Rightarrow\quad z=0.
```

随后训练不断推动 $`m_\theta`$ 超过 $`m_{\mathrm{ref}}`$。

## 正例概率不一定上升

这是 DPO 很容易误解的地方。

假设训练前：

```math
\log\pi_\theta(y_w\mid x)=-2,\qquad \log\pi_\theta(y_l\mid x)=-4.
```

此时差值为：

```math
-2-(-4)=2.
```

训练后变成：

```math
\log\pi_\theta(y_w\mid x)=-3,\qquad \log\pi_\theta(y_l\mid x)=-6.
```

此时差值为：

```math
-3-(-6)=3.
```

虽然正例概率下降了，但负例概率下降得更多，所以正例的**相对优势**仍然增加了。

因此，更准确的说法是：

> DPO 直接约束正负例之间的相对关系，不分别保证正例概率一定上升、负例概率一定下降。

不过实践中的 DPO 往往也会提高正例概率，只是这不是损失函数严格保证的性质。

## 语言模型中把轨迹的概率分解为逐token

对于回答：

```math
y=(t_1,t_2,\ldots,t_T),
```

自回归语言模型给出：

```math
\pi_\theta(y\mid x) = \prod_{k=1}^{T} \pi_\theta(t_k\mid x,t_{\lt k}).
```

取对数后：

```math
\log\pi_\theta(y\mid x) = \sum_{k=1}^{T} \log\pi_\theta(t_k\mid x,t_{\lt k}).
```

所以 DPO 的计算流程是：

1. 分别计算正例所有生成 token 的 log probability 并求和；
2. 对负例做同样的计算；
3. 用参考模型再分别计算一次；
4. 代入 $`z`$ 和 DPO loss；
5. 只对当前模型 $`\pi_\theta`$ 反向传播，参考模型保持冻结。

即使两个回答长度不同，也不需要逐 token 对齐。

## 推导

先定义我们的优化目标, 假如我们有很多轨迹, 然后我们知道一个奖励函数 $`r(x,y)`$, 应该怎么优化?

> 但是通常我们不知道真的奖励函数, 这里先假装知道

(注意这个式子用于推导, 不用于训练)

最简单的办法就是最大化奖励的期望, 毕竟我们已经知道奖励了:

```math
\max_\pi\;
\mathbb E_{x\sim\mathcal D}
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)].
```

为了防止模型崩溃或者刷分, 我们还要约束策略模型离参考模型不要太远:

```math
\boxed{
\max_\pi\;
\mathbb E_{x\sim\mathcal D}
\left[
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-
\beta D_{\mathrm{KL}}
\bigl(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\bigr)
\right]
}
```

第二项是对参考策略的 KL 惩罚

注意 KL 的方向, 是我们要去接近参考:

```math
D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}})
=
\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
```

于是得到一个约束优化问题:

```math
\max_\pi\;\mathbb E[r],
\qquad
\text{s.t. }\mathbb E_x[D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}})]
\leq\varepsilon.
```

使用**拉格朗日乘数法**来求解这个优化问题的最优解:

注意为了简便, 这里先省略输入 $`x`$, 后面会加回来

对于固定的一个输入 $`x`$, 按照上面的按轨迹拆解:

```math
J(\pi)
=
\sum_y\pi(y)r(y)
-
\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}.
```

这里需要约束 $`\pi(y)`$ 是一个合法的概率分布(而不是任意函数):

```math
\sum_y\pi(y)=1.
```

构造拉格朗日函数:

```math
\mathcal F(\pi,\lambda)
=
\sum_y\pi(y)r(y)
-
\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}
+
\lambda\left(\sum_y\pi(y)-1\right).
```

对于任意某一个 $`\pi(y)`$ 求导:

```math
\frac{\partial\mathcal F}{\partial\pi(y)}
=
r(y)
-
\beta\left(
\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}+1
\right)
+
\lambda.
```

令其为 $`0`$:

```math
\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}
=
\frac{r(y)}{\beta}+\frac{\lambda}{\beta}-1.
```

解一下此时我们的策略:

```math
\pi(y)
=
\pi_{\mathrm{ref}}(y)
\exp\left(\frac{r(y)}{\beta}\right)
\exp\left(\frac{\lambda}{\beta}-1\right).
```

接下来还要满足约束, 代入 $`\sum_y\pi(y)=1`$:

```math
\sum_y\pi(y)
=e^{\lambda/\beta-1}
\sum_y \pi_{\mathrm{ref}}(y)e^{r(y)/\beta} = 1.
```

得到:

```math
e^{\lambda/\beta-1}
=
\frac{1}{\displaystyle\sum_y\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}}.
```

此后我们可以把这个回代到拉格朗日乘数法解出的最佳策略中消去拉格朗日乘数:

为了方便做记号, 把它的分母记为:

```math
Z(x)
=
\sum_y\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
```

它叫归一化函数, 因为它的出现是**为了满足分布概率合为1**

刚刚求出的就是最优策略了, 转换一下记号加一个*, 再代入归一化函数去掉 $`\lambda`$:

```math
\boxed{
\pi_r^*(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)
}
```

这里使用下标 $`r`$, 强调它是**已知奖励 $`r`$ 下的最优策略**

### 验证它是最优解

我们接下来要验证头顶那个公式是我们的最优解

从我们的目标出发(只是换了一个形式):

```math
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y\vert{}x)} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y\vert{}x)}{\pi_{\text{ref}}(y\vert{}x)} \right]
```

转换为最小化(取负然后除以一个常数 $`\beta`$):

```math
\begin{aligned}
& \min_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y\vert{}x)} \left[ \log \frac{\pi_\theta(y\vert{}x)}{\pi_{\text{ref}}(y\vert{}x)} - \frac{1}{\beta} r(x, y) \right] \\
&= \min_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y\vert{}x)} \left[ \log \frac{\pi_\theta(y\vert{}x)}{\pi_{\text{ref}}(y\vert{}x) \exp\left(\frac{1}{\beta} r(x, y)\right)} \right] \end{aligned}
```

看分母这是啥, 这是 $`Z(x)\pi^*(x,y)`$, 也就是最优策略乘了一个归一化函数

构造一下(同除一个):

```math
\begin{aligned} &\min_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y\vert{}x)} \left[ \log \frac{\pi_\theta(y\vert{}x)}{\frac{1}{Z(x)} \pi_{\text{ref}}(y\vert{}x) \exp\left(\frac{1}{\beta} r(x, y)\right)} - \log Z(x) \right] \\
&= \min_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{D}_{\text{KL}}\left( \pi_\theta(y\vert{}x) \,\middle\Vert{}\, \frac{1}{Z(x)} \pi_{\text{ref}}(y\vert{}x) \exp\left(\frac{1}{\beta} r(x, y)\right) \right) - \log Z(x) \right] \end{aligned}
```

注意到归一化函数:

```math
Z(x)
=
\sum_y\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
```

不依赖于当前策略, 没办法优化, 并且 KL 散度大于等于 $`0`$, 在分布相等时取 $`0`$, 只能优化这个 KL 散度, 最小那肯定是取 $`0`$ 的时候, 全局最优为:

```math
\pi_r(y\vert{}x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y\vert{}x) \exp\left(\frac{1}{\beta} r(x, y)\right)
```

### 偏好模型

这里可能是令人激动的: 我们知道了这个问题具体的最优解, 接下来直接求出来不就好了?

那我们会面临一个问题, 就是这个奖励函数 $`r(x,y)`$ 是我们假装知道的

这个 $`Z(x)`$ 也不简单, 它是对所有的 $`y`$ 求和, 对于轨迹级任务, 它就是对一整个 token 树求和, 这是一个 NP 难问题, **Intractable**

我们先看看手里有什么: 我们有我们更喜欢哪条回答的数据:

```math
\mathcal D=\{(x_i,y_{w,i},y_{l,i})\}_{i=1}^N.
```

它告诉我们“两个回答中，人更喜欢哪一个”，并没有给每条回答一个绝对分数

我们希望从这里面能推导出每个选择背后的奖励分数 $`r`$ (是的这是一个假设)

---

假设同一个问题 $`x`$ 有两条回答 $`A,B`$

标注员可以选 $`A`$, 但通常不会告诉你 $`A`$ 的奖励是 $`8`$, $`B`$ 是 $`5`$, Bradley–Terry 模型给这个二选一过程建模：给每条回答一个奖励，把两个奖励做 softmax，得到选择概率:

```math
P(A\succ B\mid x)
=
\frac{e^{r(x,A)}}{e^{r(x,A)}+e^{r(x,B)}}
=
\sigma\bigl(r(x,A)-r(x,B)\bigr).
```

偏好数据就这样提供了关于**奖励差**的信息

**Bradley–Terry 是对偏好数据生成机制的建模假设**, 也就是说它是假设我们这样偏好, 背后的奖励是怎么样的

> [!NOTE]
> 到这里，其实已经有一条训练路线：**先用 BT 拟合奖励 $`r`$，再求这个奖励对应的最优策略**

传统的做法我们先做一个奖励模型 $`r_\phi`$, 对目前的偏好最大似然估计奖励:

```math
\max_\phi
\prod_{i=1}^N
P_\phi(y_{w,i}\succ y_{l,i}\mid x_i).
```

最大似然估计的老套路变化:

```math
\mathcal L_{\mathrm{RM}}(\phi)
=
-\frac1N\sum_{i=1}^N
\log\sigma\left(
r_\phi(x_i,y_{w,i})-r_\phi(x_i,y_{l,i})
\right).
```

狠狠训练 $`r_\phi`$, 然后代入:

```math
\boxed{
\max_\pi\;
\mathbb E_{x\sim\mathcal D}
\left[
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-
\beta D_{\mathrm{KL}}
\bigl(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\bigr)
\right]
}
```

用 PPO 就可以训练策略

---

但是我们还面临那个 $`Z(x)`$ 的问题, 需要干掉它

DPO 的想法是，既然最终要训练的是策略，能不能让**策略自己给出 BT 需要的奖励差**？

看看我们的最优策略:

```math
\pi(y\vert{}x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y\vert{}x) \exp\left(\frac{1}{\beta} r(x, y)\right)
```

反过来写:

```math
r(x,y)
=
\beta\log\frac{\pi_r^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
```

看起来怎么还是有 $`Z(x)`$?

别急我们有 BT 模型:

```math
p^*(y_w \succ y_l \vert{} x) = \sigma\big( r^*(x, y_w) - r^*(x, y_l) \big)
```

把奖励代入进去:

```math
\begin{aligned} r^*(x, y_w) - r^*(x, y_l) &= \left( \beta \log \frac{\pi^*(y_w\vert{}x)}{\pi_{\text{ref}}(y_w\vert{}x)} + \beta \log Z(x) \right) \\
&\quad - \left( \beta \log \frac{\pi^*(y_l\vert{}x)}{\pi_{\text{ref}}(y_l\vert{}x)} + \beta \log Z(x) \right) \\ &= \beta \log \frac{\pi^*(y_w\vert{}x)}{\pi_{\text{ref}}(y_w\vert{}x)} - \beta \log \frac{\pi^*(y_l\vert{}x)}{\pi_{\text{ref}}(y_l\vert{}x)} \end{aligned}
```

极其难算的 $`\beta \log Z(x)`$ 因为只依赖于 $`x`$, 被**完美抵消**了!

因此根据 BT 模型, 我们更喜欢一个答案的概率只由最优策略 $`\pi^*`$ 与参考策略 $`\pi_{\text{ref}}`$ 决定:

```math
p^*(y_w \succ y_l \vert{} x) = \sigma \left( \beta \log \frac{\pi^*(y_w\vert{}x)}{\pi_{\text{ref}}(y_w\vert{}x)} - \beta \log \frac{\pi^*(y_l\vert{}x)}{\pi_{\text{ref}}(y_l\vert{}x)} \right)
```

这是一对样本的概率, 我们做极大似然估计(最小化负对数似然), 得到了最终的 Loss:

```math
\begin{gathered}
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}})\\[4pt]
= -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w\vert{}x)}{\pi_{\text{ref}}(y_w\vert{}x)} - \beta \log \frac{\pi_\theta(y_l\vert{}x)}{\pi_{\text{ref}}(y_l\vert{}x)} \right) \right]
\end{gathered}
```

可以定义一个隐式奖励 $`\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y\vert{}x)}{\pi_{\text{ref}}(y\vert{}x)}`$, 这意味着**语言模型本身就是一个天然的奖励模型**

注意到我们算出来的是**最优策略**与奖励的关系, 为什么可以把 $`\pi_r^*`$ 换成任意训练中的 $`\pi_\theta`$? (注意到其实我们假设有"某一个奖励 $`r`$", 有一个最优策略, 现在我们可以知道反过来也是基本上成立的)

注意我们刚刚定义的隐式奖励, 我们知道奖励对应一个最优策略:

```math
\begin{aligned}
\pi_{\hat r_\theta}^*(y\mid x)
&=
\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{\hat r_\theta(x,y)}{\beta}\right)
\\
&=
\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)
\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\\
&=
\frac{1}{Z(x)}\pi_\theta(y\mid x).
\end{aligned}
```

注意这个 $`\pi_\theta`$ 是训练中的合法的分布,  又:

```math
Z(x)
=
\sum_y\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right) = \sum_y \pi_\theta(y\vert{}x) = 1
```

于是:

```math
\boxed{\pi_{\hat r_\theta}^*=\pi_\theta.}
```

也就是说:

> 每一个训练中的策略，都定义了一个隐式奖励；
> 而这个策略恰好就是该隐式奖励下的 KL 正则最优策略(之一)

## 从梯度看更新

先定义简化记号:

```math
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( z_\theta \right) \right]
```

```math
\boxed{
z_\theta
= \beta \left[ \log\frac{\pi_\theta(y_w\mid x)} {\pi_\theta(y_l\mid x)} - \log\frac{\pi_{\mathrm{ref}}(y_w\mid x)} {\pi_{\mathrm{ref}}(y_l\mid x)} \right],
}
```

我们固定一个偏好对来看怎么驱动梯度更新:

```math
\ell_\theta=-\log\sigma(z_\theta).
```

对 $`z_\theta`$ 求导:

```math
\frac{\partial\ell_\theta}{\partial z_\theta}
=
-\frac{\sigma'(z_\theta)}{\sigma(z_\theta)}
=
-\bigl(1-\sigma(z_\theta)\bigr)
=
-\sigma(-z_\theta).
```

```math
\nabla_\theta z_\theta
=
\beta
\left[
\nabla_\theta\log\pi_\theta(y_w\mid x)
-
\nabla_\theta\log\pi_\theta(y_l\mid x)
\right].
```

链式法则:

```math
\boxed{
\nabla_\theta\ell_\theta
=
-\beta\sigma(-z_\theta)
\left[
\nabla_\theta\log\pi_\theta(y_w\mid x)
-
\nabla_\theta\log\pi_\theta(y_l\mid x)
\right]
}
```

梯度下降:

```math
\theta
\leftarrow
\theta+
\eta\beta\sigma(-z_\theta)
\left[
\nabla_\theta\log\pi_\theta(y_w\mid x)
-
\nabla_\theta\log\pi_\theta(y_l\mid x)
\right].
```

写完整一点:

```math
\begin{gathered}
\nabla_\theta \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}})\\[4pt]
= -\beta \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \underbrace{\sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big)}_{\text{自适应权重（打分错误程度）}} \left( \underbrace{\nabla_\theta \log \pi_\theta(y_w\vert{}x)}_{\text{提高好回答概率}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l\vert{}x)}_{\text{降低坏回答概率}} \right) \right]
\end{gathered}
```

$`z_\theta\ll 0`$ 时说明判断错了, 权重接近 $`1`$, 需要大步更新, 反之权重接近 $`0`$

这里对应之前的正例概率不一定上升, 这个损失直接推动的是对应的**相对对数概率差**

参考模型虽然不参与求导，但它仍通过 $`z_\theta`$ 影响样本权重

## 结论

解析最优策略是在不受神经网络参数限制的分布空间中求出的；

偏好概率采用 Bradley–Terry 假设；

因此，这套推导建立了奖励学习与策略参数化之间的对应关系，

**并不保证有限数据上的 DPO 一定恢复真实人类奖励对应的最优策略，也不保证它与实际 PPO 训练得到同一模型**
