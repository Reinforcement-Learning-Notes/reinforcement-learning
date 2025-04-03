# **Policy Gradient & Actor-Critic**

## 🧠 **What are we trying to do?**

We have:
$$
U(\theta) = E[\sum_{t=0}^{H} R(s_t, u_t) | \pi_\theta] = \sum_{\tau} P(\tau; \theta) R(\tau)
$$

- $ \tau = (s_0, u_0, s_1, u_1, \ldots, s_H, u_H) $: trajectory, a sequence of states and actions.
- $ P(\tau; \theta) = \prod_{t=0}^{H} \pi_\theta(u_t | s_t) P(s_{t+1} | s_t, u_t) $: probability of trajectory $ \tau $ under policy $ \pi_\theta $.
- $ R(\tau) = \sum_{t=0}^{H} R(s_t, u_t) $: cumulative reward of trajectory $ \tau $.
- $ U(\theta) $: the “average score” the agent would achieve by following the policy $ \pi_\theta $.

Now we want to find policy parameters $ \theta $ that maximize the expected cumulative reward.

## **How to find the $\theta$**
### **Log-Derivative Trick (Score Function Identity):**

$$
\nabla_\theta U(\theta) = \sum_{\tau} P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) =
\mathbb{E}_{\tau \sim P(\tau;\theta)}\left[\nabla_\theta \log P(\tau;\theta) \cdot R(\tau)\right]
$$

**Empirical estimate** for m sample paths under policy $\pi_\theta$:
$$
\nabla_\theta U(\theta) \approx \frac{1}{m} 
\sum_{i=1}^{m} \nabla_\theta \log P(\tau^{(i)};\theta) \cdot R(\tau^{(i)})
$$


**What's inside  $ \log P(\tau;\theta) $?**
$$
\nabla_\theta \log P(\tau^{(i)}; \theta) = \sum_{t=0}^H \underbrace{\nabla_\theta \log \pi_\theta(u_t^{(i)} | s_t^{(i)})}_{\text{no dynamics model required!!}}
$$

**Key Insight:**
    Instead of directly differentiating $ P(\tau;\theta) $, we differentiate its logarithm:
    $\nabla_\theta P(\tau;\theta) = P(\tau;\theta) \nabla_\theta \log P(\tau;\theta)$.
    This makes it possible to estimate gradients using samples.


### **Importance Sampling:**
  - **Scenario:**
    Often, we collect data under an older policy ($ \theta_{\text{old}} $) rather than the current one.
  - **Reweighting:**
    Adjust the expectation from the old policy to the new policy:
    $\mathbb{E}_{\tau \sim P(\tau;\theta)}[f(\tau)] = \mathbb{E}_{\tau \sim P(\tau;\theta_{\text{old}})}\left[\frac{P(\tau;\theta)}{P(\tau;\theta_{\text{old}})}f(\tau)\right]$
  - **Result:**
    When evaluated at $ \theta = \theta_{\text{old}} $, the importance weight equals 1, recovering the log-likelihood ratio estimator. This shows that the basic policy gradient derivation is a special case of an importance-sampled gradient.

So far we have seen the **Likelihood Ratio Gradient Estimate**, which is the basis of REINFORCE. What are the limitations of this estimator?

### ⚠️ Unbiased by suffer from high variance
- **Unbiasedness**: This estimator is unbiased because it directly computes the gradient of the expected return $ U(\theta) $ 
using samples from $ \theta_{\text{old}} $. The expectation ensures that the true gradient is preserved over many samples.

- When we rewrite the gradient using importance sampling, we introduce the ratio:$\frac{P(\tau; \theta)}{P(\tau; \theta_{\text{old}})}$
  - If the new policy  $ \theta $ is very different from the old policy $ \theta_{\text{old}} $, some trajectories $ \tau $ may have very low probability under $ \theta_{\text{old}} $ but high probability under $ \theta $. This causes the importance weight $ \frac{P(\tau; \theta)}{P(\tau; \theta_{\text{old}})} $ to become extremely large for those trajectories.
  - Conversely, some trajectories may have very high probability under $ \theta_{\text{old}} $ but negligible probability under $ \theta $, causing their weights to approach zero.
  - These extreme values lead to **high variance** in the gradient estimate because the contributions of individual trajectories can fluctuate wildly.

- The reward $ R(\tau) $ also plays a role in amplifying variance:
  - If $ R(\tau) $ is large for trajectories with extreme importance weights, the product $ \frac{P(\tau; \theta)}{P(\tau; \theta_{\text{old}})} R(\tau) $ can explode.
  - Even if $ R(\tau) $ is small, the variance of the gradient estimate can still be high due to the inherent randomness in sampling trajectories.

### Fixes for practicality
**Baseline Subtraction:**
- Subtract a state-dependent baseline $ b(s_t) $ to compute the advantage:
  $$
  \hat{g}_t = \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \log P(\tau ^{i};\theta) \cdot \left( R(\tau ^{i}) - b(s_t^{(i)}) \right)
  $$
  This reduces variance without introducing bias.


**Temporal Structure:**

$$
\begin{aligned}
\hat{g} &= \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \log P(\tau^{(i)}; \theta) (R(\tau^{(i)}) - b) \\
&= \frac{1}{m} \sum_{i=1}^{m}\left( \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta(u_t^{(i)} | s_t^{(i)}) \right) \left( \sum_{k=0}^{H-1} R(s_k^{(i)}, u_k^{(i)}) - b \right)   \\
&= \frac{1}{m} \sum_{i=1}^{m} \left( \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta(u_t^{(i)} | s_t^{(i)}) \right) \left( \left[ \sum_{k=0}^{t-1} R(s_k^{(i)}, u_k^{(i)}) \right] + \left[ \sum_{k=t}^{H-1} R(s_k^{(i)}, u_k^{(i)}) \right] - b \right) \\
& = \frac{1}{m} \sum_{i=1}^{m} \left( \sum_{t=0}^{H-1}\nabla_\theta \log \pi_\theta(u_t^{(i)} | s_t^{(i)}) \right) \left( \sum_{k=t}^{H-1} R(s_k^{(i)}, u_k^{(i)}) - b(s_t^{(i)}) \right)
\end{aligned}
$$

Removing terms that don’t depend on current action can lower variance.



## 🔍 Step-by-step derivation 
### **Log-Derivative Trick**

1. **Take gradient of the sum:**
   $$
   \nabla_\theta U(\theta) = \nabla_\theta \sum_\tau P(\tau; \theta) R(\tau) = \sum_\tau \nabla_\theta P(\tau; \theta) R(\tau)
   $$ 
   That’s just taking the derivative inside the sum (valid because sums are linear).

2. **Now we do the trick!** Multiply and divide by $ P(\tau; \theta) $:
  $${
   = \sum_\tau P(\tau; \theta) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} R(\tau)
   }$$

3. **Trick - Recognize the identity**:
   $${
   \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} = \nabla_\theta \log P(\tau; \theta)
   }$$


4. **Now plug it in:**
   $${
   \nabla_\theta U(\theta) = \sum_\tau P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau)
   }$$

So we end up with a super useful form:
$$
\nabla_\theta U(\theta) = \mathbb{E}_{\tau \sim P(\cdot; \theta)} \left[ \nabla_\theta \log P(\tau; \theta) \cdot R(\tau) \right]
$$

🎯 What does this mean intuitively?

The gradient tells us how to nudge the parameters to **favor better trajectories**.
> "If a trajectory is **good** (high reward), then **increase** the probability of choosing it again."

### What's inside  $ \log P(\tau; \theta) $?

$$
\begin{aligned}
\nabla_\theta \log P(\tau^{(i)}; \theta) &= \nabla_\theta \log \left[ \prod_{t=0}^H \underbrace{P(s_{t+1}^{(i)} | s_t^{(i)}, u_t^{(i)})}_{\text{dynamics model}} \cdot \underbrace{\pi_\theta(u_t^{(i)} | s_t^{(i)})}_{\text{policy}} \right] \\
&= \nabla_\theta \left[ \sum_{t=0}^H \log P(s_{t+1}^{(i)} | s_t^{(i)}, u_t^{(i)}) + \sum_{t=0}^H \log \pi_\theta(u_t^{(i)} | s_t^{(i)}) \right] \\
&= \nabla_\theta \sum_{t=0}^H \log \pi_\theta(u_t^{(i)} | s_t^{(i)}) \\
&= \sum_{t=0}^H \underbrace{\nabla_\theta \log \pi_\theta(u_t^{(i)} | s_t^{(i)})}_{\text{no dynamics model required!!}}
\end{aligned}
$$

This means the full gradient estimator becomes:
$$
\nabla_\theta U(\theta) \approx \frac{1}{m} \sum_{i=1}^{m} \left[ \sum_{t=0}^{H} \nabla_\theta \log \pi_\theta(u_t^{(i)} | s_t^{(i)}) \right] R(\tau^{(i)})
$$

### Importance Sampling

1. Rewriting the expectation with importance sampling:
   $$
   U(\theta) = \mathbb{E}_{\tau \sim \theta_{\text{old}}} \left[ \frac{P(\tau; \theta)}{P(\tau; \theta_{\text{old}})} R(\tau) \right]
   $$
   $$
   U(\theta) = \sum_{\tau} P(\tau; \theta_{\text{old}}) \cdot \frac{P(\tau; \theta)}{P(\tau; \theta_{\text{old}})} \cdot R(\tau)
   $$
   Now we can use data from $\theta_{\text{old}}$ to estimate what would have happened under $\theta$.

2. Taking gradient:
   $$ 
   \nabla_\theta U(\theta) = \mathbb{E}_{\tau \sim \theta_{\text{old}}} \left[ \nabla_\theta \left( \frac{P(\tau; \theta)}{P(\tau; \theta_{\text{old}})} \right) R(\tau) \right]
   $$

3. When evaluated at  $\theta = \theta_{\text{old}}$, the ratio simplifies:
   $$
   \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta_{\text{old}})} \bigg|_{\theta = \theta_{\text{old}}} = \frac{P(\tau; \theta_{\text{old}}) \nabla_\theta \log P(\tau; \theta)}{P(\tau; \theta_{\text{old}})} = \nabla_\theta \log P(\tau; \theta)
   $$
   So we get:
   $$
   \nabla_\theta U(\theta) \bigg|_{\theta = \theta_{\text{old}}} = \mathbb{E}_{\tau \sim \theta_{\text{old}}} \left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right]
   $$

This is the same gradient estimator we saw earlier—but now we've derived it from importance sampling!