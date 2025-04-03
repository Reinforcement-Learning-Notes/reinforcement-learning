## Model-Free Prediction & Control with Temporal Difference (TD) and Q-Learning


### Learning Goals

- Understand TD(0) for prediction
- Understand SARSA for on-policy control
- Understand Q-Learning for off-policy control
- Understand the benefits of TD algorithms over MC and DP approaches
- Understand how n-step methods unify MC and TD approaches
- Understand the backward and forward view of TD($\lambda$)


### Summary

- TD-Learning is a combination of Monte Carlo and Dynamic Programming ideas. Like Monte Carlo, TD works based on samples and doesn't require a model of the environment. Like Dynamic Programming, TD uses bootstrapping to make updates.
- Whether MC or TD is better depends on the problem and there are no theoretical results that prove a clear winner.
- General Update Rule: `Q[s,a] += learning_rate * (td_target - Q[s,a])`. `td_target - Q[s,a]` is also called the TD Error.
- SARSA: On-Policy TD Control
  - TD Target for SARSA: `R[t+1] + discount_factor * Q[next_state][next_action]`
- Q-Learning: Off-policy TD Control
  - TD Target for Q-Learning: `R[t+1] + discount_factor * max(Q[next_state])`
  - Q-Learning ignores the behavior policy when updating Q-values (contrast this with SARSA). It assumes the agent will choose the best action in the next state.
  - Learning about the **optimal policy** while behaving **sub-optimally**.

- N-Step methods unify MC and TD approaches. They making updates based on n-steps instead of a single step (TD-0) or a full episode (MC).
- TD($\lambda$): Unify TD and DP approaches

(Due to technical issues, the search service is temporarily unavailable.)

Yes! Your algorithm is **TD(0)** (a **one-step Temporal Difference** method) because it updates the Q-values **immediately after each transition** \((s, a, r, s')\) using the **bootstrapped estimate** of the next state's value.  


| Method          | Update Horizon          | Update Rule (Simplified)                     |
|-----------------|-------------------------|---------------------------------------------|
| **Monte Carlo** | Full episode            | $Q(s,a) \leftarrow Q(s,a) + \alpha [G_t - Q(s,a)]$ ($G_t$ is total return). |
| **TD(0)**       | 1 step                  | $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| **n-step TD**   | n steps                 | Uses rewards from $n$ steps + value of $s_{t+n}$. |
| **TD(λ)**       | Weighted mix of all n   | Uses eligibility traces to blend 1-step, 2-step, ..., ∞-step returns. |


### Lectures & Readings

**Required:**

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) - Chapter 6: Temporal-Difference Learning
- David Silver's RL Course Lecture 4 - Model-Free Prediction ([video](https://www.youtube.com/watch?v=PnHCvfgC_ZA), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf))
- David Silver's RL Course Lecture 5 - Model-Free Control ([video](https://www.youtube.com/watch?v=0g4j2k_Ggc4), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf))

**Optional:**

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) - Chapter 7: Multi-Step Bootstrapping
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) - Chapter 12: Eligibility Traces


### Exercises

- Get familiar with the [Windy Gridworld Playground](Windy%20Gridworld%20Playground.ipynb)
- Implement SARSA
  - [Exercise](SARSA.ipynb)
  - [Solution](SARSA%20Solution.ipynb)
- Get familiar with the [Cliff Environment Playground](Cliff%20Environment%20Playground.ipynb)
- Implement Q-Learning in Python
  - [Exercise](Q-Learning.ipynb)
  - [Solution](Q-Learning%20Solution.ipynb)
