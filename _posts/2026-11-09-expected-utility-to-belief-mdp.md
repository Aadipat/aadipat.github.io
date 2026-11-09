---
layout: post
title: "From expected utility to the belief MDP"
date: 2026-11-09
description: Influence diagrams, MDPs and POMDPs derived from one principle, including the proof that a belief state is all an agent needs to remember.
tags: reinforcement-learning probability math
categories: derivations
toc:
  sidebar: left
_styles: >
  mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; }
---

Influence diagrams, MDPs and POMDPs are usually taught as three separate topics. In my notes they are one equation, maximum expected utility, applied with different answers to a single question: what does the agent know at the moment it decides? This post follows that equation from a single decision to the belief MDP, then shows what changes when the model is unknown.

## One principle

An agent takes actions in a probabilistic environment and gets rewards. Outcomes have **utility**.

One piece of notation first, because it appears everywhere below. An expectation is only a weighted sum: for any function $$f$$ and whatever we are conditioning on,

$$
E[f(x) \mid \ldots] = \sum_x P(x \mid \ldots)\, f(x)
$$

Expected utility is this with $$f = U$$ and $$x$$ the outcome. Given an action $$a$$ and evidence $$e$$:

$$
EU(a \mid e) = E[U \mid a, e] = \sum_{\text{outcomes}} P(\text{outcome} \mid a, e)\; U(\text{outcome})
$$

The principle of **maximum expected utility** (MEU) says to pick the action that maximises this:

$$
a^* = \arg\max_a EU(a \mid e), \qquad MEU[e] = \max_a EU(a \mid e)
$$

Where outcomes have utility, situations have **value**. A situation is what the agent knows or believes, and its value is the expected utility it can still collect from there onwards. For a sequence of rewards, utility is either the plain sum over an episode, or a discounted sum:

$$
U = \sum_{t=0}^{T} r_t \qquad\text{or}\qquad U = \sum_t \gamma^t r_t
$$

The three models differ only in what fills the slots:

| Model             | Evidence the agent holds   | Outcome summed over |
| ----------------- | -------------------------- | ------------------- |
| Influence diagram | $$e$$, about a state $$s$$ | $$s$$               |
| MDP               | the state $$s$$ itself     | next state $$s'$$   |
| POMDP             | the history $$h$$          | $$s, s', o$$        |

As in other probabilistic graphical models, everything below uses the same two operations: $$\oplus$$ is a sum over unobserved variables, or a max at a decision, and $$\otimes$$ is the product of local factors.

## One decision: influence diagrams

Take one timestep, $$T = 1$$. The action does not move the world, so $$s' = s$$, and $$s$$ may not be fully observable.

{% include diagram.liquid name="dm-influence" caption="An influence diagram. The dashed arrow means the evidence is available when the decision is made." %}

$$
EU(a \mid e) = \sum_s P(s \mid a, e)\, U(s, a) = \sum_s P(s \mid e)\, U(s, a)
$$

The second step uses $$s \perp a \mid e$$: the action is chosen from the evidence alone, so it says nothing further about the state. If the transition is deterministic, $$\text{Result}(s, a) = s'$$, then a utility on outcomes and a utility on state-action pairs are the same thing, $$U(s') = U(s, a)$$.

### The general formula

The same recipe works for any influence diagram, in three steps. Write $$\text{pa}(X)$$ for the parents of a node $$X$$.

**1. Chain the joint.** The joint distribution of all chance variables is the product of one local factor per chance node, each given its parents. Decisions appear only as fixed inputs to those factors:

$$
P(\text{chance vars} \mid \text{decisions}) = \prod_i P(X_i \mid \text{pa}(X_i))
$$

**2. Take the expectation.** Expected utility is the $$E[f(x) \mid \ldots]$$ sum from earlier, with $$x$$ being all the chance variables and $$f$$ being the utility:

$$
EU(\text{decisions}) = E[U \mid \text{decisions}] = \sum_{\text{chance vars}} \prod_i P(X_i \mid \text{pa}(X_i))\; U(\text{pa}(U))
$$

**3. Put each max where its decision happens.** If no decision observes anything first, one max goes outside: $$MEU = \max_{\text{decisions}} EU$$, and $$\arg\max$$ gives the decisions themselves. If decisions do observe chance variables, order everything in time. Let $$C_0$$ be the variables known before decision $$d_1$$, $$C_1$$ those revealed between $$d_1$$ and $$d_2$$, and $$C_n$$ those never seen before the last decision. Assuming each decision remembers everything before it:

$$
MEU = \underbrace{\sum_{C_0} \max_{d_1} \sum_{C_1} \cdots \max_{d_n} \sum_{C_n}}_{\text{sums and maxes in time order}}\; \underbrace{\prod_i P(X_i \mid \text{pa}(X_i))}_{\text{chained joint}}\; \underbrace{U(\text{pa}(U))}_{\text{utility}}
$$

The order is what matters. A sum outside a max is a variable that decision can see, so the choice may depend on it. A sum inside a max is still unknown when the decision is made, so the decision can only average over it.

For the diagram above, the chained joint is $$P(s)\,P(e \mid s)$$, and $$e$$ is seen before $$a$$ while $$s$$ is not, so the order is $$\sum_e \max_a \sum_s$$:

$$
\begin{aligned}
MEU &= \sum_e \max_a \sum_s P(s)\, P(e \mid s)\, U(s, a) \\
&= \sum_e P(e) \max_a \sum_s P(s \mid e)\, U(s, a) \\
&= \sum_e P(e)\, MEU[e]
\end{aligned}
$$

The sequential models below have this same form, a repeating $$\max_{a_0} \sum_{s_1} \max_{a_1} \sum_{s_2} \cdots$$, and the Bellman equation is that nesting written as a recursion.

## Full observability: the MDP

An MDP is $$\langle S, A, T, R \rangle$$ with transition model $$T(s, a, s') = P(s' \mid s, a)$$. Start from MEU, with the expected utility of an action now called $$Q$$:

$$
EU(a \mid e) = \sum_{s, s'} P(s, s' \mid a, e)\, U(s, a, s') = \sum_{s, s'} P(s \mid s', a, e)\, P(s' \mid a, e)\, U(s, a, s')
$$

Full observability means the evidence names the state exactly. So $$P(s \mid s', a, e)$$ is 1 for that state and 0 for every other, and the sum over $$s$$ disappears.

Now the utility. The **value** $$V(s)$$ is the expected utility from $$s$$ onwards if the agent acts optimally, which makes it $$MEU[s]$$ for the whole remaining sequence:

$$
V(s) = \max_\pi E\Big[ \sum_t \gamma^t r_t \;\Big|\; s_0 = s, \pi \Big]
$$

$$Q(s, a)$$ is the same expected utility with the first action fixed to $$a$$, so $$Q(s, a) = EU(a \mid s)$$. The utility of an outcome $$s'$$ is therefore the immediate reward plus the expected utility from $$s'$$ onwards:

$$
U(s, a, s') = R(s, a, s') + \gamma V(s')
$$

Substituting gives the action value, and acting by MEU gives the state value. Together they are the **Bellman equation**:

$$
Q(s, a) = \sum_{s'} T(s, a, s') \big[ R(s, a, s') + \gamma V(s') \big], \qquad V(s) = \max_a Q(s, a)
$$

{% include diagram.liquid name="dm-backup" caption="One Bellman backup. A max over the actions, then an expectation over the next states. The highlighted branch is the action MEU would choose." %}

Splitting the bracket shows the two parts of an action's value, the expected immediate return and the expected future return, with $$R(s, a) = \sum_{s'} T(s, a, s') R(s, a, s')$$:

$$
Q(s, a) = R(s, a) + \gamma \sum_{s'} T(s, a, s')\, V(s')
$$

### Solving it

**Value iteration** turns the Bellman equation into an update, starting from any assumed $$V_0$$:

$$
V_{k+1}(s) = \max_a Q_k(s, a)
$$

**Policy iteration** alternates two steps. A policy $$\pi(a \mid s)$$ is a distribution over actions, or a single action $$\pi(s)$$ if deterministic. **Policy evaluation** computes its value:

$$
V^\pi_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s'} T(s, a, s') \big[ R(s, a, s') + \gamma V^\pi_k(s') \big]
$$

With no max in it, this is a linear system in $$V^\pi$$, which can be solved directly or approximated with a few sweeps. **Policy improvement** then applies MEU, $$\pi'(s) = \arg\max_a Q(s, a)$$, and we stop when $$\pi' = \pi$$.

Both methods solve for $$Q$$ through $$V$$ and extract $$\pi$$ by MEU. In an MDP a deterministic policy is enough, because a max is never smaller than an average:

$$
\max_a Q(s, a) \;\geq\; \sum_a \pi(a \mid s)\, Q(s, a)
$$

## Hidden state: the POMDP

A POMDP $$\langle S, A, T, R, O \rangle$$ removes direct access to $$s$$. Actions now also produce observations, through $$O(o, s', a) = P(o \mid s', a)$$, and the outcome of a step is the pair $$(s', o)$$.

{% include diagram.liquid name="dm-pomdp" caption="POMDP dynamics. States are hidden (dashed) and the agent only has its observations and its own actions." %}

The evidence is now the history of everything the agent has done and seen:

$$
h_t = (a_0, o_1, \ldots, a_{t-1}, o_t), \qquad h' = \langle h, a, o \rangle
$$

Each history is a new situation, and histories have value the way states do in an MDP. The derivations below lean on three independences read straight from the model:

- $$o \perp s, h \mid s', a$$, which is the definition of $$O$$.
- $$s' \perp h \mid s, a$$, which is the definition of $$T$$.
- $$s \perp a \mid h$$, because the action is chosen from the history alone.

### Histories form an MDP

The state only matters through what the history says about it, so $$\langle H, A, T, R \rangle$$ is an MDP with $$R(h, a) = \sum_s P(s \mid h) R(s, a)$$. Its transition reduces to the probability of the next observation, since $$P(A, B \mid A) = P(B \mid A)$$ gives $$T(h, a, h') = P(\langle h, a, o \rangle \mid h, a) = P(o \mid h, a)$$. Expanding that over the hidden states:

$$
\begin{aligned}
P(o \mid h, a)
&= \sum_{s, s'} P(o \mid s, s', h, a)\, P(s, s' \mid h, a) \\
&= \sum_{s, s'} P(o \mid s', a)\, P(s' \mid s, h, a)\, P(s \mid h, a) \\
&= \sum_s P(s \mid h) \sum_{s'} O(o, s', a)\, T(s, a, s')
\end{aligned}
$$

This MDP is correct but unusable. The number of histories at depth $$T$$ is $$(\lvert A \rvert \lvert O \rvert)^T$$, exponential in the horizon, and the number of policies over that tree is doubly exponential.

### The Q function over histories

Apply MEU with $$e = h$$. Only the history has value here, since it is all the agent knows, so the utility of an outcome is $$R(s, a, s') + \gamma V(h, a, o)$$:

$$
\begin{aligned}
Q(h, a)
&= \sum_{s, s', o} P(s, s', o \mid a, h) \big[ R(s, a, s') + \gamma V(h, a, o) \big] \\
&= \sum_{s, s', o} P(o \mid s', a)\, P(s' \mid s, a)\, P(s \mid h) \big[ R(s, a, s') + \gamma V(h, a, o) \big] \\
&= \sum_s P(s \mid h) \sum_{s'} T(s, a, s') \sum_o O(o, s', a) \big[ R(s, a, s') + \gamma V(h, a, o) \big]
\end{aligned}
$$

Since $$\sum_o O(o, s', a) = 1$$, the reward term loses its sum over $$o$$, and $$Q$$ splits as $$Q(h, a) = R(h, a) + F(h, a)$$ with

$$
R(h, a) = \sum_s P(s \mid h) \sum_{s'} T(s, a, s')\, R(s, a, s')
$$

$$
F(h, a) = \gamma \sum_o V(h, a, o) \sum_s P(s \mid h) \sum_{s'} O(o, s', a)\, T(s, a, s')
$$

### Introducing the belief

Look at where $$h$$ appears. Apart from $$V(h, a, o)$$, it is only ever inside $$P(s \mid h)$$. So define the **belief** and its successor:

$$
b(s) = P(s \mid h), \qquad b'(s') = P(s' \mid h'), \quad h' = \langle h, a, o \rangle
$$

$$R(h, a)$$ uses $$h$$ only through $$b$$, so $$R(h, a) \triangleq R(b, a)$$, where $$\triangleq$$ means the history-indexed quantity is determined by the belief. $$F$$ still contains $$V(h, a, o) = V(h')$$. To remove it we need two things: a way to compute $$b'$$ without the history, and a proof that $$V$$ depends on the history only through the belief.

### The belief update

Apply Bayes' rule to the new observation, then the independences:

$$
\begin{aligned}
b'(s') = P(s' \mid h, a, o)
&= \frac{P(s', o \mid h, a)}{P(o \mid h, a)}
= \frac{P(o \mid s', a)\, P(s' \mid h, a)}{P(o \mid h, a)} \\
&= \frac{P(o \mid s', a)}{P(o \mid h, a)} \sum_s P(s' \mid s, h, a)\, P(s \mid h, a) \\
&= \frac{P(o \mid s', a)}{P(o \mid h, a)} \sum_s T(s, a, s')\, b(s)
\end{aligned}
$$

The denominator is the transition we already derived, which only needs $$b$$:

$$
P(o \mid h, a) = \sum_{s, s'} b(s)\, O(o, s', a)\, T(s, a, s') \triangleq P(o \mid b, a)
$$

It is only a normalising constant, so the update is

$$
\boxed{\, b'(s') = \frac{O(o, s', a) \sum_s T(s, a, s')\, b(s)}{\sum_{s'} O(o, s', a) \sum_s T(s, a, s')\, b(s)} \,}
$$

It predicts the state forward through $$T$$, reweights by how well each state explains the observation, and normalises. It needs only $$(b, a, o)$$.

### Proof that the belief is sufficient

We want $$V(h) \triangleq V(b)$$, by induction backwards from the last stage. Recall $$V(h) = \max_a Q(h, a)$$.

**Base case**, $$t = T - 1$$. No further action is allowed, so there is no future term:

$$
Q(h, a) = R(h, a) + 0 = R(b, a) \triangleq Q(b, a) \quad\Rightarrow\quad V(h) \triangleq V(b)
$$

**Inductive step.** Assume $$V(h') = V(b')$$ at time $$t+1$$ and prove it at $$t$$:

$$
\begin{aligned}
V(h) &= \max_a \Big[ R(b, a) + \gamma \sum_o V(h') \sum_s b(s) \sum_{s'} O(o, s', a)\, T(s, a, s') \Big] \\
&= \max_a \Big[ R(b, a) + \gamma \sum_o V(b') \sum_s b(s) \sum_{s'} O(o, s', a)\, T(s, a, s') \Big] \triangleq V(b)
\end{aligned}
$$

Because $$b'$$ is a function of $$(b, a, o)$$, the whole expression depends on $$h$$ only through $$b$$. The same argument without the max gives $$Q(h, a) \triangleq Q(b, a)$$. The belief is a sufficient statistic of the history for acting optimally.

### The belief MDP

So we can throw away histories and write a Bellman equation directly over beliefs:

$$
Q(b, a) = R(b, a) + \gamma \sum_o P(o \mid b, a)\, V(b'), \qquad V(b) = \max_a Q(b, a)
$$

$$
R(b, a) = \sum_{s, s'} b(s)\, T(s, a, s')\, R(s, a, s')
$$

It has the same shape as the MDP backup, with the belief as the state and the observation as the source of randomness. It also has far fewer distinct reachable situations, because many histories lead to the same belief, so the tree folds into something closer to a DAG.

{% include diagram.liquid name="dm-tree-dag" caption="The tiger problem (Kaelbling, Littman and Cassandra, 1998): a tiger is behind one of two doors, and listening reports the correct side 85% of the time. Each node shows P(tiger left) after repeated listening. Hearing left then right gives the same belief as right then left." %}

The catch is that $$b$$ is continuous, so we cannot brute-force a table over it.

### Solving it

Two approximations reuse the solution of the underlying MDP $$\langle S, A, T, R \rangle$$, giving $$V(s)$$, $$Q(s, a)$$ and $$\pi(s)$$, while tracking $$b' = \text{update}(b, a, o)$$ at run time:

- **Most likely state:** $$s_{ML} = \arg\max_s b(s)$$, then $$\pi(b) = \pi(s_{ML})$$.
- **QMDP:** $$Q(b, a) = \sum_s b(s) Q(s, a)$$, then $$\pi(b) = \arg\max_a Q(b, a)$$.

Both act as if the uncertainty will be gone after the current step, so neither will take an action purely to gain information.

The exact solution uses the structure of $$V$$. A plan with $$k$$ steps to go has an expected reward from each state, which is a vector $$\alpha(s)$$. With one step to go, $$\alpha(s) = R(s, a)$$. Let $$\Gamma_k$$ be the finite set of such plans. Then

$$
V_k(b) = \max_{\alpha \in \Gamma_k} \sum_s \alpha(s)\, b(s)
$$

This form survives a backup because of a cancellation. In $$Q(b, a)$$, the value $$V(b')$$ is multiplied by $$P(o \mid b, a)$$, which is exactly the denominator of the belief update:

$$
P(o \mid b, a) \sum_{s'} \alpha(s')\, b'(s') = \sum_{s'} \alpha(s')\, O(o, s', a) \sum_s T(s, a, s')\, b(s)
$$

The right side is linear in $$b$$, so each new plan is again a linear function of the belief, and value iteration can run on the sets $$\Gamma_k$$ until it converges to within $$\epsilon$$. Each plan is a straight line over beliefs, and the max switches between lines in different regions of $$b$$. That makes $$V$$ **piecewise linear and convex** (PWLC).

{% include diagram.liquid name="dm-pwlc" caption="Exact value iteration on the tiger problem with a discount of 0.95, listening at a cost of 1, a correct door worth +10 and the tiger −100. Grey lines are the plans, clipped to the plot. The coloured curve is their upper envelope, and it is the value function." %}

## Without the model: reinforcement learning

Now go back to the fully observable $$\langle S, A, T, R \rangle$$, but suppose the tables $$T$$ and $$R$$ are unknown. All we have is experience: tuples of $$s, a, r, s'$$ and sometimes $$a'$$.

Two distinctions organise the methods. **Passive** learning estimates the value $$V^\pi$$ of a given policy, and **active** learning looks for $$\pi^*, Q^*, V^*$$. **Model-based** methods estimate $$T$$ and $$R$$ and then solve the MDP (adaptive dynamic programming), and **model-free** methods go straight for $$Q^*, V^*$$ or $$\pi^*$$. The target matters as well: getting a policy from $$V$$ needs $$T$$ for the one-step lookahead, but getting one from $$Q$$ does not.

The model-free move is to replace the expectation in the Bellman backup by a single sample:

$$
\underbrace{\sum_{s'} T(s, a, s') \big[ R(s, a, s') + \gamma V(s') \big]}_{\text{model}} \;\longrightarrow\; \underbrace{r + \gamma V(s')}_{\text{sample, bootstrapped}}
$$

Each algorithm then nudges its estimate towards that sample by a step size $$\alpha$$ (unrelated to the $$\alpha$$-vectors above). The bracketed difference is the temporal-difference error.

**TD learning** updates after each transition $$(s, a, r, s')$$. It bootstraps on its own estimate, so it is biased.

$$
V^\pi(s) \leftarrow V^\pi(s) + \alpha \big[ r + \gamma V^\pi(s') - V^\pi(s) \big]
$$

**Monte Carlo** does not bootstrap. It runs trajectories to the end and averages the observed returns $$G_t$$, which is unbiased but needs much more data.

**Q-learning** also updates after $$(s, a, r, s')$$. The max makes it off-policy: it learns the greedy values whatever the agent actually did.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big]
$$

**Sarsa** waits for the next action too, using $$(s, a, r, s', a')$$. It is on-policy and converges to $$Q^\pi$$, so $$\pi$$ has to improve along the way to reach $$\pi^*$$.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma Q(s', a') - Q(s, a) \big]
$$

Off-policy learning is the more attractive idea, but Sarsa tends to be more stable and more realistic about what the agent will actually do.

Since the agent's own actions determine its data, it has to explore:

- **$$\epsilon$$-greedy** takes the greedy action with probability $$1 - \epsilon$$ and a random one otherwise.
- **R-MAX** is optimistic. Until $$(s, a)$$ has been tried $$m$$ times, it assumes $$T(s, a, s') = \delta_{ss'}$$ and $$R(s, a) = R_{\max}$$. After that it switches to the maximum-likelihood estimates $$\hat{T}(s, a, s') = N(s, a, s') / N(s, a)$$ and the mean observed reward.
- **UCB1** adds a bonus for rarely tried actions: $$a_t = \arg\max_a \big[ \hat{Q}(a) + c \sqrt{\ln N / N_a} \big]$$.

When the tables are too large, $$Q$$ or $$V$$ become deep networks, trained by the [backward pass from my backprop post](/blog/2026/backprop-one-scalar-at-a-time/). This introduces the **deadly triad**: bootstrapping, off-policy learning and function approximation, which together can make the estimates diverge. A different route parametrises the policy itself, $$\pi(a \mid s; \omega)$$, and follows the gradient of the return, as in REINFORCE:

$$
\omega_{t+1} = \omega_t + \alpha \gamma^t G_t \nabla_\omega \log \pi(a_t \mid s_t; \omega_t)
$$

Actor-critic methods combine the two, with a learned value function supporting a learned policy.
