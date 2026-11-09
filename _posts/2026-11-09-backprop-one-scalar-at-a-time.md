---
layout: post
title: "Backpropagation, one scalar at a time"
date: 2026-11-09
description: Deriving the gradient rules for linear and activation layers from the plain chain rule, treating every tensor entry as its own variable.
tags: deep-learning math
categories: derivations
toc:
  sidebar: left
_styles: >
  mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; }
---

`loss.backward()` hides a small amount of calculus. I wanted to write that calculus out without memorising any matrix-calculus identities, so this post is the derivation from my notes. The idea is to stop looking at tensors and treat every tensor entry as a separate scalar variable. The only tool left is the ordinary chain rule, and formulas like $$g W^T$$ and $$x^T g$$ come out at the end.

## From layers to one large graph

A deep architecture is a sequence of simple layers, with each layer's output wired into the next layer's input. Joining them gives one large computational graph of tensors that ends in the scalar loss. It is a directed acyclic graph (DAG): values only flow forward, so nothing depends on itself.

{% include diagram.liquid name="bp-architecture" caption="A deep architecture joins layers in sequence, which resolves into a DAG of tensors, including the weights and biases each Linear layer brings in. Every edge has its own local equation. They are written here without indices, and the sections below derive them entry by entry." %}

Each edge comes with a local equation, the derivative of its output with respect to its input. It only involves that one operation and knows nothing about the rest of the network.

Backpropagation needs exactly these two things:

1. **The DAG**, which says what feeds what, and so which paths lead to the loss and in what order to visit them.
2. **The local derivative on every edge.**

Given those, the chain rule does the rest. Each tensor is also a grid of entries, so underneath this graph sits an even larger DAG of scalars, and that is the level where the chain rule is simplest to write.

## The graph is made of scalars

Draw the computation graph at its finest level. Every entry of every input, parameter and intermediate tensor is a node, and the last node is a single scalar, the loss $$\mathcal{L}$$.

For any node $$v$$, write $$\bar{v}$$ for the derivative of the loss with respect to it. So $$\bar{y} = \partial \mathcal{L} / \partial y$$, and at the end of the graph $$\bar{\mathcal{L}} = \partial \mathcal{L} / \partial \mathcal{L} = 1$$.

{% include diagram.liquid name="bp-scalar-graph" caption="A linear layer followed by an activation and a loss, drawn entry by entry for one batch row. Coloured labels are the bar value of each node. Each edge carries a local derivative, and along the highlighted path the bar values multiply back from the loss." %}

The bar of any node comes from the chain rule. A node $$v$$ can only affect the loss through the nodes it feeds directly, so its derivative is a sum over those children, each term being the child's own bar times the local derivative on the edge:

$$
\bar{v} = \frac{\partial \mathcal{L}}{\partial v} = \sum_{c \,\in\, \text{children}(v)} \frac{\partial \mathcal{L}}{\partial c} \cdot \frac{\partial c}{\partial v} = \sum_{c \,\in\, \text{children}(v)} \bar{c} \cdot \frac{\partial c}{\partial v}
$$

This recursion is the chain rule applied at every node of the DAG, and that is all backpropagation is. To evaluate it at $$v$$ we need $$\bar{c}$$ for every child first, so we start at the loss, where $$\bar{\mathcal{L}} = 1$$, and visit the nodes in reverse order. Written out in full, the chain rule is a sum over every path from $$v$$ to the loss of the product of the edge derivatives along that path. The recursion gives the same number but reuses each $$\bar{c}$$ instead of walking every path separately. Each edge is used once, so the backward pass costs about as much as the forward pass.

I'll name directions by how the gradient travels. It starts at the loss, so for any layer the output side is **upstream** and the input side is **downstream**.

## Index notation

For a tensor $$A$$ of shape $$(I, J)$$, $$A_{ij}$$ is one element, and a matrix product is a contraction over the shared index:

$$
(A \cdot B)_{ij} = \sum_k A_{ik} B_{kj}
$$

Every entry is its own independent variable, so the derivative of one entry with respect to another is 1 if they are the same entry and 0 otherwise:

$$
\frac{\partial y_{b'j'}}{\partial y_{bj}} = \delta_{bb'}\delta_{jj'}, \qquad
\frac{\partial W_{i'j'}}{\partial W_{ij}} = \delta_{ii'}\delta_{jj'}, \qquad
\frac{\partial B_{j'}}{\partial B_{j}} = \delta_{jj'}
$$

These deltas are what make the sums collapse. Any sum over a primed index that contains a delta on that index has only one surviving term.

A gradient has the same shape as the tensor it is taken against. The one we will pass between layers gets its own letter, $$g$$, which is the bar of a layer's output:

$$
g = \bar{y} = \frac{\partial \mathcal{L}}{\partial y}, \qquad g_{bj} = \bar{y}_{bj} = \frac{\partial \mathcal{L}}{\partial y_{bj}}
$$

## The linear layer

The graph is too large to reason about node by node, so we group nodes into layers. A layer receives $$g$$ from upstream and has to produce two things: the gradient for its input, to send downstream, and the gradients for its own parameters.

{% include diagram.liquid name="bp-layer" caption="What a linear layer takes in and gives back. N is the batch size, I the input width, J the output width. The coloured arrows are the backward pass." %}

The forward pass, in matrix form and by index, with $$b$$ running over the batch:

$$
y = xW + B, \qquad y_{bj} = \sum_{i'} x_{bi'} W_{i'j} + B_j
$$

### Gradient for the input

Differentiating $$y_{bj}$$ by an input entry in the same batch row:

$$
\frac{\partial y_{bj}}{\partial x_{bi}} = \sum_{i'} \delta_{ii'} W_{i'j} = W_{ij}
$$

Different batch rows do not interact, so the chain rule only sums over the outputs $$j$$:

$$
\frac{\partial \mathcal{L}}{\partial x_{bi}}
= \sum_j \frac{\partial \mathcal{L}}{\partial y_{bj}} \frac{\partial y_{bj}}{\partial x_{bi}}
= \sum_j g_{bj} W_{ij}
\quad\Rightarrow\quad
\boxed{\bar{x} = \frac{\partial \mathcal{L}}{\partial x} = g W^T}
$$

The shapes check: $$(N, J) \cdot (J, I) = (N, I)$$. This is what the layer sends downstream, and the next layer down receives it as its own $$g$$.

### Gradient for the weights

Here the indices need care. Fix the weight $$W_{ij}$$ we are differentiating against, and use primes for the entries of $$y$$ that might depend on it. Every batch row and every output is a candidate, so the chain rule sums over both:

$$
\frac{\partial \mathcal{L}}{\partial W_{ij}} = \sum_{b, j'} g_{bj'} \frac{\partial y_{bj'}}{\partial W_{ij}}
$$

The local derivative uses the deltas from above, and the bias does not depend on $$W$$:

$$
\frac{\partial y_{bj'}}{\partial W_{ij}}
= \sum_{i'} x_{bi'} \frac{\partial W_{i'j'}}{\partial W_{ij}} + \frac{\partial B_{j'}}{\partial W_{ij}}
= \sum_{i'} x_{bi'} \delta_{ii'} \delta_{jj'} + 0
= x_{bi} \delta_{jj'}
$$

Substituting, the $$j'$$ sum collapses and only the batch sum is left:

$$
\frac{\partial \mathcal{L}}{\partial W_{ij}} = \sum_b x_{bi}\, g_{bj}
\quad\Rightarrow\quad
\boxed{\bar{W} = \frac{\partial \mathcal{L}}{\partial W} = x^T g}
$$

Again the shapes check: $$(I, N) \cdot (N, J) = (I, J)$$. The input $$x$$ appears in this result, and it belongs to the forward pass. So the layer has to **cache $$x$$** on the way forward to be able to compute its weight gradient on the way back.

### Gradient for the bias

$$B_j$$ enters each $$y_{bj'}$$ additively, so $$\partial y_{bj'} / \partial B_j = \delta_{jj'}$$, and

$$
\frac{\partial \mathcal{L}}{\partial B_j} = \sum_{b, j'} g_{bj'} \delta_{jj'} = \sum_b g_{bj}
\quad\Rightarrow\quad
\boxed{\bar{B} = \frac{\partial \mathcal{L}}{\partial B} = \sum_b g}
$$

Nothing needs caching here. The bias gradient is just the upstream gradient summed over the batch.

### Where each sum comes from

The form of each result is decided by one question: if I nudge this variable, which entries of $$y$$ move?

{% include diagram.liquid name="bp-fanout" caption="The grid is y, with one row per batch element and one column per output. The highlighted cells are the only ones that depend on the variable being nudged, and the chain rule sums over exactly those. The last panel is the activation layer, derived next." %}

## The activation layer

An activation has no parameters and applies a scalar function $$f$$ to each entry, so $$x$$ and $$y$$ have the same shape:

$$
y_{bj} = f(x_{bj})
$$

{% include diagram.liquid name="bp-activation" caption="What an activation layer takes in and gives back. With no parameters, its only backward job is to turn g into the gradient for its input." %}

For the local derivative, take any output entry against any input entry:

$$
\frac{\partial y_{b'j'}}{\partial x_{bj}}
= f'(x_{b'j'}) \frac{\partial x_{b'j'}}{\partial x_{bj}}
= f'(x_{bj})\, \delta_{bb'} \delta_{jj'}
$$

Only $$y_{bj}$$ is affected by $$x_{bj}$$. The chain rule's sum over $$b', j'$$ has one surviving term:

$$
\frac{\partial \mathcal{L}}{\partial x_{bj}}
= \sum_{b', j'} g_{b'j'} \frac{\partial y_{b'j'}}{\partial x_{bj}}
= g_{bj} f'(x_{bj})
\quad\Rightarrow\quad
\boxed{\bar{x} = \frac{\partial \mathcal{L}}{\partial x} = g \odot f'(x)}
$$

where $$\odot$$ is the element-wise product. $$f'$$ is evaluated at the input, so this layer also needs its cached $$x$$.

## All the rules

| Layer      | Forward        | Sent downstream, $$\bar{x}$$ | Parameter gradients                                   | Cached |
| ---------- | -------------- | ---------------------------- | ----------------------------------------------------- | ------ |
| Linear     | $$y = xW + B$$ | $$g W^T$$                    | $$\bar{W} = x^T g$$, $$\bar{B} = \textstyle\sum_b g$$ | $$x$$  |
| Activation | $$y = f(x)$$   | $$g \odot f'(x)$$            | none                                                  | $$x$$  |

A network is a chain of these. The forward pass fills the caches, the loss produces the first $$g$$, and each layer in reverse order converts the $$g$$ it receives into the $$g$$ for the layer below it.

{% include diagram.liquid name="bp-pass" caption="A full pass. Each layer only talks to its neighbours: it needs its own cache and the gradient arriving from upstream." %}

The rules translate almost symbol for symbol:

```python
import numpy as np

class Linear:
    def __init__(self, I, J):
        self.W = np.random.randn(I, J)
        self.B = np.random.randn(J)

    def forward(self, x):                # x: (N, I)
        self.x = x                       # needed later for dW
        return x @ self.W + self.B       # y: (N, J)

    def backward(self, g):               # g = dL/dy: (N, J)
        self.dW = self.x.T @ g           # (I, J)
        self.dB = g.sum(axis=0)          # (J)
        return g @ self.W.T              # dL/dx: (N, I)

class Activation:
    def __init__(self, f, df):
        self.f, self.df = f, df

    def forward(self, x):
        self.x = x                       # f' is evaluated at the input
        return self.f(x)

    def backward(self, g):
        return g * self.df(self.x)       # element-wise
```

I checked this against central finite differences on a Linear, tanh, Linear network with a squared-error loss. Over every entry of the input, weights and biases, the largest disagreement with the analytic gradient was about $$4 \times 10^{-9}$$.

## What never gets built

The full Jacobian $$\partial y / \partial x$$ of a linear layer is a tensor of shape $$(N, J, N, I)$$. Backpropagation never constructs it. Each layer only computes the product of $$g$$ with that Jacobian, and the index work above shows the product is one matrix multiply, or one element-wise multiply. The memory cost is the caches.

This is also what an autograd engine does. It records the operations during the forward pass, and for each operation it holds a rule of exactly this form: given the gradient of the output, return the gradient of the inputs. My [Java MLP from school](/projects/JavaNN_project/) has the same backward pass, written by hand for one architecture. Autograd generalises it to any graph of operations that have such a rule.
