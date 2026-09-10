---
layout: page
title: MLP package for JAVA
description: A multilayer perceptron built from scratch in Java — matrices, backpropagation, and all
img:
importance: 1
category: fun
related_publications: false
github: https://github.com/Aadipat/Neural-Network-MLP-
---

`Neural-Network-MLP-` is a Multi-Layer Perceptron (MLP) library I wrote from scratch in Java back in 10th grade, with no ML frameworks involved — just matrices and calculus.

### What it does

The library is a textbook implementation of a feedforward neural network trained by **backpropagation** and **gradient descent**:

- **`matrix/`** — a small linear-algebra layer (matrix multiplication, transposition, element-wise operations) that everything else is built on, since Java has no NumPy equivalent built in.
- **`neuralnetwork/`** — the network builder itself: configurable layer sizes, activation functions, forward propagation, and the backward pass that computes gradients layer by layer via the chain rule.
- **`saved_models/`** — serialization so a trained network's weights can be saved and reloaded instead of retraining from scratch every run.
- **`TestNN.java`** — a runnable demo that builds a `[2, 2, 1]` network and trains it to learn the **NAND** gate.

One thing I'm still a little proud of: rather than trusting a single random weight initialisation, `find_best_trained_model` trains many independently-initialised networks and keeps the best one — a crude but genuinely sensible way to dodge bad local minima, for a 10th-grader who hadn't yet heard the term "random restarts":

```java
// NeuralNetwork.java
public static NeuralNetwork find_best_trained_model(int number_of_random_models,
        List<Matrix> inputs, List<Matrix> targets,
        int epochs, double learning_rate, int[] nodes_array, int learnPerEpoch) {

    NeuralNetwork[] nns = new NeuralNetwork[number_of_random_models];
    for (int i = 0; i < nns.length; i++) nns[i] = new NeuralNetwork(nodes_array);

    Matrix[] costs = new Matrix[nns.length];
    for (int i = 0; i < nns.length; i++) {
        nns[i].trainModel(inputs, targets, epochs, learning_rate, learnPerEpoch, false);
        costs[i] = nns[i].getCost(inputs, targets, true);
        System.out.println("Random model " + i + ": " + costs[i]);
    }
    // ...picks the model with lowest cost out of costs[]
}
```

Compiling and running `TestNN.java` today, unmodified, against 1000 random restarts of 1000 epochs each:

```text
$ javac TestNN.java matrix/Matrix.java neuralnetwork/NeuralNetwork.java && java testnn
[[ 0.6420457879740459]]   <- untrained network's 4 NAND outputs, before training
[[ 0.6157170892840446]]
[[ 0.6370588862589173]]
[[ 0.6066007502370196]]
Random model 0: [[ 0.8053762221242432]]
Random model 1: [[ 0.7941905248023039]]
...                                       (1000 random restarts, logged)
Random model 999: [[ 0.7559249641656174]]
Best model is of index: 780
It has cost [[ 0.6032754191047551]]
[[ 0.6551095388541704]]   <- best model's 4 NAND outputs, after training
[[ 0.7163811881922523]]
[[ 0.7342605682034293]]
[[ 0.7969148407206734]]
```

Worth being honest about: that's not a converged NAND gate — the outputs should land near `[1, 1, 1, 0]` for inputs `(1,1), (1,0), (0,1), (0,0)`, and a cost of 0.60 after 1000 restarts says the default hyperparameters in `TestNN.java` just aren't enough to get there. The mechanics (forward pass, backprop, gradient descent, random-restart selection) all genuinely work; whether the _specific_ configuration in the repo converges on NAND is a separate question the code itself doesn't quite answer.

### Why it's here

This was one of my first real programming projects — the point where "I understand how neural networks work in theory" turned into "I can derive the backprop equations myself and get a working classifier out the other end." Implementing forward and backward passes by hand, without `autograd` to lean on, made the mechanics of training (gradients, learning rate, the vanishing-gradient failure mode when you get initialization wrong) very concrete in a way that using a framework doesn't. It's also the earliest thread connecting to what I work on now — the ML/AI side of my current research on learned simulator selection.
