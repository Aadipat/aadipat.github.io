---
layout: page
title: ML models playground
description: Classic ML algorithms implemented from scratch in NumPy — some finished, some very much not
img: assets/img/projects/ml_lda_boundary.png
importance: 2
category: fun
related_publications: false
github: https://github.com/Aadipat/ML
---

A repository of classic machine learning algorithms implemented from scratch in NumPy — no scikit-learn, just the maths. Fair warning going in: this one never got finished. Some files are complete, working classifiers; others are just stub functions I never got back to. I think that's honestly a more useful thing to show than a polished demo, so here's exactly where it stands.

### What actually works

`pde.py` implements three generative classifiers the textbook way — model each class as a Gaussian, then classify by comparing class-conditional densities:

```python
# pde.py
class LDA:
    def __init__(self, features, targets):
        self.c1 = features[np.where(targets == 0)]
        self.c2 = features[np.where(targets == 1)]
        self.mean1, self.cov_m1 = MLE_Mean(self.c1), MLE_Covariance_Matrix(self.c1)
        self.mean2, self.cov_m2 = MLE_Mean(self.c2), MLE_Covariance_Matrix(self.c2)
        # LDA shares one covariance matrix between classes -> linear boundary
        self.cov_m = np.add(self.cov_m1, self.cov_m2) / 2
        self.discriminant = lambda x: (
            (len(self.c1)/(len(self.c1)+len(self.c2))) * class_conditional_pdf(x, self.mean1, self.cov_m)
            - (len(self.c2)/(len(self.c1)+len(self.c2))) * class_conditional_pdf(x, self.mean2, self.cov_m)
        )
```

`QDA` is the same idea but keeps each class's own covariance matrix, which is what lets its boundary curve instead of staying a straight line. Running both of these on the file's own toy dataset and its own `draw()` function (I just redirected the plot to a file instead of `plt.show()`) gives real output:

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/projects/ml_lda_boundary.png" class="img-fluid rounded z-depth-1" zoomable=true caption="LDA: shared covariance forces a straight decision boundary." %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/projects/ml_qda_boundary.png" class="img-fluid rounded z-depth-1" zoomable=true caption="QDA: separate covariances let the boundary curve around the near cluster." %}
  </div>
</div>

The fuzzy black region rather than a crisp line is a genuine artefact of how `draw()` finds the boundary — it scatters every grid point where `abs(discriminant(x))` is under some tiny tolerance, which also lights up far from the data wherever both class densities are near-zero. A real bug, left exactly as it shipped.

`npde.py` does the non-parametric side of the same problem — Parzen-window density estimation and k-nearest-neighbours — and `nn.py` is a small from-scratch NumPy MLP (the same backprop-by-hand approach as my [Java MLP project](/projects/JavaNN_project/), just in Python). Training it on XOR for 3000 epochs gives a real, if slightly slow, convergence curve:

{% include figure.liquid path="assets/img/projects/ml_nn_xor_loss.png" class="img-fluid rounded z-depth-1" zoomable=true caption="nn.py's XOR training curve — mean squared error over 3000 epochs, learning rate 0.5." %}

### What doesn't

`lsdis.py`, `nlsdis.py`, and `usup.py` are the honest "never got finished" part — empty function stubs I'd scaffolded out and never filled in:

```python
# lsdis.py
def linear_regression():
    return

def logistic_regression():
    return

def support_vector_machine():
    return

# nlsdis.py
class decision_tree():
    def __init__():
        return

def random_forest():
    return

# usup.py
def K_Means():
    return

def PCA():
    return
```

The plan was clearly to build out the full "greatest hits" of classic ML — linear/logistic regression, SVMs, decision trees, k-means, PCA — alongside the generative and non-parametric classifiers that did get finished. It's a good snapshot of a repo built for learning rather than shipping: the algorithms I sat down and actually derived (LDA/QDA, Parzen/kNN, backprop) are complete and correct; the ones I already understood conceptually and was just going to "quickly implement later" are, predictably, still `return`.
