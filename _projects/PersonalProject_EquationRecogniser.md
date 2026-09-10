---
layout: page
title: Handwritten equation solver
description: IB Computer Science Personal Project — draw an equation, two CNNs read it, Python solves it
img: assets/img/projects/personal_project_equation_sample.jpg
importance: 5
category: fun
related_publications: false
github: https://github.com/Aadipat/FinalProductPersonalProject
---

My IB Computer Science Personal Project, kept short since it was very much a school project: draw a simple equation like `2 + 3` by hand in the browser, and have it recognised and solved.

{% include figure.liquid path="assets/img/projects/personal_project_equation_sample.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="An actual sample from the repo's FINAL_PRODUCT_USERDRAWN folder — the p5.js canvas I drew on." %}

### How it works

A [p5.js](https://p5js.org/) canvas (`final_product.html` + `load_drawn_digit_data.js`) splits the drawing area into three guided panels — digit, operator, digit — and saves each as a JPG when I hit "Predict". From there, two separate Keras CNNs take over: one trained on MNIST for digits, one trained on a small hand-drawn dataset I made myself for the four operators.

```python
# making_digitrecog_model.py — digit CNN, trained on MNIST
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

The operator model is the same shape, trained instead on `operators/` — a folder of `divide` / `minus` / `multiply` / `plus` symbols I drew myself with `image_dataset_from_directory`, since no MNIST-style dataset exists for "+" and "×" drawn by hand.

Once both models are trained, `final_recog_program.py` crops each saved panel to 28×28, feeds digit panels to one model and operator panels to the other, and stitches the predicted sequence into an answer:

```python
# functions_for_equation_recog.py
def find_equation_answer(equation):
    answer = 0
    previous_operator = 'none'
    for i in range(len(equation)):
        if isinstance(equation[i], str):
            previous_operator = equation[i]
        elif isinstance(equation[i], int):
            if previous_operator == 'plus':
                answer = answer + equation[i]
            elif previous_operator == 'minus':
                answer = answer - equation[i]
            elif previous_operator == 'multiply':
                answer = answer * equation[i]
            elif previous_operator == 'divide':
                answer = answer / equation[i]
            else:
                answer = answer + equation[i]
    return answer
```

Not much more to say about it — a fairly standard "two small CNNs glued together with some string" project, but it was my first time building a full pipeline from a hand-rolled dataset through to a working end-to-end prediction, which is really what the IB Personal Project was for.
