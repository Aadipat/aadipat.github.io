---
layout: post
title: "QEC Model Zoo accepted at NeurIPS 2026"
date: 2026-09-25
description: A benchmark for comparing AI decoders for quantum error correction on equal terms.
tags: quantum-computing machine-learning research
categories: publications
related_publications: true
---

Our paper **QEC Model Zoo: Democratizing AI-enhanced Quantum Error Correction** has been accepted at NeurIPS 2026, in the Track on Evaluations and Datasets. It's joint work with Yuandou Wang and Max de Groot (the three of us contributed equally), Floris Geerts and Rihan Hai.

### The problem

Quantum error correction (QEC) spreads one logical qubit across many physical qubits and repeatedly measures parity checks (stabilizers). The outcomes, called syndromes, show where errors probably happened without measuring the protected state. A **decoder** reads the syndrome history and predicts whether a logical error occurred. As a learning problem, that's supervised classification: input $$x$$ is a binary measurement record, label $$y$$ is a logical flip, and the decoder learns $$\hat{y} = f(x)$$.

Many architectures have been tried for $$f$$: MLPs, LSTMs, GNNs, transformers, and reinforcement learning. The issue is that papers use different datasets, input representations, preprocessing and metrics, so reported improvements are hard to compare. It's often unclear whether a gain comes from the model or from the setup around it.

### What the Model Zoo does

- **One benchmark over real hardware data.** Nine public experimental datasets (Google, DiCarlo, Lukin and Vandersypen labs), covering repetition, surface and color codes on superconducting, neutral-atom and spin-qubit platforms, plus three simulated settings including bivariate bicycle (qLDPC) codes.
- **Separate tasks for full decoders and pre-decoders.** Full decoders predict the logical error directly. Pre-decoders clean up local errors and hand the rest to a classical decoder such as PyMatching or Chromobius. These need different protocols and metrics, so we keep them apart.
- **Shot-level feature mappings.** Each experimental shot becomes a tabular row of engineered features (rates, timing, spatial coincidences, leakage), while the raw measurement sequence is kept for temporal models. This lets tree ensembles, tabular foundation models and neural sequence models be compared on the same data.
- **Standardised metadata.** Datasets are described with the Croissant schema, so they can be found, loaded and reused under one evaluation protocol.

### Main findings

- **Representation matters as much as architecture.** On a 200,000-shot DiCarlo repetition-code dataset, adding engineered features and LSTM encodings raises TabM accuracy from 57.4% to 79.2%.
- **There's no single best decoder.** LSTMs and classical decoders win in several settings. But among decoders under 1 µs per shot in batched inference, TabM and tree ensembles are the most accurate on eight of the nine real datasets.
- **Hybrid pipelines help.** On color codes, a learned pre-decoder in front of a classical decoder lowers both the logical error rate and total decoding time.

The practical result is a way to choose a decoder for a given code, hardware platform and latency budget, instead of comparing numbers across papers that aren't comparable.

### Code and data

- Code: [github.com/InfiniData-Lab/QEC-model-zoo](https://github.com/InfiniData-Lab/QEC-model-zoo)
- Processed data: [Kaggle](https://www.kaggle.com/datasets/infinidata/qec-model-zoo-data)

This builds on [QEC lake](/blog/2026/qec-lake/), our data lake for QEC datasets. The lake collects and prepares the data, and the Model Zoo is where decoders are trained and compared on it.

Paper, abstract and PDF are on the [publications page](/publications/#wang2026qecmodelzoo) {% cite wang2026qecmodelzoo %}.
