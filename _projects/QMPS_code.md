---
layout: page
title: Lightweight Quantum MPS simulator
description: A from-scratch tensor-network simulator for quantum circuits, built around Matrix Product States
img:
importance: 1
category: work
related_publications: false
github: https://github.com/Aadipat/QuantumTNSimLib_V1
---

Built in my BSc second year, `QuantumTNSimLib_V1` is a Python library for simulating quantum circuits using **tensor networks** — specifically, **Matrix Product States (MPS)** — instead of a full state vector.

### Why not just use a state vector?

The naive way to simulate a quantum circuit is to track the full statevector: an $n$-qubit system needs $2^n$ complex amplitudes, so exact simulation becomes infeasible well before $n = 50$. Most interesting circuits, however, don't need that much bookkeeping — entanglement between distant qubits is often limited, and MPS exploits exactly that. Each qubit is represented as a small tensor, chained together into a 1-D network, and the amount of entanglement the state can hold is capped by a tunable **bond dimension** $\chi$. Truncating $\chi$ trades a controllable amount of accuracy for a large reduction in memory and compute — the same idea that lets DMRG-style methods simulate large 1-D quantum systems on a laptop.

### What's in the repo

- **`qtn_sim`** — the core simulator: MPS state representation, gate application via tensor contraction and SVD-based truncation, and measurement/sampling routines.
- **`CircuitEmbedding`** — utilities for compiling a gate-level circuit description into the sequence of local tensor operations the simulator executes.
- **`baseLineExperiments`** — benchmarks comparing the MPS simulator against exact/statevector baselines on standard circuits, to check where the tensor-network approach actually pays off.
- **`metricEvolution`** — tracking how fidelity, bond dimension, and truncation error evolve as circuit depth grows, which is the key diagnostic for whether MPS is a good fit for a given circuit.
- **`testSuite.py`** — correctness tests against known circuit outputs.

### Checking it against the real thing

Every simulator is only as good as its correctness tests, so `testSuite.py` builds the same circuit twice — once on `QuantumMPS`, once on Qiskit — and checks the resulting probabilities match:

```python
# testSuite.py
def testGHZ(self):
    n, bond_dimension = 3, 3
    q = QuantumMPS(n, bond_dimension)
    q.apply(HGate(), [0])
    q.apply(CNOTGate(), [0, 1])
    q.apply(CNOTGate(), [1, 2])
    o = svQiskitStyleToMine(q.get_state_vector())

    # Check with qiskit
    circ = QuantumCircuit(n)
    circ.h(0); circ.cx(0, 1); circ.cx(1, 2)
    result = np.square(Statevector(circ).data)

    for p in range(len(result)):
        self.assertTrue(np.isclose(result[p], o[p]))
```

Installing the published wheel and running the suite today, it still passes cleanly against Qiskit's exact statevector — GHZ and W states, QFT (with and without adjacent-swap optimisation), Toffoli, deep mixed circuits, and randomised circuits included:

```text
$ pip install qtn_sim-0.1.0-py3-none-any.whl qiskit opt_einsum && python testSuite.py -v
test1XQubit ... ok
test2SWAPGateClass ... ok
test2SWAPQubit ... ok
test3TOFFOLIQubits ... ok
testCU ... ok
testDeeperCircuit ... ok
testDeeperWholeCircuit ... ok
testGHZ ... ok
testLargeRandom ... ok
testNLargeHs ... ok
testOptEinsumMPS_ghz ... ok
testQFT ... ok
testQFTWithAdjacentSwaps ... ok
testQFTWithAdjacentSwapsAndWithout ... ok
testRY ... ok
testSequentialMPS_ghz ... ok
testSwapCnotHCircuit ... ok
testToffoliCnotH ... ok
testW ... ok

Ran 19 tests in 1.510s

OK
```

### Why it mattered later

This project was my first hands-on encounter with the core question that now sits at the center of my honours research: *which classical representation should you use to simulate a given quantum circuit?* MPS is one answer for low-entanglement circuits; relational/SQL-based simulation (the subject of [InferQ](/publications/#ilinescu2026inferq)) is another, competitive for sparse, highly structured circuits. Both are really about the same underlying idea — picking the right data structure and execution engine for the workload — just applied to very different corners of quantum simulation.
