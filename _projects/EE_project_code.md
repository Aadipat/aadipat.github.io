---
layout: page
title: Rubik's cube simulation API
description: An IB Extended Essay project modelling NxN Rubik's cube rotations with vectors and matrices, in Python and C++
img: assets/img/projects/ee_cube_solved.png
importance: 4
category: fun
related_publications: false
github: https://github.com/Aadipat/EEProject
---

This was the code behind my IB Extended Essay: modelling how an NxN Rubik's cube transforms under face turns, using vectors, planes, and rotation matrices. Nothing fancy — just the maths I was writing up in the essay, turned into something I could actually scramble and watch move.

### The model

Every sticker on the cube is a small vector attached to a "piece" position — a `(x, y, z)` coordinate on an `n × n × n` grid centred at the origin. A face turn (say, the right face) is a 90° rotation of every piece whose coordinate satisfies a plane equation like $x = \frac{n-1}{2}$, around that plane's normal axis. So "turning a face" is just: select the pieces on one side of a plane, multiply their position and sticker vectors by a 3×3 rotation matrix.

```python
class Transform:
    def __init__(self, rotation_plane_coefficients, direction):
        self.rotation_plane_coefficients = rotation_plane_coefficients
        self.direction = direction

    def find_rotation_matrix(self, c):
        # c tells us which axis (x/y/z) this face turn rotates around
        if c == 0:
            rotation_matrix = np.array([[1, 0, 0],
                                         [0, 0, -1],
                                         [0, 1, 0]])
            if self.direction == -1:
                rotation_matrix = la.inv(rotation_matrix)
        # ... same idea for c == 1 (y-axis) and c == 2 (z-axis)
        return rotation_matrix
```

Scrambling the cube is just applying a handful of random `Transform`s from the set of all possible face turns, and re-grouping pieces into layers afterwards so the next turn selects the right slice again. Here's what that actually looks like, rendered straight from `cube.py`'s own `Piece`/`Rubiks_Cube` classes — each arrow is one sticker's colour vector:

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/projects/ee_cube_solved.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Solved 3×3 — every sticker vector still points along its original face normal." %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/projects/ee_cube_scrambled.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Same cube after 6 random face turns, produced by cube.py's own rotation matrices." %}
  </div>
</div>

### A second attempt, in C++

I later rewrote the whole thing in C++ (`cube.cpp`/`cube.h`) with proper classes for `Position`, `Colour`, `Face`, `Piece`, `Matrix`, `Transformation`, and `Cube` — partly for speed, partly to see the same rotation logic look completely different in a statically-typed, OOP style:

{% raw %}

```cpp
Transformation::Transformation(string layer_move) : layer_move{layer_move} {
    // "R" turns the right layer clockwise, "R*" counter-clockwise, etc.
    // (is_p is parsed from a trailing '*' just above this, trimmed here)
    if (layer_move[0] == 'R' || layer_move[0] == 'L') {
        transformation_matrix = is_p
            ? Matrix({{1,0,0}, {0,0,-1}, {0,1,0}})
            : Matrix({{1,0,0}, {0,0,1}, {0,-1,0}});
    }
    // ... U/D and F/B layers follow the same pattern
}
```

{% endraw %}

I also tried to push further than the EE actually needed and write a `solve()` — a beginner's-method layer-by-layer solver. It's honest to say it's unfinished: compiling `cube.cpp` and running it today confirms the rotation engine is solid (it correctly detects a solved vs. scrambled state), but the solver itself doesn't yet get there:

```text
$ g++ -std=c++17 -O2 -o demo_solve demo_solve.cpp cube.cpp && ./demo_solve
solved 3x3 built. is_solved() = true
Scramble:
R1 D1* F1 U2 U1 U2 F1* F2*
after scramble(8), is_solved() = false
human_solve() found a solution using 0 face turns
after solve(), is_solved() = false
```

Which is a pretty accurate snapshot of an EE project: the part I needed for the essay — modelling rotations as matrices and proving the group structure behind face turns — works properly, and the "wouldn't it be cool if it could solve itself too" stretch goal is exactly as unfinished as you'd expect from a high-schooler chasing a side quest.
