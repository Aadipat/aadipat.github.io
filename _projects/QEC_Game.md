---
layout: page
title: "QEC Tiles: Stabilizer Rush"
description: A browser game that teaches quantum error correction decoding, built for the VLDB 2026 QEC lake demo
img:
importance: 2
category: work
related_publications: true
github: https://github.com/Aadipat/VLDB_Demo_QEC_Game
---

When we demoed [QEC lake](/publications/#patwardhan2026qeclake) at VLDB 2026, we needed a way to explain quantum error correction (QEC) to conference attendees in under a minute — without a whiteboard lecture on stabilizer codes. So I built **QEC Tiles: Stabilizer Rush**, a small browser game where you *are* the decoder.

### The idea behind the game

In a QEC code, physical **data qubits** hold the encoded information, and a second set of **ancilla (check) qubits** are periodically measured to detect errors without directly measuring — and destroying — the data. The pattern of ancilla outcomes is called the **syndrome**, and decoding is the task of inferring which data qubits flipped from that syndrome alone. This is exactly the problem AI-based decoders (the kind [QEC lake](/publications/#patwardhan2026qeclake) supplies training data for) are trained to solve.

QEC Tiles turns that into a real-time game: a tile representing a round of syndrome measurements falls down the screen, and you have to lock in which data qubits you think flipped before it lands. Play across difficulty levels with tunable error rates and speeds, and the game tracks your accuracy, streaks, and survival across rounds on a leaderboard.

### How it's built

It's a self-contained vanilla JS/HTML/CSS single-page app (`index.html`, `style.css`, `game.js`) with six screens — tutorial, menu, countdown, game, results, and leaderboard — and a built-in tutorial mode that walks new players through the difference between data and check qubits using a live demo board before they play for score.

The part that matters most for the pedagogy isn't the falling-tile animation, it's that the syndrome the player sees is computed exactly the way a real QEC code computes it: an ancilla "fires" if it sits next to an **odd** number of flipped data qubits — a parity check, XOR under the hood:

```javascript
// game.js — a hidden set of flipped data qubits ("truth") becomes
// the syndrome the player actually sees: which ancillas fired.
function computeFiredAncillas(truthSet) {
  const fired = [];
  session.geo.ancillas.forEach((a, aIdx) => {
    let count = 0;
    a.qubits.forEach(q => { if (truthSet.has(q)) count++; });
    if (count % 2 === 1) fired.push(aIdx);
  });
  return fired;
}
```

The game only ever shows the player `fired`, never `truthSet` — the same information asymmetry a real decoder faces: you see which checks complained, not which qubits actually flipped, and two different errors can trigger the exact same syndrome. That's precisely why decoding is hard enough to need AI in the first place, and why it's fun (or at least tense) as a game.

### Why it worked as a demo

Letting people *play* a few rounds of decoding, rather than describing syndromes and stabilizers abstractly, made the core difficulty of QEC decoding click almost immediately: you're inferring hidden errors from indirect, noisy evidence, under time pressure — which is precisely why this is a machine learning problem in practice, and precisely the kind of training data QEC lake exists to provide {% cite patwardhan2026qeclake %}.
