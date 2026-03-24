# 🧬 Cellular Automata - A Comprehensive Guide to learn

## Table of Contents

1. [What Is a Cellular Automaton?](#1-what-is-a-cellular-automaton)
2. [1D Elementary CA - Wolfram Rules](#2-1d-elementary-ca--wolfram-rules)
3. [Neighbourhoods &amp; Boundary Conditions](#3-neighbourhoods--boundary-conditions)
4. [2D CA - Conway&#39;s Game of Life](#4-2d-ca--conways-game-of-life)
5. [Totalistic &amp; Outer-Totalistic Rules](#5-totalistic--outer-totalistic-rules)
6. [CA Classification (Wolfram Classes)](#6-ca-classification-wolfram-classes)
7. [Entropy, Complexity &amp; Measuring CA Behaviour](#7-entropy-complexity--measuring-ca-behaviour)
8. [Beyond Binary - Multi-State CA](#8-beyond-binary--multi-state-ca)
9. [Reversible CA &amp; Conservation Laws](#9-reversible-ca--conservation-laws)
10. [Continuous CA &amp; Reaction-Diffusion Systems](#10-continuous-ca--reaction-diffusion-systems)

---

## Prerequisites

```bash
pip install numpy matplotlib scipy imageio
```

All code runs in Python 3.10+. Each section builds on the last, but you can jump around if you already know the basics.

---

## 1. What Is a Cellular Automaton?

A **Cellular Automaton (CA)** is a discrete computational model defined by four things:

| Component               | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| **Grid**          | A lattice of cells (1D, 2D, or nD)                          |
| **States**        | Each cell holds one value from a finite set, e.g.`{0, 1}` |
| **Neighbourhood** | The set of nearby cells that influence an update            |
| **Rule**          | A function:`(neighbourhood states) → new state`          |

Time moves in **discrete steps** - every cell updates simultaneously based on the rule.

The magic: impossibly complex global patterns can emerge from *trivially simple* local rules.

```
t=0:  . . . . ■ . . . .
t=1:  . . . ■ ■ ■ . . .
t=2:  . . ■ . . . ■ . .
t=3:  . ■ ■ . . . ■ ■ .
```

---

### 🔥 Challenge 1 - Think Before You Code

Before writing a single line, answer these:

1. If a CA has 2 states and looks at 3 cells (itself + 2 neighbours), how many distinct rules are possible?
2. What if it has 3 states and a 5-cell neighbourhood?
3. Why must *all* cells update at the same time (synchronously)?

*(Answers are at the bottom of this section — try first!)*

<details>
<summary>Answers</summary>

1. 3 cells × 2 states = 8 possible neighbourhood patterns → 2 possible outputs each = **2⁸ = 256 rules**
2. 5 cells × 3 states = 3⁵ = 243 neighbourhood patterns → 3 outputs each = **3²⁴³ ≈ 10^116 rules**
3. If cells updated sequentially, the order would matter and the rule would be less "local" - the result would depend on position in the update sequence, breaking spatial symmetry.

</details>

---

## 2. 1D Elementary CA - Wolfram Rules

Stephen Wolfram catalogued all 256 possible 1D binary CA with a 3-cell neighbourhood (left, self, right). Each rule is named by its **rule number** (0–255), derived from its binary representation.

### How Rule Numbers Work

```
Neighbourhood:  111  110  101  100  011  010  001  000
Rule 30 output:  0    0    0    1    1    1    1    0
Binary:                                         = 00011110 = 30
```

### Code: Visualise Any Elementary CA Rule

```python
import numpy as np
import matplotlib.pyplot as plt

def elementary_ca(rule_number: int, width: int = 101, steps: int = 50) -> np.ndarray:
    """
    Simulate a 1D elementary CA.
  
    Args:
        rule_number: Wolfram rule (0-255)
        width:       Number of cells
        steps:       Number of time steps
  
    Returns:
        2D array of shape (steps, width) — rows are time steps
    """
    # Decode rule into a lookup table
    rule_bits = np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)
  
    # Initialise: single live cell in the centre
    grid = np.zeros((steps, width), dtype=np.uint8)
    grid[0, width // 2] = 1
  
    for t in range(1, steps):
        # Pad with wrap-around (periodic boundary)
        row = np.pad(grid[t - 1], 1, mode='wrap')
  
        # Compute neighbourhood index for each cell
        # left=4, self=2, right=1  →  index into rule_bits
        left  = row[:-2]
        mid   = row[1:-1]
        right = row[2:]
        idx   = (left << 2) | (mid << 1) | right
  
        grid[t] = rule_bits[idx]
  
    return grid


def plot_ca(grid: np.ndarray, rule_number: int):
    plt.figure(figsize=(12, 6))
    plt.imshow(grid, cmap='binary', interpolation='nearest', aspect='auto')
    plt.title(f'Elementary CA — Rule {rule_number}', fontsize=14)
    plt.xlabel('Cell index')
    plt.ylabel('Time step →')
    plt.tight_layout()
    plt.savefig(f'rule_{rule_number}.png', dpi=150)
    plt.show()


# Try the famous ones
for rule in [30, 90, 110, 184]:
    grid = elementary_ca(rule_number=rule, width=201, steps=100)
    plot_ca(grid, rule)
```

### Must-See Rules

| Rule               | Famous For                                                         |
| ------------------ | ------------------------------------------------------------------ |
| **Rule 30**  | Chaotic, used by Mathematica's random number generator             |
| **Rule 90**  | Sierpiński triangle - self-similar fractal from XOR of neighbours |
| **Rule 110** | Proven**Turing complete** - can compute anything!           |
| **Rule 184** | Models traffic flow (cars moving right, can't overlap)             |
| **Rule 254** | Boring - every cell turns on once any neighbour is on              |

---

### 🔥 Challenge 2 — Rule Algebra

```python
# 1. Implement a function that takes TWO rules and returns their XOR-composite rule:
#    composite(rule_a, rule_b): for each neighbourhood, apply rule_a THEN rule_b
#    Does the order matter?

# 2. Find all "totalistic" rules among the 256 elementary rules —
#    i.e., rules whose output depends only on the SUM of the neighbourhood,
#    not the left-right arrangement.
#    Hint: neighbourhoods (1,0,0) and (0,0,1) have the same sum.

# 3. Starting from a random initial condition (50% alive),
#    which rules stabilise quickly? Which stay chaotic forever?
#    Write code to measure the "change rate" over time:
#    change_rate[t] = fraction of cells that differ between step t-1 and step t
```

---

## 3. Neighbourhoods & Boundary Conditions

### Neighbourhood Types (2D)

```
Von Neumann (r=1)      Moore (r=1)        Von Neumann (r=2)
    . X .               X X X               . . X . .
    X O X               X O X               . X X X .
    . X .               X X X               X X O X X
                                            . X X X .
   4 neighbours        8 neighbours         . . X . .
                                           12 neighbours
```

For **1D**, the r-neighbourhood is just the `2r+1` cells centred on the current cell.

### Boundary Conditions

```python
import numpy as np

def apply_boundary(grid: np.ndarray, mode: str = 'periodic') -> np.ndarray:
    """
    Pad a 2D grid to handle boundary conditions.
  
    Modes:
      'periodic'  — wrap around (toroidal)
      'fixed-0'   — dead cells outside
      'fixed-1'   — live cells outside
      'mirror'    — reflect at edges
    """
    if mode == 'periodic':
        return np.pad(grid, 1, mode='wrap')
    elif mode == 'fixed-0':
        return np.pad(grid, 1, mode='constant', constant_values=0)
    elif mode == 'fixed-1':
        return np.pad(grid, 1, mode='constant', constant_values=1)
    elif mode == 'mirror':
        return np.pad(grid, 1, mode='reflect')
    else:
        raise ValueError(f"Unknown boundary mode: {mode}")


# Visualise how boundary choice changes Rule 90 after 100 steps
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
modes = ['periodic', 'fixed-0', 'fixed-1', 'mirror']

for ax, mode in zip(axes, modes):
    # Adapt elementary_ca to use different boundaries
    rule_bits = np.array([(90 >> i) & 1 for i in range(8)], dtype=np.uint8)
    width, steps = 101, 50
    grid = np.zeros((steps, width), dtype=np.uint8)
    grid[0, width // 2] = 1
  
    for t in range(1, steps):
        if mode == 'periodic':
            row = np.pad(grid[t-1], 1, mode='wrap')
        elif mode == 'fixed-0':
            row = np.pad(grid[t-1], 1, constant_values=0)
        elif mode == 'fixed-1':
            row = np.pad(grid[t-1], 1, constant_values=1)
        else:  # mirror
            row = np.pad(grid[t-1], 1, mode='reflect')
  
        l, m, r = row[:-2], row[1:-1], row[2:]
        grid[t] = rule_bits[(l << 2) | (m << 1) | r]
  
    ax.imshow(grid, cmap='binary', aspect='auto')
    ax.set_title(f'Boundary: {mode}')
    ax.axis('off')

plt.suptitle('Rule 90 under different boundary conditions', fontsize=13)
plt.tight_layout()
plt.savefig('boundary_comparison.png', dpi=150)
plt.show()
```

---

### 🔥 Challenge 3 - Neighbourhood Explorer

```python
# 1. Implement a generalised r-neighbourhood for 1D CA:
#    step(row, r, rule_dict)  where rule_dict maps tuples → states.
#    Test it with r=2 (5-cell neighbourhood).

# 2. For 2D: implement BOTH Von Neumann and Moore step functions.

# 3. Create a "null boundary" — a grid that grows by 1 cell on each side
#    each step (infinite grid, but only track non-zero cells).
#    Hint: use a dict {(x,y): state} instead of a numpy array.
```

---

## 4. 2D CA - Conway's Game of Life

Conway's Game of Life (1970) is a 2D CA with Moore neighbourhood (8 neighbours) and 4 rules:

| Condition                            | Outcome                          |
| ------------------------------------ | -------------------------------- |
| Live cell, 2 or 3 live neighbours    | **Survives**               |
| Live cell, < 2 live neighbours       | **Dies** (underpopulation) |
| Live cell, > 3 live neighbours       | **Dies** (overcrowding)    |
| Dead cell, exactly 3 live neighbours | **Born**                   |

### Efficient Convolution-Based Step

```python
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def life_step(grid: np.ndarray) -> np.ndarray:
    """One step of Conway's Game of Life using convolution."""
    # Kernel counts live neighbours (excludes centre cell)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
  
    neighbours = convolve2d(grid, kernel, mode='same', boundary='wrap')
  
    # Apply rules
    birth    = (grid == 0) & (neighbours == 3)
    survival = (grid == 1) & ((neighbours == 2) | (neighbours == 3))
  
    return (birth | survival).astype(np.uint8)


def random_grid(rows: int, cols: int, density: float = 0.3) -> np.ndarray:
    return (np.random.random((rows, cols)) < density).astype(np.uint8)


def animate_life(rows: int = 60, cols: int = 80, steps: int = 100, density: float = 0.3):
    grid = random_grid(rows, cols, density)
  
    fig, ax = plt.subplots(figsize=(10, 7))
    img = ax.imshow(grid, cmap='binary', interpolation='nearest')
    ax.axis('off')
    title = ax.set_title('Game of Life — Step 0', fontsize=12)
  
    def update(frame):
        nonlocal grid
        grid = life_step(grid)
        img.set_data(grid)
        title.set_text(f'Game of Life — Step {frame}')
        return [img, title]
  
    anim = animation.FuncAnimation(fig, update, frames=steps,
                                    interval=80, blit=True)
    plt.tight_layout()
    plt.show()
    return anim


animate_life()
```

### Famous Patterns

```python
# Define classic Game of Life patterns as numpy arrays
PATTERNS = {
    'glider': np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ]),
    'blinker': np.array([
        [1, 1, 1]
    ]),
    'block': np.array([
        [1, 1],
        [1, 1]
    ]),
    'beacon': np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]),
    'r_pentomino': np.array([
        [0, 1, 1],
        [1, 1, 0],
        [0, 1, 0]
    ]),
    'glider_gun': np.array([  # Gosper's glider gun (partial — look up full 36-cell version)
        # ... (36 cells, spawn gliders endlessly)
    ])
}

def place_pattern(grid: np.ndarray, pattern: np.ndarray,
                   row: int, col: int) -> np.ndarray:
    """Stamp a pattern onto a grid at position (row, col)."""
    h, w = pattern.shape
    grid[row:row+h, col:col+w] = pattern
    return grid


# Watch a glider travel across the grid
def demo_glider():
    grid = np.zeros((30, 50), dtype=np.uint8)
    grid = place_pattern(grid, PATTERNS['glider'], 2, 2)
  
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(grid, cmap='binary', interpolation='nearest')
        ax.set_title(f'Step {i * 4}')
        ax.axis('off')
        for _ in range(4):
            grid = life_step(grid)
  
    plt.suptitle('Glider — moves diagonally every 4 steps', fontsize=13)
    plt.tight_layout()
    plt.savefig('glider.png', dpi=150)
    plt.show()

demo_glider()
```

---

### 🔥 Challenge 4 - Life Experiments

```python
# 1. PATTERN CLASSIFIER
#    Run any initial pattern for 500 steps. 
#    Classify the final state as one of:
#      - "still life"   (nothing changes)
#      - "oscillator"   (period p: repeats every p steps)
#      - "spaceship"    (pattern translates across grid)
#      - "chaotic"      (still changing at step 500)
#    Hint: store a history of grids and check for equality.

# 2. DENSITY PHASE TRANSITION
#    For initial densities from 0.01 to 0.99, run 200 steps.
#    Plot: final alive-cell fraction vs. initial density.
#    Is there a "critical density" where behaviour changes abruptly?

# 3. LIFE VARIANTS — implement these rule changes and observe:
#    a) "HighLife":  born on 3 or 6, survive on 2 or 3
#    b) "Day & Night": born on 3,6,7,8; survive on 3,4,6,7,8
#    c) "Seeds":    born on 2, nothing survives
#    How do the dynamics differ from standard Life?
#    Hint: parametrise life_step(grid, birth_set, survival_set)
```

---

## 5. Totalistic & Outer-Totalistic Rules

A **totalistic** rule depends only on the **sum** of all cell states in the neighbourhood (including self).

An **outer-totalistic** rule depends on the **current cell state** + **sum of neighbours** - this is what Game of Life uses.

### B/S Notation

Life rules are written as **B{born}/S{survive}**: e.g., Life = **B3/S23**

```python
def make_outer_totalistic_rule(birth: set, survive: set):
    """
    Returns a step function for any 2D outer-totalistic rule
    with Moore neighbourhood, periodic boundary.
  
    Example:
        step = make_outer_totalistic_rule(birth={3}, survive={2, 3})
        # This IS Conway's Game of Life
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # exclude self from neighbour count
  
    def step(grid: np.ndarray) -> np.ndarray:
        n = convolve2d(grid, kernel, mode='same', boundary='wrap')
        new_grid = np.zeros_like(grid)
        # Dead cells that should be born
        new_grid[(grid == 0) & np.isin(n, list(birth))] = 1
        # Live cells that survive
        new_grid[(grid == 1) & np.isin(n, list(survive))] = 1
        return new_grid
  
    return step


# Compare multiple rules side by side
rules = {
    'Life (B3/S23)':       make_outer_totalistic_rule({3}, {2, 3}),
    'HighLife (B36/S23)':  make_outer_totalistic_rule({3, 6}, {2, 3}),
    'Day&Night (B3678/S34678)': make_outer_totalistic_rule({3,6,7,8}, {3,4,6,7,8}),
    'Seeds (B2/S)':        make_outer_totalistic_rule({2}, set()),
    'Maze (B3/S12345)':    make_outer_totalistic_rule({3}, {1,2,3,4,5}),
}

np.random.seed(42)
initial = random_grid(50, 50, density=0.35)

fig, axes = plt.subplots(1, len(rules), figsize=(18, 4))
for ax, (name, step_fn) in zip(axes, rules.items()):
    g = initial.copy()
    for _ in range(50):
        g = step_fn(g)
    ax.imshow(g, cmap='binary')
    ax.set_title(name, fontsize=9)
    ax.axis('off')

plt.suptitle('Same initial grid, 50 steps, different B/S rules', fontsize=12)
plt.tight_layout()
plt.savefig('rule_comparison.png', dpi=150)
plt.show()
```

---

### 🔥 Challenge 5 - Rule Space Explorer

```python
# 1. RANDOM RULE SEARCH
#    Sample 1000 random B/S rules (random subsets of {0..8} for birth and survive).
#    For each, run 100 steps from a random 50% density grid.
#    Measure: final density, spatial variance, mean cluster size.
#    Plot a scatter of birth-set-size vs survive-set-size, 
#    coloured by final density.

# 2. IMPLEMENT 1D OUTER-TOTALISTIC CA
#    For radius r=1 (3 neighbours), the totalistic sum ranges from 0–3.
#    Enumerate ALL outer-totalistic rules and classify them
#    by Wolfram class (see Section 6).

# 3. ISOTROPIC RULES
#    A rule is "isotropic" if it's symmetric under rotation and reflection.
#    How many of the 2^18 possible outer-totalistic Moore-neighbourhood rules
#    are isotropic? Write a function to CHECK if a given rule is isotropic.
```

---

## 6. CA Classification (Wolfram Classes)

Wolfram identified four behavioural classes for CA, based on long-run dynamics:

```
Class I   — Fixed point.      All patterns collapse to uniform state.   e.g. Rule 0, 255
Class II  — Periodic.         Stable or oscillating structures.          e.g. Rule 4, 108
Class III — Chaotic.          Pseudo-random, sensitive to initial cond.  e.g. Rule 30, 126
Class IV  — Complex / Edge of chaos.  Long-lived, structure-rich.       e.g. Rule 110, Life
```

### Measuring Class Automatically

```python
def classify_ca_rule(rule_number: int, 
                     width: int = 200, 
                     steps: int = 200,
                     n_trials: int = 5) -> dict:
    """
    Heuristic classification of a 1D elementary CA rule.
  
    Returns a dict with measured features:
      - mean_density:        average fraction of live cells after burnin
      - transient_length:    steps until behaviour stabilises
      - lambda_param:        fraction of outputs that are 1 (Langton's λ)
      - change_rate_variance: variance of step-to-step change rates (high → chaotic)
    """
    rule_bits = np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)
  
    # Langton's lambda: fraction of rule outputs = 1
    lambda_param = rule_bits.sum() / 8.0
  
    change_rates = []
    densities = []
  
    for _ in range(n_trials):
        row = (np.random.random(width) < 0.5).astype(np.uint8)
  
        for t in range(steps):
            padded = np.pad(row, 1, mode='wrap')
            idx = (padded[:-2] << 2) | (padded[1:-1] << 1) | padded[2:]
            new_row = rule_bits[idx]
  
            change_rates.append(np.mean(new_row != row))
            densities.append(new_row.mean())
            row = new_row
  
    return {
        'lambda':               round(lambda_param, 3),
        'mean_density':         round(np.mean(densities[-50:]), 3),
        'change_rate_variance': round(np.var(change_rates[-50:]), 5),
        'mean_change_rate':     round(np.mean(change_rates[-50:]), 3),
    }


# Classify all 256 rules and plot lambda vs complexity
results = {}
for r in range(256):
    results[r] = classify_ca_rule(r, width=100, steps=100, n_trials=3)

lambdas      = [results[r]['lambda'] for r in range(256)]
change_vars  = [results[r]['change_rate_variance'] for r in range(256)]
densities    = [results[r]['mean_density'] for r in range(256)]

plt.figure(figsize=(10, 5))
sc = plt.scatter(lambdas, change_vars, c=densities, cmap='viridis', 
                  alpha=0.7, s=30)
plt.colorbar(sc, label='Final density')
plt.xlabel("Langton's λ")
plt.ylabel('Change-rate variance (proxy for chaos)')
plt.title('All 256 Elementary CA Rules — λ vs Complexity')

# Annotate famous rules
for rule in [30, 90, 110, 184, 0, 255]:
    plt.annotate(f'R{rule}', 
                  (results[rule]['lambda'], results[rule]['change_rate_variance']),
                  fontsize=8, ha='center')

plt.tight_layout()
plt.savefig('rule_classification.png', dpi=150)
plt.show()
```

---

### 🔥 Challenge 6 — Edge of Chaos

```python
# The "edge of chaos" hypothesis (Langton, 1990):
# Maximum computational complexity occurs near λ ≈ 0.5.

# 1. Plot: for each λ value (bin into 20 buckets),
#    compute average change-rate VARIANCE across all rules in that bucket.
#    Does the peak variance align with λ ≈ 0.5?

# 2. Implement a "damage spreading" experiment:
#    - Run two identical grids from the SAME initial condition.
#    - Flip ONE cell in grid B.
#    - Run both grids for 200 steps, measure Hamming distance over time.
#    - For Class III rules, distance → 50% quickly.
#    - For Class I/II, distance → 0.
#    - For Class IV (Rule 110), what happens?

# 3. Relate to Lyapunov exponents:
#    The spreading rate of a single-bit perturbation is analogous to
#    the Lyapunov exponent in continuous dynamical systems.
#    Define a "CA Lyapunov exponent" and compute it for rules 30, 90, 110, 184.
```

---

## 7. Entropy, Complexity & Measuring CA Behaviour

### Shannon Entropy of CA States

```python
from collections import Counter
import math

def local_entropy(row: np.ndarray, k: int = 3) -> float:
    """
    Compute Shannon entropy of k-cell word frequencies in a row.
    Higher = more random/disordered.
    """
    words = [tuple(row[i:i+k]) for i in range(len(row) - k + 1)]
    counts = Counter(words)
    total = len(words)
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return entropy


def excess_entropy(rows: list, k: int = 8) -> float:
    """
    Excess entropy (a.k.a. effective complexity, or statistical complexity).
    Measures how much past states reduce uncertainty about future states.
  
    E = H(past) + H(future) - H(past, future)
      = Mutual Information between past and future.
  
    Here we approximate it as H(k-gram) - H(1-gram) * k
    which captures 'structure above raw randomness'.
    """
    all_words = []
    for row in rows:
        all_words.extend([tuple(row[i:i+k]) for i in range(len(row) - k + 1)])
  
    counts = Counter(all_words)
    total  = len(all_words)
    h_k    = -sum((c/total) * math.log2(c/total) for c in counts.values())
    h_1    = local_entropy(np.array(list(rows[0])), k=1) if rows else 0
  
    return max(0.0, h_1 * k - h_k)


# Compare entropy profiles of different Wolfram classes
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
rule_examples = [(0, 'Class I'), (4, 'Class II'), (30, 'Class III'), (110, 'Class IV')]

for ax, (rule, label) in zip(axes.flat, rule_examples):
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    width, steps = 200, 100
  
    grid = np.zeros((steps, width), dtype=np.uint8)
    grid[0, width // 2] = 1
    for t in range(1, steps):
        row = np.pad(grid[t-1], 1, mode='wrap')
        l, m, r = row[:-2], row[1:-1], row[2:]
        grid[t] = rule_bits[(l << 2) | (m << 1) | r]
  
    entropies = [local_entropy(grid[t], k=3) for t in range(steps)]
  
    ax.plot(entropies, color='steelblue', linewidth=1.5)
    ax.set_title(f'Rule {rule} — {label}')
    ax.set_xlabel('Time step')
    ax.set_ylabel('3-gram Shannon entropy')
    ax.set_ylim(0, 3.1)
    ax.grid(True, alpha=0.3)

plt.suptitle('Entropy over time — Wolfram Class comparison', fontsize=13)
plt.tight_layout()
plt.savefig('entropy_profiles.png', dpi=150)
plt.show()
```

---

### 🔥 Challenge 7 — Information Theory of CA

```python
# 1. LEMPEL-ZIV COMPLEXITY
#    The lz_complexity of a binary string measures "algorithmic complexity"
#    (how many distinct substrings appear, normalised by string length).
#    Implement it and compare across rules 0, 4, 30, 110.
#    Rule 30 should be close to a random string. Rule 0 should be near 0.

# 2. MUTUAL INFORMATION IN SPACE
#    For a 2D CA (Game of Life), compute the MI between
#    cell state at position (x, y) and (x + dx, y + dy) as a function of (dx, dy).
#    This is the "spatial correlation function".
#    Plot it as a heatmap. Does MI decay with distance?

# 3. CAUSAL STATES & ε-MACHINES
#    The theory of Computational Mechanics (Crutchfield & Young, 1989)
#    defines "causal states" — equivalence classes of histories that predict the future.
#    Given Rule 110 output, empirically cluster past k-grams by conditional
#    futures and count how many causal states emerge.
#    This is a research-level challenge!
```

---

## 8. Beyond Binary — Multi-State CA

### Cyclic CA

In a **Cyclic CA**, states are integers `0, 1, ..., k-1`. A cell advances from state `s` to `s+1 mod k` if it has at least one neighbour already at state `s+1`. This produces beautiful rotating spirals.

```python
def cyclic_ca_step(grid: np.ndarray, k: int, threshold: int = 1) -> np.ndarray:
    """
    One step of 2D cyclic CA.
  
    A cell at state s advances to (s+1) % k if it has
    >= threshold neighbours at state (s+1) % k.
  
    Args:
        grid:      2D int array with values in {0, ..., k-1}
        k:         number of states
        threshold: minimum excited neighbours to advance
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # Moore neighbourhood, 8 neighbours
    new_grid = grid.copy()
  
    for s in range(k):
        next_s = (s + 1) % k
        # Count how many neighbours are at state next_s
        is_next = (grid == next_s).astype(np.uint8)
        n_excited = convolve2d(is_next, kernel, mode='same', boundary='wrap')
  
        # Cells currently at s that have enough excited neighbours
        advance = (grid == s) & (n_excited >= threshold)
        new_grid[advance] = next_s
  
    return new_grid


def animate_cyclic(k: int = 6, threshold: int = 2, size: int = 100, steps: int = 150):
    grid = np.random.randint(0, k, (size, size))
    cmap = plt.cm.get_cmap('hsv', k)
  
    fig, ax = plt.subplots(figsize=(7, 7))
    img = ax.imshow(grid, cmap=cmap, vmin=0, vmax=k-1, interpolation='nearest')
    ax.axis('off')
    title = ax.set_title(f'Cyclic CA  k={k}  threshold={threshold}  step=0')
  
    def update(frame):
        nonlocal grid
        grid = cyclic_ca_step(grid, k, threshold)
        img.set_data(grid)
        title.set_text(f'Cyclic CA  k={k}  threshold={threshold}  step={frame}')
        return [img, title]
  
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=60, blit=True)
    plt.tight_layout()
    plt.show()

animate_cyclic(k=8, threshold=2)
```

### Brian's Brain

A 3-state CA: **OFF (0) → ON (1) → DYING (2) → OFF**. A dying cell always goes to OFF. An off cell turns on if it has exactly 2 live (ON) neighbours. Produces fast-moving "sparks".

```python
def brians_brain_step(grid: np.ndarray) -> np.ndarray:
    """
    States: 0=OFF, 1=ON, 2=DYING
    Rules:
      ON → DYING
      DYING → OFF
      OFF → ON  iff exactly 2 ON neighbours (Moore)
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
  
    on_neighbours = convolve2d((grid == 1).astype(np.uint8), kernel,
                                mode='same', boundary='wrap')
    new_grid = np.zeros_like(grid)
    new_grid[grid == 1] = 2          # ON → DYING
    # DYING → OFF (already 0)
    new_grid[(grid == 0) & (on_neighbours == 2)] = 1   # OFF → ON
    return new_grid
```

---

### 🔥 Challenge 8 — Multi-State Dynamics

```python
# 1. CYCLIC CA PHASE DIAGRAM
#    For all combinations of k ∈ {3,4,5,6,8,12} and threshold ∈ {1,2,3},
#    run 200 steps and classify the final pattern:
#      - "spiral waves" (self-organising, beautiful)
#      - "frozen"       (dynamics stall)
#      - "turbulent"    (no coherent structure)
#    What (k, threshold) pairs produce spirals?

# 2. WIRE WORLD
#    A 4-state CA that simulates electron flow in wires.
#    States: EMPTY, WIRE, ELECTRON_HEAD, ELECTRON_TAIL
#    Rules:
#      EMPTY        → EMPTY (always)
#      ELECTRON_HEAD → ELECTRON_TAIL
#      ELECTRON_TAIL → WIRE
#      WIRE          → ELECTRON_HEAD  iff 1 or 2 neighbouring HEADS; else WIRE
#    Implement it and build a simple "AND gate" out of wire patterns.

# 3. LENIA (preview of continuous CA)
#    Lenia is a multi-state CA with continuous states and smooth kernels.
#    As a stepping stone, implement a 16-state CA with the same
#    outer-totalistic structure as Life (but states 0-15).
#    What new behaviours emerge that binary CA cannot show?
```

---

## 9. Reversible CA & Conservation Laws

A CA is **reversible** if you can always reconstruct the previous state from the current one — i.e., the update function is a bijection.

### Second-Order (Margolus) CA

```python
def critters_step(grid: np.ndarray, phase: int) -> np.ndarray:
    """
    Critters rule: a reversible 2D CA using Margolus neighbourhood.
  
    The grid is divided into 2x2 blocks (phase 0: even, phase 1: odd blocks).
    Each 2x2 block is mapped to a new 2x2 block by a lookup table.
  
    The Critters rule conserves particle count (number of 1-cells).
    """
    rows, cols = grid.shape
    new_grid = grid.copy()
  
    # Critters lookup table (maps 4-bit block to 4-bit block)
    # Ordered as: [top-left, top-right, bottom-left, bottom-right] → same
    # Here we use a simple rotation rule for demonstration
    critters_table = {
        0b0000: 0b0000, 0b0001: 0b0010, 0b0010: 0b0001, 0b0011: 0b0011,
        0b0100: 0b1000, 0b0101: 0b1010, 0b0110: 0b1001, 0b0111: 0b1011,  # ← Corrected
        0b1000: 0b0100, 0b1001: 0b0110, 0b1010: 0b0101, 0b1011: 0b0111,
        0b1100: 0b1100, 0b1101: 0b1101, 0b1110: 0b1110, 0b1111: 0b1111,
    }
  
    offset = phase  # 0 or 1
    for r in range(offset, rows - 1, 2):
        for c in range(offset, cols - 1, 2):
            tl = grid[r,   c]
            tr = grid[r,   c+1]
            bl = grid[r+1, c]
            br = grid[r+1, c+1]
  
            block = (tl << 3) | (tr << 2) | (bl << 1) | br
            new_block = critters_table[block]
  
            new_grid[r,   c]   = (new_block >> 3) & 1
            new_grid[r,   c+1] = (new_block >> 2) & 1
            new_grid[r+1, c]   = (new_block >> 1) & 1
            new_grid[r+1, c+1] = new_block & 1
  
    return new_grid


# Verify reversibility: run forward N steps, then reverse N steps
def verify_reversibility(rule_fn, grid, steps=20):
    """Test if a CA is reversible by running forward then backward."""
    original = grid.copy()
    history = [grid.copy()]
  
    for _ in range(steps):
        grid = rule_fn(grid)
        history.append(grid.copy())
  
    # A reversible CA: reconstruct history from the end
    # For Critters, we store phase, so we run the same rule backward
    reconstructed = grid.copy()
    for t in range(steps - 1, -1, -1):
        reconstructed = rule_fn(reconstructed)  # Critters IS its own inverse!
  
    is_reversible = np.array_equal(reconstructed, original)
    print(f"Reversible: {is_reversible}")
    print(f"Max difference: {np.abs(reconstructed.astype(int) - original.astype(int)).max()}")
    return is_reversible
```

---

### 🔥 Challenge 9 — Conservation & Reversibility

```python
# 1. VERIFY PARTICLE CONSERVATION in Critters:
#    For 1000 random grids, check that sum(grid) is unchanged after one step.
#    Any exceptions?

# 2. SECOND-ORDER CA (Takesue scheme):
#    A non-Margolus way to achieve reversibility:
#      next[t+1] = rule(grid[t]) XOR grid[t-1]
#    This makes any rule reversible! Implement it for Rule 30 and verify.
#    (You'll need to store two time steps.)

# 3. ENTROPY IN REVERSIBLE CA:
#    Since reversible CAs have no attractor (they're bijections),
#    does their Shannon entropy stay constant over time?
#    Test empirically. What does this imply about the "arrow of time"?
```

---

## 10. Continuous CA & Reaction-Diffusion Systems

Continuous CA replace discrete states with real-valued fields, often using convolution with a smooth kernel and a non-linear activation.

### Lenia — A Continuous Life

```python
def lenia_step(grid: np.ndarray, kernel: np.ndarray, 
               dt: float = 0.1, mu: float = 0.14, sigma: float = 0.017) -> np.ndarray:
    """
    One step of Lenia (Chan, 2019).
  
    grid:   2D float array in [0, 1]
    kernel: convolution kernel (smooth, radially symmetric)
    dt:     time step
    mu:     growth function mean
    sigma:  growth function width
  
    Update: A[t+1] = clip(A[t] + dt * growth(conv(A[t], K)), 0, 1)
    Growth: G(u) = 2 * exp(-(u - mu)^2 / (2*sigma^2)) - 1
    """
    from scipy.signal import fftconvolve
  
    # Normalise kernel
    kernel = kernel / kernel.sum()
  
    # Convolve (periodic boundary)
    u = fftconvolve(grid, kernel, mode='same')
  
    # Growth function: bell curve centred at mu
    growth = 2.0 * np.exp(-((u - mu) ** 2) / (2 * sigma ** 2)) - 1.0
  
    # Update
    new_grid = np.clip(grid + dt * growth, 0.0, 1.0)
    return new_grid


def make_lenia_kernel(size: int = 25, beta: list = None) -> np.ndarray:
    """
    Create Lenia's ring-shaped kernel.
    beta: shell weights (list of floats)
    """
    if beta is None:
        beta = [1.0]
  
    cx, cy = size // 2, size // 2
    y, x   = np.ogrid[-cy:size-cy, -cx:size-cx]
    r      = np.sqrt(x**2 + y**2) / (size // 2)
  
    # Sum of Gaussian shells
    kernel = np.zeros((size, size))
    n      = len(beta)
    for i, b in enumerate(beta):
        peak = (i + 0.5) / n
        kernel += b * np.exp(-((r - peak) ** 2) / (2 * (0.15 ** 2)))
  
    kernel[r > 1] = 0
    return kernel


def demo_lenia():
    size   = 100
    kernel = make_lenia_kernel(size=25, beta=[1.0])
  
    # "Orbium" — a glider-like creature in Lenia
    grid = np.zeros((size, size))
    cx, cy = size // 2, size // 2
    r = 7
    y, x = np.ogrid[-cy:size-cy, -cx:size-cx]
    grid[x**2 + y**2 < r**2] = 1.0
  
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(grid, cmap='inferno', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(f'Step {i * 30}')
        ax.axis('off')
        for _ in range(30):
            grid = lenia_step(grid, kernel)
  
    plt.suptitle('Lenia — Continuous CA Life-like System', fontsize=13)
    plt.tight_layout()
    plt.savefig('lenia.png', dpi=150)
    plt.show()

demo_lenia()
```

### Reaction-Diffusion (Gray-Scott Model)

```python
def gray_scott_step(U: np.ndarray, V: np.ndarray,
                     Du: float = 0.16, Dv: float = 0.08,
                     F: float = 0.035, k: float = 0.060,
                     dt: float = 1.0) -> tuple:
    """
    One step of the Gray-Scott reaction-diffusion system.
  
    U, V: concentration fields for two chemical species
    Du, Dv: diffusion rates
    F:  feed rate (U is replenished)
    k:  kill rate (V is removed)
  
    dU/dt = Du * ∇²U - U*V² + F*(1-U)
    dV/dt = Dv * ∇²V + U*V² - (F+k)*V
    """
    def laplacian(Z):
        """Discrete Laplacian with periodic boundary."""
        return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
                np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z)
  
    uvv = U * V * V
  
    dU = Du * laplacian(U) - uvv + F * (1 - U)
    dV = Dv * laplacian(V) + uvv - (F + k) * V
  
    return np.clip(U + dt * dU, 0, 1), np.clip(V + dt * dV, 0, 1)


def demo_gray_scott(size=100, steps=2000, F=0.035, k=0.060):
    # Initialise: mostly U, small square of V
    U = np.ones((size, size))
    V = np.zeros((size, size))
  
    # Seed V in the centre
    cx, cy = size // 2, size // 2
    r = 5
    U[cx-r:cx+r, cy-r:cy+r] = 0.5
    V[cx-r:cx+r, cy-r:cy+r] = 0.25
    U += 0.02 * np.random.randn(size, size)
    V += 0.02 * np.random.randn(size, size)
  
    snapshots = []
    for t in range(steps):
        U, V = gray_scott_step(U, V, F=F, k=k)
        if t % 400 == 0:
            snapshots.append(V.copy())
  
    fig, axes = plt.subplots(1, len(snapshots), figsize=(14, 4))
    for ax, snap in zip(axes, snapshots):
        ax.imshow(snap, cmap='plasma', vmin=0, vmax=0.4)
        ax.axis('off')
  
    plt.suptitle(f'Gray-Scott Reaction-Diffusion  F={F}  k={k}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'gray_scott_F{F}_k{k}.png', dpi=150)
    plt.show()


# Spots pattern
demo_gray_scott(F=0.035, k=0.065)

# Stripes/maze pattern  
demo_gray_scott(F=0.060, k=0.062)

# Worms
demo_gray_scott(F=0.078, k=0.061)
```

---

### 🔥 Challenge 10 — Continuous Worlds

```python
# 1. GRAY-SCOTT PHASE DIAGRAM
#    Map the (F, k) parameter space by running Gray-Scott for many (F, k) pairs.
#    For each, classify the final pattern by computing:
#      - mean(V), variance(V), Fourier dominant frequency
#    Plot a 2D heatmap of V-variance over the (F, k) plane.
#    You'll see distinct "pattern regions" (spots, stripes, worms, chaos, death).

# 2. LENIA CREATURE ZOO
#    Lenia supports many "creatures" (Orbium, Aquarium, Scutium, etc.)
#    that behave like living organisms.
#    Find the parameters (mu, sigma, beta, kernel size) for Orbium (a glider)
#    and Hydrogeminium (a rotating disk).
#    Hint: see Chan's original Lenia paper (2019) for parameter tables.

# 3. TURING PATTERN LINK
#    Alan Turing's 1952 paper showed how reaction-diffusion produces
#    animal coat patterns (spots, stripes).
#    Run Gray-Scott with parameters that produce:
#      a) Cheetah-like spots  (F≈0.025, k≈0.06)
#      b) Zebra-like stripes  (F≈0.06,  k≈0.062)
#      c) Labyrinthine maze   (F≈0.04,  k≈0.063)
#    Save each as a high-resolution PNG.
```

---

## Online Resources

- **Golly:** the best CA simulator: `https://golly.sourceforge.net`
- **ConwayLife wiki:** database of Life patterns: `https://conwaylife.com/wiki`
- **Lenia Portal**: `https://chakazul.github.io/lenia`
- **Distill NCA article**: `https://distill.pub/2020/growing-ca/`
- **Wolfram MathWorld CA**: `https://mathworld.wolfram.com/CellularAutomaton.html`
