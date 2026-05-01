# Three-Way Decision Systems - A Study Guide

> Reading notes on **Three-Way Decisions (3WD)**, the framework introduced by **Yiyu Yao**
> (UofR, 2010) that generalises Pawlak's rough sets and Bayesian decision theory by
> replacing the *accept / reject* binary with a richer *accept / defer / reject* trio.

---

## 1. Why three-way?

Most decision frameworks force a binary verdict. In real cognition we constantly use a
**third option** - *"I don't have enough information yet, defer the call"* - and revise
when more evidence arrives. 3WD turns this everyday cognitive move into a formal
mathematical theory by:

1. **Trisecting** the universe of objects into three disjoint regions:
   - **POS** (positive / acceptance region)
   - **NEG** (negative / rejection region)
   - **BND** (boundary / non-commitment / *deferment* region)
2. **Acting** differently on each region.
3. **Evaluating the outcome** and (optionally) iterating.

This is the **TAO model**: *Trisect → Act → Outcome*. It is the unifying meta-pattern for
the whole field.

> **Slogan.** *"Thinking, problem solving and computing in threes."* - Yao.

---

## 2. From rough sets to three-way decisions

### 2.1 Pawlak rough sets (recap)

Given a universe `U`, an equivalence relation `R`, and a target concept `X ⊆ U`:

- **Lower approximation**:  `apr_R(X) = { x ∈ U : [x]_R ⊆ X }`
- **Upper approximation**:  `apr̄_R(X) = { x ∈ U : [x]_R ∩ X ≠ ∅ }`

These give a *qualitative* trisection:

```
POS(X) = apr_R(X)              -- definitely in X
NEG(X) = U \ apr̄_R(X)          -- definitely not in X
BND(X) = apr̄_R(X) \ apr_R(X)   -- uncertain
```

This is **already three-way** - but in a *strict* sense (probabilities are 0 or 1). 3WD's
contribution is to make the three regions **graded** and **decision-theoretic**.

### 2.2 Probabilistic / decision-theoretic rough sets (DTRS)

Replace strict containment with a probability:

```
P(X | [x]_R) = |[x]_R ∩ X|  /  |[x]_R|
```

Choose two thresholds `0 ≤ β < α ≤ 1` and define:

```
POS_(α,β)(X) = { x : P(X | [x]_R) ≥ α }
NEG_(α,β)(X) = { x : P(X | [x]_R) ≤ β }
BND_(α,β)(X) = { x : β < P(X | [x]_R) < α }
```

Pawlak rough sets are recovered with `α = 1, β = 0`. The whole game is now:
**how do we choose α and β?**

### 2.3 Yao's Bayesian derivation of α and β (the heart of DTRS)

Define a **2×3 loss matrix** for two states (`X`, `¬X`) and three actions
(`a_P` = accept, `a_B` = defer, `a_N` = reject):

| state \ action | a_P (accept) | a_B (defer) | a_N (reject) |
|----------------|--------------|-------------|--------------|
| in X           | λ_PP         | λ_BP        | λ_NP         |
| not in X       | λ_PN         | λ_BN        | λ_NN         |

Reasonable monotonicity: `λ_PP ≤ λ_BP ≤ λ_NP` (accepting a true positive is cheaper than
deferring it, which is cheaper than rejecting it) and dually `λ_NN ≤ λ_BN ≤ λ_PN`.

For an object `x` with `p = P(X | [x]_R)`, the Bayes expected losses are:

```
R(a_P | x) = λ_PP · p + λ_PN · (1 − p)
R(a_B | x) = λ_BP · p + λ_BN · (1 − p)
R(a_N | x) = λ_NP · p + λ_NN · (1 − p)
```

Choose the action with **minimum expected loss**. Solving the inequalities yields three
threshold expressions:

```
α = (λ_PN − λ_BN)
   ──────────────────────────
   (λ_PN − λ_BN) + (λ_BP − λ_PP)

β = (λ_BN − λ_NN)
   ──────────────────────────
   (λ_BN − λ_NN) + (λ_NP − λ_BP)

γ = (λ_PN − λ_NN)
   ──────────────────────────
   (λ_PN − λ_NN) + (λ_NP − λ_PP)
```

Decision rules:

```
if   p ≥ α       →  POS  (accept)
if   p ≤ β       →  NEG  (reject)
if   β < p < α   →  BND  (defer)
```

`γ` is the threshold of the *two-way* (no-deferment) Bayes rule and equals α = β when the
deferment cost balances out - a useful sanity check.

> **Why this matters.** α and β are no longer chosen by hand - they are **derived from
> economically meaningful losses**. This is the chief reason DTRS has been so widely
> adopted: domain experts can elicit losses (medical mistreatment cost, financial fraud
> cost, …) and the thresholds follow.

---

## 3. The TAO model (the modern unifying view)

Yao's later writing (2018-2023) reframes 3WD around three explicit modules:

### 3.1 **Trisecting**

Define an *evaluation function* `v : U → V` where `V` is a totally / partially ordered
*evaluation space*. Pick a *designated subset* `D_α, D_β` of `V` (e.g. `[α, 1]` and
`[0, β]` for `V = [0, 1]`). Then:

```
POS = { x : v(x) ∈ D_α }
NEG = { x : v(x) ∈ D_β }
BND = U \ (POS ∪ NEG)
```

`v` is intentionally generic: it can be a probability (DTRS), a fuzzy membership, a
similarity score, a neural-net softmax, or a rank.

### 3.2 **Acting**

Each region gets its own *strategy*:

| Region | Typical action                                           |
|--------|----------------------------------------------------------|
| POS    | commit; emit positive rule; admit to cluster core         |
| NEG    | commit; emit negative rule; exclude from cluster          |
| BND    | gather more data; query oracle; pass to next granular level |

The **deferment action is not "do nothing"** - it is *"escalate to a different process"*.
This is what makes 3WD operationally distinct from confidence thresholding.

### 3.3 **Outcome**

Evaluate the trisection with an *effectiveness measure* (overall expected loss, accuracy
on POS∪NEG, ratio of |BND|, change measures, …) and revise `v`, the thresholds, or the
loss matrix accordingly.

---

## 4. Sequential three-way decisions (S3WD)

The most active research direction. The idea: instead of a single trisection, use a
**sequence of trisections at increasing levels of granularity / information**.

```
Level 1:  trisect with cheap evidence
          → POS_1 ∪ NEG_1 are committed
          → BND_1 carries forward
Level 2:  trisect BND_1 with richer evidence
          → POS_2 ∪ NEG_2 committed
          → BND_2 carries forward
…
Level k:  forced two-way decision (no more deferment)
```

Why this is appealing:

- **Test-cost-sensitive**: cheap features used first; expensive ones only on hard cases.
- **Graceful degradation**: each level reduces uncertainty by exactly the amount its
  features warrant.
- **Natural fit for streaming / multi-resolution data** (image pyramids, time-series at
  multiple scales, hierarchical attribute reduction).

Recent S3WD variants (2024-2025):

- **Adaptive thresholds** - α, β learned per level from validation data
  ([Sequential 3WD with adaptive thresholds, Expert Systems with Applications, 2025](https://www.sciencedirect.com/science/article/abs/pii/S0957417425018147)).
- **Game-theoretic S3WD** - Stackelberg game between accuracy and cost
  ([Discover Computing, 2025](https://link.springer.com/article/10.1007/s10791-025-09729-5)).
- **Cost-sensitive multi-class S3WD**
  ([IJMLC, 2025](https://link.springer.com/article/10.1007/s13042-025-02839-y)).
- **Multilevel info-gain + regret optimisation** for classification (Liang et al., 2024).

---

## 5. Three-way classification

A 3WD classifier replaces the standard hard-label output with a triplet:

```
predict(x) ∈ {ω_+, ω_?, ω_−}
                 │      │
                 │      └── deferred (not enough confidence)
                 └────────── confidently positive / negative
```

Three implementation patterns appear in the literature:

1. **Threshold a probabilistic model** - train any classifier producing `P(y=1|x)`, then
   apply DTRS thresholds α, β derived from a loss matrix. Simple but powerful.
2. **Trisect attribute reducts** - choose attribute subsets per region, accept under one
   reduct, reject under another, defer otherwise.
3. **Hierarchical classifiers** - sequential 3WD: a fast/coarse model handles POS/NEG;
   borderline cases (BND) escalate to a slower/finer model. Efficient at inference.

Empirically: 3WD classifiers shine on **cost-sensitive** tasks (medical screening, fraud
detection, fault diagnosis) where mis-acceptance and mis-rejection costs are *asymmetric*
and a "needs follow-up" verdict is operationally useful.

---

## 6. Three-way clustering

Standard clustering forces every point into exactly one cluster. Three-way clustering
represents each cluster `C_k` as a pair `(Co(C_k), Fr(C_k))`:

- **Core region `Co(C_k)`** - points that *definitely belong*.
- **Fringe region `Fr(C_k)`** - points that *partially belong*.
- Anything outside `Co ∪ Fr` *does not belong*.

Cores are pairwise disjoint; fringes are allowed to overlap (an object can be on the
fringe of several clusters). Reported benefits:

- Better treatment of boundary-ambiguous points than hard partitions.
- Compatible with k-means, density-based, hierarchical, fuzzy bases.
- Used in stream clustering (TWStream, 2024), text clustering, image segmentation.

See Yu et al., *Three-way clustering: foundations, survey and challenges*, **Applied Soft
Computing**, 2024.

---

## 7. Extensions to richer uncertainty calculi

3WD has been ported into virtually every uncertainty formalism, because the TAO pattern
is parametric in `v`:

- **Fuzzy 3WD** - `v` is a fuzzy membership; thresholds are fuzzy numbers.
- **Intuitionistic fuzzy 3WD** - membership + non-membership with hesitancy slack.
  See *Three-way decisions in generalized intuitionistic fuzzy environments: survey and
  challenges* (Springer AI Review, 2024).
- **Hesitant fuzzy 3WD** - multiple plausible memberships per object.
  Survey: Zhan, Wang, Ding, Yao - IEEE/CAA JAS, 2023.
- **Dual hesitant / Pythagorean / q-rung fuzzy 3WD** - generalised membership grades.
- **Interval-valued / Z-number 3WD** - uncertainty on the thresholds themselves.
- **Neutrosophic 3WD** - truth/indeterminacy/falsity grades.
- **Conflict analysis 3WD** - Pawlak-style conflict graphs trisected by alliance level.
- **Game-theoretic rough sets (GTRS)** - α, β as equilibria of a game between
  *accuracy* and *generality* players (Herbert & Yao).

The pattern: choose a richer `V`, plug into TAO, derive new threshold formulas.

---

## 8. Application areas (selected)

- **Medical decision support** - accept treatment / further test / reject diagnosis;
  COVID-19 mild-symptom triage (PMC9132434, 2022).
- **Credit risk / fraud** - three-way Sequential 3WD (Li et al., 2024).
- **Recommendation systems** - recommend / consider / hide.
- **Information retrieval** - relevant / borderline / irrelevant document partitions.
- **Email & spam filtering** - spam / quarantine / inbox is a textbook 3WD use-case.
- **Anomaly detection** - normal / suspicious / anomaly (3WADSP, 2025).
- **Conflict resolution and dilemma reasoning** - agree / negotiate / disagree.
- **Cloud workload prediction**, **fault diagnosis**, **document summarisation**, **MCDM**.

---

## 9. Strengths and open problems

### Strengths

- **Cost-aware by construction** - losses are first-class citizens.
- **Interpretable** - each region maps to a real-world action.
- **Modular** - TAO works with any evaluation function, including modern neural
  classifiers (use softmax probability as `v`).
- **Reduces operational risk** - by deferring borderline cases for human review or extra
  evidence.

### Open problems / criticisms

- **Loss elicitation is hard** - practitioners often guess λ values, undoing the
  decision-theoretic rigour. *Inverse* 3WD (learn losses from data) is an active topic.
- **Threshold instability under distribution shift** - α, β computed offline may be
  miscalibrated at deploy time; adaptive S3WD partially addresses this.
- **BND-region semantics in deep learning** - what does deferment *mean* when the model
  is a black box? Emerging work links 3WD to selective classification and conformal
  prediction.
- **Multi-class 3WD** - most theory is binary; multi-class requires class-by-class
  trisections or one-vs-rest reductions.
- **Empirical bake-offs** - comparisons to selective classification, abstention learning,
  and conformal predictors (which solve overlapping problems from different angles) are
  still rare and badly needed.

---

## 10. Suggested reading order

### Foundations
1. **Yao, Y. Y.** *Decision-theoretic rough set models.* Rough Sets and Knowledge
   Technology, LNCS 4481, 1-12, **2007**.
2. **Yao, Y. Y.** *Three-way decisions with probabilistic rough sets.* **Information
   Sciences** 180(3):341-353, **2010**. *(the canonical reference)*
3. **Yao, Y. Y.** *An outline of a theory of three-way decisions.* RSCTC, LNCS 7413,
   1-17, **2012**.
   PDF: https://www2.cs.uregina.ca/~yyao/PAPERS/a_theory_of_three_way_decisions.pdf

### Modern unifying view
4. **Yao, Y. Y.** *Three-way decision and granular computing.* **International Journal of
   Approximate Reasoning**, 103:107-123, **2018**.
5. **Yao, Y. Y.** *Tri-level thinking: models of three-way decision.* **International
   Journal of Machine Learning and Cybernetics**, **2019**.
6. **Yao, Y. Y.** *The Dao of three-way decision and three-world thinking.* **IJAR**,
   **2023**. https://www.sciencedirect.com/science/article/abs/pii/S0888613X23001639

### Surveys (start here for a quick map of the field)
7. **Zhan, J.; Wang, J.; Ding, W.; Yao, Y.** *Three-way behavioral decision making with
   hesitant fuzzy information systems: survey and challenges.* **IEEE/CAA Journal of
   Automatica Sinica**, 10(2):330-350, **2023**.
   https://www.ieee-jas.net/article/doi/10.1109/JAS.2022.106061
8. *Three-way decisions in generalized intuitionistic fuzzy environments: survey and
   challenges.* **AI Review** (Springer), **2024**.
   https://link.springer.com/article/10.1007/s10462-023-10647-5
9. **Yu, H. et al.** *Three-way clustering: foundations, survey and challenges.*
   **Applied Soft Computing**, **2024**.

### Sequential / dynamic 3WD
10. **Yao, Y. Y.** *Sequential three-way decisions with probabilistic rough sets.* IEEE
    ICCI*CC, **2011**.
11. **Sequential three-way decision with adaptive thresholds and its applications in two
    binary classifiers.** *Expert Systems with Applications*, **2025**.
    https://www.sciencedirect.com/science/article/abs/pii/S0957417425018147
12. **A game-theoretic sequential three-way decision using probabilistic rough sets and
    multiple levels of granularity.** *Discover Computing*, **2025**.
    https://link.springer.com/article/10.1007/s10791-025-09729-5
13. **A cost-sensitive sequential multi-class three-way decision model.** *International
    Journal of Machine Learning and Cybernetics*, **2025**.
    https://link.springer.com/article/10.1007/s13042-025-02839-y

### Applications & extensions
14. **Three-way Investment Decisions with Decision-theoretic Rough Sets.** IJCIS, 2011.
    https://link.springer.com/content/pdf/10.2991/ijcis.2011.4.1.6.pdf
15. **A novel behavioral three-way decision model with application to the treatment of
    mild symptoms of COVID-19.** PMC, **2022**.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9132434/
16. **TWStream: Three-Way Stream Clustering.** **2024**.
    https://dumingjing.github.io/files/paper-11_2024-02-21/paper.pdf
17. **Decision-Theoretic Rough Sets for Three-Way Decision-Making in Dilemma Reasoning
    and Conflict Resolution.** *Mathematics* (MDPI), **2025**.
    https://www.mdpi.com/2227-7390/13/13/2111
18. **Three-way unsupervised anomaly detection of sequential patterns.** *International
    Journal of Machine Learning and Cybernetics*, **2025**.
    https://link.springer.com/article/10.1007/s13042-025-02533-z

### Related complementary frameworks (worth knowing)
19. **Selective classification / classification with reject option** - Chow (1970);
    Cortes et al. (2016).
20. **Conformal prediction** - Vovk, Gammerman, Shafer.
21. **Abstaining classifiers** - modern deep-learning angle on the "defer" action.

---

## 11. TL;DR cheat-sheet

```
3WD core idea          Trisect → Act → Outcome  (TAO)
Three regions          POS (accept) | BND (defer) | NEG (reject)
Pawlak rough sets      special case: α = 1, β = 0
DTRS thresholds        derived from 2x3 loss matrix (Bayes minimum risk)
                       α, β depend on λ_PP, λ_PN, λ_BP, λ_BN, λ_NP, λ_NN
Decision rule          p ≥ α → POS;  p ≤ β → NEG;  else BND
Sequential 3WD         iterate trisection across granularity levels;
                       BND of level i is the input of level i+1
3W Clustering          cluster = (Core, Fringe);  Cores disjoint, Fringes can overlap
3W Classification      output ∈ {accept, defer, reject};  great for cost-sensitive tasks
Killer apps            medical triage, fraud detection, conflict analysis,
                       recommender systems, anomaly detection
Hot 2024-25 topics     adaptive S3WD thresholds, game-theoretic 3WD,
                       deep-learning-compatible 3WD, multi-class & cost-sensitive 3WD
Foundational paper     Yao 2010 - "Three-way decisions with probabilistic rough sets",
                       Information Sciences 180(3):341-353
```

---

*Companion files in this folder:*
- `tolerance_near_sets_classifier.md` - sister theory (Peters/Ramanna near sets).
- `tns_classifier.ipynb` - a working classifier built on those near-set ideas; the same
  TAO discipline could be wrapped around it to obtain a *three-way* tolerance-near-set
  classifier (POS = nearest representative + high confidence; BND = ambiguous between
  representatives; NEG = far from all). Worth experimenting with as a follow-up.
