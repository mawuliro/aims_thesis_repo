# Private Decoupling of Quantum Information

### AIMS Ghana · MSc Mathematical Sciences · Research Code Repository

> *Can a quantum channel simultaneously hide information from every receiver?  
> This repository traces the geometry of that question — from a two-qubit  
> algebraic criterion to high-dimensional Werner and Bell-mixture families.*

---

## Overview

This repository contains all numerical research notebooks for the MSc thesis
**"Private Decoupling of Quantum Information"** (AIMS Ghana, 2025–2026).
The central object of study is the **hiding capacity** of a bipartite quantum
state $\rho_{RA}$: the minimum total information leakage achievable by any
isometry acting on subsystem $A$,

$$\mathcal{C}(\rho_{RA}) \;=\; \min_{V}\,\bigl[I(R:B) + I(R:E)\bigr],$$

where $V : \mathcal{H}_A \to \mathcal{H}_B \otimes \mathcal{H}_E$ is the
isometry that splits $A$ into a *Bob* part $B$ and an *Eve* part $E$.
When $\mathcal{C}(\rho_{RA}) = 0$ the state admits **perfect hiding**: there
exists a channel that simultaneously decouples $R$ from both outputs.

The notebooks are ordered by conceptual depth and were developed sequentially
across three research objectives.

---

## Repository Structure

```
aims_thesis_repo/
│
├── Perfect_hiding_condition_checking.ipynb
│       Objective 1 — VQA + KAK decomposition, qubit case.
│       Verifies perfect hiding (I(R:B) = I(R:E) = 0) numerically
│       via SLSQP / L-BFGS-B optimisation over the full 16-parameter
│       U(4) KAK ansatz.  Establishes the coherent-information condition
│       I_c(R▷A) ≤ 0 as the necessary and sufficient criterion.
│
├── Perfect_hiding_condition_cheching_including_operator_subspace_condition.ipynb
│       Objective 1 (extended) — adds the operator subspace criterion.
│       Proves and numerically validates: perfect hiding is achievable
│       if and only if dim(S) ≤ 3, where S = span{A, D, B, B†} is
│       the operator subspace of ρ_RA, computed via the rank of the
│       4×4 Pauli coefficient matrix C.
│
├── bounds_computation.ipynb
│       Objective 2 — Pareto frontier tracing and analytical bounds.
│       Computes C(ρ_RA) as the convex hull of (I(R:B), I(R:E)) pairs
│       sampled over KAK isometries, and overlays four bounds: the SSA
│       upper bound, the Buscemi exact lower bound 2I_c, the discord
│       upper bound (conjecture), and the conjectured closed-form
│       C_hide lower bound — verified numerically to ~10⁻¹³ bits.
│
├── Werner_Isotropic_states_high_dimension.ipynb
│       Objective 3 — Werner and isotropic families, d ∈ {2, 3, 5, 7}.
│       Scales the optimisation to qudits using JAX + Adam (GPU-ready),
│       Gell-Mann / Cayley parametrisation of U(d²), and analytical
│       bounds for the full mixing-parameter range.
│
└── Bell_mixture_state_high_dimension.ipynb
        Objective 3 — Bell mixtures, d ∈ {2, 3, 5, 7}.
        Computes I(R:A), D(R|A), L_min = 2D(R|A), the lower bound, and
        cost = C(ρ_RA) numerically at every dimension.  Figure 4 tracks
        how D_min and L_min scale with d at mixing parameter p = 0.5.
```

---

## The Three Research Objectives

### Objective 1 — Perfect Hiding: When Can Everything Be Hidden?

**Question.** For which qubit states $\rho_{RA}$ does there exist an isometry
that drives both $I(R:B)$ and $I(R:E)$ to zero simultaneously?

**Answer (proved and verified).** Perfect hiding is achievable if and only if

$$I_c(R \triangleright A) = S(A) - S(RA) \;\leq\; 0
\quad\Longleftrightarrow\quad
\dim(\mathcal{S}) \leq 3,$$

where $\mathcal{S} = \operatorname{span}\{A,\, D,\, B,\, B^\dagger\}$ is the
**operator subspace** extracted from the block decomposition of $\rho_{RA}$
over the reference basis:

$$\rho_{RA} = \begin{pmatrix} A & B \\ B^\dagger & D \end{pmatrix},
\qquad A = \langle 0|_R\,\rho_{RA}\,|0\rangle_R,\quad
D = \langle 1|_R\,\rho_{RA}\,|1\rangle_R,\quad
B = \langle 0|_R\,\rho_{RA}\,|1\rangle_R.$$

#### The Pauli Coefficient Matrix

The dimension of $\mathcal{S}$ is computed as the rank of the $4\times 4$
real matrix $C$ whose rows are the Hilbert–Schmidt coordinates of
$A$, $D$, $\operatorname{Re}(B)$, $\operatorname{Im}(B)$ in the Pauli basis
$\{\,I,\,X,\,Y,\,Z\,\}$:

$$C = \begin{pmatrix}
a_0 & a_1 & a_2 & a_3 \\
d_0 & d_1 & d_2 & d_3 \\
b^{\rm re}_0 & b^{\rm re}_1 & b^{\rm re}_2 & b^{\rm re}_3 \\
b^{\rm im}_0 & b^{\rm im}_1 & b^{\rm im}_2 & b^{\rm im}_3
\end{pmatrix} \in \mathbb{R}^{4\times 4}, \qquad
\operatorname{rank}(C) = \dim(\mathcal{S}).$$

**Practical rule:** if $\operatorname{rank}(C) \leq 3$ → hiding succeeds;
if $\operatorname{rank}(C) = 4$ → hiding fails.  The Bell state $|\Phi^+\rangle$
serves as the canonical negative example: its $C$ has full rank 4, confirming
that maximally entangled states cannot be perfectly hidden.

#### KAK Implementation

The isometry is embedded as $V|\psi\rangle_A = U(|\psi\rangle_A\otimes|0\rangle_E)$
with $U\in U(4)$ parametrised by the **KAK (Cartan) decomposition**:

$$U(\boldsymbol{\theta}) = e^{i\phi}\,
(A_1\otimes A_2)\cdot
\exp\!\bigl[i(c_1\,XX + c_2\,YY + c_3\,ZZ)\bigr]\cdot
(A_3\otimes A_4),$$

where each $A_k\in SU(2)$ uses **ZYZ Euler angles**
$A(\alpha,\beta,\gamma) = R_Z(\alpha)\,R_Y(\beta)\,R_Z(\gamma)$, giving
16 real parameters total:

$$\boldsymbol{\theta} =
(\underbrace{\alpha_1,\beta_1,\gamma_1}_{A_1},\,
\underbrace{\alpha_2,\beta_2,\gamma_2}_{A_2},\,
\underbrace{\alpha_3,\beta_3,\gamma_3}_{A_3},\,
\underbrace{\alpha_4,\beta_4,\gamma_4}_{A_4},\,
\underbrace{c_1,c_2,c_3}_{\Omega},\,
\underbrace{\phi}_{\rm phase})
\;\in\;[-\pi,\pi]^{16}.$$

---

### Objective 2 — Bounds: The Geometry of Leakage

**Question.** When perfect hiding fails, what is the minimum achievable
total leakage $\mathcal{C}(\rho_{RA})$?  What are the exact and conjectured
bounds on the Pareto tradeoff curve between $I(R:B)$ and $I(R:E)$?

The notebook `bounds_computation.ipynb` traces the **Pareto frontier** of the
achievable leakage set

$$\mathcal{A}(\rho_{RA}) = \bigl\{(I(R:B),\, I(R:E)) :
V\in\mathrm{Isom}(\mathcal{H}_A,\,\mathcal{H}_B\otimes\mathcal{H}_E)\bigr\}$$

via convex-hull extraction over dense KAK samples, and overlays four
analytically motivated bounds:

| Bound | Expression | Status |
|-------|-----------|--------|
| **DPI box** | $0 \leq I(R:B),\;I(R:E) \leq I(R:A)$ | Exact (data-processing inequality) |
| **SSA upper bound** | $I(R:B)+I(R:E) \leq S(R) + 2\,S(A) - S(RA)$ | Exact (strong subadditivity) |
| **Buscemi lower bound** | $I(R:B)+I(R:E) \geq 2\max(0,\,I_c)$ | Exact |
| **Discord upper bound** | $I(R:B)+I(R:E) \leq I(R:A) + |I(R:A)-2D_{\min}|$ | Conjectured |
| **$C_{\rm hide}$ lower bound** | $\mathcal{C}(\rho_{RA}) \geq I(R:A) - |I(R:A)-2D_{\min}|$ | Conjectured |

where $D_{\min} = \min\bigl(D(R|A),\,D(A|R)\bigr)$ is the smaller of the two
asymmetric quantum discords.

#### The Conjectured Closed-Form Formula

The principal conjecture of the thesis is that the hiding capacity admits the
closed-form expression

$$\boxed{\mathcal{C}(\rho_{RA}) \;=\; I(R:A) - \bigl|I(R:A) - 2\min\bigl(D(R|A),\,D(A|R)\bigr)\bigr|,}$$

verified numerically to $\sim 10^{-13}$ bits precision across all tested state
families.  This formula reveals a direct structural link between
**hiding capacity** and **quantum discord**:

- **Zero discord** ($D = 0$, classical-quantum states): $\mathcal{C} = 0$,
  perfect hiding is achievable.
- **Symmetric discord** ($D(R|A) = D(A|R) = I(R:A)/2$, e.g. maximally
  entangled states): $\mathcal{C} = I(R:A)$, no hiding is possible at all.
- **General mixed states:** the gap $|I(R:A) - 2D_{\min}|$ measures exactly
  how much of the mutual information remains hideable.

#### State Families Tested

Canonical states surveyed across the Pareto plots:

- Bell states $|\Phi^\pm\rangle$, $|\Psi^\pm\rangle$ (pure, maximally entangled)
- Werner states $\rho_W(p) = p\,|\Phi^+\rangle\langle\Phi^+| + (1-p)\,\tfrac{I}{4}$
- Separable classical-quantum (CQ) states
- Noisy-channel outputs: dephasing, amplitude damping, bit-flip channels
- Bell mixtures $p\,|\Phi^+\rangle\langle\Phi^+| + (1-p)\,|\Phi^-\rangle\langle\Phi^-|$

All figures use the **Wong colorblind-safe palette** following
Nature/Science publication standards.

---

### Objective 3 — High Dimensions: Scaling to Qudits

**Question.** How do the hiding capacity, quantum discord, and the
$\mathcal{C}_{\rm hide}$ lower bound behave for Werner, isotropic, and
Bell-mixture states as the local dimension $d$ grows?

#### Qudit State Families

$$\rho_W(p,d) = p\,|\Phi^+_d\rangle\langle\Phi^+_d| + (1-p)\,\frac{I_{d^2}}{d^2}
\qquad\text{(Werner)}$$

$$\rho_{\rm iso}(p,d) = p\,|\Phi^+_d\rangle\langle\Phi^+_d|
+ \frac{1-p}{d^2-1}\,\bigl(I_{d^2} - |\Phi^+_d\rangle\langle\Phi^+_d|\bigr)
\qquad\text{(Isotropic)}$$

$$\rho_{\rm Bell}(p,d) = p\,|\Phi^+_d\rangle\langle\Phi^+_d|
+ (1-p)\,|\Phi^-_d\rangle\langle\Phi^-_d|
\qquad\text{(Bell mixture)}$$

computed at $d\in\{2,3,5,7\}$ over the full range of mixing parameter $p$.

#### JAX GPU Pipeline

Both high-dimensional notebooks replace the SciPy/KAK qubit optimizer with a
fully JIT-compiled JAX pipeline:

- **Parametrisation:** $U\in U(d^2)$ via the **Cayley map**
  $U = (I - iH)^{-1}(I + iH)$ with $H = \sum_\alpha \theta_\alpha G_\alpha$
  expanded in $d^2\times d^2$ **Gell-Mann generators**.
- **Optimiser (Phase 1):** Adam with warmup-cosine schedule, `vmap`-batched
  over all restarts and all states simultaneously.
- **Optimiser (Phase 2):** Tight-Adam refinement ($\mathrm{lr}=10^{-4}$),
  seeded from best Phase-1 solution plus fresh random restarts.
- **Discord:** same two-phase Adam pipeline applied to the classical mutual
  information maximisation over rank-1 projective measurements on $A$.
- **GPU fallback:** `jax.default_backend()` automatically selects CUDA GPU
  when available; CPU execution is supported without code changes.

Per-dimension optimisation budgets (restarts, learning rate, iterations) are
tuned individually to ensure convergence at each Hilbert-space dimension.

#### Key Scaling Result

Figure 4 of `Bell_mixture_state_high_dimension.ipynb` plots $D_{\rm min}$
and $L_{\rm min} = 2D_{\rm min}$ as functions of $d\in\{2,\ldots,19\}$ at
$p=0.5$, probing whether the discord–hiding correspondence survives dimension
growth and how the minimum leakage floor scales with the local Hilbert-space
dimension.

---

## Global Conventions

| Symbol | Definition |
|--------|-----------|
| $\rho_{RA}$ | Bipartite reference-system state; **big-endian** ordering: $\rho_{RA} \equiv \mathrm{kron}(R,A)$, basis $\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}$ |
| $I(R:A)$ | Quantum mutual information $S(R)+S(A)-S(RA)$ |
| $I_c(R\triangleright A)$ | Coherent information $S(A)-S(RA)$ (can be negative) |
| $D(R\|A)$ | Quantum discord measured on $A$: $I(R:A) - \max_{\{\Pi_k\}} I_{\rm cl}(R:\{\Pi_k\})$ |
| $V$ | Isometry $\mathcal{H}_A\to\mathcal{H}_B\otimes\mathcal{H}_E$, embedded as $V|\psi\rangle_A = U(|\psi\rangle_A\otimes|0\rangle_E)$ |
| $S(\rho)$ | Von Neumann entropy $-\operatorname{Tr}(\rho\log_2\rho)$ |
| $\mathcal{S}$ | Operator subspace $\operatorname{span}\{A,D,B,B^\dagger\}\subseteq\mathcal{L}(\mathcal{H}_A)$ |

**Index convention:** subsystem 0 = $R$, subsystem 1 = $A$, subsystem 2 = $E$
throughout all partial-trace operations.  No hidden index reversal occurs
anywhere in the codebase.

---

## Dependencies

**Objectives 1 & 2** — qubit, CPU:

```bash
pip install numpy scipy matplotlib
```

**Objective 3** — qudit, GPU-accelerated:

```bash
pip install numpy matplotlib optax jaxopt
pip install "jax[cuda12]"    # GPU (recommended)
# or
pip install "jax[cpu]"       # CPU fallback
```

Google Colab with a GPU runtime is the recommended environment for the
high-dimensional notebooks.  Drive-mount save paths
(`/content/drive/MyDrive/...`) are declared at the top of each notebook
and can be redirected to a local output directory without further changes.

---

## Theoretical References

| Reference | Role in this work |
|-----------|------------------|
| Braunstein & Pati (2007), *PRL* — *Quantum information cannot be completely hidden in correlations* | No-hiding theorem; motivates the hiding capacity problem |
| Buscemi (2009) — *All entangled quantum states are nonlocal* | Source of the exact Buscemi lower bound $\mathcal{C}\geq 2\max(0,I_c)$ |
| Modi, Cabrera, Paterek, Piani & Vedral, *RMP* (2012) — *The classical-quantum boundary for correlations* | Discord framework; connects $D=0$ to classical-quantum states and to the perfect-hiding condition |
| Dupuis, Fawzi & Wehner (2014) — *Decoupling with unitary approximate two-designs* | Decoupling framework motivating the isometry ansatz and the Stinespring picture |

---

## Author

**Mawulikplimi Roland Hounkpe**  
MSc Mathematical Sciences · AIMS Ghana  
Mastercard Foundation Scholar  
 
Research context: quantum information hiding, in connection with the group
of Dr. Eric Chitambar

---

*"The quantum world does not simply hide information —  
it negotiates between what Bob learns and what Eve loses."*
