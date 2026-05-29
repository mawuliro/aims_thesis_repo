# Private Decoupling of Quantum Information

### AIMS Ghana · MSc Mathematical Sciences · Research Code Repository

> *Can a quantum channel simultaneously hide information from every receiver?*  
> *This repository traces the geometry of that question — from a two-qubit*  
> *algebraic criterion to high-dimensional Werner and Bell-mixture families.*

---

## Overview

This repository contains all numerical research notebooks for the MSc thesis
**"Private Decoupling of Quantum Information"** (AIMS Ghana, 2025–2026).
The central object of study is the **hiding capacity** of a bipartite quantum
state ρ_RA: the minimum total information leakage achievable by any isometry
acting on subsystem A,

```
C(ρ_RA) = min_V [ I(R:B) + I(R:E) ]
```

where V : H_A → H_B ⊗ H_E is the isometry that splits A into a *Bob* part B
and an *Eve* part E.  When C(ρ_RA) = 0 the state admits **perfect hiding**:
there exists a channel that simultaneously decouples R from both outputs.

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
│       upper bound, the Buscemi exact lower bound 2·I_c, the discord
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
        Computes I(R:A), D(R|A), L_min = 2·D(R|A), the lower bound, and
        cost = C(ρ_RA) numerically at every dimension.  Figure 4 tracks
        how D_min and L_min scale with d at mixing parameter p = 0.5.
```

---

## The Three Research Objectives

### Objective 1 — Perfect Hiding: When Can Everything Be Hidden?

**Question.** For which qubit states ρ_RA does there exist an isometry that
drives both I(R:B) and I(R:E) to zero simultaneously?

**Answer (Conjectured based on numerical evidence).** Perfect hiding is achievable if and only if

```
dim(S) ≤ 3
```

where S = span{A, D, B, B†} is the **operator subspace** extracted from the
block decomposition of ρ_RA over the reference basis:

```
        ┌         ┐
ρ_RA =  │  A    B │,    A = ⟨0|_R  ρ_RA  |0⟩_R
        │  B†   D │     D = ⟨1|_R  ρ_RA  |1⟩_R
        └         ┘     B = ⟨0|_R  ρ_RA  |1⟩_R
```

#### The Pauli Coefficient Matrix

The dimension of S is computed as the rank of the 4×4 real matrix C whose
rows are the Hilbert–Schmidt coordinates of A, D, Re(B), Im(B) in the Pauli
basis {I, X, Y, Z}:

```
    ┌                           ┐
    │  a₀   a₁   a₂   a₃       │   ← coords of A
C = │  d₀   d₁   d₂   d₃       │   ← coords of D
    │  b⁰ᵣₑ b¹ᵣₑ b²ᵣₑ b³ᵣₑ    │   ← coords of Re(B)
    │  b⁰ᵢₘ b¹ᵢₘ b²ᵢₘ b³ᵢₘ    │   ← coords of Im(B)
    └                           ┘

rank(C) = dim(S)
```

**Practical rule:**  `rank(C) ≤ 3` → hiding succeeds;  `rank(C) = 4` → hiding
fails.  The Bell state |Φ⁺⟩ is the canonical negative example: its C has full
rank 4, confirming that maximally entangled states cannot be perfectly hidden.

#### KAK Implementation

The isometry is embedded as `V|ψ⟩_A = U(|ψ⟩_A ⊗ |0⟩_E)` with U ∈ U(4)
parametrised by the **KAK (Cartan) decomposition**:

```
U(θ) = e^{iφ} · (A₁ ⊗ A₂) · exp[i(c₁·XX + c₂·YY + c₃·ZZ)] · (A₃ ⊗ A₄)
```

Each A_k ∈ SU(2) uses **ZYZ Euler angles**
`A(α,β,γ) = R_Z(α) · R_Y(β) · R_Z(γ)`, giving 16 real parameters total:

```
θ = (α₁,β₁,γ₁,  α₂,β₂,γ₂,  α₃,β₃,γ₃,  α₄,β₄,γ₄,  c₁,c₂,c₃,  φ)  ∈ [-π, π]¹⁶
     \___ A₁ ___/ \___ A₂ ___/ \___ A₃ ___/ \___ A₄ ___/ \__ Ω __/  \phase/
```

---

### Objective 2 — Bounds: The Geometry of Leakage

**Question.** When perfect hiding fails, what is the minimum achievable total
leakage C(ρ_RA)?  What are the exact and conjectured bounds on the Pareto
tradeoff curve between I(R:B) and I(R:E)?

The notebook `bounds_computation.ipynb` traces the **Pareto frontier** of the
achievable leakage set

```
A(ρ_RA) = { (I(R:B), I(R:E)) : V isometry on H_A → H_B ⊗ H_E }
```

via convex-hull extraction over dense KAK samples, and overlays four
analytically motivated bounds:

| Bound | Expression | Status |
|-------|-----------|--------|
| **DPI box** | `0 ≤ I(R:B), I(R:E) ≤ I(R:A)` | Exact — data-processing inequality |
| **SSA upper bound** | `I(R:B) + I(R:E) ≤ S(R) + 2·S(A) − S(RA)` | Exact — strong subadditivity |
| **Buscemi lower bound** | `I(R:B) + I(R:E) ≥ 2·max(0, I_c)` | Exact |
| **Discord upper bound** | `I(R:B) + I(R:E) ≤ I(R:A) + \|I(R:A) − 2·D_min\|` | Conjectured |
| **C_hide lower bound** | `C(ρ_RA) ≥ I(R:A) − \|I(R:A) − 2·D_min\|` | Conjectured |

where `D_min = min( D(R|A), D(A|R) )` is the smaller of the two asymmetric
quantum discords.

#### The Conjectured Closed-Form Formula

The principal conjecture of the thesis is that the hiding capacity admits the
closed-form expression

```
C(ρ_RA)  =  I(R:A)  −  | I(R:A) − 2·min( D(R|A), D(A|R) ) |
```

verified numerically to ~10⁻¹³ bits precision across all tested state
families.  This formula reveals a direct structural link between **hiding
capacity** and **quantum discord**:

- **Zero discord** — D = 0 (classical-quantum states): C = 0, perfect hiding
  is achievable.
- **Symmetric discord** — D(R|A) = D(A|R) = I(R:A)/2 (e.g. maximally
  entangled states): C = I(R:A), no hiding is possible at all.
- **General mixed states** — the gap `|I(R:A) − 2·D_min|` measures exactly
  how much of the mutual information remains hideable.

#### State Families Tested

Canonical states surveyed across the Pareto plots:

- Bell states |Φ±⟩, |Ψ±⟩ — pure, maximally entangled
- Werner states `ρ_W(p) = p·|Φ⁺⟩⟨Φ⁺| + (1−p)·I/4`
- Separable classical-quantum (CQ) states
- Noisy-channel outputs: dephasing, amplitude damping, bit-flip channels
- Bell mixtures `p·|Φ⁺⟩⟨Φ⁺| + (1−p)·|Φ⁻⟩⟨Φ⁻|`

All figures use the **Wong colorblind-safe palette** following Nature/Science
publication standards.

---

### Objective 3 — High Dimensions: Scaling to Qudits

**Question.** How do the hiding capacity, quantum discord, and the C_hide
lower bound behave for Werner, isotropic, and Bell-mixture states as the
local dimension d grows?

#### Qudit State Families

```
Werner:    ρ_W(p,d)   = p·|Φ⁺_d⟩⟨Φ⁺_d| + (1−p)·I_{d²}/d²

Isotropic: ρ_iso(p,d) = p·|Φ⁺_d⟩⟨Φ⁺_d| + (1−p)/(d²−1) · (I_{d²} − |Φ⁺_d⟩⟨Φ⁺_d|)

Bell mix:  ρ_Bell(p,d) = p·|Φ⁺_d⟩⟨Φ⁺_d| + (1−p)·|Φ⁻_d⟩⟨Φ⁻_d|
```

computed at d ∈ {2, 3, 5, 7} over the full range of mixing parameter p.

#### JAX GPU Pipeline

Both high-dimensional notebooks replace the SciPy/KAK qubit optimizer with a
fully JIT-compiled JAX pipeline:

- **Parametrisation:** U ∈ U(d²) via the **Cayley map**
  `U = (I − iH)⁻¹(I + iH)` with `H = Σ_α θ_α · G_α` expanded in d²×d²
  **Gell-Mann generators**.
- **Optimiser — Phase 1:** Adam with warmup-cosine schedule, `vmap`-batched
  over all restarts and all states simultaneously.
- **Optimiser — Phase 2:** Tight-Adam refinement (lr = 10⁻⁴), seeded from
  the best Phase-1 solution plus fresh random restarts.
- **Discord:** the same two-phase Adam pipeline applied to the classical
  mutual information maximisation over rank-1 projective measurements on A.
- **GPU fallback:** `jax.default_backend()` automatically selects the CUDA
  GPU when available; CPU execution is supported without code changes.

Per-dimension optimisation budgets (restarts, learning rate, iterations) are
tuned individually to ensure convergence at each Hilbert-space dimension.

#### Key Scaling Result

Figure 4 of `Bell_mixture_state_high_dimension.ipynb` plots D_min and
L_min = 2·D_min as functions of d ∈ {2, …, 19} at p = 0.5, probing whether
the discord–hiding correspondence survives dimension growth and how the
minimum leakage floor scales with local Hilbert-space dimension.

---

## Global Conventions

| Symbol | Definition |
|--------|-----------|
| ρ_RA | Bipartite reference-system state; **big-endian** ordering — ρ_RA ≡ kron(R, A), basis {\|00⟩, \|01⟩, \|10⟩, \|11⟩} |
| I(R:A) | Quantum mutual information: S(R) + S(A) − S(RA) |
| I_c(R▷A) | Coherent information: S(A) − S(RA) (can be negative) |
| D(R\|A) | Quantum discord measured on A: I(R:A) − max_{Π_k} I_cl(R:{Π_k}) |
| V | Isometry H_A → H_B ⊗ H_E, embedded as V\|ψ⟩_A = U(\|ψ⟩_A ⊗ \|0⟩_E) |
| S(ρ) | Von Neumann entropy: −Tr(ρ · log₂ρ) |
| S | Operator subspace span{A, D, B, B†} ⊆ L(H_A) |

**Qubit index convention:** subsystem 0 = R, subsystem 1 = A, subsystem 2 = E
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
(`/content/drive/MyDrive/...`) are declared at the top of each notebook and
can be redirected to a local output directory without further changes.

---

## Theoretical References

| Reference | Role in this work |
|-----------|------------------|
| Braunstein & Pati (2007), *PRL* — *Quantum information cannot be completely hidden in correlations* | No-hiding theorem; motivates the hiding capacity problem |
| Buscemi (2009) — *All entangled quantum states are nonlocal* | Source of the exact lower bound C ≥ 2·max(0, I_c) |
| Modi, Cabrera, Paterek, Piani & Vedral, *RMP* (2012) — *The classical-quantum boundary for correlations* | Discord framework; connects D = 0 to classical-quantum states and to the perfect-hiding condition |
| Dupuis, Fawzi & Wehner (2014) — *Decoupling with unitary approximate two-designs* | Decoupling framework motivating the isometry ansatz and the Stinespring picture |

---

## Author

**Mawulikplimi Roland Hounkpe**  
MSc Mathematical Sciences · AIMS Ghana  
Mastercard Foundation Scholar  

Supervisor: **Prof. Eric Chitambar** (University of Illinois)  

---

*"The quantum world does not simply hide information —*  
*it negotiates between what Bob learns and what Eve loses."*