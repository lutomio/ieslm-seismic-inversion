# iES-LM for 1D acoustic seismic inversion

Python implementation of the **iterative ensemble smoother with
Levenberg-Marquardt** (iES-LM, called ES-LM in the original paper), applied to
one-dimensional acoustic impedance inversion and compared against **ES-MDA**.

> Ma, X. and Bi, L. (2019). *A robust iterative ensemble smoother method for
> efficient history matching and uncertainty quantification.*
> Computational Geosciences 23:415–442.

The two methods differ in how they regularize the ensemble update step. ES-MDA
uses a **fixed** sequence of inflation factors chosen in advance; iES-LM adapts
its regularization parameter at every iteration through a trust-region rule
driven by the gain ratio. Comparing those two strategies under identical
conditions is the point of this repository.

Written as the practical part of an undergraduate thesis in Computer Science at
UFSC (Universidade Federal de Santa Catarina).

## Install

Requires Python 3.9+ and the [SeReMpy](https://github.com/dariograna/SeReMpy)
library, which supplies the forward-model primitives and the reference ES-MDA
implementation. SeReMpy is **not** modified or vendored here.

```bash
git clone https://github.com/lutomio/ieslm-seismic-inversion.git
cd ieslm-seismic-inversion
pip install -r requirements.txt
```

Then make SeReMpy available, either by placing it next to this project:

```bash
git clone https://github.com/dariograna/SeReMpy.git ../SeReMpy-main
```

or by pointing an environment variable at it:

```bash
export SEREMPY_PATH=/path/to/SeReMpy
```

## Run

The comparison experiment prints the metrics and writes the figures:

```bash
python experimento.py
```

The test suite:

```bash
python -m pytest tests/ -v
```

## Layout

| File | Contents |
|---|---|
| `ieslm.py` | **The algorithm** — Algorithm 2 of Ma & Bi (2019) |
| `forward.py` | Acoustic forward model, `g(Z) = W (1/2) D ln Z` |
| `prior.py` | Prior ensemble of impedance profiles |
| `dados.py` | Data loading and SeReMpy lookup |
| `experimento.py` | iES-LM vs ES-MDA driver and figures |
| `tests/` | 67 tests, see below |
| `data/` | Two data files redistributed from SeReMpy (MIT) |

Source and identifiers are in Portuguese, since this is a thesis project. The
mapping to the paper's notation is in the table below.

`ieslm.py` is **domain-agnostic**: it takes the forward model as a callable
`g`, so the same core runs both the paper's synthetic example and the seismic
problem. That is what makes it possible to validate the implementation against
a published result.

## Paper → code

| Symbol | Eq. | Location |
|---|---|---|
| C_MD, C_DD | 30, 31 | `covariancias()` |
| model update | 32 | `M + C_MD @ V` in `ieslm()` |
| per-member objective | 34 | `objetivo_por_membro()` (perturbed data) |
| gain ratio ρ_j | 37, 38 | `reducao_real / reducao_prevista` |
| average mismatch Ō | 39 | `desajuste_medio()` (unperturbed data) |
| γ, α update | 40, 41 | `fator_lm()` + median rule |
| discrepancy stop | 42, 43 | `desajuste_absoluto()`, `fator_ruido` |

## Validation

The anchor test reproduces the synthetic linear example of Section 5.1 of the
paper, whose maximum-likelihood solution is known analytically (4.76543):

| N_e | This implementation | Paper (Fig. 1) |
|---|---|---|
| 10 | 4.76640 | 4.76552 |
| 100 | 4.76599 | 4.76528 |
| 500 | 4.76385 | 4.76540 |

The forward model is checked against the real seismic trace (correlation
0.998), and every equation is unit-tested against hand-computable values.

## Two notes on the paper

**Eq. 38 as printed omits the 1/2 factor.** Eq. 34 defines the objective with a
1/2, and the paper states that `L(mⁱ) = O(mⁱ)`. Deriving `L` from Eq. 36 using
`Ḡ C_MD = C_DD` yields `L = (α²/2) vᵀ C_D v`. The factor is verifiable
independently: under a linear forward model the linearization is exact, so ρ
must equal 1. Measured: `1.00000000` with the factor, `1.103` without it.

**The Eq. 43 stopping rule is what prevents overfitting.** iES-LM minimizes a
maximum-likelihood objective with no prior term. Left to run on the Section 4.3
criteria alone it keeps reducing the mismatch until it fits the noise: in the
experiment the mismatch reached 9e-10 while the RMSE against the reference well
degraded to 3.86, five times worse than the initial guess, and the ensemble
collapsed. With the discrepancy rule (`fator_ruido`) the result reverses.

## Scope

This implements Algorithm 2 (Section 4) of the paper. The variants for unknown
diagonal covariance (Section 6, aES-LM) and for robust regression (Section 7,
rES-LM) are not implemented.

## License

MIT, see `LICENSE`. Third-party notices in `THIRD_PARTY.md`.
