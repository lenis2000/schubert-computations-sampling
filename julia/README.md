# Independent Julia cross-check of the certified n=17 values

This directory contains a small Julia script that independently verifies the
certified C++ values of $\Upsilon_w = \mathfrak{S}_w(1, \ldots, 1)$ for the
three $n=17$ permutations relevant to the disproof of the Merzon--Smirnov
conjecture in Anderson--Panova--Petrov,
*Computation and sampling for Schubert specializations*.

The Julia path uses **`SchubertPolynomials.jl`** (D. Anderson, OSU), function
`nschub`. Reference:
[github.com/pseudoeffective/SchubertPolynomials.jl](https://github.com/pseudoeffective/SchubertPolynomials.jl).

## Files

| File | What it is |
|---|---|
| `run_n17_check.jl` | The verification script. Hard-codes the three permutations and the certified C++ values; calls `nschub` and asserts equality. |
| `Project.toml` | Declares `SchubertPolynomials` as a dependency. |
| `n17_julia_check.log` | Output of one run on a 2026-04-30 macOS Apple-silicon laptop, Julia 1.12.6. All three values match digit-for-digit. |

## Reproducing

Requires Julia $\ge 1.10.9$ (per `SchubertPolynomials.jl`'s declared compat).
From this directory:

```bash
# First-time install (the package is not on the General registry):
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/pseudoeffective/SchubertPolynomials.jl")'

# Run the cross-check:
julia --project=. run_n17_check.jl
```

The first invocation precompiles `Nemo`/`AbstractAlgebra` and takes ~1 minute.
The verification itself completes in ~13 seconds wall time on M2 Pro: about
6 s each for $w^*$ and $u^*$, and a fraction of a second for the layered
comparator.

## Permutations

In one-line notation:

| Name | Permutation | $\Upsilon_w$ |
|---|---|---|
| $w^*$ | `1,3,2,7,6,5,17,4,16,15,14,13,12,11,10,9,8` | $3{,}272{,}424{,}600{,}397{,}137{,}120{,}000$ |
| $u^*$ | `1,3,2,8,6,5,17,4,16,15,14,13,12,11,10,9,7` | $3{,}528{,}445{,}515{,}842{,}977{,}489{,}500$ |
| $w(1,2,4,10)$ | `1,3,2,7,6,5,4,17,16,15,14,13,12,11,10,9,8` | $3{,}050{,}684{,}475{,}186{,}219{,}300{,}000$ |

The $w^*$ and $u^*$ values exceed the layered comparator $\Upsilon_{w(1,2,4,10)}$,
disproving the conjecture that the maximum of $\Upsilon_w$ over $S_n$ is
attained on a layered permutation.

## Relationship to the C++ certification

The primary certification trail is the digit-for-digit agreement between the
two independent C++ implementations of the cotransition (Knutson) and
transition (Lascoux--Schützenberger) recurrences in `../schubert.cpp`,
recorded for $n = 17, 18, 19, 20$. This Julia script provides a third,
independently-implemented check at $n=17$ using a completely different
computational pathway (BPDs / divided differences / Nemo polynomial
expansion).
