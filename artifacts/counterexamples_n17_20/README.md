# Counterexamples to Merzon-Smirnov at n=17,18,19,20

Certification trail for Theorem 0.2 (main paper) and the
values reported for $n=18,19,20$.

## Status

| $n$ | Permutation | cotrans (exact) | transition (exact) | layered comparator | $\Upsilon_w$ |
|---|---|---|---|---|---|
| 17 | $w^*$ | done (`logs/n17_wstar_cotrans.log`) | done | product (`logs/n17_layered_product.log`) | $3{,}272{,}424{,}600{,}397{,}137{,}120{,}000$ |
| 17 | $u^*$ | done (`logs/n17_ustar_cotrans.log`) | done | — | $3{,}528{,}445{,}515{,}842{,}977{,}489{,}500$ |
| 17 | $w(1,2,4,10)$ | — | — | done | $3{,}050{,}684{,}475{,}186{,}219{,}300{,}000$ |
| 18 | $s_7 w(1,2,4,11)$ | in progress | in progress | done (`logs/n18_layered_product.log`) | to fill |
| 19 | $s_8 w(1,2,5,11)$ | done (`logs/n19_counterex_cotrans.log`, laptop) | done (`logs/n19_counterex_transition_big.log`, Rivanna 384 GB, 5:29 min) | done | $2{,}635{,}009{,}155{,}895{,}571{,}409{,}093{,}282{,}000$ (ratio 1.0944) |
| 20 | $s_8 w(1,2,5,12)$ | done (`logs/n20_counterex_cotrans.log`, Rivanna 200 GB, 1 h wall) | done (`logs/n20_counterex_transition_big.log`, Rivanna 384 GB, 24:28 min) | done | $4{,}432{,}622{,}773{,}682{,}098{,}875{,}213{,}559{,}764{,}000$ (ratio 1.0793) |

Digit-for-digit cross-check between cotrans and transition at $n=17$:
both implementations return the same 22-digit integer for $w^*$ and the
same 22-digit integer for $u^*$. Product formula for $w(1,2,4,10)$
agrees with main paper `main_pnas.tex:440`.

A separate Julia cross-check is deferred; the two independent C++
formulas serve as the primary certification. Julia logs will be
added under `julia/` if/when re-run for the formal release snapshot.

## One-line permutations

Derived from `main_pnas.tex:428-453, 1090` and the layered construction.

```
n=17 w*:            1,3,2,7,6,5,17,4,16,15,14,13,12,11,10,9,8
n=17 u*:            1,3,2,8,6,5,17,4,16,15,14,13,12,11,10,9,7
n=17 layered:       1,3,2,7,6,5,4,17,16,15,14,13,12,11,10,9,8     (= w(1,2,4,10))
n=18 counterex:     1,3,2,7,6,5,18,4,17,16,15,14,13,12,11,10,9,8  (= s_7 * w(1,2,4,11))
n=18 layered:       1,3,2,7,6,5,4,18,17,16,15,14,13,12,11,10,9,8  (= w(1,2,4,11))
n=19 counterex:     1,3,2,8,7,6,5,19,4,18,17,16,15,14,13,12,11,10,9  (= s_8 * w(1,2,5,11))
n=19 layered:       1,3,2,8,7,6,5,4,19,18,17,16,15,14,13,12,11,10,9  (= w(1,2,5,11))
n=20 counterex:     1,3,2,8,7,6,5,20,4,19,18,17,16,15,14,13,12,11,10,9  (= s_8 * w(1,2,5,12))
n=20 layered:       1,3,2,8,7,6,5,4,20,19,18,17,16,15,14,13,12,11,10,9  (= w(1,2,5,12))
```

## How the existing logs were produced

From the repo root, with `code/schubert` built per `CLAUDE.md`:

```
./code/schubert --exact --cotrans   '<one-line>'        # counterexample cotrans
./code/schubert --exact --transition '<one-line>'        # counterexample transition
./code/schubert layered:b1,b2,b3,b4                      # layered comparator (product)
```

Each invocation writes stdout+stderr to
`logs/<n><name>_<method>.log`. Wall time appears in the last line
(`Computation time:` for cotrans/transition, `Time:` for the product).

## Rivanna submission (n=19 transition cross-check + n=20 full cert)

Slurm script: `rivanna_certify_n19_n20.slurm`.

Submission recipe (from a logged-in Rivanna session, with the repo
mirrored into `$SCHUBERT_DIR`):

```
cp code/schubert.cpp $SCHUBERT_DIR/
cp PNAS/artifacts/counterexamples_n17_20/rivanna_certify_n19_n20.slurm \
   $SCHUBERT_DIR/
cd $SCHUBERT_DIR && sbatch rivanna_certify_n19_n20.slurm
# ... wait for completion (expected: n=19 transition ~1h, n=20 cotrans
# several hours, n=20 transition possibly overnight) ...
# after completion, pull the three logs back:
scp rivanna:$SCHUBERT_DIR/n19_counterex_transition.log \
    rivanna:$SCHUBERT_DIR/n20_counterex_cotrans.log \
    rivanna:$SCHUBERT_DIR/n20_counterex_transition.log \
    PNAS/artifacts/counterexamples_n17_20/logs/
```

The Slurm script requests 8 CPUs, 48 GB RAM, 24 h wall. It bumps
`MEMO_HARD_CAP` to `1ULL << 28` (256M entries, ~25 GB RAM) before
compiling, which is the ceiling appropriate for a node with 48 GB.
The shipped `schubert.cpp` is untouched; the patched source lives
only inside the job in `schubert_rivanna.cpp`.

The laptop-side cotransition run at n=19 produced the correct integer
(see `logs/n19_counterex_cotrans.log`); the Rivanna transition log
serves as the independent cross-check.

## Still needed (after Rivanna returns)

- Overwrite `logs/n19_counterex_transition.log` and place the two
  `logs/n20_counterex_*.log` files under version control.
- `build_certificate_csv.py`: parse every `logs/*.log`, emit
  `certificate.csv` (long form) and `certificate.tex` (one row per
  permutation).
- Wire `\input{artifacts/counterexamples_n17_20/certificate.tex}` into
  SI Section 2 under a new subsection
  `\subsection{Certificate of the counterexamples}`.
