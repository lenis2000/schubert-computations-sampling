# Rivanna runbook: certify n=19 transition + n=20 cotrans/transition

Preconditions: UVA VPN is on; `lap5r` can ssh to Rivanna.

Slurm script: `artifacts/counterexamples_n17_20/rivanna_certify_n19_n20.slurm`
(8 CPU, 48 GB RAM, 24 h wall; `MEMO_HARD_CAP` bumped to `1<<28` in-job).

## 1. Upload source + Slurm script

```bash
RIV_DIR=/scratch/lap5r/certify

ssh lap5r@login.hpc.virginia.edu "mkdir -p $RIV_DIR"

scp schubert.cpp \
    artifacts/counterexamples_n17_20/rivanna_certify_n19_n20.slurm \
    lap5r@login.hpc.virginia.edu:$RIV_DIR/
```

## 2. Submit the job

```bash
ssh lap5r@login.hpc.virginia.edu \
    "cd /scratch/lap5r/certify && sbatch rivanna_certify_n19_n20.slurm"
```

Note the printed `Submitted batch job <JOBID>`.

## 3. Monitor

```bash
# queue position / running status
ssh lap5r@login.hpc.virginia.edu "squeue -u lap5r"

# live job log (tail -f)
ssh lap5r@login.hpc.virginia.edu \
    "tail -f /scratch/lap5r/certify/certify_n19_n20_*.out"

# current per-run logs (populate incrementally)
ssh lap5r@login.hpc.virginia.edu "ls -la /scratch/lap5r/certify/*.log"
```

## 4. Pull logs back when done

```bash
scp "lap5r@login.hpc.virginia.edu:/scratch/lap5r/certify/n19_counterex_transition.log" \
    "lap5r@login.hpc.virginia.edu:/scratch/lap5r/certify/n20_counterex_cotrans.log" \
    "lap5r@login.hpc.virginia.edu:/scratch/lap5r/certify/n20_counterex_transition.log" \
    artifacts/counterexamples_n17_20/logs/
```

## 5. Verify results on laptop

```bash
grep -E 'Result:|S_w\s*=' artifacts/counterexamples_n17_20/logs/n19_counterex_transition.log \
                          artifacts/counterexamples_n17_20/logs/n20_counterex_cotrans.log \
                          artifacts/counterexamples_n17_20/logs/n20_counterex_transition.log
```

Expected (from already-certified cotrans at n=19):
- `n19_counterex_transition.log`: `Result: S_w(1^19) = 2635009155895571409093282000` (digit-for-digit match with laptop cotrans).
- `n20_counterex_cotrans.log` / `n20_counterex_transition.log`: two identical 31-digit integers. The ratio over the layered product $4{,}107{,}026{,}814{,}014{,}206{,}081{,}132{,}873{,}740{,}000$ should be ~1.08 (+8%, per `main_pnas.tex:1090`).

## 6. After logs are in place

```bash
git add artifacts/counterexamples_n17_20/logs/
git commit -m "Rivanna: certify n=19 transition and n=20 cotrans/transition"
```

Then proceed with the plan's Task 3 / Task 4 (build_certificate_csv.py + SI wiring).
