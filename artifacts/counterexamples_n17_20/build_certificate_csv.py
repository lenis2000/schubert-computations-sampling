#!/usr/bin/env python3
"""
Build certificate.csv and certificate.tex from the log files under logs/.

Parses each logs/*.log file, extracts the exact integer Upsilon_w and the
wall time, groups runs by (n, permutation_label), verifies cross-method
agreement (cotrans, transition, descent, product formula), and emits:

    certificate.csv   -- long form: one row per (n, permutation, method)
    certificate.tex   -- compact:   one row per counterexample permutation,
                                    with the layered comparator and ratio,
                                    and "OK" if all methods at that n agree.

Usage (from repo root):
    python3 artifacts/counterexamples_n17_20/build_certificate_csv.py

To check byte-identical regeneration:
    python3 artifacts/counterexamples_n17_20/build_certificate_csv.py --stdout \
        | diff - artifacts/counterexamples_n17_20/certificate.tex

Assumes log filenames of the form:
    n<N>_<LABEL>_<METHOD>[_suffix].log
where LABEL in {wstar, ustar, counterex, layered} and
METHOD in {cotrans, transition, descent, product}.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

FNAME_RE = re.compile(
    r"^n(?P<n>\d+)_(?P<label>[a-z]+)_(?P<method>cotrans|transition|descent|product)"
    r"(?:_[a-z0-9]+)?\.log$"
)

UPSILON_RES = [
    re.compile(r"Result:\s+S_w\(1\^\d+\)\s*=\s*(\d+)"),   # cotrans/transition/descent
    re.compile(r"^S_w\s*=\s*(\d+)", re.M),                # product formula
]

WALL_RES = [
    re.compile(r"Computation time:\s*([\d.]+)\s*seconds"),   # schubert cotrans/transition
    re.compile(r"^Time:\s*([\d.]+)s", re.M),                 # product formula
]


@dataclass
class LogEntry:
    n: int
    label: str          # wstar, ustar, counterex, layered
    method: str         # cotrans, transition, descent, product
    filename: str       # basename of the log file
    upsilon: Optional[str] = None      # exact integer as decimal string, or None if missing
    wall_seconds: Optional[float] = None
    sha256: str = ""


def parse_log(path: str) -> Optional[LogEntry]:
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        return None
    n = int(m["n"])
    label = m["label"]
    method = m["method"]

    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")

    ups = None
    for pat in UPSILON_RES:
        mm = pat.search(text)
        if mm:
            ups = mm.group(1)
            break

    wall = None
    for pat in WALL_RES:
        mm = pat.search(text)
        if mm:
            wall = float(mm.group(1))
            break

    return LogEntry(
        n=n, label=label, method=method, filename=base,
        upsilon=ups, wall_seconds=wall, sha256=sha,
    )


# ---------------------------------------------------------------------------
# Pretty formatting helpers
# ---------------------------------------------------------------------------

# Human-readable labels for the TeX table.
LABEL_LATEX = {
    "wstar":    r"$w^{*}$",
    "ustar":    r"$u^{*}$",
    "counterex": r"$w_n^{*}$",   # generic counterexample; annotated per n
    "layered":  r"layered",
}

# Canonical one-line permutations (match the ones in the plan/README).
PERMS = {
    (17, "wstar"):    "1,3,2,7,6,5,17,4,16,15,14,13,12,11,10,9,8",
    (17, "ustar"):    "1,3,2,8,6,5,17,4,16,15,14,13,12,11,10,9,7",
    (17, "layered"):  "1,3,2,7,6,5,4,17,16,15,14,13,12,11,10,9,8",
    (18, "counterex"): "1,3,2,7,6,5,18,4,17,16,15,14,13,12,11,10,9,8",
    (18, "layered"):   "1,3,2,7,6,5,4,18,17,16,15,14,13,12,11,10,9,8",
    (19, "counterex"): "1,3,2,8,7,6,5,19,4,18,17,16,15,14,13,12,11,10,9",
    (19, "layered"):   "1,3,2,8,7,6,5,4,19,18,17,16,15,14,13,12,11,10,9",
    (20, "counterex"): "1,3,2,8,7,6,5,20,4,19,18,17,16,15,14,13,12,11,10,9",
    (20, "layered"):   "1,3,2,8,7,6,5,4,20,19,18,17,16,15,14,13,12,11,10,9",
    (21, "counterex"): "1,3,2,8,7,6,5,21,4,20,19,18,17,16,15,14,13,12,11,10,9",
    (21, "layered"):   "1,3,2,8,7,6,5,4,21,20,19,18,17,16,15,14,13,12,11,10,9",
}


def fmt_seconds(s: Optional[float]) -> str:
    if s is None:
        return "--"
    if s < 60:
        return f"{s:.2f}\\,s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}\\,m\\,{sec:.0f}\\,s"
    h, rem = divmod(m, 60)
    return f"{int(h)}\\,h\\,{int(rem):02d}\\,m"


def ratio_decimal(num: str, den: str) -> str:
    if num is None or den is None:
        return "--"
    from decimal import Decimal, getcontext
    getcontext().prec = 10
    return f"{(Decimal(num) / Decimal(den)):.4f}"


def upsilon_short(s: str) -> str:
    """Render the full integer with thousands-separators as TeX comma."""
    if s is None:
        return "--"
    # group digits in threes from the right and join with TeX commas
    groups = []
    while s:
        groups.append(s[-3:])
        s = s[:-3]
    return "$" + "{,}".join(reversed(groups)) + "$"


# ---------------------------------------------------------------------------
# CSV + TeX emission
# ---------------------------------------------------------------------------

def emit_csv(entries: list[LogEntry], path: str) -> None:
    import csv
    entries_sorted = sorted(entries, key=lambda e: (e.n, e.label, e.method))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "n", "permutation_label", "permutation", "method",
            "upsilon", "wall_seconds", "log_file", "sha256",
        ])
        for e in entries_sorted:
            perm = PERMS.get((e.n, e.label), "")
            w.writerow([
                e.n, e.label, perm, e.method,
                e.upsilon or "",
                f"{e.wall_seconds:.4f}" if e.wall_seconds is not None else "",
                e.filename, e.sha256,
            ])


def agreement_ok(rows: list[LogEntry]) -> bool:
    vals = {e.upsilon for e in rows if e.upsilon is not None}
    return len(vals) == 1


def emit_tex(entries: list[LogEntry]) -> str:
    # Group by (n, label) and pick methods.
    by_group: dict[tuple[int, str], list[LogEntry]] = {}
    for e in entries:
        by_group.setdefault((e.n, e.label), []).append(e)

    def val(n: int, label: str) -> Optional[str]:
        """Representative Upsilon string for (n, label), None if unavailable."""
        rows = by_group.get((n, label), [])
        for e in rows:
            if e.upsilon is not None:
                return e.upsilon
        return None

    def methods_string(n: int, label: str) -> str:
        rows = by_group.get((n, label), [])
        ok = [e.method for e in rows if e.upsilon is not None]
        # normalize to 3-letter abbreviations
        return ", ".join({"cotrans":"co", "transition":"tr", "descent":"de",
                          "product":"prod"}[m] for m in ok) or "--"

    def ratio_cell(n: int, label: str) -> str:
        num = val(n, label)
        den = val(n, "layered")
        return ratio_decimal(num, den) if num and den else "--"

    lines: list[str] = []
    lines.append(r"% Auto-generated by build_certificate_csv.py. DO NOT EDIT BY HAND.")
    lines.append(r"\begin{tabular}{rlrllll}")
    lines.append(r"\toprule")
    lines.append(
        r"$n$ & perm. & $\ell(w)$ & $\Upsilon_w$ & ratio vs.\ layered & "
        r"methods agreed & max wall \\"
    )
    lines.append(r"\midrule")

    def ell_of(perm_str: Optional[str]) -> object:
        if perm_str is None:
            return "--"
        perm_list = [int(x) for x in perm_str.split(",")]
        return sum(1 for i in range(len(perm_list))
                   for j in range(i + 1, len(perm_list))
                   if perm_list[i] > perm_list[j])

    def emit_row(n: int, label: str, allow_pending: bool = False) -> None:
        rows = by_group.get((n, label), [])
        if not rows and not allow_pending:
            return
        u = val(n, label)
        u_str = upsilon_short(u) if u else ("pending" if allow_pending else "--")
        ratio = "1.0000" if label == "layered" else ratio_cell(n, label)
        ok = agreement_ok([e for e in rows if e.upsilon is not None])
        meths = methods_string(n, label)
        meths_tex = meths
        # Add checkmark when >=2 methods agreed (not for single-method rows).
        n_methods = sum(1 for e in rows if e.upsilon is not None)
        if ok and n_methods >= 2:
            meths_tex = meths + r"\,\checkmark"
        walls = [e.wall_seconds for e in rows if e.wall_seconds is not None]
        wall = fmt_seconds(max(walls)) if walls else "--"
        ell = ell_of(PERMS.get((n, label)))
        label_tex = LABEL_LATEX.get(label, label)
        if label == "counterex":
            label_tex = rf"$w_{{{n}}}^{{*}}$"
        lines.append(
            rf"{n} & {label_tex} & {ell} & {u_str} & {ratio} & {meths_tex} & {wall} \\"
        )

    # Table layout per n: all counterexample permutations for that n, then the
    # single layered-comparator row.
    groups_per_n = [
        (17, [("wstar", False), ("ustar", False)]),
        (18, [("counterex", False)]),
        (19, [("counterex", False)]),
        (20, [("counterex", False)]),
        # n=21 dropped: cotrans with memo 2^30 and frontier ~700M entries
        # overflowed the 384G standard-node limit (std::bad_alloc at level
        # 143/210 after 2h 5m wall). See reference_rivanna_n21_infeasible
        # in memory. The layered product for n=21 is still in logs/ as a
        # historical artifact but is not tabled here.
    ]
    for i, (n, cx) in enumerate(groups_per_n):
        for label, allow_pending in cx:
            emit_row(n, label, allow_pending=allow_pending)
        emit_row(n, "layered")
        if i < len(groups_per_n) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=LOG_DIR)
    ap.add_argument("--out-csv", default=os.path.join(HERE, "certificate.csv"))
    ap.add_argument("--out-tex", default=os.path.join(HERE, "certificate.tex"))
    ap.add_argument("--stdout", action="store_true",
                    help="emit TeX to stdout instead of file; for diff checks")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    entries: list[LogEntry] = []
    for name in sorted(os.listdir(args.log_dir)):
        p = os.path.join(args.log_dir, name)
        if not os.path.isfile(p):
            continue
        e = parse_log(p)
        if e is None:
            if args.verbose:
                print(f"  (skip unrecognized filename: {name})", file=sys.stderr)
            continue
        entries.append(e)
        if args.verbose:
            print(f"  parsed n={e.n:2d} {e.label:<10} {e.method:<11} "
                  f"ups={'yes' if e.upsilon else 'no ':<3} "
                  f"wall={f'{e.wall_seconds:.2f}' if e.wall_seconds is not None else '--':<8} "
                  f"{e.filename}", file=sys.stderr)

    if not entries:
        print("error: no logs parsed", file=sys.stderr)
        return 1

    # Cross-method agreement check
    errors = []
    groups: dict[tuple[int, str], list[LogEntry]] = {}
    for e in entries:
        groups.setdefault((e.n, e.label), []).append(e)
    for (n, label), rows in sorted(groups.items()):
        vals = {e.upsilon for e in rows if e.upsilon is not None}
        if len(vals) > 1:
            errors.append(f"DISAGREEMENT at n={n} {label}: values = {vals}")
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 2

    complete_entries = [e for e in entries if e.upsilon is not None]
    skipped = [e for e in entries if e.upsilon is None]
    if skipped:
        print(
            "warning: ignoring log files with no parsed Result line: "
            + ", ".join(e.filename for e in skipped),
            file=sys.stderr,
        )

    tex = emit_tex(complete_entries)

    if args.stdout:
        sys.stdout.write(tex)
    else:
        emit_csv(complete_entries, args.out_csv)
        with open(args.out_tex, "w") as f:
            f.write(tex)
        print(f"wrote {args.out_csv}")
        print(f"wrote {args.out_tex}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
