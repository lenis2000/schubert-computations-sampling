#!/usr/bin/env python3
"""
Performance benchmark suite for Schubert polynomial implementations.

Tests all 6 implementations
(descent_double, descent_rational, cotrans_double, cotrans_exact, transition_double, transition_exact)
across layered permutations (MPP3 validation) and random permutations from CFTP sampler.

Usage:
    python benchmark_suite.py --list-sizes                    # Show available random sizes
    python benchmark_suite.py --layered --max-n 16
    python benchmark_suite.py --random --random-size 15 --samples 50
    python benchmark_suite.py --random --random-size 20 --samples 10 --binaries cotrans_double cotrans_exact
    python benchmark_suite.py --all --binaries descent_double descent_rational

    # For large n, skip slow exact/rational arithmetic:
    python benchmark_suite.py --random --random-size 20 --samples 5 --double-only
    python benchmark_suite.py --random --random-size 25 --samples 5 --double-only --timeout 120

    # With deadlines:
    python benchmark_suite.py --random --random-size 20 --samples 100 --deadline 300  # Stop after 5 min
"""

import argparse
import csv
import json
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR
RESULTS_DIR = SCRIPT_DIR / "benchmark_results"

# Unified binary path and variant flags
UNIFIED_BINARY = SCRIPT_DIR / "schubert"

VARIANT_FLAGS = {
    "descent_double": ["--double", "--descent"],
    "descent_rational": ["--exact", "--descent"],
    "cotrans_double": ["--double", "--cotrans"],
    "cotrans_exact": ["--exact", "--cotrans"],
    "transition_double": ["--double", "--transition"],
    "transition_exact": ["--exact", "--transition"],
}

# MPP3 data file
MPP3_DATA_FILE = PROJECT_ROOT / "data_MPP3.tex"

# Default permutation directory (can be overridden via CLI)
DEFAULT_PERMS_DIR = SCRIPT_DIR / "permutation_data"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BinaryResult:
    """Result from running a single binary on a permutation."""
    binary: str
    permutation: str
    n: int
    value: Optional[int]
    time_seconds: float
    success: bool
    error: Optional[str] = None
    memo_size: Optional[int] = None  # Number of memo entries
    iterations: Optional[int] = None  # Total iterations
    iter_per_sec: Optional[float] = None  # Iterations per second
    log2_normalized: Optional[float] = None  # log2(S_w) / n^2


@dataclass
class LayeredEntry:
    """Entry from MPP3 layered permutation data."""
    n: int
    layers: Tuple[int, ...]
    f_n: float  # log2(v(n)) / n^2


@dataclass
class BenchmarkResult:
    """Result from benchmarking one permutation across all binaries."""
    permutation: str
    n: int
    ell: Optional[int]
    results: Dict[str, BinaryResult] = field(default_factory=dict)


# =============================================================================
# Binary Runner
# =============================================================================

class BinaryRunner:
    """Unified execution and parsing for Schubert polynomial variants via unified binary."""

    # Match "Result: S_w(1^n) = X" (actual value)
    RESULT_VALUE_PATTERN = re.compile(r"Result:\s*S_w\(1\^(\d+)\)\s*=\s*([0-9eE+.-]+)")
    # Match "Result: S_w(1^n) has ~X digits" (cotrans_double with product formula)
    RESULT_DIGITS_PATTERN = re.compile(r"Result:\s*S_w\(1\^(\d+)\)\s*has ~(\d+) digits")
    # Match log2 value for approximate computation: "log2(S_w) / n^2 = X" or "~ X"
    LOG2_PATTERN = re.compile(r"log2\(S_w\)\s*/\s*n\^2\s*[=~]\s*([0-9.]+)")
    TIME_PATTERN = re.compile(r"Computation time:\s*([0-9.]+)\s*seconds?")
    ELL_PATTERN = re.compile(r"Length ell\(w\)\s*=\s*(\d+)")
    # Match "Memo size: X entries" or "Memo: X" (with K/M suffixes)
    MEMO_PATTERN = re.compile(r"Memo(?:\s*size)?:\s*([0-9.]+)([KMB]?)\s*(?:entries)?")
    # Match "Total iterations: X" or "Iterations: X" (with K/M suffixes)
    ITER_PATTERN = re.compile(r"(?:Total\s+)?[Ii]terations:\s*([0-9.]+)([KMB]?)")
    # Match "X iter/sec"
    ITER_SEC_PATTERN = re.compile(r"([0-9.]+)\s*iter/sec")

    def __init__(self, variant_name: str, flags: List[str]):
        self.name = variant_name
        self.flags = flags
        self.is_double = "double" in variant_name

    @staticmethod
    def parse_with_suffix(value_str: str, suffix: str) -> int:
        """Parse a number with optional K/M/B suffix."""
        multipliers = {'': 1, 'K': 1000, 'M': 1000000, 'B': 1000000000}
        return int(float(value_str) * multipliers.get(suffix, 1))

    def run(self, perm_str: str, timeout: Optional[float] = None, extra_flags: Optional[List[str]] = None,
            stream_output: bool = False) -> BinaryResult:
        """Run the unified binary with variant flags and parse results.

        Args:
            perm_str: Comma-separated permutation string
            timeout: Optional timeout in seconds
            extra_flags: Additional CLI flags
            stream_output: If True, print stdout lines in real-time
        """
        if not UNIFIED_BINARY.exists():
            return BinaryResult(
                binary=self.name,
                permutation=perm_str,
                n=len(perm_str.split(",")),
                value=None,
                time_seconds=0,
                success=False,
                error=f"Unified binary not found: {UNIFIED_BINARY}"
            )

        try:
            flags = self.flags + (extra_flags or [])
            start = time.perf_counter()

            # Use Popen for streaming output, otherwise use simpler run()
            if stream_output:
                proc = subprocess.Popen(
                    [str(UNIFIED_BINARY)] + flags + [perm_str],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(SCRIPT_DIR),
                )

                # Accumulate output while streaming
                stdout_lines = []
                try:
                    while True:
                        line = proc.stdout.readline()
                        if not line and proc.poll() is not None:
                            break
                        if line:
                            stdout_lines.append(line)
                            # Print progress lines in-place (overwrite previous)
                            if "Iter:" in line or "Progress:" in line or "Memo:" in line:
                                print(f"\r    {line.rstrip():<80}", end="", flush=True)
                        # Check timeout
                        if timeout and (time.perf_counter() - start) > timeout:
                            proc.kill()
                            proc.wait()
                            raise subprocess.TimeoutExpired(cmd=perm_str, timeout=timeout)
                except subprocess.TimeoutExpired:
                    raise

                # Clear the progress line
                print("\r" + " " * 90 + "\r", end="", flush=True)
                stdout = "".join(stdout_lines)
                stderr = proc.stderr.read()
                returncode = proc.returncode
            else:
                result = subprocess.run(
                    [str(UNIFIED_BINARY)] + flags + [perm_str],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(SCRIPT_DIR),
                )
                stdout = result.stdout
                stderr = result.stderr
                returncode = result.returncode

            wall_time = time.perf_counter() - start

            if returncode != 0:
                return BinaryResult(
                    binary=self.name,
                    permutation=perm_str,
                    n=len(perm_str.split(",")),
                    value=None,
                    time_seconds=wall_time,
                    success=False,
                    error=f"Exit code {returncode}: {stderr}"
                )

            # Parse result - try value pattern first, then digits pattern
            value_match = self.RESULT_VALUE_PATTERN.search(stdout)
            digits_match = self.RESULT_DIGITS_PATTERN.search(stdout)
            log2_match = self.LOG2_PATTERN.search(stdout)
            time_match = self.TIME_PATTERN.search(stdout)

            n = len(perm_str.split(","))
            value = None

            if value_match:
                # Have actual value: "Result: S_w(1^n) = X"
                n = int(value_match.group(1))
                raw_value = value_match.group(2)
                if self.is_double:
                    value = int(round(float(raw_value)))
                else:
                    value = int(raw_value)
            elif digits_match and log2_match:
                # Have digits only: "Result: S_w(1^n) has ~X digits"
                # Compute approximate value from log2
                n = int(digits_match.group(1))
                f_n = float(log2_match.group(1))
                # value = 2^(f_n * n^2), but round to int
                approx_log2 = f_n * n * n
                value = int(round(math.pow(2, approx_log2)))
            else:
                return BinaryResult(
                    binary=self.name,
                    permutation=perm_str,
                    n=n,
                    value=None,
                    time_seconds=wall_time,
                    success=False,
                    error=f"Could not parse result from output:\n{stdout}"
                )

            # Parse reported time (prefer this over wall time for accuracy)
            if time_match:
                reported_time = float(time_match.group(1))
            else:
                reported_time = wall_time

            # Parse log2 normalized value
            log2_normalized = None
            if log2_match:
                log2_normalized = float(log2_match.group(1))

            # Parse memo size (find last occurrence for final value)
            memo_size = None
            for memo_match in self.MEMO_PATTERN.finditer(stdout):
                memo_size = self.parse_with_suffix(memo_match.group(1), memo_match.group(2))

            # Parse iterations (find last occurrence for final value)
            iterations = None
            for iter_match in self.ITER_PATTERN.finditer(stdout):
                iterations = self.parse_with_suffix(iter_match.group(1), iter_match.group(2))

            # Parse iter/sec (find last occurrence)
            iter_per_sec = None
            for ips_match in self.ITER_SEC_PATTERN.finditer(stdout):
                iter_per_sec = float(ips_match.group(1))

            return BinaryResult(
                binary=self.name,
                permutation=perm_str,
                n=n,
                value=value,
                time_seconds=reported_time,
                success=True,
                memo_size=memo_size,
                iterations=iterations,
                iter_per_sec=iter_per_sec,
                log2_normalized=log2_normalized
            )

        except subprocess.TimeoutExpired:
            return BinaryResult(
                binary=self.name,
                permutation=perm_str,
                n=len(perm_str.split(",")),
                value=None,
                time_seconds=timeout,
                success=False,
                error=f"Timeout after {timeout}s"
            )
        except Exception as e:
            return BinaryResult(
                binary=self.name,
                permutation=perm_str,
                n=len(perm_str.split(",")),
                value=None,
                time_seconds=0,
                success=False,
                error=str(e)
            )

    @staticmethod
    def parse_ell(output: str) -> Optional[int]:
        """Extract length ell(w) from binary output."""
        match = BinaryRunner.ELL_PATTERN.search(output)
        return int(match.group(1)) if match else None


# =============================================================================
# MPP3 Data Loader
# =============================================================================

class MPP3DataLoader:
    """Parse layered permutation data from MPP3 paper (data_MPP3.tex)."""

    # Pattern: n & (layers) & f(n)
    # e.g., "5&(1, 1, 3)&0.152294\\"
    ENTRY_PATTERN = re.compile(r"(\d+)&\(([^)]+)\)&([0-9.]+)")

    def __init__(self, data_file: Path = MPP3_DATA_FILE):
        self.data_file = data_file
        self._cache: Dict[int, LayeredEntry] = {}

    def load(self) -> Dict[int, LayeredEntry]:
        """Load all layered permutation data from the file."""
        if self._cache:
            return self._cache

        if not self.data_file.exists():
            raise FileNotFoundError(f"MPP3 data file not found: {self.data_file}")

        with open(self.data_file, "r") as f:
            content = f.read()

        for match in self.ENTRY_PATTERN.finditer(content):
            n = int(match.group(1))
            layers_str = match.group(2)
            f_n = float(match.group(3))

            # Parse layers tuple (comma-separated integers)
            layers = tuple(int(x.strip()) for x in layers_str.split(","))

            self._cache[n] = LayeredEntry(n=n, layers=layers, f_n=f_n)

        return self._cache

    def get_entry(self, n: int) -> Optional[LayeredEntry]:
        """Get layered permutation data for a specific n."""
        if not self._cache:
            self.load()
        return self._cache.get(n)

    @staticmethod
    def layers_to_permutation(layers: Tuple[int, ...]) -> List[int]:
        """
        Convert layer sizes to permutation array.

        Layers are given as (b_k, ..., b_2, b_1) where:
        - b_k is the SMALLEST layer (leftmost in notation)
        - b_1 is the LARGEST layer (rightmost in notation)

        A layered permutation has:
        - Layer k occupies positions 1..b_k with values 1..b_k in DECREASING order
        - Layer k-1 occupies next positions with next values in DECREASING order
        - etc.

        Example: (1, 2) for n=3 gives [1, 3, 2]
        - Layer of size 1: value 1 at position 1
        - Layer of size 2: values 3, 2 at positions 2, 3
        """
        perm = []
        offset = 0

        for layer_size in layers:
            # This layer contains values from offset+1 to offset+layer_size
            # in DECREASING order
            layer_values = list(range(offset + layer_size, offset, -1))
            perm.extend(layer_values)
            offset += layer_size

        return perm

    @staticmethod
    def permutation_to_string(perm: List[int]) -> str:
        """Convert permutation list to comma-separated string."""
        return ",".join(map(str, perm))


# =============================================================================
# Permutation Loader
# =============================================================================

class PermutationLoader:
    """Load permutations from CFTP sampler result files."""

    # Pattern: {1,2,3,...}
    PERM_PATTERN = re.compile(r"\{([0-9,]+)\}")
    # Pattern for filename: perms_n{N}_B{count}.txt
    FILENAME_PATTERN = re.compile(r"perms_n(\d+)_B(\d+)\.txt")

    @staticmethod
    def discover_available_sizes(base_dir: Path) -> Dict[int, int]:
        """Discover available sizes from perms files in directory.

        Returns dict mapping n -> max_samples available.
        """
        available = {}
        if not base_dir.exists():
            return available

        for f in base_dir.iterdir():
            match = PermutationLoader.FILENAME_PATTERN.match(f.name)
            if match:
                n = int(match.group(1))
                count = int(match.group(2))
                # Keep the largest sample count for each n
                if n not in available or count > available[n]:
                    available[n] = count

        return available

    @staticmethod
    def load_file(filepath: Path, max_count: Optional[int] = None) -> List[str]:
        """Load permutations from file, returning comma-separated strings."""
        if not filepath.exists():
            raise FileNotFoundError(f"Permutation file not found: {filepath}")

        perms = []
        with open(filepath, "r") as f:
            for line in f:
                match = PermutationLoader.PERM_PATTERN.search(line)
                if match:
                    perms.append(match.group(1))
                    if max_count and len(perms) >= max_count:
                        break

        return perms

    @staticmethod
    def load_perms_for_n(n: int, base_dir: Path, max_count: Optional[int] = None) -> List[str]:
        """Load permutations for a specific n from the given directory.

        Looks for perms_n{n}_B*.txt, preferring larger sample counts.
        """
        # Find all matching files for this n
        candidates = []
        if base_dir.exists():
            for f in base_dir.iterdir():
                match = PermutationLoader.FILENAME_PATTERN.match(f.name)
                if match and int(match.group(1)) == n:
                    candidates.append((int(match.group(2)), f))

        if not candidates:
            raise FileNotFoundError(f"No permutation file found for n={n} in {base_dir}")

        # Use file with most samples
        candidates.sort(reverse=True)
        filepath = candidates[0][1]
        return PermutationLoader.load_file(filepath, max_count)


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Orchestrate benchmarking across all implementations."""

    def __init__(self, verbose: bool = True, perms_dir: Path = DEFAULT_PERMS_DIR,
                 timeout: Optional[float] = None, deadline: Optional[float] = None):
        self.verbose = verbose
        self.perms_dir = perms_dir
        self.timeout = timeout
        self.deadline = deadline
        self.start_time: Optional[float] = None
        self.runners = {
            name: BinaryRunner(name, flags)
            for name, flags in VARIANT_FLAGS.items()
        }
        self.mpp3_loader = MPP3DataLoader()

    def check_deadline(self) -> bool:
        """Return True if deadline exceeded."""
        if self.deadline is None or self.start_time is None:
            return False
        return time.perf_counter() - self.start_time > self.deadline

    def elapsed(self) -> float:
        """Return elapsed time since start."""
        if self.start_time is None:
            return 0
        return time.perf_counter() - self.start_time

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def run_single(self, perm_str: str, binaries: Optional[List[str]] = None) -> BenchmarkResult:
        """Run all specified binaries on a single permutation."""
        if binaries is None:
            binaries = list(self.runners.keys())

        n = len(perm_str.split(","))
        result = BenchmarkResult(permutation=perm_str, n=n, ell=None)

        for binary_name in binaries:
            if binary_name in self.runners:
                result.results[binary_name] = self.runners[binary_name].run(perm_str)

        return result

    def benchmark_layered(self, max_n: int = 16, selected_binaries: Optional[List[str]] = None, timeout: Optional[float] = None) -> List[Dict]:
        """Benchmark on MPP3 layered permutations for validation and timing."""
        self.log(f"=== Layered Permutation Benchmark (n=1..{max_n}) ===")

        mpp3_data = self.mpp3_loader.load()
        results = []

        # Filter runners if specific binaries selected
        runners_to_use = {k: v for k, v in self.runners.items()
                         if selected_binaries is None or k in selected_binaries}

        for n in range(1, max_n + 1):
            entry = mpp3_data.get(n)
            if not entry:
                self.log(f"  n={n}: No MPP3 data available")
                continue

            # Convert layers to permutation
            perm = MPP3DataLoader.layers_to_permutation(entry.layers)
            perm_str = MPP3DataLoader.permutation_to_string(perm)

            # Compute ell(w) = number of inversions
            ell = sum(1 for i in range(len(perm)) for j in range(i+1, len(perm)) if perm[i] > perm[j])

            # Get exact value from product formula for comparison
            product_runner = self.runners.get("cotrans_exact") or self.runners.get("cotrans_double")
            if product_runner:
                product_result = product_runner.run(perm_str, stream_output=False)  # Uses product formula by default
                exact_product_value = product_result.value
            else:
                exact_product_value = None

            self.log(f"  n={n:2d} layers={entry.layers} ell={ell} exact={exact_product_value}")

            result_row = {
                "n": n,
                "layers": entry.layers,
                "layers_str": str(entry.layers),
                "permutation": perm_str,
                "ell": ell,
                "expected_f_n": entry.f_n,
                "exact_product_value": exact_product_value,
            }

            # Run selected binaries (--no-product forces algorithm, disables layered shortcut)
            # Enable streaming for n >= 13 where computations take longer
            extra_flags = ["--no-product"]
            for binary_name, runner in runners_to_use.items():
                br = runner.run(perm_str, timeout=timeout, extra_flags=extra_flags, stream_output=(n >= 13))
                result_row[f"{binary_name}_time"] = br.time_seconds
                result_row[f"{binary_name}_value"] = br.value
                result_row[f"{binary_name}_success"] = br.success
                result_row[f"{binary_name}_memo"] = br.memo_size
                result_row[f"{binary_name}_iter"] = br.iterations
                result_row[f"{binary_name}_iter_per_sec"] = br.iter_per_sec
                result_row[f"{binary_name}_log2_norm"] = br.log2_normalized

                # Check against exact product value
                value_match = (br.value == exact_product_value) if (br.value is not None and exact_product_value is not None) else None
                result_row[f"{binary_name}_match"] = value_match

                if br.success:
                    match_str = "" if value_match else " [MISMATCH!]" if value_match is False else ""
                    self.log(f"       {binary_name}: {br.time_seconds:.4f}s value={br.value}{match_str}")
                else:
                    self.log(f"       {binary_name}: TIMEOUT or FAILED - {br.error}")

            # Compute relative errors (double vs exact)
            exact_fallback = (
                result_row.get("descent_rational_value")
                or result_row.get("cotrans_exact_value")
                or result_row.get("transition_exact_value")
            )
            paired_variants = [
                ("descent_double", "descent_rational"),
                ("cotrans_double", "cotrans_exact"),
                ("transition_double", "transition_exact"),
            ]
            for dbl_name, exact_name in paired_variants:
                dbl_val = result_row.get(f"{dbl_name}_value")
                exact_val = result_row.get(f"{exact_name}_value")
                if exact_val is None:
                    exact_val = exact_fallback
                if dbl_val is not None and exact_val is not None and exact_val != 0:
                    rel_err = abs(dbl_val - exact_val) / exact_val
                    result_row[f"{dbl_name}_rel_error"] = rel_err

            results.append(result_row)

        return results

    def benchmark_random(self, n: int, num_samples: int, selected_binaries: Optional[List[str]] = None) -> List[Dict]:
        """Benchmark on random permutations from CFTP sampler."""
        self.log(f"=== Random Permutation Benchmark (n={n}, samples={num_samples}) ===")
        timeout_str = f"{self.timeout}s" if self.timeout else "none"
        deadline_str = f", Deadline: {self.deadline}s total" if self.deadline else ""
        self.log(f"    Timeout: {timeout_str} per binary{deadline_str}")

        try:
            perms = PermutationLoader.load_perms_for_n(n, self.perms_dir, max_count=num_samples)
        except FileNotFoundError as e:
            self.log(f"  Error: {e}")
            return []

        if len(perms) < num_samples:
            self.log(f"  Warning: Only {len(perms)} permutations available (requested {num_samples})")

        # Filter runners if specific binaries selected
        runners_to_use = {k: v for k, v in self.runners.items()
                         if selected_binaries is None or k in selected_binaries}

        self.log(f"    Binaries: {', '.join(runners_to_use.keys())}")
        self.log("")

        # Track timing for progress estimates
        self.start_time = time.perf_counter()
        sample_times: List[float] = []
        timeout_counts: Dict[str, int] = {name: 0 for name in runners_to_use}

        results = []
        actual_samples = min(num_samples, len(perms))

        for i, perm_str in enumerate(perms[:actual_samples]):
            # Check deadline before starting new sample
            if self.check_deadline():
                self.log(f"\n  DEADLINE REACHED after {self.elapsed():.1f}s - stopping at sample {i}/{actual_samples}")
                break

            sample_start = time.perf_counter()
            perm = [int(x) for x in perm_str.split(",")]
            ell = sum(1 for ii in range(len(perm)) for jj in range(ii+1, len(perm)) if perm[ii] > perm[jj])

            result_row = {
                "sample_index": i,
                "n": n,
                "permutation": perm_str,
                "ell": ell,
            }

            # Progress header for this sample
            if self.verbose:
                elapsed = self.elapsed()
                if sample_times:
                    avg_time = sum(sample_times) / len(sample_times)
                    eta = avg_time * (actual_samples - i)
                    eta_str = f", ETA {eta:.0f}s"
                else:
                    eta_str = ""
                print(f"  [{i+1}/{actual_samples}] ell={ell} (elapsed {elapsed:.1f}s{eta_str})", end="", flush=True)

            for binary_name, runner in runners_to_use.items():
                short_name = binary_name.replace("descent_", "d_").replace("cotrans_", "c_")

                # Show which binary is running
                if self.verbose:
                    print(f" [{short_name}...", end="", flush=True)

                br = runner.run(perm_str, timeout=self.timeout)
                result_row[f"{binary_name}_time"] = br.time_seconds
                result_row[f"{binary_name}_value"] = br.value
                result_row[f"{binary_name}_success"] = br.success
                result_row[f"{binary_name}_memo"] = br.memo_size
                result_row[f"{binary_name}_iter"] = br.iterations
                result_row[f"{binary_name}_iter_per_sec"] = br.iter_per_sec
                result_row[f"{binary_name}_log2_norm"] = br.log2_normalized

                # Track timeouts
                if not br.success and br.error and "Timeout" in br.error:
                    timeout_counts[binary_name] += 1

                # Show result
                if self.verbose:
                    if br.success:
                        print(f"{br.time_seconds:.2f}s]", end="", flush=True)
                    else:
                        err_type = "TIMEOUT" if "Timeout" in (br.error or "") else "FAIL"
                        print(f"{err_type}]", end="", flush=True)

            if self.verbose:
                print("", flush=True)  # newline after sample

            # Compute relative errors (double vs exact)
            exact_fallback = (
                result_row.get("descent_rational_value")
                or result_row.get("cotrans_exact_value")
                or result_row.get("transition_exact_value")
            )
            paired_variants = [
                ("descent_double", "descent_rational"),
                ("cotrans_double", "cotrans_exact"),
                ("transition_double", "transition_exact"),
            ]
            for dbl_name, exact_name in paired_variants:
                dbl_val = result_row.get(f"{dbl_name}_value")
                exact_val = result_row.get(f"{exact_name}_value")
                if exact_val is None:
                    exact_val = exact_fallback
                if dbl_val is not None and exact_val is not None and exact_val != 0:
                    rel_err = abs(dbl_val - exact_val) / exact_val
                    result_row[f"{dbl_name}_rel_error"] = rel_err

            results.append(result_row)
            sample_times.append(time.perf_counter() - sample_start)

        # Summary
        self.log(f"\n  Completed {len(results)}/{actual_samples} samples in {self.elapsed():.1f}s")
        if any(c > 0 for c in timeout_counts.values()):
            timeout_str = ", ".join(f"{k}:{v}" for k, v in timeout_counts.items() if v > 0)
            self.log(f"  Timeouts: {timeout_str}")

        return results


# =============================================================================
# Result Formatter
# =============================================================================

class ResultFormatter:
    """Format benchmark results for output."""

    @staticmethod
    def to_json(results: Dict, filepath: Path):
        """Write results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

    @staticmethod
    def to_csv(rows: List[Dict], filepath: Path):
        """Write results to CSV file."""
        if not rows:
            return

        # Collect all unique fieldnames from all rows (preserves order from first row, adds new ones)
        fieldnames = list(rows[0].keys())
        seen = set(fieldnames)
        for row in rows[1:]:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def compute_statistics(rows: List[Dict]) -> Dict:
        """Compute summary statistics from benchmark rows."""
        if not rows:
            return {}

        binary_names = [
            "descent_double", "descent_rational",
            "cotrans_double", "cotrans_exact",
            "transition_double", "transition_exact",
        ]
        stats = {}

        for binary in binary_names:
            times = [r[f"{binary}_time"] for r in rows if r.get(f"{binary}_success", False)]
            if times:
                stats[binary] = {
                    "count": len(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times),
                }

        return stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Performance benchmark suite for Schubert polynomial implementations"
    )
    # Benchmark type selection
    parser.add_argument("--layered", action="store_true", help="Run layered permutation benchmark")
    parser.add_argument("--random", action="store_true", help="Run random permutation benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")

    # Binary selection
    parser.add_argument("--binaries", nargs="+", choices=list(VARIANT_FLAGS.keys()),
                        help="Select specific binaries to run (default: all)")

    # Layered benchmark options
    parser.add_argument("--max-n", type=int, default=16, help="Max n for layered benchmark (default: 16)")

    # Random benchmark options
    parser.add_argument("--random-size", type=int,
                        help="Size n for random benchmark (use --list-sizes to see available)")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples to use (default: 50)")
    parser.add_argument("--perms-dir", type=Path, default=DEFAULT_PERMS_DIR,
                        help=f"Permutations directory (default: {DEFAULT_PERMS_DIR})")
    parser.add_argument("--list-sizes", action="store_true",
                        help="List available random sizes and exit")

    # Performance options
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-binary timeout in seconds (default: no timeout)")
    parser.add_argument("--double-only", action="store_true",
                        help="Only run double-precision binaries (skip exact/rational)")
    parser.add_argument("--deadline", type=float, default=None,
                        help="Total benchmark deadline in seconds (stops when exceeded)")

    # Output options
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Discover available sizes
    available_sizes = PermutationLoader.discover_available_sizes(args.perms_dir)

    # Handle --list-sizes
    if args.list_sizes:
        if available_sizes:
            print("Available random sizes (n -> max samples):")
            for n in sorted(available_sizes.keys()):
                print(f"  n={n}: {available_sizes[n]} samples")
        else:
            print(f"No permutation files found in {args.perms_dir}")
        return

    # Validate random-size if random benchmark is requested
    if args.random or args.all:
        if args.random_size is None:
            # Default to smallest available size
            if available_sizes:
                args.random_size = min(available_sizes.keys())
            else:
                print(f"Error: No permutation files found in {args.perms_dir}")
                sys.exit(1)
        elif args.random_size not in available_sizes:
            print(f"Error: --random-size {args.random_size} not available.")
            print(f"Available sizes: {sorted(available_sizes.keys())}")
            sys.exit(1)

    # Default to --all if no specific benchmark is selected
    if not (args.layered or args.random or args.all):
        args.all = True

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Handle --double-only: filter to only double-precision binaries
    if args.double_only:
        if args.binaries is None:
            args.binaries = ["descent_double", "cotrans_double", "transition_double"]
        else:
            args.binaries = [b for b in args.binaries if "double" in b]
        print(f"Double-only mode: using binaries {args.binaries}")

    runner = BenchmarkRunner(
        verbose=not args.quiet,
        perms_dir=args.perms_dir,
        timeout=args.timeout,
        deadline=args.deadline
    )

    # Timestamp for this run (used in filenames and metadata)
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    timestamp_iso = now.isoformat()

    metadata = {
        "timestamp": timestamp_iso,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "selected_binaries": args.binaries or list(VARIANT_FLAGS.keys()),
    }

    all_results = {"metadata": metadata}

    # Run benchmarks
    if args.layered or args.all:
        layered_results = runner.benchmark_layered(max_n=args.max_n, selected_binaries=args.binaries, timeout=args.timeout)
        all_results["layered"] = layered_results
        ResultFormatter.to_csv(layered_results, args.output_dir / f"{timestamp_str}_layered.csv")
        print(f"\nLayered results: {len(layered_results)} entries")

    if args.random or args.all:
        n = args.random_size
        random_results = runner.benchmark_random(n=n, num_samples=args.samples,
                                                  selected_binaries=args.binaries)
        all_results[f"random_n{n}"] = random_results
        ResultFormatter.to_csv(random_results, args.output_dir / f"{timestamp_str}_random_n{n}.csv")
        print(f"\nRandom n={n} results: {len(random_results)} entries")

        stats = ResultFormatter.compute_statistics(random_results)
        all_results[f"stats_n{n}"] = stats

    # Write combined JSON output
    json_path = args.output_dir / f"{timestamp_str}_benchmark.json"
    ResultFormatter.to_json(all_results, json_path)
    print(f"\nResults written to: {json_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    stats_key = f"stats_n{args.random_size}"
    if stats_key in all_results and all_results[stats_key]:
        print(f"\nn={args.random_size} Random Permutations:")
        for binary, stats in all_results[stats_key].items():
            print(f"  {binary}: mean={stats['mean']:.4f}s min={stats['min']:.4f}s max={stats['max']:.4f}s (n={stats['count']})")


if __name__ == "__main__":
    main()
