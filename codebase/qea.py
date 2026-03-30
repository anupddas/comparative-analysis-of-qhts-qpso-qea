#!/usr/bin/env python3
"""
qea.py

A modular, flexible Python implementation of a Quantum-Inspired Evolutionary Algorithm (QEA).

Features:
- Accepts CLI inputs for cost function (builtin or custom), problem dimensionality, population size,
  number of iterations, and bounds.
- Binary encoding of continuous variables with configurable bits per variable.
- Q-bit representation (probability amplitudes) with rotation gate updates towards the best solution.
- Several builtin cost functions (sphere, rastrigin, ackley, rosenbrock) and support for a custom
  Python file that defines a function `cost(x: List[float]) -> float`.
- Error handling for invalid inputs and helpful messages.
- Well-commented and modular for easy extension.

Usage examples (from command line):

# Use a builtin function (Rastrigin), 10 dims, pop 40, 500 iterations, same bound for all dims
python qea.py --func rastrigin --dim 10 --pop 40 --iters 500 --bounds -5.12 5.12

# Use custom cost defined in my_cost.py (must contain `def cost(x): ...`) with 3 dimensions
python qea.py --custom my_cost.py --dim 3 --pop 30 --iters 200 --bounds -10 10

"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np


# ------------------------- Utility / Error classes -------------------------
class QEAError(Exception):
    pass


# ------------------------- Builtin cost functions ---------------------------
# Note: these functions safely convert input to numpy arrays where vector ops are used.

def sphere(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sum(arr * arr))


def manhattan_norm(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(arr)))


def quartic(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sum((np.arange(1, arr.size + 1) * (arr ** 4))))


def sinusoidal(x: List[float]) -> float:
    # uses Python math.sin elementwise for clarity
    arr = np.asarray(x, dtype=float)
    return float(np.sum((arr - 1.0) ** 2 * np.sin(arr)))


def composite_sine(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(arr) + 10.0 * np.sin(arr)))


def rastrigin(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    n = arr.size
    return float(10.0 * n + np.sum(arr ** 2 - 10.0 * np.cos(2.0 * math.pi * arr)))


def sine_deviation(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sum(np.sin(arr) ** 2 + 0.1 * (arr ** 2 - 1.0) ** 2))


def exponential_squared(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sum(np.exp(arr) - arr ** 2))


def sine_cosine(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sum((arr - 1.0) ** 2 * (np.sin(arr) + np.cos(arr))))


def sum_square(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sum(np.arange(1, arr.size + 1) * (arr ** 2)))


def dixon(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size < 2:
        # Dixon function is defined for n>=2; for n==1 we use (x1-1)^2
        return float((arr[0] - 1.0) ** 2)
    term1 = (arr[0] - 1.0) ** 2
    term2 = np.sum(np.arange(2, arr.size + 1) * (2.0 * arr[1:] ** 2 - arr[:-1] - 1.0) ** 2)
    return float(term1 + term2)


def zakharov(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    i = np.arange(1, arr.size + 1)
    sum1 = np.sum(arr ** 2)
    sum2 = 0.5 * np.sum(i * arr)
    return float(sum1 + sum2 ** 2 + sum2 ** 4)


_BUILTINS = {
    'sphere': sphere,
    'rastrigin': rastrigin,
    'manhattan': manhattan_norm,
    'quartic': quartic,
    'sinusoidal': sinusoidal,
    'composite_sine': composite_sine,
    'sine_deviation': sine_deviation,
    'exponential_squared': exponential_squared,
    'sine_cosine': sine_cosine,
    'sum_square': sum_square,
    'dixon': dixon,
    'zakharov': zakharov,
}


# ---------------------- Encoding / Decoding utilities ----------------------

def validate_bounds(bounds: Tuple[float, float]) -> None:
    if bounds[0] >= bounds[1]:
        raise QEAError(f"Invalid bounds: lower bound {bounds[0]} must be < upper bound {bounds[1]}")


def int_to_bits(val: int, bits: int) -> List[int]:
    return [(val >> i) & 1 for i in reversed(range(bits))]


def bits_to_int(bits_list: List[int]) -> int:
    val = 0
    for b in bits_list:
        val = (val << 1) | (1 if b else 0)
    return val


# ---------------------- Q-bit and population helpers -----------------------
class QBit:
    """A Q-bit represented by the probability p of being 1.

    In the standard QEA literature the Q-bit is often represented as amplitudes (alpha, beta)
    with |alpha|^2 + |beta|^2 = 1. Here we simplify by storing p = Prob(bit==1) which is
    equivalent (p = |beta|^2). Rotations will adjust p toward 0 or 1.
    """

    def __init__(self, p: float = 0.5):
        if not (0.0 <= p <= 1.0):
            raise ValueError("QBit probability must be in [0,1]")
        self.p = float(p)

    def measure(self) -> int:
        """Collapse/measure the qubit to a classical bit (0/1)."""
        return 1 if random.random() < self.p else 0

    def rotate_towards(self, target_bit: int, delta: float) -> None:
        """Rotate probability towards target_bit by 'delta'.

        If target_bit==1, increase p by delta (capped at 1); else decrease p by delta (floored at 0).
        This is a simplified rotation gate approximation.
        """
        if target_bit not in (0, 1):
            raise ValueError("target_bit must be 0 or 1")
        if not (0.0 <= delta <= 1.0):
            raise ValueError("delta must be in [0,1]")
        if target_bit == 1:
            self.p = min(1.0, self.p + abs(delta))
        else:
            self.p = max(0.0, self.p - abs(delta))


class QPopulation:
    """Population of quantum individuals. Each individual is a list of QBits representing
    the binary encoding of all variables concatenated.
    """

    def __init__(self, num_individuals: int, total_bits: int):
        if num_individuals <= 0:
            raise ValueError("num_individuals must be > 0")
        if total_bits <= 0:
            raise ValueError("total_bits must be > 0")
        self.num_individuals = num_individuals
        self.total_bits = total_bits
        # Initialize q-bits uniformly with p=0.5
        self.qbits = [[QBit(0.5) for _ in range(total_bits)] for _ in range(num_individuals)]

    def measure_population(self) -> List[List[int]]:
        """Measure the entire population into classical binary strings."""
        return [[qb.measure() for qb in indiv] for indiv in self.qbits]

    def update_towards_best(self, classical_pop: List[List[int]], best_bits: List[int],
                             rotation_delta: float) -> None:
        """Update each QBit using a rotation towards the corresponding bit in best_bits.

        classical_pop is not used for direction here (simple global-best nudging).
        """
        if len(best_bits) != self.total_bits:
            raise QEAError("best_bits length mismatch with population bit-length")
        for i in range(self.num_individuals):
            for j in range(self.total_bits):
                target = best_bits[j]
                # Optionally skip updating the best individual itself to reduce premature convergence.
                self.qbits[i][j].rotate_towards(target, rotation_delta)


# ---------------------- QEA main implementation ----------------------------
class QuantumInspiredEA:
    def __init__(self,
                 cost_fn: Callable[[List[float]], float],
                 dim: int,
                 bounds: Tuple[float, float],
                 pop_size: int = 50,
                 iterations: int = 200,
                 bits_per_var: int = 16,
                 rotation_delta: float = 0.01,
                 seed: Optional[int] = None):
        """Initialize the QEA.

        cost_fn: function accepting a list of floats and returning a float (lower is better)
        dim: number of real-valued decision variables
        bounds: (lower, upper) bounds applied to each dimension (same for all dims)
        pop_size: number of individuals
        iterations: search iterations
        bits_per_var: binary encoding resolution per variable
        rotation_delta: how strongly to nudge q-bits toward best (0 < delta < 1)
        seed: optional PRNG seed
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if dim <= 0:
            raise QEAError("dim must be > 0")
        if pop_size <= 0:
            raise QEAError("pop_size must be > 0")
        if iterations <= 0:
            raise QEAError("iterations must be > 0")
        validate_bounds(bounds)
        if bits_per_var <= 0:
            raise QEAError("bits_per_var must be > 0")
        if not (0.0 < rotation_delta < 1.0):
            raise QEAError("rotation_delta must be in (0,1)")

        self.cost_fn = cost_fn
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.iterations = iterations
        self.bits_per_var = bits_per_var
        self.total_bits = bits_per_var * dim
        self.rotation_delta = float(rotation_delta)

        # Create quantum population
        self.qpop = QPopulation(pop_size, self.total_bits)

        # Keep track of best solution seen
        self.best_solution: Optional[List[float]] = None
        self.best_cost: float = float('inf')

    def decode(self, bitstring: List[int]) -> List[float]:
        """Decode concatenated bitstring into a list of real values inside bounds.

        We treat each block as an unsigned integer and map to [lower, upper].
        """
        if len(bitstring) != self.total_bits:
            raise QEAError("bitstring length does not match expected total bits")
        lower, upper = self.bounds
        values: List[float] = []
        max_int = (1 << self.bits_per_var) - 1
        for i in range(self.dim):
            start = i * self.bits_per_var
            end = start + self.bits_per_var
            block = bitstring[start:end]
            integer_val = bits_to_int(block)
            # Map integer to real
            real_val = lower + (upper - lower) * (integer_val / max_int)
            values.append(float(real_val))
        return values

    def evaluate_population(self, classical_pop: List[List[int]]) -> List[float]:
        costs: List[float] = []
        for bits in classical_pop:
            x = self.decode(bits)
            try:
                cost_val = float(self.cost_fn(x))
            except Exception as e:
                raise QEAError(f"Cost function raised an error for x={x}: {e}")
            costs.append(cost_val)
        return costs

    def run(self, verbose: bool = True) -> Tuple[List[float], float]:
        """Run the QEA optimization and return (best_solution, best_cost)."""
        for it in range(1, self.iterations + 1):
            classical_pop = self.qpop.measure_population()
            costs = self.evaluate_population(classical_pop)

            # Update best
            for bits, cost in zip(classical_pop, costs):
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = self.decode(bits)

            # Prepare best_bits (binary representation of current best)
            if self.best_solution is None:
                best_bits = classical_pop[0]
            else:
                best_bits = []
                lower, upper = self.bounds
                max_int = (1 << self.bits_per_var) - 1
                for val in self.best_solution:
                    # Clamp and convert
                    v = float(min(max(val, lower), upper))
                    # safe division
                    ratio = 0.0 if upper == lower else (v - lower) / (upper - lower)
                    int_val = int(round(ratio * max_int))
                    int_val = max(0, min(max_int, int_val))
                    best_bits.extend(int_to_bits(int_val, self.bits_per_var))

            # Update q-bits toward best bits
            self.qpop.update_towards_best(classical_pop, best_bits, self.rotation_delta)

            if verbose and (it % max(1, self.iterations // 10) == 0 or it == 1):
                print(f"Iter {it}/{self.iterations} -- best_cost = {self.best_cost:.6g}")

        if self.best_solution is None:
            raise QEAError("No feasible solution found during run")
        return (self.best_solution, self.best_cost)


# ---------------------- CLI / custom module loader ------------------------

def load_custom_cost(path: str) -> Callable[[List[float]], float]:
    """Dynamically load a file containing a `cost(x)` function and return it.

    The file must define a top-level callable named `cost` that accepts a list of floats
    and returns a float. Example file contents:

    def cost(x):
        return sum(xi*xi for xi in x)

    """
    if not os.path.isfile(path):
        raise QEAError(f"Custom cost file not found: {path}")
    spec = importlib.util.spec_from_file_location("custom_cost_module", path)
    if spec is None or spec.loader is None:
        raise QEAError(f"Could not load custom cost module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'cost') or not callable(module.cost):
        raise QEAError("Custom cost module must define a callable `cost(x)`")
    return module.cost


def parse_bounds(bounds_list: List[float], dim: int) -> Tuple[float, float]:
    """Parse bounds argument. Accepts two floats which apply to all dims (lower upper).

    For simplicity this script uses same bounds for all dimensions. You can extend it in the
    future to accept per-dimension bounds.
    """
    if len(bounds_list) != 2:
        raise QEAError("--bounds requires exactly two numbers: LOWER UPPER")
    lower, upper = float(bounds_list[0]), float(bounds_list[1])
    validate_bounds((lower, upper))
    return (lower, upper)


def positive_int(val: str) -> int:
    i = int(val)
    if i <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return i


def nonnegative_int(val: str) -> int:
    i = int(val)
    if i < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return i


def float_in_open_interval(val: str) -> float:
    f = float(val)
    if f <= 0.0 or f >= 1.0:
        raise argparse.ArgumentTypeError("rotation-delta must be in (0,1)")
    return f


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quantum-Inspired Evolutionary Algorithm (QEA)")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--func', type=str, choices=list(_BUILTINS.keys()),
                       help=f"Builtin cost function. Choices: {', '.join(_BUILTINS.keys())}")
    group.add_argument('--custom', type=str,
                       help="Path to a Python file that defines `def cost(x): ...`")

    p.add_argument('--dim', type=positive_int, required=True, help='Number of real-valued dimensions')
    p.add_argument('--pop', type=positive_int, default=50, help='Population size (default: 50)')
    p.add_argument('--iters', type=positive_int, default=200, help='Iterations (default: 200)')
    p.add_argument('--bits', type=positive_int, default=16, help='Bits per variable (default: 16)')
    p.add_argument('--bounds', type=float, nargs=2, required=True,
                   help='Lower and upper bound for all dimensions (two floats)')
    p.add_argument('--seed', type=nonnegative_int, default=None, help='Random seed (optional)')
    p.add_argument('--rotation-delta', type=float_in_open_interval, default=0.01,
                   help='Rotation delta in (0,1) used to nudge q-bits (default 0.01)')
    p.add_argument('--no-verbose', dest='verbose', action='store_false', help='Suppress progress prints')
    return p


# ---------------------- Example main for demonstration --------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Load cost function (builtin or custom)
    if args.custom:
        try:
            cost_fn = load_custom_cost(args.custom)
        except QEAError as e:
            print(f"Error loading custom cost: {e}", file=sys.stderr)
            return 2
        benchmarks = {"custom": cost_fn}
    else:
        benchmarks = _BUILTINS

    # Parse and validate bounds
    try:
        bounds = parse_bounds(args.bounds, args.dim)
    except QEAError as e:
        print(f"Invalid bounds: {e}", file=sys.stderr)
        return 3

    results = {}

    # Iterate over all benchmarks
    for benchmark_name, cost_fn in benchmarks.items():
        print(f"\nRunning benchmark: {benchmark_name}")
        best_scores = []
        total_runtime = 0.0

        for run in range(10):
            print(f"Run {run + 1}...")
            try:
                qea = QuantumInspiredEA(cost_fn=cost_fn,
                                        dim=args.dim,
                                        bounds=bounds,
                                        pop_size=args.pop,
                                        iterations=args.iters,
                                        bits_per_var=args.bits,
                                        rotation_delta=args.rotation_delta,
                                        seed=args.seed)
            except QEAError as e:
                print(f"Configuration error: {e}", file=sys.stderr)
                return 4

            start_time = time.time()
            try:
                _, best_cost = qea.run(verbose=args.verbose)
            except QEAError as e:
                print(f"Runtime error during QEA: {e}", file=sys.stderr)
                return 5
            except Exception as e:
                print(f"Unexpected error: {e}", file=sys.stderr)
                return 10
            end_time = time.time()

            runtime = end_time - start_time
            total_runtime += runtime
            best_scores.append(best_cost)
            print(f"Run {run + 1} completed in {runtime} seconds with best cost {best_cost}")
        # Calculate averages
        avg_best_score = np.mean(best_scores)
        avg_runtime = total_runtime / 10
        results[benchmark_name] = (avg_best_score, avg_runtime)

        print(f"\nBenchmark: {benchmark_name}")
        print(f"Average best cost: {avg_best_score}")
        print(f"Average runtime: {avg_runtime} seconds")

    # Print summary for all benchmarks
    print("\nSummary of all benchmarks:")
    for benchmark_name, (avg_best_score, avg_runtime) in results.items():
        print(f"{benchmark_name}: Average best cost = {avg_best_score}, Average runtime = {avg_runtime} seconds")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())