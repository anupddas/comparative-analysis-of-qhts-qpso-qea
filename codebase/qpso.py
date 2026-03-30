#!/usr/bin/env python3
"""
Quantum-Inspired Particle Swarm Optimization (QPSO) Implementation
Author: ChatGPT

Description:
    - Modular and flexible implementation of QPSO
    - Adjustable parameters (cost function, dimensionality, population size, iterations, bounds)
    - Built-in benchmark functions (sphere, rastrigin, rosenbrock, ackley)
    - Option to load a custom cost function from a Python file
    - Error handling for invalid inputs
    - CLI support for runtime configuration

Usage Examples:
    python qpso.py --func rastrigin --dim 10 --pop 40 --iters 500 --bounds -5.12 5.12
    python qpso.py --custom my_cost.py --dim 3 --pop 30 --iters 200 --bounds -10 10
"""

import argparse
import importlib.util
import numpy as np
import os
import sys
import time


# =======================
# Built-in Cost Functions
# =======================
def sphere(x):
    x = np.array(x, dtype=float)
    return np.sum(x ** 2)

def manhattan_norm(x):
    x = np.array(x, dtype=float)
    return np.sum(np.abs(x))

def quartic(x):
    x = np.array(x, dtype=float)
    return np.sum(x**4 - 16*x**2 + 5*x)

def sinusoidal(x):
    x = np.array(x, dtype=float)
    return np.sum(np.sin(x)**2)

def composite_sine(x):
    x = np.array(x, dtype=float)
    return np.sum(np.abs(x) + 10 * np.sin(x))

def rastrigin(x):
    x = np.array(x, dtype=float)
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def sine_deviation(x):
    x = np.array(x, dtype=float)
    return np.sum(np.sin(x)**2 + 0.1 * (x**2 - 1)**2)

def exponential_squared(x):
    x = np.array(x, dtype=float)
    return np.sum(np.exp(x) - x**2)

def sine_cosine(x):
    x = np.array(x, dtype=float)
    return np.sum((x - 1)**2 * (np.sin(x) + np.cos(x))) 

def sum_square(x):
    x = np.array(x, dtype=float)
    return np.sum(np.arange(1, len(x) + 1) * x**2)

def dixon(x):
    x = np.array(x, dtype=float)
    if len(x) < 2:
        raise ValueError("Dixon function requires at least 2 dimensions.")
    term1 = (x[0] - 1)**2
    term2 = np.sum(np.arange(2, len(x) + 1) * (2 * x[1:]**2 - x[:-1] - 1)**2)
    return term1 + term2

def zakharov(x):
    x = np.array(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2**2 + sum2**4


BENCHMARKS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "manhattan": manhattan_norm,
    "quartic": quartic,
    "sinusoidal": sinusoidal,  
    "composite_sine": composite_sine,
    "sine_deviation": sine_deviation,
    "exponential_squared": exponential_squared,
    "sine_cosine": sine_cosine,
    "sum_square": sum_square,
    "dixon": dixon,
    "zakharov": zakharov,
}


# =======================
# QPSO Implementation
# =======================
class QPSOError(Exception):
    """Custom exception class for QPSO errors."""
    pass


class QPSO:
    def __init__(self, cost_func, dim, pop_size, iterations, bounds, beta=0.75, seed=None):
        if dim <= 0:
            raise QPSOError("Dimension must be positive.")
        if pop_size <= 0:
            raise QPSOError("Population size must be positive.")
        if iterations <= 0:
            raise QPSOError("Number of iterations must be positive.")
        if bounds[0] >= bounds[1]:
            raise QPSOError("Invalid bounds: lower bound must be less than upper bound.")
        if not callable(cost_func):
            raise QPSOError("Cost function must be callable.")

        self.cost_func = cost_func
        self.dim = dim
        self.pop_size = pop_size
        self.iterations = iterations
        self.bounds = bounds
        self.beta = beta

        if seed is not None:
            np.random.seed(seed)

        # Initialize population
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        self.pbest = np.copy(self.population)
        self.pbest_scores = np.array([self.cost_func(ind) for ind in self.population])
        best_idx = np.argmin(self.pbest_scores)
        self.gbest = np.copy(self.pbest[best_idx])
        self.gbest_score = self.pbest_scores[best_idx]

    def optimize(self):
        for t in range(self.iterations):
            mbest = np.mean(self.pbest, axis=0)

            for i in range(self.pop_size):
                u = np.random.rand(self.dim)
                phi = np.random.rand(self.dim)

                p = phi * self.pbest[i] + (1 - phi) * self.gbest
                # QPSO position update
                self.population[i] = p + ((-1) ** np.random.randint(2)) * self.beta * np.abs(mbest - self.population[i]) * np.log(1 / u)

                # Boundary check
                self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])

                # Evaluate new solution
                score = self.cost_func(self.population[i])
                if score < self.pbest_scores[i]:
                    self.pbest[i] = self.population[i]
                    self.pbest_scores[i] = score

                    if score < self.gbest_score:
                        self.gbest = self.population[i]
                        self.gbest_score = score

        return self.gbest, self.gbest_score


# =======================
# Helper: Load Custom Cost Function
# =======================
def load_custom_function(path):
    if not os.path.exists(path):
        raise QPSOError(f"Custom cost file '{path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "cost"):
        raise QPSOError("Custom cost file must define a 'cost(x)' function.")

    return module.cost


# =======================
# Main CLI Entry Point
# =======================
def main():
    parser = argparse.ArgumentParser(description="Quantum-Inspired Particle Swarm Optimization (QPSO)")
    parser.add_argument("--func", type=str, choices=BENCHMARKS.keys(), help="Benchmark function to optimize.")
    parser.add_argument("--custom", type=str, help="Path to custom cost function (Python file with cost(x) defined).")
    parser.add_argument("--dim", type=int, required=True, help="Dimensionality of the problem.")
    parser.add_argument("--pop", type=int, required=True, help="Population size.")
    parser.add_argument("--iters", type=int, required=True, help="Number of iterations.")
    parser.add_argument("--bounds", type=float, nargs=2, required=True, metavar=("LOW", "HIGH"), help="Search space bounds.")
    parser.add_argument("--beta", type=float, default=0.75, help="QPSO contraction-expansion coefficient (default=0.75).")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    args = parser.parse_args()

    try:
        if args.custom:
            cost_func = load_custom_function(args.custom)
        elif args.func:
            cost_func = BENCHMARKS[args.func]
        else:
            raise QPSOError("Either --func or --custom must be specified.")

        best_scores = []
        start_time = time.time()
        for run in range(10):
            print(f"Run {run + 1}...")
            qpso = QPSO(cost_func, args.dim, args.pop, args.iters, args.bounds, beta=args.beta, seed=args.seed)
            _, best_score = qpso.optimize()
            best_scores.append(best_score)
            print(f"Run {run+1}: Best score = {best_score}")

        end_time = time.time()
        avg_score = np.mean(best_scores)
        print("\nAll best scores:", best_scores)

        print(f"\n\nAverage runtime for 10 runs: {(end_time - start_time)/10} seconds")
        print(f"\nAverage of best scores: {avg_score}")

    except QPSOError as e:
        print("QPSO Error:", e)
    except Exception as e:
        print("Unexpected error:", e)


if __name__ == "__main__":
    main()
