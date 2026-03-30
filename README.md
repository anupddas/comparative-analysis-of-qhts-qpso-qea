# Comparative Analysis of QHTS, QPSO, and QEA Optimization Algorithms

## Overview

This repository presents a comparative experimental analysis of three quantum-inspired optimization algorithms:

* Quantum Hazelnut Tree Search (QHTS)
* Quantum Particle Swarm Optimization (QPSO)
* Quantum Evolutionary Algorithm (QEA)

The objective of this project is to evaluate the optimization performance of these algorithms on standard numerical benchmark functions and analyze their convergence characteristics, stability, and solution quality.

This work is intended as an experimental research implementation for studying quantum-inspired metaheuristic optimization techniques.

---

## Motivation

Metaheuristic optimization algorithms are widely used for solving complex nonlinear optimization problems where classical deterministic methods may fail. Quantum-inspired algorithms extend these approaches by incorporating probabilistic behavior, population diversity strategies, and stochastic exploration mechanisms.

Quantum Particle Swarm Optimization (QPSO), for example, improves classical PSO by modeling particle behavior using quantum probability distributions instead of deterministic trajectories. These approaches are often studied for their convergence stability and global search capability.

This repository investigates how QHTS compares with established quantum-inspired optimizers such as QPSO and QEA.

---

## Objectives

The main goals of this project include:

* Implementation of QHTS algorithm
* Implementation of QPSO and QEA baselines
* Performance comparison on benchmark functions
* Convergence behavior analysis
* Evaluation of exploration vs exploitation capability
* Visualization of optimization performance

---

## Algorithms Compared

### Quantum Hazelnut Tree Search (QHTS)

QHTS is a quantum-inspired extension of the Hazelnut Tree Search optimization algorithm incorporating:

* Probabilistic search operators
* Adaptive diversification
* Quantum-inspired exploration strategies
* Chaotic search behavior

---

### Quantum Particle Swarm Optimization (QPSO)

QPSO is a quantum-inspired variant of Particle Swarm Optimization where particle movement is modeled using probability distributions rather than velocity updates.

Key characteristics:

* Global convergence behavior
* Reduced parameter dependency
* Quantum delta potential well model
* Improved exploration capability

---

### Quantum Evolutionary Algorithm (QEA)

QEA applies quantum computing concepts such as:

* Q-bit representation
* Superposition states
* Probability amplitudes
* Quantum rotation gates

These techniques help maintain diversity while guiding convergence.

---

## Technology Stack

### Programming Language

* Python 3.x

### Libraries

* NumPy
* Matplotlib
* SciPy (optional)
* Random
* Math

### Tools

* Jupyter Notebook (optional)
* Git
* GitHub

---

## Repository Structure

comparative-analysis-of-qhts-qpso-qea/
│
├── QHTS.py
│ Implementation of Quantum Hazelnut Tree Search algorithm
│
├── QPSO.py
│ Implementation of Quantum Particle Swarm Optimization
│
├── QEA.py
│ Implementation of Quantum Evolutionary Algorithm
│
├── benchmarks/
│ Benchmark optimization functions used for testing
│
├── plots/
│ Convergence graphs and performance comparison figures
│
├── results/
│ Experimental output data and comparisons
│
├── README.md
│ Project documentation
│
└── LICENSE
License information

The algorithms are tested on standard optimization functions such as:

* Sphere Function
* Rastrigin Function
* Rosenbrock Function
* Zakharov Function
* Sum Square Function

These functions evaluate:

* Convergence speed
* Accuracy of global optimum detection
* Stability across iterations
* High-dimensional performance

---

## Installation

### Clone repository

git clone https://github.com/anupddas/comparative-analysis-of-qhts-qpso-qea.git

cd comparative-analysis-of-qhts-qpso-qea

### Install dependencies

pip install numpy matplotlib

---

## Running Experiments

Run individual algorithms:

### Run QHTS

python QHTS.py

### Run QPSO

python QPSO.py

### Run QEA

python QEA.py

---

## Performance Evaluation Criteria

Algorithms are compared using:

* Best fitness value
* Mean fitness value
* Convergence rate
* Stability across runs
* Iteration efficiency
* Exploration capability

---

## Results

The results include:

* Convergence curves
* Performance comparison plots
* Cost vs iteration graphs
* Best solution comparisons

These visualizations are available in the **plots** directory.

---

## Observations

General experimental findings may include:

* Differences in convergence speed
* Stability variations across algorithms
* Exploration behavior differences
* Performance trade-offs between algorithms

(Detailed observations should be interpreted from experimental plots.)

---

## Limitations

Current limitations include:

* Limited benchmark set
* No statistical testing across multiple runs
* No real-world optimization problems included
* No parallel optimization implementation

---

## Future Work

Possible improvements:

* Statistical significance testing
* Hybrid algorithm development
* Real engineering optimization problems
* Parallel implementations
* Parameter sensitivity analysis
* Comparison with GA, DE, PSO, and GWO

---

## Research Applications

These algorithms are applicable in:

* Engineering optimization
* Machine learning hyperparameter tuning
* Network optimization
* Scheduling problems
* Resource allocation
* Computational intelligence research

---

## Reproducibility

To reproduce results:

1 Run each algorithm with identical benchmark settings  
2 Use same population size  
3 Use same iteration limits  
4 Compare convergence plots  

---

## Author

Anup Das  
B.Tech Computer Science Engineering

GitHub:
https://github.com/anupddas

---

## Citation

If you use this implementation for research purposes, please cite the associated work or repository.

---

## License

This project is licensed under the GPL-3.0 License.

---

## Project Status

Research Implementation  
Experimental evaluation ongoing.

---

## Contact

For questions, suggestions, or collaboration:

Open an Issue in the repository.
