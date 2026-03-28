#!/usr/bin/env python3

import numpy as np
import time

# ==========================================================
# Hamiltonian / Benchmark Operators
# ==========================================================
def get_hamiltonian(name):
    """
    Retrieves a benchmark Hamiltonian (cost function) and its parameters.

    Args:
        name (str): Name of the Hamiltonian (e.g., 'H1').

    Returns:
        tuple: (operator_function, lower bound, upper bound, global_optimum, dimension)
    """
    ops = {
        'H1': (lambda psi: np.sum(psi**2, axis=1), -10, 10, 0, 10),   # Sphere
        'H2': (lambda psi: np.sum(np.abs(psi), axis=1), -10, 10, 0, 50),  # Manhattan
        'H3': (lambda psi: np.sum(psi**4 - 16*psi**2 + 5*psi, axis=1), -5, 5, 0, 50),  # Quartic
        'H4': (lambda psi: np.sum(np.sin(psi)**2, axis=1), -5, 5, 0, 50),  # Sinusoidal
        'H5': (lambda psi: np.sum(np.abs(psi) + 10*np.sin(psi), axis=1), -5, 5, 0, 50),  # Abs + Sine
        'H6': (lambda psi: np.sum(psi**2 - 10*np.cos(2*np.pi*psi) + 10, axis=1), -5, 5, 0, 50),  # Rastrigin
        'H7': (lambda psi: np.sum(np.sin(psi)**2 + 0.1*(psi**2 - 1)**2, axis=1), -5, 5, 0.737, 50),
        'H8': (lambda psi: np.sum(np.exp(psi) - psi**2, axis=1), -5, 5, 1.535, 50),
        'H9': (lambda psi: np.sum((psi - 1)**2 * (np.sin(psi) + np.cos(psi)), axis=1), -5, 5, 0, 50),
        'H10': (lambda psi: np.sum(np.arange(1, psi.shape[1] + 1) * psi**2, axis=1), -100, 100, 0, 50),  # Sum Squares
        'H11': (lambda psi: (psi[:, 0] - 1)**2 + np.sum(np.arange(2, psi.shape[1] + 1) * 
                (2*psi[:, 1:]**2 - psi[:, 1:] - 1)**2, axis=1), -10, 10, 0, 50),  # Dixon
        'H12': (lambda psi: np.sum(psi**2, axis=1) + 
                np.sum(0.5*np.arange(1, psi.shape[1] + 1)*psi, axis=1)**2 + 
                np.sum(0.5*np.arange(1, psi.shape[1] + 1)*psi, axis=1)**4, -5, 5, 0, 50),  # Zakharov
    }

    if name not in ops:
        raise ValueError(f"Unknown Hamiltonian name '{name}'")

    return ops[name]


# ==========================================================
# Stochastic Noise (chaos)
# ==========================================================
def stochastic_noise(model_no, steps):
    """Generates a stochastic noise vector (chaos analog)."""
    if model_no == 1:  # Gaussian noise
        return np.random.normal(0, 1, steps)
    return np.random.uniform(-1, 1, steps)  # Uniform noise


# ==========================================================
# Wavepacket Initialization (population init)
# ==========================================================
def initialize_wavepacket(simulation_params, system_params):
    """Initializes an ensemble of wavepackets (candidate states)."""
    return np.random.uniform(
        system_params['VarMin'],
        system_params['VarMax'],
        (simulation_params['NumOfStates'], system_params['NQubits'])
    )


# ==========================================================
# Quantum Evolutionary Operators
# ==========================================================
def superposition_evolution(ensemble, system_params, amplitude, step):
    """Superposition-like evolution operator (growth)."""
    factor = amplitude / (step + 1)
    evolved = ensemble[:, :-1] + factor * np.random.uniform(-1, 1, ensemble[:, :-1].shape)
    evolved = np.clip(evolved, system_params['VarMin'], system_params['VarMax'])
    return np.hstack((evolved, system_params['Hamiltonian'](evolved).reshape(-1, 1)))


def tunneling_operator(trial, system_params, ground_state, amplitude):
    """Tunneling operator: probabilistic scattering around the ground state."""
    phi = np.random.uniform(0, 1)
    scatter_scale = amplitude * np.random.uniform(0, 1, trial[:, :-1].shape)
    tunneled = phi * ground_state + (1 - phi) * scatter_scale * np.random.uniform(-1, 1, trial[:, :-1].shape)
    tunneled = np.clip(tunneled, system_params['VarMin'], system_params['VarMax'])
    return np.hstack((tunneled, system_params['Hamiltonian'](tunneled).reshape(-1, 1)))


def entanglement_spread(trial, system_params, amplitude, noise_vec, ground_state):
    """Entanglement-like spreading influenced by chaotic noise."""
    phi = np.random.uniform(0, 1)
    spread = (amplitude * noise_vec[:trial.shape[0]])[:, np.newaxis] * np.random.uniform(-1, 1, trial[:, :-1].shape)
    entangled = phi * ground_state + (1 - phi) * (trial[:, :-1] + spread)
    entangled = np.clip(entangled, system_params['VarMin'], system_params['VarMax'])
    return np.hstack((entangled, system_params['Hamiltonian'](entangled).reshape(-1, 1)))


# ==========================================================
# Main Quantum-Style Simulation Loop
# ==========================================================
def run_quantum_simulation():
    """Runs the quantum-style optimization and returns the ground-state energy (best cost)."""
    # System setup (use H12 by default, same as original)
    ham, vmin, vmax, _, dim = get_hamiltonian('H1')
    system_params = {
        'Hamiltonian': ham,
        'VarMin': np.array([vmin] * dim),
        'VarMax': np.array([vmax] * dim),
        'NQubits': dim
    }

    # Simulation control (amplitude = mutation amplitude)
    amplitude, amplitude_decay = 0.2, 0.98
    simulation_params = {'NumOfStates': 100, 'NumOfSteps': 10000}

    # Noise (chaos) vector
    noise_vector = stochastic_noise(1, simulation_params['NumOfSteps'])

    # Initialize ensemble (wavepackets)
    wavefunctions = initialize_wavepacket(simulation_params, system_params)
    energies = system_params['Hamiltonian'](wavefunctions)
    ensemble = np.hstack((wavefunctions, energies.reshape(-1, 1)))
    min_energies = np.zeros(simulation_params['NumOfSteps'])

    # Evolution loop
    for step in range(simulation_params['NumOfSteps']):
        energies = ensemble[:, -1]
        min_energies[step] = np.min(energies)
        ground_state = ensemble[np.argmin(energies), :-1]

        # Apply quantum-style operators
        trial_states = superposition_evolution(ensemble, system_params, amplitude, step+1)
        trial_states = tunneling_operator(trial_states, system_params, ground_state, amplitude)
        trial_states = entanglement_spread(trial_states, system_params, amplitude, noise_vector, ground_state)

        # Measurement & selection: accept lower-energy trials
        improved = trial_states[:, -1] < ensemble[:, -1]
        ensemble[improved] = trial_states[improved]

        # Amplitude (mutation) decay -> collapse-like narrowing
        amplitude *= amplitude_decay

    # Final measurement: return best energy
    best_idx = np.argmin(ensemble[:, -1])
    return ensemble[best_idx, -1]


def main():
    runs = 1
    best_energies = []

    start_time = time.time()
    for r in range(runs):
        print(f"Simulation run {r + 1}...")
        best_energy = run_quantum_simulation()
        best_energies.append(best_energy)
        print(f"Best energy for run {r + 1}: {best_energy}")

    end_time = time.time()

    avg_time = (end_time - start_time) / runs
    average_best = np.mean(best_energies)

    print("\nBest energies from all runs:", best_energies)
    print(f"Average runtime for {runs} runs: {avg_time} seconds")
    print(f"Average of best energies: {average_best}")


if __name__ == "__main__":
    main()
