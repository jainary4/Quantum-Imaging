# Quantum Target Detection in High Noise Environments

A quantum optics simulation project for target detection using Bell states and entanglement-based sensing.

## Overview

This repository contains quantum circuit simulations and machine learning models for detecting targets in high-noise environments using quantum entanglement. The project leverages Bell states and quantum correlations to achieve superior detection capabilities compared to classical methods.

## Project Goals

- **Quantum Circuit Simulation**: Build and simulate quantum circuits that implement quantum target detection protocols based on theoretical quantum sensing frameworks
- **Bell State Implementation**: Utilize maximally entangled Bell states for enhanced sensitivity in noisy environments
- **Data Generation**: Generate synthetic quantum state data from simulations to capture target presence/absence signatures
- **Classification Model**: Develop machine learning models that classify target detection from quantum state metadata and measurement statistics

## Theoretical Background

The project is based on quantum target detection theory, which exploits:
- Quantum entanglement for noise resilience
- Bell state correlations for enhanced signal-to-noise ratio
- Quantum measurement collapse for target signature extraction
- Quantum advantage over classical sensing in high-noise regimes

## Repository Structure

```
.
├── circuits/           # Quantum circuit implementations
├── simulations/        # Simulation scripts and experiments
├── data/              # Generated quantum state data
├── models/            # Machine learning classification models
├── analysis/          # Data analysis and visualization
├── notebooks/         # Jupyter notebooks for exploration
└── utils/             # Helper functions and utilities
```

## Dependencies

- Python 3.8+
- Qiskit / Cirq / Pennylane (quantum simulation framework)
- NumPy, SciPy (numerical computation)
- Scikit-learn / TensorFlow / PyTorch (machine learning)
- Matplotlib, Seaborn (visualization)

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PHYD02-Code.git
cd PHYD02-Code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run Quantum Simulations

```bash
python simulations/run_bell_state_detection.py
```

### 2. Generate Training Data

```bash
python data/generate_quantum_data.py --num_samples 10000
```

### 3. Train Classification Model

```bash
python models/train_classifier.py --data data/quantum_states.npz
```

### 4. Evaluate Performance

```bash
python analysis/evaluate_detection.py --model models/classifier.pkl
```

## Methodology

1. **Quantum State Preparation**: Initialize Bell states and apply noise models
2. **Target Interaction**: Simulate target-induced perturbations on quantum states
3. **Measurement**: Perform quantum measurements to extract state metadata
4. **Feature Extraction**: Extract relevant features from measurement outcomes
5. **Classification**: Train ML models to distinguish target presence/absence

## Key Features

- Configurable noise models (depolarizing, amplitude damping, phase damping)
- Multiple Bell state configurations (Φ+, Φ-, Ψ+, Ψ-)
- Quantum circuit optimization for target sensitivity
- Scalable data generation pipeline
- Multiple classifier architectures (SVM, Random Forest, Neural Networks)


## License

MIT License

## Authors

PHYD02 Research Team

## Acknowledgments

This project is part of quantum optics research in target detection and quantum sensing applications.



---

*Last updated: October 2025*

