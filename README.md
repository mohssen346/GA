# Genetic Algorithm Implementation

A comprehensive implementation of Genetic Algorithms (GA) in Python, featuring both custom implementations and library-based approaches for optimization and feature selection problems.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Optimization](#basic-optimization)
  - [Feature Selection](#feature-selection)
- [Examples](#examples)
- [Algorithm Components](#algorithm-components)
- [Requirements](#requirements)

## 🔍 Overview

This repository contains implementations of Genetic Algorithms for:
1. **Custom GA Implementation**: A from-scratch implementation demonstrating core GA concepts
2. **Library-based GA**: Using `geneticalgorithm` library for complex optimization problems
3. **Feature Selection**: Using `genetic_selection` for ML feature selection with scikit-learn

## ✨ Features

- **Binary encoding** for optimization problems
- **Tournament selection** mechanism
- **Single-point crossover** operator
- **Bit-flip mutation** operator
- **Multi-objective optimization** support
- **Feature selection** for machine learning models
- **Visualization** of convergence

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genetic-algorithm.git
cd genetic-algorithm

# Install required packages
pip install numpy pandas matplotlib scikit-learn
pip install geneticalgorithm
pip install sklearn-genetic
```

## 📦 Requirements

```python
numpy
pandas
matplotlib
scikit-learn
geneticalgorithm
sklearn-genetic
```

## 🚀 Usage

### Basic Optimization

Custom implementation for maximizing sum of binary array:

```python
# Define objective function (minimization)
def f(x):
    return -sum(x)

# Set parameters
n_iter = 2000
n_bits = 20
n_pop = 100
r_cross = 0.9
r_mut = 1/float(n_bits)

# Run GA
best, score = genetic_algorithm(f, n_bits, n_iter, n_pop, r_cross, r_mut)
print(f'Best solution: {best}')
print(f'Best score: {score}')
```

### Library-based Optimization

Using `geneticalgorithm` library for constrained optimization:

```python
from geneticalgorithm import geneticalgorithm

# Define objective function
def g(x):
    return x[0] + x[1] + x[2]

# Define variable bounds
bounds = np.array([[-5, 100]] * 3)

# Configure GA
model = geneticalgorithm(
    function=g,
    dimension=3,
    variable_type='real',
    variable_boundaries=bounds,
    algorithm_parameters={
        'max_num_iteration': 1000,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform'
    }
)

# Run optimization
model.run()
```

### Feature Selection

Using GA for feature selection with Iris dataset:

```python
from sklearn import datasets, linear_model
from genetic_selection import GeneticSelectionCV

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Configure classifier
clf = linear_model.LogisticRegression(
    solver='liblinear', 
    multi_class='auto'
)

# Configure GA feature selector
selector = GeneticSelectionCV(
    clf,
    cv=5,
    scoring='accuracy',
    max_features=5,
    n_population=300,
    crossover_proba=0.5,
    mutation_proba=0.2,
    n_generations=40,
    verbose=1
)

# Fit and select features
selector.fit(X, y)
print(f"Selected features: {selector.support_}")
```

## 📊 Examples

### Example 1: Binary Optimization

Maximize the sum of a 20-bit binary string:

```python
# Results show progression towards optimal solution
# pop = [1,1,1,...,1,0,1] score = -19
# Final best score: -20 (all ones)
```

### Example 2: Multi-objective Optimization

Optimize multiple objective functions:

```python
def f1(x): return x[0] + x[1] + x[2]  # min
def f2(x): return x[0] - x[1] - x[2]  # max
def f3(x): return x[0] * x[1] * x[2]  # min

def g(x):
    return f1(x) - f2(x) - f3(x)
```

### Example 3: Feature Selection on Noisy Data

Select relevant features from Iris dataset with added noise:

```python
# Add 30 noise features
E = np.random.uniform(0, 0.1, size=(len(iris.data), 30))
X = np.hstack((iris.data, E))

# GA successfully identifies original 4 features
# Selected features: [True, True, True, True, False, False, ...]
```

## 🧬 Algorithm Components

### Selection
- **Tournament Selection**: Selects best individual from random subset
- Tournament size: 3 (configurable)

### Crossover
- **Single-point Crossover**: Swaps genetic material at random point
- Crossover rate: 90%

### Mutation
- **Bit-flip Mutation**: Flips bits with probability 1/n_bits
- Adaptive mutation rates available

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_pop` | Population size | 100 |
| `n_iter` | Number of generations | 2000 |
| `r_cross` | Crossover probability | 0.9 |
| `r_mut` | Mutation probability | 1/n_bits |
| `n_bits` | Chromosome length | 20 |

## 📈 Performance

The algorithm shows convergence characteristics:
- Initial diversity in population
- Gradual improvement over generations
- Convergence to optimal or near-optimal solutions
- Visualization available through built-in plotting

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Notes

- The custom implementation uses binary encoding
- Library implementations support real-valued optimization
- Feature selection integrates with scikit-learn pipelines
- Convergence plots help visualize optimization progress

## 🔗 References

- Holland, J. H. (1992). Genetic Algorithms. Scientific American.
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This implementation is for educational and research purposes. For production use, consider optimization and parameter tuning based on your specific problem domain.
