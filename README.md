# Genetic Algorithm Implementation

A comprehensive implementation of Genetic Algorithms (GA) in Python, featuring both custom implementations and library-based approaches for optimization and feature selection problems.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Basic Optimization](#basic-optimization)
  - [Library-based Optimization](#library-based-optimization)
  - [Feature Selection](#feature-selection)
- [Examples](#examples)
  - [Binary Optimization](#binary-optimization)
  - [Multi-objective Optimization](#multi-objective-optimization)
  - [Feature Selection with Noisy Data](#feature-selection-with-noisy-data)
- [Algorithm Components](#algorithm-components)
  - [Selection](#selection)
  - [Crossover](#crossover)
  - [Mutation](#mutation)
  - [Parameters](#parameters)
- [Performance](#performance)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Overview

This repository contains implementations of Genetic Algorithms for:
1. **Custom GA Implementation**: A from-scratch implementation demonstrating core GA concepts
2. **Library-based GA**: Using `geneticalgorithm` library for complex optimization problems
3. **Feature Selection**: Using `genetic_selection` for ML feature selection with scikit-learn

## Features

- ✅ **Binary encoding** for optimization problems
- ✅ **Tournament selection** mechanism
- ✅ **Single-point crossover** operator
- ✅ **Bit-flip mutation** operator
- ✅ **Multi-objective optimization** support
- ✅ **Feature selection** for machine learning models
- ✅ **Visualization** of convergence
- ✅ **Real-valued optimization** support

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genetic-algorithm.git
cd genetic-algorithm

# Install required packages
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install numpy pandas matplotlib scikit-learn
pip install geneticalgorithm
pip install sklearn-genetic
```

## Requirements

```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
scikit-learn>=0.23.0
geneticalgorithm>=1.0.2
sklearn-genetic>=0.5.0
```

## Usage

### Basic Optimization

Custom implementation for maximizing sum of binary array:

```python
import numpy as np
from numpy.random import randint, rand

# Define objective function (minimization)
def f(x):
    return -sum(x)

# Tournament selection
def selection(pop, scores, k=3):
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# Single-point crossover
def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        pt = randint(1, len(p1)-2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# Bit-flip mutation
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]
    return bitstring

# Genetic algorithm
def genetic_algorithm(f, n_bits, n_iter, n_pop, r_cross, r_mut):
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    best, best_eval = 0, f(pop[0])
    
    for gen in range(n_iter):
        scores = [f(c) for c in pop]
        
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(f'Generation {gen}, New Best: {pop[i]}, Score: {scores[i]}')
        
        selected = [selection(pop, scores) for _ in range(n_pop)]
        children = list()
        
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
        
        pop = children
    
    return [best, best_eval]

# Set parameters
n_iter = 2000
n_bits = 20
n_pop = 100
r_cross = 0.9
r_mut = 1/float(n_bits)

# Run GA
best, score = genetic_algorithm(f, n_bits, n_iter, n_pop, r_cross, r_mut)
print(f'\nBest Solution: {best}')
print(f'Best Score: {score}')
```

### Library-based Optimization

Using `geneticalgorithm` library for constrained optimization:

```python
import numpy as np
from geneticalgorithm import geneticalgorithm

# Define objective function with constraints
def f(x):
    penalty = 0
    if x[0] + x[1] < 2:
        penalty = -2
    return sum(x) + penalty

# Define variable bounds
bounds = np.array([[-5, 100]] * 3)

# Configure GA
model = geneticalgorithm(
    function=f,
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
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None
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

# Add noise features
E = np.random.uniform(0, 0.1, size=(len(iris.data), 30))
X = np.hstack((iris.data, E))

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
    tournament_size=3,
    verbose=1
)

# Fit and select features
selector.fit(X, y)
print(f"Selected features: {selector.support_}")
print(f"Number of features selected: {sum(selector.support_)}")
```

## Examples

### Binary Optimization

Maximize the sum of a 20-bit binary string:

```python
# Objective: Find binary string with maximum sum (all 1s)
# Result progression:
# Generation 0: [0,0,1,0,1,...] Score: -14
# Generation 5: [1,1,1,1,1,...] Score: -19
# Final: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] Score: -20
```

### Multi-objective Optimization

Optimize multiple conflicting objective functions:

```python
def f1(x): 
    return x[0] + x[1] + x[2]  # Minimize

def f2(x): 
    return x[0] - x[1] - x[2]  # Maximize (minimize negative)

def f3(x): 
    return x[0] * x[1] * x[2]  # Minimize

def g(x):
    return f1(x) - f2(x) - f3(x)

# Best solution found: [99.98, 99.96, 99.94]
# Objective function value: -998397.35
```

### Feature Selection with Noisy Data

Select relevant features from Iris dataset with added noise:

```python
# Original dataset: 4 features
# Added noise: 30 features
# Total: 34 features

# GA successfully identifies the 4 original features
# Selected features: [True, True, True, True, False, False, ...]
# Accuracy with selected features: ~96%
```

## Algorithm Components

### Selection

**Tournament Selection**: Selects the best individual from a random subset of the population.

- **Tournament size**: 3 (default, configurable)
- **Process**: 
  1. Randomly select k individuals
  2. Choose the one with best fitness
  3. Return selected individual

```python
def selection(pop, scores, k=3):
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]
```

### Crossover

**Single-point Crossover**: Combines two parent solutions to create offspring.

- **Crossover rate**: 90% (default)
- **Process**:
  1. Select random crossover point
  2. Swap genetic material after point
  3. Create two offspring

```python
def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        pt = randint(1, len(p1)-2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]
```

### Mutation

**Bit-flip Mutation**: Randomly flips bits in the chromosome.

- **Mutation rate**: 1/n_bits (default)
- **Process**: Each bit has independent probability of flipping

```python
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]
    return bitstring
```

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_pop` | Population size | 100 | 50-500 |
| `n_iter` | Number of generations | 2000 | 100-10000 |
| `r_cross` | Crossover probability | 0.9 | 0.5-1.0 |
| `r_mut` | Mutation probability | 1/n_bits | 0.001-0.1 |
| `n_bits` | Chromosome length | 20 | 10-100 |
| `tournament_size` | Tournament size | 3 | 2-5 |

## Performance

The algorithm demonstrates:
- ✅ **Initial diversity**: Random population generation
- ✅ **Gradual improvement**: Fitness increases over generations
- ✅ **Convergence**: Reaches optimal/near-optimal solutions
- ✅ **Visualization**: Built-in plotting for progress tracking

**Typical convergence behavior**:
- Generations 0-10: Rapid improvement
- Generations 10-50: Moderate improvement
- Generations 50+: Fine-tuning, slow improvement

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## References

1. Holland, J. H. (1992). *Genetic Algorithms*. Scientific American, 267(1), 66-73.
2. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
3. Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This implementation is for educational and research purposes. For production use, consider parameter tuning and optimization based on your specific problem domain.

**Developed with ❤️ for the optimization community**
