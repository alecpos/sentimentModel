# Robustness

DOCUMENTATION STATUS: COMPLETE

This directory contains the robustness module for machine learning models in the WITHIN ML Prediction System.

## Purpose

The robustness module provides capabilities for:
- Adversarial attack implementations and defenses
- Certification methods for model robustness
- Gradient masking detection
- Noise resilience assessment tools
- Input perturbation testing utilities

## Directory Structure

- **__init__.py**: Module initialization with component exports
- **attacks.py**: Implementation of adversarial attack algorithms
- **certification.py**: Tools for certifying model robustness against perturbations

## Key Components

### RandomizedSmoothingCertifier

`RandomizedSmoothingCertifier` is responsible for providing provable robustness guarantees for ML models using randomized smoothing techniques.

**Key Features:**
- Provides provable robustness guarantees against L2-norm bounded perturbations
- Implements Cohen et al.'s randomized smoothing approach
- Supports both classification and regression models
- Calculates certified radius for individual predictions

**Parameters:**
- `model` (nn.Module): The base model to certify
- `sigma` (float): Standard deviation of Gaussian noise
- `n_samples` (int): Number of noise samples to use
- `confidence` (float): Confidence level for certification (0.0-1.0)

**Methods:**
- `certify(x)`: Returns the certified prediction and radius for input x
- `predict(x, n_samples)`: Makes a prediction using the smoothed classifier
- `certify_batch(batch_x)`: Certifies a batch of inputs
- `get_certified_accuracy(test_loader, epsilon)`: Calculates certified accuracy at radius epsilon

### detect_gradient_masking

`detect_gradient_masking` is a function responsible for detecting if a model exhibits gradient masking, a phenomenon that can give a false sense of robustness.

**Key Features:**
- Detects gradient masking in ML models
- Tests multiple gradient-based and decision-based attacks
- Compares attack success rates to identify discrepancies
- Provides a detailed diagnostic report

**Parameters:**
- `model` (nn.Module): The model to test
- `test_data` (torch.utils.data.Dataset): Test dataset
- `epsilons` (List[float]): Perturbation magnitudes to test
- `verbose` (bool): Whether to print detailed logs

**Returns:**
- `Dict[str, Any]`: Diagnostic report containing attack success rates and gradient masking detection

### AutoAttack

`AutoAttack` is responsible for implementing a state-of-the-art ensemble of attacks for robustness evaluation.

**Key Features:**
- Combines multiple complementary attacks (APGD, FAB, Square Attack)
- Parameter-free implementation requiring minimal tuning
- Supports multiple norm constraints (L1, L2, Linf)
- Customizable attack versions (standard, plus, rand, custom)

**Parameters:**
- `model` (nn.Module): The model to attack
- `norm` (str): Norm for perturbation constraint ('Linf', 'L2', 'L1')
- `eps` (float): Maximum perturbation size
- `version` (str): Attack version ('standard', 'plus', 'rand', 'custom')

**Methods:**
- `run_standard_attacks(x, y)`: Runs the standard suite of attacks
- `run_plus_attacks(x, y)`: Runs enhanced version with additional attacks
- `run_custom_attacks(x, y, attacks_to_run)`: Runs a custom subset of attacks
- `get_adversarial_examples()`: Returns generated adversarial examples

### BoundaryAttack

`BoundaryAttack` is responsible for implementing a decision-based adversarial attack that doesn't require gradients.

**Key Features:**
- Works with black-box models that only provide decision outputs
- Requires no access to model gradients or probabilities
- Generates adversarial examples with minimal perturbation
- Uses a random walk along the decision boundary

**Parameters:**
- `model` (nn.Module): The model to attack
- `iterations` (int): Number of iterations to run
- `spherical_step` (float): Size of the step along the sphere
- `source_step` (float): Size of the step towards the original input

**Methods:**
- `attack(x, y)`: Attacks a single input to generate an adversarial example
- `attack_batch(x_batch, y_batch)`: Attacks a batch of inputs
- `perturb(x, y)`: Generates a perturbation for the input
- `initialize(x, y)`: Initializes the attack with a valid adversarial example

## Usage Examples

### RandomizedSmoothingCertifier Usage

```python
from app.models.ml.robustness import RandomizedSmoothingCertifier
import torch

# Initialization
certifier = RandomizedSmoothingCertifier(
    model=pytorch_model,
    sigma=0.25,
    n_samples=1000,
    confidence=0.95
)

# Using key methods
test_input = torch.randn(1, 3, 32, 32)  # Example input
prediction, certified_radius = certifier.certify(test_input)

print(f"Prediction: {prediction}, Certified Radius: {certified_radius}")
```

### AutoAttack Usage

```python
from app.models.ml.robustness import AutoAttack
import torch

# Initialization
attack = AutoAttack(
    model=pytorch_model,
    norm='Linf',
    eps=0.03,
    version='standard'
)

# Using key methods
images = torch.randn(10, 3, 32, 32)  # Example inputs
labels = torch.randint(0, 10, (10,))  # Example labels
adversarial_images = attack.run_standard_attacks(images, labels)

# Evaluate robustness
with torch.no_grad():
    clean_acc = (pytorch_model(images).argmax(dim=1) == labels).float().mean()
    adv_acc = (pytorch_model(adversarial_images).argmax(dim=1) == labels).float().mean()
    
print(f"Clean Accuracy: {clean_acc.item()}, Adversarial Accuracy: {adv_acc.item()}")
```

## Integration Points

- **ML Training Pipeline**: Used during model training to assess and improve robustness
- **Model Validation**: Integrates with validation pipeline to certify model robustness
- **Model Registry**: Provides robustness metrics for registered models
- **Monitoring System**: Supplies robustness metrics for deployed models
- **Compliance Documentation**: Generates robustness documentation for regulatory compliance

## Dependencies

- **PyTorch**: Core dependency for implementing attacks and defenses
- **NumPy**: Used for numerical operations
- **SciPy**: Required for statistical tests and optimizations
- **Matplotlib**: Visualization of attack results and robustness metrics
- **AutoAttack (library)**: Optional dependency for extended attack functionality
