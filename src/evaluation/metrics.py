"""
Evaluation Metrics for ARC Tasks

Metrics for evaluating prediction quality and energy function discrimination.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..energy.base import EnergyFunction


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    exact_match: float
    cell_accuracy: float
    shape_match: float
    energy_separation: Optional[float] = None
    auc: Optional[float] = None
    per_task_results: Optional[Dict] = None

    def summary(self) -> str:
        """Return summary string."""
        lines = [
            "Evaluation Results",
            "=" * 40,
            f"Exact Match:    {self.exact_match:.2%}",
            f"Cell Accuracy:  {self.cell_accuracy:.2%}",
            f"Shape Match:    {self.shape_match:.2%}",
        ]
        if self.energy_separation is not None:
            lines.append(f"Energy Sep:     {self.energy_separation:.3f}")
        if self.auc is not None:
            lines.append(f"AUC:            {self.auc:.3f}")
        return "\n".join(lines)


def exact_match_accuracy(
    predictions: List[np.ndarray],
    targets: List[np.ndarray]
) -> float:
    """
    Compute exact match accuracy.

    Returns fraction of predictions that exactly match targets.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    if len(predictions) == 0:
        return 0.0

    matches = sum(
        1 for pred, target in zip(predictions, targets)
        if np.array_equal(pred, target)
    )
    return matches / len(predictions)


def cell_accuracy(
    predictions: List[np.ndarray],
    targets: List[np.ndarray]
) -> float:
    """
    Compute average cell-wise accuracy.

    For shape mismatches, accuracy is 0 for that example.
    """
    if len(predictions) == 0:
        return 0.0

    accuracies = []
    for pred, target in zip(predictions, targets):
        if pred.shape == target.shape:
            acc = np.mean(pred == target)
        else:
            acc = 0.0
        accuracies.append(acc)

    return np.mean(accuracies)


def shape_accuracy(
    predictions: List[np.ndarray],
    targets: List[np.ndarray]
) -> float:
    """Compute fraction of predictions with correct shape."""
    if len(predictions) == 0:
        return 0.0

    matches = sum(
        1 for pred, target in zip(predictions, targets)
        if pred.shape == target.shape
    )
    return matches / len(predictions)


def transformation_identification_accuracy(
    predictions: List[str],
    targets: List[str]
) -> float:
    """
    Compute accuracy of transformation type identification.

    Args:
        predictions: List of predicted transformation names
        targets: List of ground truth transformation names
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    if len(predictions) == 0:
        return 0.0

    matches = sum(1 for p, t in zip(predictions, targets) if p == t)
    return matches / len(predictions)


def energy_discrimination_metrics(
    energy_fn: EnergyFunction,
    correct_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    incorrect_outputs_fn=None,
    n_negative_samples: int = 5,
) -> Dict[str, float]:
    """
    Evaluate how well an energy function discriminates correct from incorrect.

    Args:
        energy_fn: Energy function to evaluate
        correct_pairs: List of (input, correct_output, params) tuples
        incorrect_outputs_fn: Function that generates incorrect outputs
        n_negative_samples: Number of negative samples per example

    Returns:
        Dictionary with discrimination metrics
    """
    correct_energies = []
    incorrect_energies = []

    for input_grid, correct_output, *rest in correct_pairs:
        params = rest[0] if rest else None

        # Energy of correct output
        e_correct = energy_fn.compute_energy(correct_output, input_grid, params)
        correct_energies.append(e_correct)

        # Generate incorrect outputs
        if incorrect_outputs_fn is not None:
            incorrect_outputs = incorrect_outputs_fn(
                input_grid, correct_output, n_negative_samples
            )
            for inc_out in incorrect_outputs:
                e_inc = energy_fn.compute_energy(inc_out, input_grid, params)
                incorrect_energies.append(e_inc)
        else:
            # Default: random perturbations
            for _ in range(n_negative_samples):
                perturbed = perturb_grid(correct_output)
                e_inc = energy_fn.compute_energy(perturbed, input_grid, params)
                incorrect_energies.append(e_inc)

    correct_mean = np.mean(correct_energies)
    incorrect_mean = np.mean(incorrect_energies)
    separation = incorrect_mean - correct_mean

    # Compute AUC
    auc = compute_auc(correct_energies, incorrect_energies)

    return {
        "correct_mean_energy": correct_mean,
        "incorrect_mean_energy": incorrect_mean,
        "energy_separation": separation,
        "auc": auc,
        "correct_energies": correct_energies,
        "incorrect_energies": incorrect_energies,
    }


def compute_auc(
    correct_energies: List[float],
    incorrect_energies: List[float]
) -> float:
    """
    Compute AUC for energy-based discrimination.

    Lower energy should indicate correct outputs, so we count
    how often correct < incorrect.
    """
    n_correct = len(correct_energies)
    n_incorrect = len(incorrect_energies)

    if n_correct == 0 or n_incorrect == 0:
        return 0.5

    concordant = 0
    for ce in correct_energies:
        for ie in incorrect_energies:
            if ce < ie:
                concordant += 1
            elif ce == ie:
                concordant += 0.5

    return concordant / (n_correct * n_incorrect)


def perturb_grid(
    grid: np.ndarray,
    n_cells: int = 3,
    max_color: int = 9
) -> np.ndarray:
    """
    Create a perturbed version of a grid.

    Args:
        grid: Input grid
        n_cells: Number of cells to perturb
        max_color: Maximum color value

    Returns:
        Perturbed grid
    """
    result = grid.copy()
    h, w = grid.shape

    for _ in range(n_cells):
        i = np.random.randint(0, h)
        j = np.random.randint(0, w)
        # Change to a different color
        current = result[i, j]
        new_color = np.random.randint(0, max_color + 1)
        while new_color == current:
            new_color = np.random.randint(0, max_color + 1)
        result[i, j] = new_color

    return result


def generate_negative_samples(
    input_grid: np.ndarray,
    correct_output: np.ndarray,
    n_samples: int,
    perturbation_levels: List[int] = [1, 3, 5, 10]
) -> List[np.ndarray]:
    """
    Generate negative (incorrect) output samples.

    Generates outputs with varying levels of perturbation.
    """
    samples = []

    for _ in range(n_samples):
        level = np.random.choice(perturbation_levels)
        perturbed = perturb_grid(correct_output, n_cells=level)
        samples.append(perturbed)

    return samples


def evaluate_on_task(
    energy_fn: EnergyFunction,
    task,
    generate_candidates_fn=None,
) -> Dict:
    """
    Evaluate energy function on a single ARC task.

    Args:
        energy_fn: Energy function to evaluate
        task: ARCTask object
        generate_candidates_fn: Function to generate candidate outputs

    Returns:
        Evaluation dictionary
    """
    results = {
        "task_id": task.task_id,
        "train_examples": [],
        "test_examples": [],
    }

    # Evaluate on training examples
    for i, ex in enumerate(task.train):
        energy = energy_fn.compute_energy(ex.output, ex.input)
        results["train_examples"].append({
            "example_idx": i,
            "correct_energy": energy,
        })

    # Evaluate on test examples
    for i, ex in enumerate(task.test):
        energy = energy_fn.compute_energy(ex.output, ex.input)
        results["test_examples"].append({
            "example_idx": i,
            "correct_energy": energy,
        })

    return results


def evaluate_transformation_identification(
    tasks: List,
    energy_functions: Dict[str, EnergyFunction],
) -> Dict:
    """
    Evaluate ability to identify correct transformation type.

    Args:
        tasks: List of ARCTask objects
        energy_functions: Dictionary mapping transform names to energy functions

    Returns:
        Evaluation results
    """
    correct = 0
    total = 0
    per_transform_accuracy = {name: {"correct": 0, "total": 0} for name in energy_functions}

    for task in tasks:
        for ex in task.train:
            # Find which energy function gives lowest energy
            best_name = None
            best_energy = float('inf')

            for name, energy_fn in energy_functions.items():
                e = energy_fn.compute_energy(ex.output, ex.input)
                if e < best_energy:
                    best_energy = e
                    best_name = name

            # This is a simplified evaluation - in practice, we'd need
            # ground truth labels for transformation types
            total += 1

    return {
        "total_evaluated": total,
        "per_transform": per_transform_accuracy,
    }
