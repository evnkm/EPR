"""
Energy Function Base Classes

Defines the interface for compositional energy functions that score
how well an output grid matches a transformed input.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class EnergyFunction(ABC):
    """
    Abstract base class for energy functions.

    Energy functions map (output, input, params) -> scalar energy value.
    Lower energy indicates better match (higher probability).

    The probability is proportional to exp(-E(output | input, params)).
    """

    def __init__(self, name: str, temperature: float = 1.0):
        """
        Initialize energy function.

        Args:
            name: Human-readable name for this energy function
            temperature: Temperature parameter for softmax (default 1.0)
        """
        self.name = name
        self.temperature = temperature

    @abstractmethod
    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        Compute energy E(output | input, params).

        Args:
            output: Candidate output grid
            input_grid: Input grid
            params: Optional parameters for the transformation

        Returns:
            Energy value (lower = better match)
        """
        pass

    def __call__(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """Shorthand for compute_energy."""
        return self.compute_energy(output, input_grid, params)

    def compute_probability(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        Compute unnormalized probability exp(-E/T).

        Args:
            output: Candidate output grid
            input_grid: Input grid
            params: Optional parameters

        Returns:
            Unnormalized probability (higher = more likely)
        """
        energy = self.compute_energy(output, input_grid, params)
        return np.exp(-energy / self.temperature)

    def score_candidates(
        self,
        candidates: List[np.ndarray],
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Score multiple candidate outputs and return sorted by energy.

        Args:
            candidates: List of candidate output grids
            input_grid: Input grid
            params: Optional parameters

        Returns:
            List of (energy, candidate) tuples, sorted by energy (lowest first)
        """
        scored = [
            (self.compute_energy(c, input_grid, params), c)
            for c in candidates
        ]
        return sorted(scored, key=lambda x: x[0])

    def gradient_estimate(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None,
        cell_idx: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Estimate gradient of energy w.r.t. discrete output.

        For discrete grids, this estimates the change in energy
        from flipping each cell to each possible color.

        Args:
            output: Current output grid
            input_grid: Input grid
            params: Optional parameters
            cell_idx: If provided, only compute gradient for this cell

        Returns:
            Array of shape (H, W, 10) where entry [i,j,c] is the
            energy if cell (i,j) is changed to color c
        """
        h, w = output.shape
        base_energy = self.compute_energy(output, input_grid, params)

        if cell_idx is not None:
            # Only compute for specified cell
            gradient = np.zeros((1, 1, 10))
            i, j = cell_idx
            for c in range(10):
                if output[i, j] == c:
                    gradient[0, 0, c] = base_energy
                else:
                    modified = output.copy()
                    modified[i, j] = c
                    gradient[0, 0, c] = self.compute_energy(modified, input_grid, params)
            return gradient

        # Compute for all cells
        gradient = np.zeros((h, w, 10))

        for i in range(h):
            for j in range(w):
                for c in range(10):
                    if output[i, j] == c:
                        gradient[i, j, c] = base_energy
                    else:
                        modified = output.copy()
                        modified[i, j] = c
                        gradient[i, j, c] = self.compute_energy(modified, input_grid, params)

        return gradient


class ComposedEnergy(EnergyFunction):
    """
    Composes multiple energy functions.

    Supports different composition modes:
    - 'additive': E_total = sum(E_i)
    - 'product': p_total = prod(exp(-E_i)) -> E_total = sum(E_i)
    - 'weighted': E_total = sum(w_i * E_i)
    - 'max': E_total = max(E_i)
    - 'min': E_total = min(E_i)
    """

    def __init__(
        self,
        components: List[EnergyFunction],
        mode: str = "additive",
        weights: Optional[List[float]] = None,
        temperature: float = 1.0
    ):
        """
        Initialize composed energy function.

        Args:
            components: List of energy functions to compose
            mode: Composition mode ('additive', 'product', 'weighted', 'max', 'min')
            weights: Weights for each component (only used in 'weighted' mode)
            temperature: Temperature parameter
        """
        name = f"Composed({mode})[{', '.join(c.name for c in components)}]"
        super().__init__(name, temperature)

        self.components = components
        self.mode = mode

        if weights is None:
            self.weights = [1.0] * len(components)
        else:
            assert len(weights) == len(components)
            self.weights = weights

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """Compute composed energy."""
        energies = [
            comp.compute_energy(output, input_grid, params)
            for comp in self.components
        ]

        if self.mode == "additive" or self.mode == "product":
            return sum(energies)

        elif self.mode == "weighted":
            return sum(w * e for w, e in zip(self.weights, energies))

        elif self.mode == "max":
            return max(energies)

        elif self.mode == "min":
            return min(energies)

        else:
            raise ValueError(f"Unknown composition mode: {self.mode}")

    def get_component_energies(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Get individual energy contributions from each component."""
        return {
            comp.name: comp.compute_energy(output, input_grid, params)
            for comp in self.components
        }


class ParameterizedEnergy(EnergyFunction):
    """
    Energy function with learnable or searchable parameters.

    Base class for energy functions that need to search over
    transformation parameters (e.g., which rotation angle, which color map).
    """

    def __init__(
        self,
        name: str,
        param_space: Dict[str, Any],
        temperature: float = 1.0
    ):
        """
        Initialize parameterized energy.

        Args:
            name: Name of energy function
            param_space: Dictionary defining parameter search space
            temperature: Temperature parameter
        """
        super().__init__(name, temperature)
        self.param_space = param_space

    @abstractmethod
    def enumerate_params(self) -> List[Dict]:
        """Enumerate all valid parameter combinations."""
        pass

    def find_best_params(
        self,
        output: np.ndarray,
        input_grid: np.ndarray
    ) -> Tuple[Dict, float]:
        """
        Find parameters that minimize energy.

        Returns:
            (best_params, min_energy) tuple
        """
        best_params = None
        best_energy = float('inf')

        for params in self.enumerate_params():
            energy = self.compute_energy(output, input_grid, params)
            if energy < best_energy:
                best_energy = energy
                best_params = params

        return best_params, best_energy


class DiscriminativeEnergy(EnergyFunction):
    """
    Energy function designed to discriminate correct from incorrect outputs.

    Provides methods for evaluation metrics and threshold tuning.
    """

    def __init__(self, name: str, temperature: float = 1.0):
        super().__init__(name, temperature)
        self._threshold = 0.0  # Energy threshold for classification

    def set_threshold(self, threshold: float):
        """Set energy threshold for binary classification."""
        self._threshold = threshold

    def classify(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> bool:
        """
        Classify output as correct (True) or incorrect (False).

        Returns True if energy < threshold.
        """
        energy = self.compute_energy(output, input_grid, params)
        return energy < self._threshold

    def evaluate_discrimination(
        self,
        correct_outputs: List[np.ndarray],
        incorrect_outputs: List[np.ndarray],
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Evaluate discrimination between correct and incorrect outputs.

        Returns:
            Dictionary with:
            - 'correct_energies': mean energy of correct outputs
            - 'incorrect_energies': mean energy of incorrect outputs
            - 'separation': incorrect_mean - correct_mean
            - 'auc': approximate AUC score
        """
        correct_energies = [
            self.compute_energy(o, input_grid, params)
            for o in correct_outputs
        ]
        incorrect_energies = [
            self.compute_energy(o, input_grid, params)
            for o in incorrect_outputs
        ]

        correct_mean = np.mean(correct_energies)
        incorrect_mean = np.mean(incorrect_energies)

        # Simple AUC approximation
        n_correct = len(correct_energies)
        n_incorrect = len(incorrect_energies)
        concordant = 0
        for ce in correct_energies:
            for ie in incorrect_energies:
                if ce < ie:
                    concordant += 1
                elif ce == ie:
                    concordant += 0.5

        auc = concordant / (n_correct * n_incorrect) if n_correct * n_incorrect > 0 else 0.5

        return {
            'correct_mean': correct_mean,
            'incorrect_mean': incorrect_mean,
            'separation': incorrect_mean - correct_mean,
            'auc': auc,
            'correct_energies': correct_energies,
            'incorrect_energies': incorrect_energies,
        }
