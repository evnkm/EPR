"""
Energy Composition Module

Methods for composing multiple energy functions following Du et al.'s
compositional EBM framework (product-of-experts).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import EnergyFunction, ComposedEnergy


class AdditiveComposition(ComposedEnergy):
    """
    Additive energy composition: E_total = Σ E_i

    This is equivalent to product-of-experts in probability space:
    p(x) ∝ exp(-Σ E_i) = Π exp(-E_i)
    """

    def __init__(
        self,
        components: List[EnergyFunction],
        weights: Optional[List[float]] = None,
        temperature: float = 1.0,
    ):
        super().__init__(
            components=components,
            mode="additive",
            weights=weights,
            temperature=temperature,
        )

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """Compute sum of component energies."""
        if self.weights is None or all(w == 1.0 for w in self.weights):
            return sum(
                comp.compute_energy(output, input_grid, params)
                for comp in self.components
            )
        return sum(
            w * comp.compute_energy(output, input_grid, params)
            for w, comp in zip(self.weights, self.components)
        )


class ProductOfExpertsComposition(ComposedEnergy):
    """
    Product-of-experts composition following Du et al.

    The key insight: combining energy functions additively creates
    a joint distribution that satisfies ALL constraints simultaneously.

    For ARC: E_total = E_rotation + E_color_map allows searching for
    outputs that are both rotated AND color-mapped from input.
    """

    def __init__(
        self,
        components: List[EnergyFunction],
        temperature: float = 1.0,
        normalize: bool = False,
    ):
        """
        Initialize product-of-experts composition.

        Args:
            components: List of energy functions (experts)
            temperature: Temperature for sampling
            normalize: If True, normalize by number of components
        """
        super().__init__(
            components=components,
            mode="product",
            temperature=temperature,
        )
        self.normalize = normalize

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """Compute product-of-experts energy."""
        total = sum(
            comp.compute_energy(output, input_grid, params)
            for comp in self.components
        )
        if self.normalize:
            return total / len(self.components)
        return total

    def compute_joint_probability(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        Compute unnormalized joint probability.

        p(output | input) ∝ Π_i exp(-E_i / T)
        """
        energy = self.compute_energy(output, input_grid, params)
        return np.exp(-energy / self.temperature)


class SequentialComposition(EnergyFunction):
    """
    Sequential composition for multi-step transformations.

    Models: input -> T1 -> intermediate -> T2 -> output

    The energy is computed by finding the best intermediate state.
    """

    def __init__(
        self,
        step1_energy: EnergyFunction,
        step2_energy: EnergyFunction,
        temperature: float = 1.0,
        search_intermediates: bool = True,
    ):
        """
        Initialize sequential composition.

        Args:
            step1_energy: Energy function for first transformation
            step2_energy: Energy function for second transformation
            temperature: Temperature parameter
            search_intermediates: If True, search for best intermediate
        """
        name = f"Sequential({step1_energy.name} -> {step2_energy.name})"
        super().__init__(name, temperature)

        self.step1 = step1_energy
        self.step2 = step2_energy
        self.search_intermediates = search_intermediates

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        Compute sequential composition energy.

        E(output | input) = min_intermediate [E1(intermediate | input) + E2(output | intermediate)]
        """
        params = params or {}

        # If intermediate is provided, use it
        if "intermediate" in params:
            intermediate = params["intermediate"]
            e1 = self.step1.compute_energy(intermediate, input_grid)
            e2 = self.step2.compute_energy(output, intermediate)
            return e1 + e2

        # Otherwise, we need to search or use heuristics
        if not self.search_intermediates:
            # Without search, assume direct composition
            # This is an approximation
            return self.step1.compute_energy(output, input_grid)

        # Search over candidate intermediates
        # This is expensive - generate candidates from step1
        return self._search_intermediates(output, input_grid)

    def _search_intermediates(
        self,
        output: np.ndarray,
        input_grid: np.ndarray
    ) -> float:
        """Search for best intermediate state."""
        # Generate candidate intermediates by applying step1 transformations
        candidates = self._generate_intermediate_candidates(input_grid)

        best_energy = float('inf')
        for candidate in candidates:
            e1 = self.step1.compute_energy(candidate, input_grid)
            e2 = self.step2.compute_energy(output, candidate)
            total = e1 + e2
            best_energy = min(best_energy, total)

        return best_energy

    def _generate_intermediate_candidates(
        self,
        input_grid: np.ndarray
    ) -> List[np.ndarray]:
        """Generate candidate intermediate states."""
        # This is a placeholder - in practice, this should be customized
        # based on the types of transformations in step1
        from .transformations import (
            rotate_90, rotate_180, rotate_270,
            flip_horizontal, flip_vertical,
        )

        candidates = [
            input_grid.copy(),
            rotate_90(input_grid),
            rotate_180(input_grid),
            rotate_270(input_grid),
            flip_horizontal(input_grid),
            flip_vertical(input_grid),
        ]
        return candidates


class MinEnergyComposition(EnergyFunction):
    """
    Take minimum energy across components.

    Useful when only one transformation type is correct.
    """

    def __init__(
        self,
        components: List[EnergyFunction],
        temperature: float = 1.0,
    ):
        names = [c.name for c in components]
        super().__init__(f"Min({', '.join(names)})", temperature)
        self.components = components

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """Return minimum energy across all components."""
        return min(
            comp.compute_energy(output, input_grid, params)
            for comp in self.components
        )

    def identify_best_component(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> Tuple[EnergyFunction, float]:
        """Identify which component has lowest energy."""
        best_comp = None
        best_energy = float('inf')

        for comp in self.components:
            energy = comp.compute_energy(output, input_grid, params)
            if energy < best_energy:
                best_energy = energy
                best_comp = comp

        return best_comp, best_energy


class ConditionalComposition(EnergyFunction):
    """
    Conditional composition based on input properties.

    Different energy functions are applied based on input characteristics.
    """

    def __init__(
        self,
        condition_fn,
        true_energy: EnergyFunction,
        false_energy: EnergyFunction,
        temperature: float = 1.0,
    ):
        """
        Initialize conditional composition.

        Args:
            condition_fn: Function that takes input_grid and returns bool
            true_energy: Energy function if condition is True
            false_energy: Energy function if condition is False
        """
        name = f"Conditional({true_energy.name}, {false_energy.name})"
        super().__init__(name, temperature)

        self.condition_fn = condition_fn
        self.true_energy = true_energy
        self.false_energy = false_energy

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """Compute energy based on condition."""
        if self.condition_fn(input_grid):
            return self.true_energy.compute_energy(output, input_grid, params)
        else:
            return self.false_energy.compute_energy(output, input_grid, params)


def compose_transformations(
    energy_functions: List[EnergyFunction],
    mode: str = "additive"
) -> EnergyFunction:
    """
    Factory function to compose energy functions.

    Args:
        energy_functions: List of energy functions to compose
        mode: Composition mode ('additive', 'product', 'min', 'sequential')

    Returns:
        Composed energy function
    """
    if len(energy_functions) == 0:
        raise ValueError("Need at least one energy function")

    if len(energy_functions) == 1:
        return energy_functions[0]

    if mode == "additive":
        return AdditiveComposition(energy_functions)

    elif mode == "product":
        return ProductOfExpertsComposition(energy_functions)

    elif mode == "min":
        return MinEnergyComposition(energy_functions)

    elif mode == "sequential":
        if len(energy_functions) != 2:
            raise ValueError("Sequential composition requires exactly 2 functions")
        return SequentialComposition(energy_functions[0], energy_functions[1])

    else:
        raise ValueError(f"Unknown composition mode: {mode}")
