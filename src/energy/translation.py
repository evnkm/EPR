"""
Translation Energy Function

Energy function that scores how well an output matches a translated input.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import DiscriminativeEnergy, ParameterizedEnergy
from .transformations import translate, find_translation, grid_similarity


class TranslationEnergy(ParameterizedEnergy, DiscriminativeEnergy):
    """
    Energy function for translation transformations.

    Computes E(output | input, dx, dy) based on how well the output
    matches the translated input.
    """

    def __init__(
        self,
        dx: Optional[int] = None,
        dy: Optional[int] = None,
        max_shift: int = 10,
        mode: str = "soft",
        temperature: float = 1.0,
        mismatch_penalty: float = 10.0,
        fill_value: int = 0,
    ):
        """
        Initialize translation energy function.

        Args:
            dx: Horizontal shift. If None, searches.
            dy: Vertical shift. If None, searches.
            max_shift: Maximum shift to search in each direction
            mode: 'exact' or 'soft'
            temperature: Temperature parameter
            mismatch_penalty: Penalty for poor matches
            fill_value: Value for cells that shift out of bounds
        """
        param_space = {
            "dx": list(range(-max_shift, max_shift + 1)),
            "dy": list(range(-max_shift, max_shift + 1)),
        }
        name = f"TranslationEnergy(dx={dx}, dy={dy}, mode={mode})"

        ParameterizedEnergy.__init__(self, name, param_space, temperature)
        DiscriminativeEnergy.__init__(self, name, temperature)

        self.dx = dx
        self.dy = dy
        self.max_shift = max_shift
        self.mode = mode
        self.mismatch_penalty = mismatch_penalty
        self.fill_value = fill_value

    def enumerate_params(self) -> List[Dict]:
        """Enumerate all translation parameters."""
        if self.dx is not None and self.dy is not None:
            return [{"dx": self.dx, "dy": self.dy}]

        params = []
        for dy in range(-self.max_shift, self.max_shift + 1):
            for dx in range(-self.max_shift, self.max_shift + 1):
                if dy != 0 or dx != 0:  # Skip identity
                    params.append({"dx": dx, "dy": dy})
        return params

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        Compute translation energy.

        If dx, dy are specified, computes energy for that translation.
        Otherwise, returns minimum energy over search space.
        """
        params = params or {}
        dx = params.get("dx", self.dx)
        dy = params.get("dy", self.dy)

        # Check shape compatibility
        if input_grid.shape != output.shape:
            return self.mismatch_penalty

        if dx is not None and dy is not None:
            return self._compute_single_energy(output, input_grid, dx, dy)

        # Search for best translation
        best_energy = float('inf')
        for search_dy in range(-self.max_shift, self.max_shift + 1):
            for search_dx in range(-self.max_shift, self.max_shift + 1):
                energy = self._compute_single_energy(
                    output, input_grid, search_dx, search_dy
                )
                best_energy = min(best_energy, energy)

        return best_energy

    def _compute_single_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        dx: int,
        dy: int
    ) -> float:
        """Compute energy for specific translation."""
        translated = translate(input_grid, dy, dx, self.fill_value)

        if self.mode == "exact":
            if np.array_equal(translated, output):
                return 0.0
            return self.mismatch_penalty

        elif self.mode == "soft":
            similarity = grid_similarity(translated, output)
            if similarity == 1.0:
                return 0.0
            elif similarity == 0.0:
                return self.mismatch_penalty
            else:
                mismatch_fraction = 1.0 - similarity
                return mismatch_fraction * self.mismatch_penalty

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def identify_translation(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Identify the translation that best transforms input to output.

        Returns:
            ((dy, dx), energy) tuple
        """
        best_shift = None
        best_energy = float('inf')

        for dy in range(-self.max_shift, self.max_shift + 1):
            for dx in range(-self.max_shift, self.max_shift + 1):
                energy = self._compute_single_energy(
                    output_grid, input_grid, dx, dy
                )
                if energy < best_energy:
                    best_energy = energy
                    best_shift = (dy, dx)

        return best_shift, best_energy

    def compute_cross_correlation(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> np.ndarray:
        """
        Compute normalized cross-correlation to find translation.

        Returns:
            2D array of correlation values for each shift
        """
        h, w = input_grid.shape
        corr_h = 2 * self.max_shift + 1
        corr_w = 2 * self.max_shift + 1
        correlation = np.zeros((corr_h, corr_w))

        for i, dy in enumerate(range(-self.max_shift, self.max_shift + 1)):
            for j, dx in enumerate(range(-self.max_shift, self.max_shift + 1)):
                translated = translate(input_grid, dy, dx, self.fill_value)
                correlation[i, j] = grid_similarity(translated, output_grid)

        return correlation

    def find_peak_translation(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Tuple[int, int, float]:
        """
        Find translation with highest cross-correlation.

        Returns:
            (dy, dx, correlation) tuple
        """
        correlation = self.compute_cross_correlation(input_grid, output_grid)
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        dy = peak_idx[0] - self.max_shift
        dx = peak_idx[1] - self.max_shift
        return dy, dx, correlation[peak_idx]


class PeriodicTranslationEnergy(DiscriminativeEnergy):
    """
    Energy function for periodic/wrapping translations.

    Unlike standard translation, content wraps around edges.
    """

    def __init__(
        self,
        mode: str = "soft",
        temperature: float = 1.0,
    ):
        super().__init__("PeriodicTranslation", temperature)
        self.mode = mode

    def _periodic_translate(
        self,
        grid: np.ndarray,
        dy: int,
        dx: int
    ) -> np.ndarray:
        """Translate with wrapping."""
        return np.roll(np.roll(grid, dy, axis=0), dx, axis=1)

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """Compute energy searching over all periodic translations."""
        if input_grid.shape != output.shape:
            return 10.0

        h, w = input_grid.shape
        best_energy = float('inf')

        for dy in range(h):
            for dx in range(w):
                translated = self._periodic_translate(input_grid, dy, dx)
                if self.mode == "exact":
                    if np.array_equal(translated, output):
                        return 0.0
                else:
                    similarity = grid_similarity(translated, output)
                    energy = (1.0 - similarity) * 10.0
                    best_energy = min(best_energy, energy)

        return best_energy if self.mode == "soft" else 10.0
