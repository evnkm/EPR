"""
Color Mapping Energy Function

Energy function that scores how well an output matches a color-remapped input.
"""

from itertools import permutations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .base import DiscriminativeEnergy, ParameterizedEnergy
from .transformations import apply_color_map, find_color_map, grid_similarity


class ColorMapEnergy(ParameterizedEnergy, DiscriminativeEnergy):
    """
    Energy function for color remapping transformations.

    Computes E(output | input, color_map) based on how well the output
    matches the color-remapped input.

    Color maps are bijective mappings from input colors to output colors.
    The search space can be:
    - Specified color map
    - Inferred from input/output
    - Search over all permutations (expensive)
    """

    NUM_COLORS = 10  # ARC uses colors 0-9

    def __init__(
        self,
        color_map: Optional[Dict[int, int]] = None,
        mode: str = "soft",
        temperature: float = 1.0,
        mismatch_penalty: float = 10.0,
        infer_map: bool = True,
    ):
        """
        Initialize color map energy function.

        Args:
            color_map: Explicit color mapping. If None, inferred from data.
            mode: 'exact' for binary match, 'soft' for cell-wise similarity
            temperature: Temperature for energy computation
            mismatch_penalty: Penalty for inconsistent mapping
            infer_map: If True, attempt to infer color map from input/output
        """
        param_space = {"color_map": "inferred"}
        name = f"ColorMapEnergy(mode={mode})"

        ParameterizedEnergy.__init__(self, name, param_space, temperature)
        DiscriminativeEnergy.__init__(self, name, temperature)

        self.color_map = color_map
        self.mode = mode
        self.mismatch_penalty = mismatch_penalty
        self.infer_map = infer_map

    def enumerate_params(self) -> List[Dict]:
        """
        Enumerate color map parameters.

        Note: Full enumeration is expensive (10! = 3.6M for all permutations).
        In practice, we infer the map from data instead.
        """
        if self.color_map is not None:
            return [{"color_map": self.color_map}]
        # Don't enumerate all permutations - use infer_map instead
        return []

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        Compute color map energy.

        If color_map is specified, uses that mapping.
        If infer_map is True, attempts to infer the optimal mapping.
        """
        params = params or {}
        color_map = params.get("color_map", self.color_map)

        # Check shape compatibility first
        if input_grid.shape != output.shape:
            return self.mismatch_penalty

        if color_map is not None:
            return self._compute_with_map(output, input_grid, color_map)

        if self.infer_map:
            # Try to infer the color map
            inferred_map = find_color_map(input_grid, output)
            if inferred_map is not None:
                return self._compute_with_map(output, input_grid, inferred_map)

        # No valid color map found
        return self.mismatch_penalty

    def _compute_with_map(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        color_map: Dict[int, int]
    ) -> float:
        """Compute energy with a specific color map."""
        mapped = apply_color_map(input_grid, color_map)

        if self.mode == "exact":
            if np.array_equal(mapped, output):
                return 0.0
            return self.mismatch_penalty

        elif self.mode == "soft":
            similarity = grid_similarity(mapped, output)
            if similarity == 1.0:
                return 0.0
            elif similarity == 0.0:
                return self.mismatch_penalty
            else:
                mismatch_fraction = 1.0 - similarity
                return mismatch_fraction * self.mismatch_penalty

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def infer_color_map(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict[int, int]]:
        """
        Infer the color map from input/output pair.

        Returns:
            Color map dictionary if consistent mapping exists, None otherwise
        """
        return find_color_map(input_grid, output_grid)

    def is_pure_color_map(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> bool:
        """Check if the transformation is purely a color remapping."""
        if input_grid.shape != output_grid.shape:
            return False
        return self.infer_color_map(input_grid, output_grid) is not None

    def get_colors_used(self, grid: np.ndarray) -> Set[int]:
        """Get set of colors used in a grid."""
        return set(np.unique(grid).tolist())

    def compute_color_histogram(self, grid: np.ndarray) -> np.ndarray:
        """Compute histogram of colors in grid."""
        hist = np.zeros(self.NUM_COLORS, dtype=np.int32)
        for c in range(self.NUM_COLORS):
            hist[c] = np.sum(grid == c)
        return hist

    def color_histogram_distance(
        self,
        grid1: np.ndarray,
        grid2: np.ndarray
    ) -> float:
        """
        Compute distance between color histograms.

        A distance of 0 means histograms are identical (necessary but not
        sufficient for color map transformation).
        """
        hist1 = self.compute_color_histogram(grid1)
        hist2 = self.compute_color_histogram(grid2)

        # Sort histograms to compare distributions
        sorted1 = np.sort(hist1)
        sorted2 = np.sort(hist2)

        return np.sum(np.abs(sorted1 - sorted2)) / (2 * grid1.size)


class ColorSwapEnergy(DiscriminativeEnergy):
    """
    Energy function specifically for color swap transformations.

    Swaps are a special case of color maps where exactly two colors
    are exchanged.
    """

    def __init__(
        self,
        c1: Optional[int] = None,
        c2: Optional[int] = None,
        mode: str = "soft",
        temperature: float = 1.0,
    ):
        """
        Initialize color swap energy.

        Args:
            c1, c2: Colors to swap. If None, searches for best swap.
            mode: 'exact' or 'soft'
            temperature: Temperature parameter
        """
        super().__init__(f"ColorSwapEnergy({c1}<->{c2})", temperature)
        self.c1 = c1
        self.c2 = c2
        self.mode = mode

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """Compute energy for color swap."""
        params = params or {}
        c1 = params.get("c1", self.c1)
        c2 = params.get("c2", self.c2)

        if input_grid.shape != output.shape:
            return 10.0

        if c1 is not None and c2 is not None:
            return self._compute_swap_energy(output, input_grid, c1, c2)

        # Search for best swap
        return self._find_best_swap_energy(output, input_grid)

    def _compute_swap_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        c1: int,
        c2: int
    ) -> float:
        """Compute energy for specific color swap."""
        swapped = input_grid.copy()
        mask1 = input_grid == c1
        mask2 = input_grid == c2
        swapped[mask1] = c2
        swapped[mask2] = c1

        if self.mode == "exact":
            return 0.0 if np.array_equal(swapped, output) else 10.0
        else:
            return (1.0 - grid_similarity(swapped, output)) * 10.0

    def _find_best_swap_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray
    ) -> float:
        """Find the color swap with lowest energy."""
        best_energy = float('inf')

        for c1 in range(10):
            for c2 in range(c1 + 1, 10):
                energy = self._compute_swap_energy(output, input_grid, c1, c2)
                best_energy = min(best_energy, energy)

        return best_energy

    def identify_swap(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Identify the color swap that transforms input to output.

        Returns:
            (c1, c2) tuple if swap found, None otherwise
        """
        for c1 in range(10):
            for c2 in range(c1 + 1, 10):
                if self._compute_swap_energy(output_grid, input_grid, c1, c2) == 0.0:
                    return (c1, c2)
        return None
