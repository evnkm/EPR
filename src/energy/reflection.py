"""
Reflection Energy Function

Energy function that scores how well an output matches a reflected input.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import DiscriminativeEnergy, ParameterizedEnergy
from .transformations import (
    TransformationType,
    flip_horizontal,
    flip_vertical,
    flip_diagonal,
    flip_antidiagonal,
    grid_similarity,
)


class ReflectionEnergy(ParameterizedEnergy, DiscriminativeEnergy):
    """
    Energy function for reflection/flip transformations.

    Computes E(output | input, reflection_type) based on how well the output
    matches the reflected input.

    Supports:
    - Horizontal flip (left-right)
    - Vertical flip (up-down)
    - Diagonal flip (transpose)
    - Anti-diagonal flip
    """

    REFLECTIONS = {
        "horizontal": flip_horizontal,
        "vertical": flip_vertical,
        "diagonal": flip_diagonal,
        "antidiagonal": flip_antidiagonal,
    }

    REFLECTION_TYPES = {
        "horizontal": TransformationType.FLIP_HORIZONTAL,
        "vertical": TransformationType.FLIP_VERTICAL,
        "diagonal": TransformationType.FLIP_DIAGONAL,
        "antidiagonal": TransformationType.FLIP_ANTIDIAGONAL,
    }

    def __init__(
        self,
        reflection_type: Optional[str] = None,
        mode: str = "soft",
        temperature: float = 1.0,
        mismatch_penalty: float = 10.0,
    ):
        """
        Initialize reflection energy function.

        Args:
            reflection_type: Type of reflection ('horizontal', 'vertical',
                            'diagonal', 'antidiagonal'). If None, searches all.
            mode: 'exact' for binary match, 'soft' for cell-wise similarity
            temperature: Temperature for energy computation
            mismatch_penalty: Penalty for shape mismatch
        """
        param_space = {"reflection_type": list(self.REFLECTIONS.keys())}
        name = f"ReflectionEnergy(type={reflection_type}, mode={mode})"

        ParameterizedEnergy.__init__(self, name, param_space, temperature)
        DiscriminativeEnergy.__init__(self, name, temperature)

        self.reflection_type = reflection_type
        self.mode = mode
        self.mismatch_penalty = mismatch_penalty

    def enumerate_params(self) -> List[Dict]:
        """Enumerate all reflection parameters."""
        if self.reflection_type is not None:
            return [{"reflection_type": self.reflection_type}]
        return [{"reflection_type": rt} for rt in self.REFLECTIONS.keys()]

    def _apply_reflection(self, grid: np.ndarray, reflection_type: str) -> np.ndarray:
        """Apply reflection to grid."""
        if reflection_type not in self.REFLECTIONS:
            raise ValueError(f"Unknown reflection type: {reflection_type}")
        return self.REFLECTIONS[reflection_type](grid)

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        Compute reflection energy.

        If reflection_type is specified, computes energy for that type.
        Otherwise, returns minimum energy over all reflection types.
        """
        params = params or {}
        reflection_type = params.get("reflection_type", self.reflection_type)

        if reflection_type is not None:
            return self._compute_single_energy(output, input_grid, reflection_type)

        # Search over all reflection types
        energies = [
            self._compute_single_energy(output, input_grid, rt)
            for rt in self.REFLECTIONS.keys()
        ]
        return min(energies)

    def _compute_single_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        reflection_type: str
    ) -> float:
        """Compute energy for a specific reflection type."""
        reflected = self._apply_reflection(input_grid, reflection_type)

        # Check shape compatibility
        if reflected.shape != output.shape:
            return self.mismatch_penalty

        if self.mode == "exact":
            if np.array_equal(reflected, output):
                return 0.0
            return self.mismatch_penalty

        elif self.mode == "soft":
            similarity = grid_similarity(reflected, output)
            if similarity == 1.0:
                return 0.0
            elif similarity == 0.0:
                return self.mismatch_penalty
            else:
                mismatch_fraction = 1.0 - similarity
                return mismatch_fraction * self.mismatch_penalty

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def identify_reflection(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Identify the reflection type that best transforms input to output.

        Returns:
            (reflection_type, energy) tuple
        """
        best_type = None
        best_energy = float('inf')

        for rt in self.REFLECTIONS.keys():
            energy = self._compute_single_energy(output_grid, input_grid, rt)
            if energy < best_energy:
                best_energy = energy
                best_type = rt

        return best_type, best_energy

    def get_reflected_candidates(
        self,
        input_grid: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Get all reflected versions of input grid."""
        return {
            rt: self._apply_reflection(input_grid, rt)
            for rt in self.REFLECTIONS.keys()
        }

    def check_symmetry(
        self,
        grid: np.ndarray,
        reflection_type: str
    ) -> bool:
        """Check if a grid has the specified symmetry."""
        reflected = self._apply_reflection(grid, reflection_type)
        return np.array_equal(grid, reflected)

    def find_symmetries(self, grid: np.ndarray) -> List[str]:
        """Find all symmetries present in the grid."""
        return [
            rt for rt in self.REFLECTIONS.keys()
            if self.check_symmetry(grid, rt)
        ]


class RotationReflectionEnergy(DiscriminativeEnergy):
    """
    Combined energy function for rotations and reflections.

    Searches over the dihedral group D4 (8 symmetries of a square):
    - 4 rotations (0, 90, 180, 270)
    - 4 reflections (horizontal, vertical, two diagonals)
    """

    def __init__(self, mode: str = "soft", temperature: float = 1.0):
        super().__init__("RotationReflectionEnergy", temperature)
        self.mode = mode
        self.rotation_energy = RotationEnergy(mode=mode, temperature=temperature)
        self.reflection_energy = ReflectionEnergy(mode=mode, temperature=temperature)

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """Compute minimum energy over all D4 transformations."""
        rotation_e = self.rotation_energy.compute_energy(output, input_grid)
        reflection_e = self.reflection_energy.compute_energy(output, input_grid)
        return min(rotation_e, reflection_e)

    def identify_transformation(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Tuple[str, Dict, float]:
        """
        Identify the best D4 transformation.

        Returns:
            (transform_name, params, energy) tuple
        """
        rot_angle, rot_energy = self.rotation_energy.identify_rotation(
            input_grid, output_grid
        )
        ref_type, ref_energy = self.reflection_energy.identify_reflection(
            input_grid, output_grid
        )

        if rot_energy <= ref_energy:
            return "rotation", {"angle": rot_angle}, rot_energy
        else:
            return "reflection", {"type": ref_type}, ref_energy
