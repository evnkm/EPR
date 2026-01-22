"""
Rotation Energy Function

Energy function that scores how well an output matches a rotated input.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import DiscriminativeEnergy, ParameterizedEnergy
from .transformations import (
    TransformationType,
    rotate_90,
    rotate_180,
    rotate_270,
    grid_similarity,
)


class RotationEnergy(ParameterizedEnergy, DiscriminativeEnergy):
    """
    Energy function for rotation transformations.

    Computes E(output | input, rotation) based on how well the output
    matches the rotated input. Lower energy = better match.

    Supports:
    - Exact match energy (0 if perfect match, high otherwise)
    - Soft energy based on cell-wise similarity
    - Search over rotation angles
    """

    ROTATIONS = {
        0: TransformationType.IDENTITY,
        90: TransformationType.ROTATE_90,
        180: TransformationType.ROTATE_180,
        270: TransformationType.ROTATE_270,
    }

    def __init__(
        self,
        angle: Optional[int] = None,
        mode: str = "soft",
        temperature: float = 1.0,
        mismatch_penalty: float = 10.0,
    ):
        """
        Initialize rotation energy function.

        Args:
            angle: Rotation angle in degrees (0, 90, 180, 270).
                   If None, searches over all angles.
            mode: 'exact' for binary match, 'soft' for cell-wise similarity
            temperature: Temperature for energy computation
            mismatch_penalty: Penalty for shape mismatch
        """
        param_space = {"angle": [0, 90, 180, 270]}
        name = f"RotationEnergy(angle={angle}, mode={mode})"

        ParameterizedEnergy.__init__(self, name, param_space, temperature)
        DiscriminativeEnergy.__init__(self, name, temperature)

        self.angle = angle
        self.mode = mode
        self.mismatch_penalty = mismatch_penalty

    def enumerate_params(self) -> List[Dict]:
        """Enumerate all rotation parameters."""
        if self.angle is not None:
            return [{"angle": self.angle}]
        return [{"angle": a} for a in [0, 90, 180, 270]]

    def _apply_rotation(self, grid: np.ndarray, angle: int) -> np.ndarray:
        """Apply rotation to grid."""
        if angle == 0:
            return grid.copy()
        elif angle == 90:
            return rotate_90(grid)
        elif angle == 180:
            return rotate_180(grid)
        elif angle == 270:
            return rotate_270(grid)
        else:
            raise ValueError(f"Invalid rotation angle: {angle}")

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        Compute rotation energy.

        If angle is specified (via init or params), computes energy for that angle.
        Otherwise, returns minimum energy over all angles.
        """
        params = params or {}
        angle = params.get("angle", self.angle)

        if angle is not None:
            return self._compute_single_energy(output, input_grid, angle)

        # Search over all angles
        energies = [
            self._compute_single_energy(output, input_grid, a)
            for a in [0, 90, 180, 270]
        ]
        return min(energies)

    def _compute_single_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        angle: int
    ) -> float:
        """Compute energy for a specific rotation angle."""
        rotated = self._apply_rotation(input_grid, angle)

        # Check shape compatibility
        if rotated.shape != output.shape:
            return self.mismatch_penalty

        if self.mode == "exact":
            # Binary: 0 if exact match, mismatch_penalty otherwise
            if np.array_equal(rotated, output):
                return 0.0
            return self.mismatch_penalty

        elif self.mode == "soft":
            # Soft: based on cell-wise mismatch
            # E = -log(similarity) = -log(matching_cells / total_cells)
            similarity = grid_similarity(rotated, output)
            if similarity == 1.0:
                return 0.0
            elif similarity == 0.0:
                return self.mismatch_penalty
            else:
                # Energy based on fraction of mismatched cells
                mismatch_fraction = 1.0 - similarity
                return mismatch_fraction * self.mismatch_penalty

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def identify_rotation(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Tuple[Optional[int], float]:
        """
        Identify the rotation angle that best transforms input to output.

        Returns:
            (angle, energy) tuple. angle is None if no good match found.
        """
        best_angle = None
        best_energy = float('inf')

        for angle in [0, 90, 180, 270]:
            energy = self._compute_single_energy(output_grid, input_grid, angle)
            if energy < best_energy:
                best_energy = energy
                best_angle = angle

        return best_angle, best_energy

    def get_rotated_candidates(
        self,
        input_grid: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Get all rotated versions of input grid."""
        return {
            angle: self._apply_rotation(input_grid, angle)
            for angle in [0, 90, 180, 270]
        }


class RotationSequenceEnergy(DiscriminativeEnergy):
    """
    Energy function for identifying sequences of rotations.

    Useful for tasks where multiple rotations are composed.
    """

    def __init__(self, max_rotations: int = 2, temperature: float = 1.0):
        super().__init__(f"RotationSequence(max={max_rotations})", temperature)
        self.max_rotations = max_rotations
        self.single_energy = RotationEnergy(mode="soft")

    def compute_energy(
        self,
        output: np.ndarray,
        input_grid: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        Compute minimum energy over all rotation sequences up to max_rotations.

        Note: Rotation sequences are equivalent to single rotations mod 360,
        but this provides a framework for composing with other transforms.
        """
        # For pure rotations, any sequence is equivalent to a single rotation
        return self.single_energy.compute_energy(output, input_grid)

    def decompose_transformation(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> List[int]:
        """
        Decompose transformation into rotation sequence.

        Returns:
            List of rotation angles
        """
        angle, energy = self.single_energy.identify_rotation(input_grid, output_grid)
        if energy == 0.0 and angle is not None:
            if angle == 0:
                return []
            return [angle]
        return []
