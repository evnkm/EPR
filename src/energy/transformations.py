"""
ARC Transformation Primitives

Core transformation operations for ARC grids including rotations,
reflections, translations, color mappings, and structural operations.
"""

from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class TransformationType(Enum):
    """Enumeration of supported transformation types."""
    # Rotations
    ROTATE_90 = auto()
    ROTATE_180 = auto()
    ROTATE_270 = auto()

    # Reflections
    FLIP_HORIZONTAL = auto()
    FLIP_VERTICAL = auto()
    FLIP_DIAGONAL = auto()
    FLIP_ANTIDIAGONAL = auto()

    # Translations
    TRANSLATE = auto()

    # Color operations
    COLOR_MAP = auto()
    COLOR_SWAP = auto()

    # Structural
    CROP = auto()
    TILE = auto()
    SCALE = auto()

    # Identity
    IDENTITY = auto()


# =============================================================================
# Rotation Operations
# =============================================================================

def rotate_90(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)


def rotate_180(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 180 degrees."""
    return np.rot90(grid, k=2)


def rotate_270(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 270 degrees clockwise (90 counter-clockwise)."""
    return np.rot90(grid, k=1)


def get_all_rotations(grid: np.ndarray) -> Dict[TransformationType, np.ndarray]:
    """Return all rotations of a grid."""
    return {
        TransformationType.IDENTITY: grid.copy(),
        TransformationType.ROTATE_90: rotate_90(grid),
        TransformationType.ROTATE_180: rotate_180(grid),
        TransformationType.ROTATE_270: rotate_270(grid),
    }


# =============================================================================
# Reflection Operations
# =============================================================================

def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    """Flip grid horizontally (left-right)."""
    return np.fliplr(grid)


def flip_vertical(grid: np.ndarray) -> np.ndarray:
    """Flip grid vertically (up-down)."""
    return np.flipud(grid)


def flip_diagonal(grid: np.ndarray) -> np.ndarray:
    """Flip grid along main diagonal (transpose)."""
    return grid.T


def flip_antidiagonal(grid: np.ndarray) -> np.ndarray:
    """Flip grid along anti-diagonal."""
    return np.rot90(grid.T, k=2)


def get_all_reflections(grid: np.ndarray) -> Dict[TransformationType, np.ndarray]:
    """Return all reflections of a grid."""
    return {
        TransformationType.FLIP_HORIZONTAL: flip_horizontal(grid),
        TransformationType.FLIP_VERTICAL: flip_vertical(grid),
        TransformationType.FLIP_DIAGONAL: flip_diagonal(grid),
        TransformationType.FLIP_ANTIDIAGONAL: flip_antidiagonal(grid),
    }


# =============================================================================
# Translation Operations
# =============================================================================

def translate(
    grid: np.ndarray,
    dy: int,
    dx: int,
    fill_value: int = 0
) -> np.ndarray:
    """
    Translate grid by (dy, dx) with wrapping or fill.

    Args:
        grid: Input grid
        dy: Vertical shift (positive = down)
        dx: Horizontal shift (positive = right)
        fill_value: Value to fill empty cells (default: 0 = black)

    Returns:
        Translated grid
    """
    h, w = grid.shape
    result = np.full_like(grid, fill_value)

    # Calculate source and destination ranges
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)

    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)

    # Only copy if there's overlap
    if src_y_end > src_y_start and src_x_end > src_x_start:
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            grid[src_y_start:src_y_end, src_x_start:src_x_end]

    return result


def find_translation(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    max_shift: int = 10
) -> Optional[Tuple[int, int]]:
    """
    Find the translation that transforms input to output.

    Returns:
        (dy, dx) if found, None otherwise
    """
    if input_grid.shape != output_grid.shape:
        return None

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if np.array_equal(translate(input_grid, dy, dx), output_grid):
                return (dy, dx)

    return None


# =============================================================================
# Color Operations
# =============================================================================

def apply_color_map(
    grid: np.ndarray,
    color_map: Dict[int, int]
) -> np.ndarray:
    """
    Apply a color mapping to the grid.

    Args:
        grid: Input grid
        color_map: Dictionary mapping old colors to new colors

    Returns:
        Grid with colors remapped
    """
    result = grid.copy()
    for old_color, new_color in color_map.items():
        result[grid == old_color] = new_color
    return result


def find_color_map(
    input_grid: np.ndarray,
    output_grid: np.ndarray
) -> Optional[Dict[int, int]]:
    """
    Find color mapping that transforms input to output.

    Returns:
        Color map dictionary if found, None if shapes don't match or
        the transformation isn't a pure color mapping.
    """
    if input_grid.shape != output_grid.shape:
        return None

    color_map = {}
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            old_color = int(input_grid[i, j])
            new_color = int(output_grid[i, j])

            if old_color in color_map:
                if color_map[old_color] != new_color:
                    return None  # Inconsistent mapping
            else:
                color_map[old_color] = new_color

    return color_map


def swap_colors(grid: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Swap two colors in the grid."""
    result = grid.copy()
    mask1 = grid == c1
    mask2 = grid == c2
    result[mask1] = c2
    result[mask2] = c1
    return result


# =============================================================================
# Structural Operations
# =============================================================================

def crop(
    grid: np.ndarray,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int
) -> np.ndarray:
    """Crop a region from the grid."""
    return grid[y_start:y_end, x_start:x_end].copy()


def crop_to_content(
    grid: np.ndarray,
    background: int = 0
) -> np.ndarray:
    """Crop grid to bounding box of non-background content."""
    mask = grid != background
    if not mask.any():
        return grid.copy()

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_start, y_end = np.where(rows)[0][[0, -1]]
    x_start, x_end = np.where(cols)[0][[0, -1]]

    return grid[y_start:y_end + 1, x_start:x_end + 1].copy()


def tile(
    grid: np.ndarray,
    repeat_y: int,
    repeat_x: int
) -> np.ndarray:
    """Tile the grid repeat_y times vertically and repeat_x times horizontally."""
    return np.tile(grid, (repeat_y, repeat_x))


def scale(
    grid: np.ndarray,
    scale_factor: int
) -> np.ndarray:
    """Scale up grid by an integer factor (each cell becomes a scale_factor x scale_factor block)."""
    return np.repeat(np.repeat(grid, scale_factor, axis=0), scale_factor, axis=1)


# =============================================================================
# Transformation Application and Identification
# =============================================================================

def apply_transformation(
    grid: np.ndarray,
    transform_type: TransformationType,
    params: Optional[Dict] = None
) -> np.ndarray:
    """
    Apply a transformation to a grid.

    Args:
        grid: Input grid
        transform_type: Type of transformation
        params: Optional parameters for parameterized transformations

    Returns:
        Transformed grid
    """
    params = params or {}

    if transform_type == TransformationType.IDENTITY:
        return grid.copy()

    elif transform_type == TransformationType.ROTATE_90:
        return rotate_90(grid)

    elif transform_type == TransformationType.ROTATE_180:
        return rotate_180(grid)

    elif transform_type == TransformationType.ROTATE_270:
        return rotate_270(grid)

    elif transform_type == TransformationType.FLIP_HORIZONTAL:
        return flip_horizontal(grid)

    elif transform_type == TransformationType.FLIP_VERTICAL:
        return flip_vertical(grid)

    elif transform_type == TransformationType.FLIP_DIAGONAL:
        return flip_diagonal(grid)

    elif transform_type == TransformationType.FLIP_ANTIDIAGONAL:
        return flip_antidiagonal(grid)

    elif transform_type == TransformationType.TRANSLATE:
        dy = params.get("dy", 0)
        dx = params.get("dx", 0)
        fill = params.get("fill_value", 0)
        return translate(grid, dy, dx, fill)

    elif transform_type == TransformationType.COLOR_MAP:
        color_map = params.get("color_map", {})
        return apply_color_map(grid, color_map)

    elif transform_type == TransformationType.CROP:
        y_start = params.get("y_start", 0)
        y_end = params.get("y_end", grid.shape[0])
        x_start = params.get("x_start", 0)
        x_end = params.get("x_end", grid.shape[1])
        return crop(grid, y_start, y_end, x_start, x_end)

    elif transform_type == TransformationType.TILE:
        repeat_y = params.get("repeat_y", 1)
        repeat_x = params.get("repeat_x", 1)
        return tile(grid, repeat_y, repeat_x)

    elif transform_type == TransformationType.SCALE:
        scale_factor = params.get("scale_factor", 1)
        return scale(grid, scale_factor)

    else:
        raise ValueError(f"Unknown transformation type: {transform_type}")


def identify_transformation(
    input_grid: np.ndarray,
    output_grid: np.ndarray
) -> List[Tuple[TransformationType, Dict, float]]:
    """
    Identify possible transformations that map input to output.

    Returns:
        List of (transformation_type, params, confidence) tuples,
        sorted by confidence (highest first).
    """
    candidates = []

    # Check identity
    if np.array_equal(input_grid, output_grid):
        candidates.append((TransformationType.IDENTITY, {}, 1.0))

    # Check rotations
    for rot_type in [TransformationType.ROTATE_90, TransformationType.ROTATE_180,
                     TransformationType.ROTATE_270]:
        transformed = apply_transformation(input_grid, rot_type)
        if transformed.shape == output_grid.shape:
            if np.array_equal(transformed, output_grid):
                candidates.append((rot_type, {}, 1.0))

    # Check reflections
    for flip_type in [TransformationType.FLIP_HORIZONTAL, TransformationType.FLIP_VERTICAL,
                      TransformationType.FLIP_DIAGONAL, TransformationType.FLIP_ANTIDIAGONAL]:
        transformed = apply_transformation(input_grid, flip_type)
        if transformed.shape == output_grid.shape:
            if np.array_equal(transformed, output_grid):
                candidates.append((flip_type, {}, 1.0))

    # Check color mapping
    if input_grid.shape == output_grid.shape:
        color_map = find_color_map(input_grid, output_grid)
        if color_map is not None:
            # Check it's not identity
            if any(k != v for k, v in color_map.items()):
                candidates.append((
                    TransformationType.COLOR_MAP,
                    {"color_map": color_map},
                    1.0
                ))

    # Check translation
    if input_grid.shape == output_grid.shape:
        translation = find_translation(input_grid, output_grid)
        if translation is not None:
            dy, dx = translation
            if dy != 0 or dx != 0:
                candidates.append((
                    TransformationType.TRANSLATE,
                    {"dy": dy, "dx": dx},
                    1.0
                ))

    # Check tiling
    out_h, out_w = output_grid.shape
    in_h, in_w = input_grid.shape
    if out_h % in_h == 0 and out_w % in_w == 0:
        repeat_y = out_h // in_h
        repeat_x = out_w // in_w
        if repeat_y > 1 or repeat_x > 1:
            tiled = tile(input_grid, repeat_y, repeat_x)
            if np.array_equal(tiled, output_grid):
                candidates.append((
                    TransformationType.TILE,
                    {"repeat_y": repeat_y, "repeat_x": repeat_x},
                    1.0
                ))

    # Check scaling
    if out_h % in_h == 0 and out_w % in_w == 0:
        scale_y = out_h // in_h
        scale_x = out_w // in_w
        if scale_y == scale_x and scale_y > 1:
            scaled = scale(input_grid, scale_y)
            if np.array_equal(scaled, output_grid):
                candidates.append((
                    TransformationType.SCALE,
                    {"scale_factor": scale_y},
                    1.0
                ))

    return sorted(candidates, key=lambda x: -x[2])


def grid_similarity(grid1: np.ndarray, grid2: np.ndarray) -> float:
    """
    Compute similarity between two grids (0 to 1).

    Returns 0 if shapes don't match, otherwise fraction of matching cells.
    """
    if grid1.shape != grid2.shape:
        return 0.0
    return float(np.mean(grid1 == grid2))
