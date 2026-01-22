"""
ARC Grid Visualization

Utilities for visualizing ARC grids and tasks using matplotlib.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import ListedColormap

# ARC color palette (matching official colors)
ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: gray
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: cyan
    "#870C25",  # 9: brown
]

ARC_CMAP = ListedColormap(ARC_COLORS)


def visualize_grid(
    grid: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_grid_lines: bool = True,
    cell_size: float = 0.5,
) -> plt.Axes:
    """
    Visualize a single ARC grid.

    Args:
        grid: 2D numpy array with values 0-9
        ax: Matplotlib axes to plot on (creates new if None)
        title: Optional title for the plot
        show_grid_lines: Whether to show grid lines
        cell_size: Size of each cell in inches (for figure sizing)

    Returns:
        The matplotlib Axes object
    """
    if ax is None:
        h, w = grid.shape
        fig, ax = plt.subplots(figsize=(w * cell_size, h * cell_size))

    # Plot the grid
    ax.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9)

    # Add grid lines
    if show_grid_lines:
        h, w = grid.shape
        for i in range(h + 1):
            ax.axhline(i - 0.5, color="white", linewidth=0.5)
        for j in range(w + 1):
            ax.axvline(j - 0.5, color="white", linewidth=0.5)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=10)

    return ax


def visualize_example(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    title: Optional[str] = None,
    cell_size: float = 0.4,
) -> plt.Figure:
    """
    Visualize an input-output example pair.

    Args:
        input_grid: Input grid array
        output_grid: Output grid array
        title: Optional title for the figure
        cell_size: Size of each cell in inches

    Returns:
        The matplotlib Figure object
    """
    # Calculate figure size
    in_h, in_w = input_grid.shape
    out_h, out_w = output_grid.shape
    max_h = max(in_h, out_h)
    total_w = in_w + out_w + 2  # +2 for arrow space

    fig, axes = plt.subplots(
        1, 2,
        figsize=(total_w * cell_size, max_h * cell_size + 0.5)
    )

    visualize_grid(input_grid, ax=axes[0], title="Input", cell_size=cell_size)
    visualize_grid(output_grid, ax=axes[1], title="Output", cell_size=cell_size)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    return fig


def visualize_task(
    task,
    max_examples: int = 5,
    cell_size: float = 0.35,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a complete ARC task with all train and test examples.

    Args:
        task: ARCTask object
        max_examples: Maximum number of examples to show
        cell_size: Size of each cell in inches
        save_path: Optional path to save the figure

    Returns:
        The matplotlib Figure object
    """
    n_train = min(len(task.train), max_examples)
    n_test = min(len(task.test), max_examples)
    n_total = n_train + n_test

    # Create subplot grid
    fig, axes = plt.subplots(
        n_total, 2,
        figsize=(10, n_total * 3),
        squeeze=False
    )

    # Plot training examples
    for i, ex in enumerate(task.train[:n_train]):
        visualize_grid(ex.input, ax=axes[i, 0], title=f"Train {i+1} - Input")
        visualize_grid(ex.output, ax=axes[i, 1], title=f"Train {i+1} - Output")

    # Plot test examples
    for i, ex in enumerate(task.test[:n_test]):
        row = n_train + i
        visualize_grid(ex.input, ax=axes[row, 0], title=f"Test {i+1} - Input")
        visualize_grid(ex.output, ax=axes[row, 1], title=f"Test {i+1} - Output (Ground Truth)")

    fig.suptitle(f"Task: {task.task_id}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_comparison(
    input_grid: np.ndarray,
    target_grid: np.ndarray,
    predicted_grid: np.ndarray,
    title: Optional[str] = None,
    cell_size: float = 0.4,
) -> plt.Figure:
    """
    Visualize input, target output, and predicted output.

    Args:
        input_grid: Input grid array
        target_grid: Ground truth output grid
        predicted_grid: Model's predicted output
        title: Optional title
        cell_size: Cell size in inches

    Returns:
        The matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    visualize_grid(input_grid, ax=axes[0], title="Input")
    visualize_grid(target_grid, ax=axes[1], title="Target")
    visualize_grid(predicted_grid, ax=axes[2], title="Predicted")

    # Highlight differences
    if target_grid.shape == predicted_grid.shape:
        diff = target_grid != predicted_grid
        if diff.any():
            # Add red overlay on differences
            for i in range(target_grid.shape[0]):
                for j in range(target_grid.shape[1]):
                    if diff[i, j]:
                        rect = patches.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            linewidth=2,
                            edgecolor="red",
                            facecolor="none"
                        )
                        axes[2].add_patch(rect)

    # Calculate accuracy
    if target_grid.shape == predicted_grid.shape:
        accuracy = (target_grid == predicted_grid).mean() * 100
        exact_match = np.array_equal(target_grid, predicted_grid)
        status = "CORRECT" if exact_match else f"{accuracy:.1f}% cells correct"
    else:
        status = "SHAPE MISMATCH"

    if title:
        fig.suptitle(f"{title} - {status}", fontsize=12)
    else:
        fig.suptitle(status, fontsize=12)

    plt.tight_layout()
    return fig


def visualize_transformation_sequence(
    grids: List[np.ndarray],
    labels: Optional[List[str]] = None,
    cell_size: float = 0.4,
) -> plt.Figure:
    """
    Visualize a sequence of grid transformations.

    Args:
        grids: List of grid arrays showing transformation steps
        labels: Optional labels for each step
        cell_size: Cell size in inches

    Returns:
        The matplotlib Figure object
    """
    n = len(grids)
    if labels is None:
        labels = [f"Step {i}" for i in range(n)]

    max_h = max(g.shape[0] for g in grids)
    total_w = sum(g.shape[1] for g in grids) + n

    fig, axes = plt.subplots(1, n, figsize=(total_w * cell_size, max_h * cell_size + 1))

    if n == 1:
        axes = [axes]

    for i, (grid, label) in enumerate(zip(grids, labels)):
        visualize_grid(grid, ax=axes[i], title=label)

    plt.tight_layout()
    return fig


def visualize_energy_landscape(
    energies: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Energy Landscape",
) -> plt.Figure:
    """
    Visualize energy values for different transformations.

    Args:
        energies: Array of energy values
        labels: Labels for each energy value
        title: Plot title

    Returns:
        The matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(energies))
    colors = plt.cm.RdYlGn_r(energies / energies.max())

    bars = ax.bar(x, energies, color=colors)

    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel("Energy")
    ax.set_title(title)

    # Highlight minimum
    min_idx = np.argmin(energies)
    bars[min_idx].set_edgecolor("black")
    bars[min_idx].set_linewidth(3)

    plt.tight_layout()
    return fig
