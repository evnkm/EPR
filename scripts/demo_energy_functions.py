#!/usr/bin/env python3
"""
Demo script for compositional energy functions.

This script demonstrates:
1. Basic energy function evaluation
2. Transformation identification
3. Energy composition
4. Discrimination between correct and incorrect outputs
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt

from src.energy import (
    RotationEnergy,
    ReflectionEnergy,
    ColorMapEnergy,
    TranslationEnergy,
    AdditiveComposition,
    ProductOfExpertsComposition,
    MinEnergyComposition,
)
from src.energy.transformations import (
    rotate_90,
    rotate_180,
    flip_horizontal,
    flip_vertical,
    apply_color_map,
    translate,
)
from src.data.visualization import (
    visualize_grid,
    visualize_comparison,
    visualize_energy_landscape,
)


def create_test_grid():
    """Create a simple test grid."""
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int8)
    return grid


def demo_rotation_energy():
    """Demonstrate rotation energy function."""
    print("\n" + "=" * 60)
    print("DEMO: Rotation Energy Function")
    print("=" * 60)

    input_grid = create_test_grid()
    rotated_90 = rotate_90(input_grid)
    rotated_180 = rotate_180(input_grid)

    # Create energy function
    rotation_energy = RotationEnergy(mode="soft")

    # Test energies
    print("\nEnergies for different candidate outputs:")
    print("-" * 40)

    # Correct 90-degree rotation
    e1 = rotation_energy.compute_energy(rotated_90, input_grid, {"angle": 90})
    print(f"Rotated 90° (correct angle=90):  {e1:.4f}")

    # Wrong angle
    e2 = rotation_energy.compute_energy(rotated_90, input_grid, {"angle": 180})
    print(f"Rotated 90° (wrong angle=180):   {e2:.4f}")

    # Search over all angles
    e3 = rotation_energy.compute_energy(rotated_90, input_grid)
    print(f"Rotated 90° (search all angles): {e3:.4f}")

    # Identify transformation
    angle, energy = rotation_energy.identify_rotation(input_grid, rotated_90)
    print(f"\nIdentified rotation: {angle}° (energy: {energy:.4f})")

    return input_grid, rotated_90


def demo_reflection_energy():
    """Demonstrate reflection energy function."""
    print("\n" + "=" * 60)
    print("DEMO: Reflection Energy Function")
    print("=" * 60)

    input_grid = create_test_grid()
    flipped_h = flip_horizontal(input_grid)
    flipped_v = flip_vertical(input_grid)

    reflection_energy = ReflectionEnergy(mode="soft")

    print("\nEnergies for different reflections:")
    print("-" * 40)

    # Test horizontal flip
    e1 = reflection_energy.compute_energy(
        flipped_h, input_grid, {"reflection_type": "horizontal"}
    )
    print(f"Horizontal flip (correct type): {e1:.4f}")

    e2 = reflection_energy.compute_energy(
        flipped_h, input_grid, {"reflection_type": "vertical"}
    )
    print(f"Horizontal flip (wrong type):   {e2:.4f}")

    # Identify transformation
    ref_type, energy = reflection_energy.identify_reflection(input_grid, flipped_h)
    print(f"\nIdentified reflection: {ref_type} (energy: {energy:.4f})")

    # Find symmetries in original grid
    symmetries = reflection_energy.find_symmetries(input_grid)
    print(f"Symmetries in input grid: {symmetries if symmetries else 'None'}")

    return input_grid, flipped_h


def demo_color_map_energy():
    """Demonstrate color map energy function."""
    print("\n" + "=" * 60)
    print("DEMO: Color Map Energy Function")
    print("=" * 60)

    input_grid = create_test_grid()
    color_map = {0: 0, 1: 3, 2: 5}  # Map blue->green, red->gray
    remapped = apply_color_map(input_grid, color_map)

    color_energy = ColorMapEnergy(mode="soft")

    print("\nEnergies for color mapping:")
    print("-" * 40)

    # With inferred map
    e1 = color_energy.compute_energy(remapped, input_grid)
    print(f"Correct color map (inferred): {e1:.4f}")

    # Infer the map
    inferred = color_energy.infer_color_map(input_grid, remapped)
    print(f"Inferred color map: {inferred}")

    # Wrong output
    wrong_output = input_grid.copy()
    wrong_output[2, 2] = 7  # Wrong color
    e2 = color_energy.compute_energy(wrong_output, input_grid)
    print(f"Wrong output energy: {e2:.4f}")

    return input_grid, remapped


def demo_translation_energy():
    """Demonstrate translation energy function."""
    print("\n" + "=" * 60)
    print("DEMO: Translation Energy Function")
    print("=" * 60)

    input_grid = create_test_grid()
    translated = translate(input_grid, dy=1, dx=2)

    trans_energy = TranslationEnergy(mode="soft", max_shift=5)

    print("\nEnergies for translation:")
    print("-" * 40)

    # Correct translation
    e1 = trans_energy.compute_energy(translated, input_grid, {"dx": 2, "dy": 1})
    print(f"Correct translation (1,2): {e1:.4f}")

    # Wrong translation
    e2 = trans_energy.compute_energy(translated, input_grid, {"dx": 0, "dy": 0})
    print(f"Wrong translation (0,0):   {e2:.4f}")

    # Identify translation
    shift, energy = trans_energy.identify_translation(input_grid, translated)
    print(f"\nIdentified translation: dy={shift[0]}, dx={shift[1]} (energy: {energy:.4f})")

    return input_grid, translated


def demo_composition():
    """Demonstrate energy composition."""
    print("\n" + "=" * 60)
    print("DEMO: Energy Composition")
    print("=" * 60)

    input_grid = create_test_grid()

    # Apply rotation + color map
    rotated = rotate_90(input_grid)
    color_map = {0: 0, 1: 3, 2: 5}
    transformed = apply_color_map(rotated, color_map)

    print("\nApplied: Rotation 90° + Color Map {1->3, 2->5}")

    # Individual energies
    rot_energy = RotationEnergy(angle=90, mode="soft")
    color_energy = ColorMapEnergy(mode="soft")

    # Composed energy (additive)
    composed = AdditiveComposition([rot_energy, color_energy])

    print("\nEnergies:")
    print("-" * 40)

    # Need intermediate state for proper evaluation
    # For rotation: check if rotated matches
    e_rot = rot_energy.compute_energy(rotated, input_grid)
    print(f"Rotation energy (on intermediate): {e_rot:.4f}")

    # For color map: check if color_map(rotated) = transformed
    e_color = color_energy.compute_energy(transformed, rotated)
    print(f"Color map energy (rotated->final): {e_color:.4f}")

    # Min composition to identify transform type
    min_composed = MinEnergyComposition([
        RotationEnergy(mode="soft"),
        ReflectionEnergy(mode="soft"),
        ColorMapEnergy(mode="soft"),
        TranslationEnergy(mode="soft", max_shift=3),
    ])

    best_fn, best_e = min_composed.identify_best_component(rotated, input_grid)
    print(f"\nBest single transform for step 1: {best_fn.name} (energy: {best_e:.4f})")

    return input_grid, transformed


def demo_discrimination():
    """Demonstrate energy discrimination between correct/incorrect outputs."""
    print("\n" + "=" * 60)
    print("DEMO: Energy Discrimination")
    print("=" * 60)

    input_grid = create_test_grid()
    correct_output = rotate_90(input_grid)

    rot_energy = RotationEnergy(angle=90, mode="soft")

    # Generate incorrect outputs
    print("\nEnergies for correct vs incorrect outputs:")
    print("-" * 40)

    e_correct = rot_energy.compute_energy(correct_output, input_grid)
    print(f"Correct output:        {e_correct:.4f}")

    # Slightly wrong
    wrong1 = correct_output.copy()
    wrong1[2, 2] = 5
    e_wrong1 = rot_energy.compute_energy(wrong1, input_grid)
    print(f"1 cell wrong:          {e_wrong1:.4f}")

    # More wrong
    wrong2 = correct_output.copy()
    wrong2[1:3, 1:3] = 7
    e_wrong2 = rot_energy.compute_energy(wrong2, input_grid)
    print(f"4 cells wrong:         {e_wrong2:.4f}")

    # Completely wrong (random)
    wrong3 = np.random.randint(0, 10, size=correct_output.shape, dtype=np.int8)
    e_wrong3 = rot_energy.compute_energy(wrong3, input_grid)
    print(f"Random output:         {e_wrong3:.4f}")

    # Wrong transform
    wrong4 = flip_horizontal(input_grid)
    e_wrong4 = rot_energy.compute_energy(wrong4, input_grid)
    print(f"Wrong transform (flip): {e_wrong4:.4f}")

    print("\nConclusion: Lower energy = better match to correct transformation")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# EPR: Energy-based Program Reasoning Demo")
    print("# Compositional Energy Functions for ARC-AGI")
    print("#" * 60)

    demo_rotation_energy()
    demo_reflection_energy()
    demo_color_map_energy()
    demo_translation_energy()
    demo_composition()
    demo_discrimination()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
