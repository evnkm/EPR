"""
ARC Dataset Loader

Handles loading and parsing of ARC-AGI tasks from JSON format.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ARCExample:
    """A single input-output example from an ARC task."""
    input: np.ndarray
    output: np.ndarray

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.input.shape

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self.output.shape

    def __post_init__(self):
        if isinstance(self.input, list):
            self.input = np.array(self.input, dtype=np.int8)
        if isinstance(self.output, list):
            self.output = np.array(self.output, dtype=np.int8)


@dataclass
class ARCTask:
    """
    An ARC task containing training examples and test examples.

    Attributes:
        task_id: Unique identifier for the task
        train: List of training input-output examples
        test: List of test input-output examples
    """
    task_id: str
    train: List[ARCExample] = field(default_factory=list)
    test: List[ARCExample] = field(default_factory=list)

    @property
    def num_train(self) -> int:
        return len(self.train)

    @property
    def num_test(self) -> int:
        return len(self.test)

    @classmethod
    def from_json(cls, task_id: str, data: Dict) -> "ARCTask":
        """Load task from parsed JSON data."""
        train = [
            ARCExample(
                input=np.array(ex["input"], dtype=np.int8),
                output=np.array(ex["output"], dtype=np.int8)
            )
            for ex in data.get("train", [])
        ]
        test = [
            ARCExample(
                input=np.array(ex["input"], dtype=np.int8),
                output=np.array(ex["output"], dtype=np.int8)
            )
            for ex in data.get("test", [])
        ]
        return cls(task_id=task_id, train=train, test=test)

    def get_all_examples(self) -> List[ARCExample]:
        """Return all examples (train + test)."""
        return self.train + self.test

    def get_input_shapes(self) -> List[Tuple[int, int]]:
        """Return shapes of all input grids."""
        return [ex.input_shape for ex in self.get_all_examples()]

    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Return shapes of all output grids."""
        return [ex.output_shape for ex in self.get_all_examples()]

    def get_colors_used(self) -> set:
        """Return set of all colors used in this task."""
        colors = set()
        for ex in self.get_all_examples():
            colors.update(np.unique(ex.input).tolist())
            colors.update(np.unique(ex.output).tolist())
        return colors


class ARCLoader:
    """
    Loader for ARC-AGI dataset.

    Expected directory structure:
        data_dir/
            training/
                task_id.json
                ...
            evaluation/
                task_id.json
                ...
    """

    # ARC color palette (0-9)
    COLORS = {
        0: "black",
        1: "blue",
        2: "red",
        3: "green",
        4: "yellow",
        5: "gray",
        6: "magenta",
        7: "orange",
        8: "cyan",
        9: "brown"
    }

    NUM_COLORS = 10

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize ARC loader.

        Args:
            data_dir: Path to ARC dataset directory
        """
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "training"
        self.evaluation_dir = self.data_dir / "evaluation"

        self._training_tasks: Optional[Dict[str, ARCTask]] = None
        self._evaluation_tasks: Optional[Dict[str, ARCTask]] = None

    def _load_task_from_file(self, filepath: Path) -> ARCTask:
        """Load a single task from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        task_id = filepath.stem
        return ARCTask.from_json(task_id, data)

    def _load_tasks_from_dir(self, task_dir: Path) -> Dict[str, ARCTask]:
        """Load all tasks from a directory."""
        tasks = {}
        if not task_dir.exists():
            return tasks

        for filepath in sorted(task_dir.glob("*.json")):
            task = self._load_task_from_file(filepath)
            tasks[task.task_id] = task

        return tasks

    @property
    def training_tasks(self) -> Dict[str, ARCTask]:
        """Lazy load training tasks."""
        if self._training_tasks is None:
            self._training_tasks = self._load_tasks_from_dir(self.training_dir)
        return self._training_tasks

    @property
    def evaluation_tasks(self) -> Dict[str, ARCTask]:
        """Lazy load evaluation tasks."""
        if self._evaluation_tasks is None:
            self._evaluation_tasks = self._load_tasks_from_dir(self.evaluation_dir)
        return self._evaluation_tasks

    def get_task(self, task_id: str) -> Optional[ARCTask]:
        """Get a specific task by ID."""
        if task_id in self.training_tasks:
            return self.training_tasks[task_id]
        if task_id in self.evaluation_tasks:
            return self.evaluation_tasks[task_id]
        return None

    def get_all_tasks(self) -> Dict[str, ARCTask]:
        """Get all tasks (training + evaluation)."""
        return {**self.training_tasks, **self.evaluation_tasks}

    def get_task_ids(self, split: str = "all") -> List[str]:
        """
        Get list of task IDs.

        Args:
            split: One of "training", "evaluation", or "all"
        """
        if split == "training":
            return list(self.training_tasks.keys())
        elif split == "evaluation":
            return list(self.evaluation_tasks.keys())
        else:
            return list(self.get_all_tasks().keys())

    def filter_tasks_by_grid_size(
        self,
        max_size: int = 30,
        split: str = "training"
    ) -> List[ARCTask]:
        """Filter tasks where all grids are at most max_size x max_size."""
        tasks = self.training_tasks if split == "training" else self.evaluation_tasks
        filtered = []

        for task in tasks.values():
            max_dim = 0
            for ex in task.get_all_examples():
                max_dim = max(max_dim, max(ex.input_shape), max(ex.output_shape))
            if max_dim <= max_size:
                filtered.append(task)

        return filtered

    def filter_tasks_by_num_colors(
        self,
        max_colors: int = 5,
        split: str = "training"
    ) -> List[ARCTask]:
        """Filter tasks that use at most max_colors distinct colors."""
        tasks = self.training_tasks if split == "training" else self.evaluation_tasks
        filtered = []

        for task in tasks.values():
            if len(task.get_colors_used()) <= max_colors:
                filtered.append(task)

        return filtered

    def summary(self) -> str:
        """Return summary statistics of the dataset."""
        lines = [
            "ARC Dataset Summary",
            "=" * 40,
            f"Data directory: {self.data_dir}",
            f"Training tasks: {len(self.training_tasks)}",
            f"Evaluation tasks: {len(self.evaluation_tasks)}",
        ]
        return "\n".join(lines)


def load_arc_from_github(target_dir: Union[str, Path] = "data/arc") -> ARCLoader:
    """
    Download ARC dataset from GitHub and return loader.

    Note: Requires git to be installed.
    """
    import subprocess

    target_dir = Path(target_dir)

    if not target_dir.exists():
        target_dir.mkdir(parents=True)

        # Clone the ARC dataset repository
        repo_url = "https://github.com/fchollet/ARC-AGI.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir / "repo")],
            check=True
        )

        # Move data to expected structure
        repo_data = target_dir / "repo" / "data"
        if repo_data.exists():
            for subdir in ["training", "evaluation"]:
                src = repo_data / subdir
                dst = target_dir / subdir
                if src.exists():
                    src.rename(dst)

    return ARCLoader(target_dir)
