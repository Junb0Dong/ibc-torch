import dataclasses
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclasses.dataclass
class ToyDatasetConfig:
    dataset_size: int = 200
    """Number of samples."""

    mode: str = "step"
    """Dataset mode: 'step', 'piecewise', or 'random'."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""


class Toy1DDataset(Dataset):
    """Synthetic 1D dataset for implicit vs explicit policy toy examples."""

    def __init__(self, config: ToyDatasetConfig) -> None:
        self.dataset_size = config.dataset_size
        self.mode = config.mode
        self.seed = config.seed

        self.rng = np.random.RandomState(self.seed)
        self._generate_data()

    def _generate_data(self):
        # Uniformly sample x ∈ [0,1]
        self._coordinates = self.rng.rand(self.dataset_size, 1).astype(np.float32)
        self._targets = self._generate_y(self._coordinates)

    def _generate_y(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "step":
            return (x >= 0.5).astype(np.float32)

        elif self.mode == "piecewise":
            y = np.zeros_like(x)
            mask1 = x < 0.33
            y[mask1] = 1.5 * x[mask1]
            mask2 = (x >= 0.33) & (x < 0.66)
            y[mask2] = -1.5 * (x[mask2] - 0.33) + 0.5
            mask3 = x >= 0.66
            y[mask3] = 0.5 + 2 * (x[mask3] - 0.66)
            return y.astype(np.float32)

        elif self.mode == "random":
            return self.rng.rand(len(x), 1).astype(np.float32)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ---------------------------
    # 与 CoordinateRegression 对齐的接口
    # ---------------------------

    def exclude(self, coordinates: np.ndarray) -> None:
        """Exclude the given coordinates, if present, and resample new ones."""
        mask = (self._coordinates == coordinates[:, None]).all(-1).any(0)
        num_matches = mask.sum()
        while mask.sum() > 0:
            # 重新采样这些点
            self._coordinates[mask] = self.rng.rand(mask.sum(), 1).astype(np.float32)
            self._targets[mask] = self._generate_y(self._coordinates[mask])
            mask = (self._coordinates == coordinates[:, None]).all(-1).any(0)
        print(f"Resampled {num_matches} data points.")

    def get_target_bounds(self) -> np.ndarray:
        """Return per-dimension target min/max."""
        # 这里返回 y 的范围 [0,1]
        return np.array([[0.0], [1.0]], dtype=np.float32)

    # ---------------------------
    # Dataset 接口
    # ---------------------------
    @property
    def coordinates(self) -> np.ndarray:
        """Return x values."""
        return self._coordinates

    @property
    def targets(self) -> np.ndarray:
        """Return y values."""
        return self._targets

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._coordinates[idx]),
            torch.from_numpy(self._targets[idx]),
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cfg = ToyDatasetConfig(dataset_size=200, seed=0, mode="piecewise")
    dataset = Toy1DDataset(cfg)

    xs = dataset.coordinates.flatten()
    ys = dataset.targets.flatten()

    plt.scatter(xs, ys, marker="x", c="blue", alpha=0.6)
    plt.title(f"Toy1DDataset - {cfg.mode}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    print("Target bounds:", dataset.get_target_bounds())
