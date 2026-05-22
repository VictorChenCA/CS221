"""World-state views.

The agent's observation always carries a `grid` field — a (2R+1)x(2R+1)
window of biome ids centered on the bot's current cell. Where that grid
comes from depends on the proposal's two settings (§2):

  * Complete knowledge: NpzWorldView slices from a pre-extracted biome
    map dump. Same data source the oracle plans on, so there's no
    train/eval drift.
  * Line-of-sight: bridge ships the grid live from `bot.world.getBiome`,
    optionally filtered by visibility. Env passes that grid through
    unchanged.

`Env` selects between them by being constructed with or without a
WorldView. Bridge stays the same in either mode (it always reports
position); in complete mode the grid it ships is ignored.
"""

from pathlib import Path
from typing import Protocol

import numpy as np

CELL_BLOCKS = 4
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

UNKNOWN_BIOME = -1


class WorldView(Protocol):
    def biome_at(self, cell_x: int, cell_z: int) -> int: ...
    def get_grid(self, cell_x: int, cell_z: int, radius_cells: int) -> np.ndarray: ...


class NpzWorldView:
    """Complete-knowledge view backed by `data/biomes_<seed>.npz`.

    File format (numpy savez_compressed):
      - biomes:      int16 array, shape (H, W)
      - origin_cell: int32 [cellX0, cellZ0] of biomes[0, 0]

    Cells outside the dump's window return UNKNOWN_BIOME. Generate the
    dump once per seed via `tools/extract_biomes.py`.
    """

    def __init__(self, seed: int, data_dir: Path = DATA_DIR):
        path = data_dir / f"biomes_{seed}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"no biome dump at {path}. Run "
                f"`python tools/extract_biomes.py --seed {seed}` first."
            )
        z = np.load(path)
        self.biomes: np.ndarray = z["biomes"]
        self.origin_cell = tuple(int(v) for v in z["origin_cell"])

    def biome_at(self, cell_x: int, cell_z: int) -> int:
        ox, oz = self.origin_cell
        i = cell_z - oz
        j = cell_x - ox
        h, w = self.biomes.shape
        if not (0 <= i < h and 0 <= j < w):
            return UNKNOWN_BIOME
        return int(self.biomes[i, j])

    def get_grid(self, cell_x: int, cell_z: int, radius_cells: int) -> np.ndarray:
        """Return a (2r+1, 2r+1) window, padding out-of-bounds with -1."""
        ox, oz = self.origin_cell
        i0 = cell_z - oz - radius_cells
        j0 = cell_x - ox - radius_cells
        size = 2 * radius_cells + 1
        out = np.full((size, size), UNKNOWN_BIOME, dtype=np.int16)
        h, w = self.biomes.shape
        # Clamp source/dest rectangles to the dump's bounds.
        si0, sj0 = max(0, i0), max(0, j0)
        si1, sj1 = min(h, i0 + size), min(w, j0 + size)
        if si1 <= si0 or sj1 <= sj0:
            return out
        di0, dj0 = si0 - i0, sj0 - j0
        out[di0:di0 + (si1 - si0), dj0:dj0 + (sj1 - sj0)] = self.biomes[si0:si1, sj0:sj1]
        return out
