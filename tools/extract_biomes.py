"""Materializer: dump a biome map for one seed to data/biomes_<seed>.npz.

Iterates a generator callable over a square cell window and writes the
result as a dense int16 array. The generator is anything that satisfies
`(cell_x, cell_z) -> int`; production uses `agent.biomegen.cubiomes_gen`,
tests use a plain dict's `.get`.

File format (numpy savez_compressed):
  biomes:           int16[H, W]
  origin_cell:      int32[2]    (cellX, cellZ of biomes[0, 0])
  seed:             int64
  cell_size_blocks: int8        (= 4, Minecraft 1.18+ quart size)
  format_version:   int8

Usage:
    python tools/extract_biomes.py --seed 1111
    python tools/extract_biomes.py --seed 1111 --radius-blocks 2048
"""

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from mdp.world import CELL_BLOCKS, DATA_DIR

DEFAULT_RADIUS_BLOCKS = 1024


def materialize(seed: int, radius_cells: int, gen: Callable[[int, int], int],
                out_path: Path) -> None:
    size = 2 * radius_cells + 1
    biomes = np.empty((size, size), dtype=np.int16)
    for i, cz in enumerate(range(-radius_cells, radius_cells + 1)):
        for j, cx in enumerate(range(-radius_cells, radius_cells + 1)):
            biomes[i, j] = gen(cx, cz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        biomes=biomes,
        origin_cell=np.array([-radius_cells, -radius_cells], dtype=np.int32),
        seed=np.int64(seed),
        cell_size_blocks=np.int8(CELL_BLOCKS),
        format_version=np.int8(1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--radius-blocks", type=int, default=DEFAULT_RADIUS_BLOCKS)
    args = ap.parse_args()

    from mdp.biomegen import cubiomes_gen
    gen = cubiomes_gen(args.seed)
    radius_cells = args.radius_blocks // CELL_BLOCKS
    out = DATA_DIR / f"biomes_{args.seed}.npz"
    materialize(args.seed, radius_cells, gen, out)
    print(f"wrote {out} ({2 * radius_cells + 1}^2 cells)")


if __name__ == "__main__":
    main()
