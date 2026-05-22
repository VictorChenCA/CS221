"""Materializer roundtrip test using a dict-backed generator."""

import numpy as np

from tools.extract_biomes import materialize


def test_materialize_roundtrip(tmp_path):
    m = {(cx, cz): (cx + cz) % 7
         for cx in range(-3, 4) for cz in range(-3, 4)}
    out = tmp_path / "biomes_42.npz"
    materialize(seed=42, radius_cells=3,
                gen=lambda cx, cz: m[(cx, cz)], out_path=out)

    z = np.load(out)
    assert z["biomes"].shape == (7, 7)
    assert tuple(z["origin_cell"].tolist()) == (-3, -3)
    assert int(z["seed"]) == 42
    assert int(z["cell_size_blocks"]) == 4
    assert int(z["format_version"]) == 1
    assert int(z["biomes"][3, 3]) == 0   # cell (0, 0) = (0+0) % 7
    assert int(z["biomes"][6, 0]) == (-3 + 3) % 7  # cell (-3, 3)
