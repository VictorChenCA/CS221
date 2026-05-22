"""Biome generators: `(cell_x, cell_z) -> biome_id` callables.

Generators are plain callables — the signature matches `BiomeFn` in
oracle.py, so they slot into oracle.plan(biome_at=...) and into the
materializer (tools/extract_biomes.py) without adapter code. Tests use
`{(cx, cz): id}.get` as the generator; no class hierarchy needed.

Only one real generator here: `cubiomes_gen(seed)`, backed by the
cubiomes C library loaded via ctypes. One-time setup:

    git clone https://github.com/Cubitect/cubiomes tools/cubiomes
    cd tools/cubiomes
    make CFLAGS='-O3 -fPIC -Wall'
    cc -shared -o libcubiomes.so *.o    # macOS: -dynamiclib -o libcubiomes.dylib

The shim assumes cubiomes' public API (generator.h / biomes.h). If the
upstream API drifts, the constants below are the only thing to touch.
"""

import ctypes
import sys
from pathlib import Path
from typing import Callable

_CUBIOMES_DIR = Path(__file__).resolve().parent.parent / "tools" / "cubiomes"
_LIB_NAME = "libcubiomes.dylib" if sys.platform == "darwin" else "libcubiomes.so"
CUBIOMES_LIB = _CUBIOMES_DIR / _LIB_NAME

# From cubiomes/biome_const.h (1.20.x version constant). Update if the
# upstream enum changes.
MC_1_20 = 0x12100
DIM_OVERWORLD = 0
SCALE_4 = 4  # one biome per 4-block cell; matches our MDP cell size

# Measured sizeof(Generator) for cubiomes 1.20.x is 27,592 bytes
# (LayerStack + BiomeNoise + NetherNoise + EndNoise). Oversize to be
# safe across upstream bumps; this is the stack of one Python object,
# so the ~5 KB overhead is irrelevant.
_GENERATOR_BYTES = 32768


def cubiomes_gen(seed: int) -> Callable[[int, int], int]:
    """Return a (cell_x, cell_z) -> biome_id callable for `seed`.

    Per-call cost is one ctypes hop into cubiomes; on the order of a
    microsecond per cell, so a full ±256 cell dump is ~0.2 s.
    """
    if not CUBIOMES_LIB.exists():
        raise FileNotFoundError(
            f"cubiomes shared lib missing at {CUBIOMES_LIB}. "
            f"See mdp/biomegen.py docstring for build steps."
        )
    lib = ctypes.CDLL(str(CUBIOMES_LIB))

    lib.setupGenerator.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    lib.setupGenerator.restype = None
    lib.applySeed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    lib.applySeed.restype = None
    lib.getBiomeAt.argtypes = [ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.getBiomeAt.restype = ctypes.c_int

    g = (ctypes.c_byte * _GENERATOR_BYTES)()
    g_ptr = ctypes.cast(g, ctypes.c_void_p)
    lib.setupGenerator(g_ptr, MC_1_20, 0)
    lib.applySeed(g_ptr, DIM_OVERWORLD, seed & 0xFFFFFFFFFFFFFFFF)

    def gen(cell_x: int, cell_z: int) -> int:
        # At scale=4, x/z args are in 4-block units (= our cells). Y is
        # the surface-ish height; biomes are 3D in 1.18+ but we only
        # ever locomote on the surface.
        return lib.getBiomeAt(g_ptr, SCALE_4, cell_x, 63, cell_z)

    return gen
