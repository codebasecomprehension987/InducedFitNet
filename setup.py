"""
Build script for the Cython rotamer sampler extension.

Usage:
    cd inducedfitnet/cython_ext
    python setup.py build_ext --inplace

Or from the repo root:
    python inducedfitnet/cython_ext/setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


ext_file = "rotamer.pyx" if USE_CYTHON else "rotamer.c"

extensions = [
    Extension(
        name="inducedfitnet.cython_ext.rotamer",
        sources=[ext_file],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
            "-fopenmp",      # OpenMP for potential future parallelism
        ],
        extra_link_args=["-fopenmp"],
        language="c",
    )
]

if USE_CYTHON:
    extensions = cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "profile": False,
        },
        annotate=False,
    )

setup(
    name="rotamer",
    ext_modules=extensions,
)
