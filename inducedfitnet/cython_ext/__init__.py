"""
Cython-compiled extensions for InducedFitNet.

Import the compiled rotamer sampler; fall back gracefully to the
pure-Python version if the extension has not been built yet.
"""

try:
    from inducedfitnet.cython_ext.rotamer import rotamer_sample  # type: ignore
    _CYTHON_AVAILABLE = True
except ImportError:
    from inducedfitnet.diffusion.ode_integrator import _python_rotamer_fallback as rotamer_sample  # type: ignore
    _CYTHON_AVAILABLE = False

__all__ = ["rotamer_sample", "_CYTHON_AVAILABLE"]
