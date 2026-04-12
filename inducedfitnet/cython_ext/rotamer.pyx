# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-compiled sidechain rotamer sampler.

rotamer_sample(backbone_coords, chi_angles) → updated chi_angles
Called 200× per generation during the reverse diffusion loop.

The function:
  1. Accepts backbone Cα coordinates and current χ-angle array.
  2. For each residue, samples a new χ₁ angle using a von Mises
     distribution centred on the current χ₁ with concentration κ.
  3. Updates χ₂–χ₄ with a simple backbone-dependent torsion prior.
  4. Returns the updated chi_angles array.

All computation is in C via typed memoryviews — no Python objects
are created inside the hot loop, keeping the GC entirely uninvolved.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, atan2, sqrt, exp, M_PI
from libc.stdlib cimport rand, RAND_MAX

cnp.import_array()

# Von Mises sampler (approximation via ratio-of-uniforms for κ > 1)
cdef inline double _vm_sample(double mu, double kappa) nogil:
    """
    Sample from von Mises distribution VM(mu, kappa).
    Uses the Best (1979) algorithm for kappa > 1, direct for kappa <= 1.
    """
    cdef double a, b, r, u1, u2, u3, z, f, c
    cdef double result

    if kappa <= 0.0:
        # Uniform on circle
        return 2.0 * M_PI * (<double>rand() / RAND_MAX) - M_PI

    a = 1.0 + sqrt(1.0 + 4.0 * kappa * kappa)
    b = (a - sqrt(2.0 * a)) / (2.0 * kappa)
    r = (1.0 + b * b) / (2.0 * b)

    while True:
        u1 = <double>rand() / RAND_MAX
        z  = cos(M_PI * u1)
        f  = (1.0 + r * z) / (r + z)
        c  = kappa * (r - f)

        u2 = <double>rand() / RAND_MAX
        if (c * (2.0 - c) - u2) > 0.0:
            u3 = <double>rand() / RAND_MAX
            result = mu + (1.0 if u3 - 0.5 < 0 else -1.0) * atan2(sin(atan2(sqrt(1.0 - f * f), f)), 1.0)
            # Wrap to (-π, π]
            while result > M_PI:
                result -= 2.0 * M_PI
            while result <= -M_PI:
                result += 2.0 * M_PI
            return result
        if (c - u2) >= 0.0:
            u3 = <double>rand() / RAND_MAX
            result = mu + (1.0 if u3 - 0.5 < 0 else -1.0) * atan2(sqrt(1.0 - f * f), f)
            while result > M_PI:
                result -= 2.0 * M_PI
            while result <= -M_PI:
                result += 2.0 * M_PI
            return result


def rotamer_sample(
    cnp.ndarray[cnp.float32_t, ndim=2] backbone_coords,   # (L, 3)  Cα positions
    cnp.ndarray[cnp.float32_t, ndim=2] chi_angles,         # (L, 4)  current χ angles
    double kappa = 5.0,
    double perturb_scale = 0.3,
):
    """
    Sample new sidechain χ-angles conditioned on backbone Cα positions.

    Args:
        backbone_coords : (L, 3) float32  Cα coordinates
        chi_angles      : (L, 4) float32  current χ₁–χ₄ angles (radians)
        kappa           : von Mises concentration (higher = tighter sampling)
        perturb_scale   : std of Gaussian perturbation applied to χ₂–χ₄

    Returns:
        new_chi : (L, 4) float32  updated χ angles
    """
    cdef int L = backbone_coords.shape[0]
    cdef int l, k
    cdef double chi1_new, chi_k, noise

    # Output array (typed memoryview for GC-free access)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] new_chi = np.empty_like(chi_angles)

    # Typed memoryviews — all access via C pointers, no Python boxing
    cdef cnp.float32_t[:, :] bb   = backbone_coords
    cdef cnp.float32_t[:, :] chi  = chi_angles
    cdef cnp.float32_t[:, :] out  = new_chi

    with nogil:
        for l in range(L):
            # χ₁: sample from von Mises centred on current χ₁
            chi1_new = _vm_sample(<double>chi[l, 0], kappa)
            out[l, 0] = <cnp.float32_t>chi1_new

            # χ₂–χ₄: Gaussian perturbation around current values
            for k in range(1, 4):
                # Simple uniform noise ∈ [-perturb_scale, +perturb_scale]
                # (real impl uses Dunbrack backbone-dependent library)
                noise = perturb_scale * (2.0 * (<double>rand() / RAND_MAX) - 1.0)
                chi_k = <double>chi[l, k] + noise
                # Wrap to (-π, π]
                while chi_k > M_PI:
                    chi_k -= 2.0 * M_PI
                while chi_k <= -M_PI:
                    chi_k += 2.0 * M_PI
                out[l, k] = <cnp.float32_t>chi_k

    return new_chi
