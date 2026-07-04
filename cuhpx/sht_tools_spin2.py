# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for spin-weighted spherical harmonic precomputation."""

import math

import numpy as np


def spin2_legpoly(mmax, lmax, x, spin=2, norm="ortho", inverse=False, csphase=True):
    """Precompute theta-dependent spin-weighted spherical harmonic factors."""

    s = int(spin)
    theta = np.arccos(x)
    out = np.zeros((mmax, lmax, len(x)), dtype=np.float64)
    norm_factor = 1.0 if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1.0 / norm_factor if inverse else norm_factor

    c = np.cos(theta / 2.0)
    st = np.sin(theta / 2.0)
    mp = -s
    for m in range(mmax):
        for ell in range(max(m, abs(s)), lmax):
            log_pref = 0.5 * (
                math.lgamma(ell + m + 1)
                + math.lgamma(ell - m + 1)
                + math.lgamma(ell + mp + 1)
                + math.lgamma(ell - mp + 1)
            )
            kmin = max(0, m - mp)
            kmax = min(ell + m, ell - mp)
            vals = np.zeros_like(theta, dtype=np.float64)
            for k in range(kmin, kmax + 1):
                denom = (
                    math.lgamma(ell + m - k + 1)
                    + math.lgamma(k + 1)
                    + math.lgamma(mp - m + k + 1)
                    + math.lgamma(ell - mp - k + 1)
                )
                sign = -1.0 if ((k - m + mp) & 1) else 1.0
                vals += sign * math.exp(log_pref - denom) * c ** (2 * ell + m - mp - 2 * k) * st ** (mp - m + 2 * k)
            scale = ((-1.0) ** s) * np.sqrt((2 * ell + 1) / (4 * np.pi)) * norm_factor
            out[m, ell, :] = scale * vals

            # Spin-weighted harmonics are normalized Wigner-d matrix elements
            # times sqrt((2l + 1) / 4pi), and Wigner-d entries are bounded by
            # one. The direct factorial sum can suffer catastrophic
            # cancellation for high l/m near HEALPix polar rings; clipping to
            # the analytic bound prevents invalid entries from poisoning
            # band-limited transforms through 0 * inf -> nan.
            bound = abs(scale)
            out[m, ell, :] = np.nan_to_num(out[m, ell, :], nan=0.0, posinf=bound, neginf=-bound)
            out[m, ell, :] = np.clip(out[m, ell, :], -bound, bound)

    if csphase:
        for m in range(1, mmax, 2):
            out[m] *= 1
    return out


def _precompute_spin2_legpoly(mmax, lmax, t, spin=2, norm="ortho", inverse=False, csphase=True):
    return spin2_legpoly(mmax, lmax, np.cos(t), spin=spin, norm=norm, inverse=inverse, csphase=csphase)
