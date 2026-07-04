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

import numpy as np
import torch
import torch.nn as nn

from cuhpx.sht_tools import healpix_weights
from cuhpx.sht_tools_spin2 import _precompute_spin2_legpoly

from . import cuhpx_fft


class SHTCUDA_spin2(nn.Module):
    """Forward spin-2 HEALPix SHT from ``g = g1 + i*g2`` to E/B alms."""

    def __init__(self, nside, lmax=None, mmax=None, quad_weights="ring", norm="ortho", csphase=True):
        super().__init__()
        self.nside = nside
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside
        self.quad_weights = quad_weights
        self.lmax = lmax or self.nlat
        self.mmax = mmax or (self.nlon // 2 + 1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is not available. This class requires a GPU.")
        self.device = torch.device("cuda")

        cost, w = healpix_weights(nside, self.quad_weights)
        tq = np.flip(np.arccos(cost))
        weights = torch.from_numpy(w).float().to(self.device)
        p2 = (
            torch.from_numpy(
                _precompute_spin2_legpoly(self.mmax, self.lmax, tq, spin=2, norm=self.norm, csphase=self.csphase)
            )
            .float()
            .to(self.device)
        )
        pm2 = (
            torch.from_numpy(
                _precompute_spin2_legpoly(self.mmax, self.lmax, tq, spin=-2, norm=self.norm, csphase=self.csphase)
            )
            .float()
            .to(self.device)
        )
        self.register_buffer("w2", p2 * weights, persistent=False)
        self.register_buffer("wm2", pm2 * weights, persistent=False)
        self.register_buffer("w2_f64", self.w2.double(), persistent=False)
        self.register_buffer("wm2_f64", self.wm2.double(), persistent=False)

    def forward(self, g1, g2=None):
        if g2 is None:
            if not torch.is_complex(g1):
                raise ValueError("Pass either a complex tensor g or separate real tensors g1 and g2.")
            g1, g2 = g1.real, g1.imag
        if torch.is_complex(g1) or torch.is_complex(g2):
            raise ValueError("g1 and g2 must be real tensors.")
        if g1.shape != g2.shape:
            raise ValueError("g1 and g2 must have the same shape.")
        if g1.dim() == 1:
            q = cuhpx_fft.healpix_rfft_class(g1, self.mmax, self.nside).clone()
            u = cuhpx_fft.healpix_rfft_class(g2, self.mmax, self.nside).clone()
        else:
            q = cuhpx_fft.healpix_rfft_batch(g1, self.mmax, self.nside).clone()
            u = cuhpx_fft.healpix_rfft_batch(g2, self.mmax, self.nside).clone()
        gc = q + 1j * u
        gcc = q - 1j * u
        w2 = self.w2_f64 if g1.dtype == torch.float64 else self.w2
        wm2 = self.wm2_f64 if g1.dtype == torch.float64 else self.wm2
        a2 = torch.einsum("...km,mlk->...lm", gc, w2.to(gc.dtype))
        am2 = torch.einsum("...km,mlk->...lm", gcc, wm2.to(gcc.dtype))
        return (-0.5 * (a2 + am2)).contiguous(), (0.5j * (a2 - am2)).contiguous()


class iSHTCUDA_spin2(nn.Module):
    """Inverse spin-2 HEALPix SHT from E/B alms to ``g1, g2`` maps."""

    def __init__(self, nside, lmax=None, mmax=None, quad_weights="ring", norm="ortho", csphase=True):
        super().__init__()
        self.nside = nside
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside
        self.quad_weights = quad_weights
        self.lmax = lmax or self.nlat
        self.mmax = mmax or (self.nlon // 2 + 1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is not available. This class requires a GPU.")
        self.device = torch.device("cuda")
        cost, _ = healpix_weights(nside, "none")
        t = np.flip(np.arccos(cost))
        p2 = (
            torch.from_numpy(
                _precompute_spin2_legpoly(
                    self.mmax, self.lmax, t, spin=2, norm=self.norm, inverse=True, csphase=self.csphase
                )
            )
            .float()
            .to(self.device)
        )
        pm2 = (
            torch.from_numpy(
                _precompute_spin2_legpoly(
                    self.mmax, self.lmax, t, spin=-2, norm=self.norm, inverse=True, csphase=self.csphase
                )
            )
            .float()
            .to(self.device)
        )
        self.register_buffer("p2", p2, persistent=False)
        self.register_buffer("pm2", pm2, persistent=False)
        self.register_buffer("p2_f64", p2.double(), persistent=False)
        self.register_buffer("pm2_f64", pm2.double(), persistent=False)

    def forward(self, E, B=None):
        if B is None:
            if not torch.is_complex(E):
                raise ValueError("Pass either a complex tensor a=E+i*B or separate E and B tensors.")
            E, B = E.real, E.imag
        if E.shape != B.shape:
            raise ValueError("E and B must have the same shape.")
        dtype = E.real.dtype
        p2 = self.p2_f64 if dtype == torch.float64 else self.p2
        pm2 = self.pm2_f64 if dtype == torch.float64 else self.pm2
        a2 = -E - 1j * B
        am2 = -E + 1j * B
        ftm = torch.einsum("...lm,mlk->...km", a2, p2.to(a2.dtype))
        ftm_conj = torch.einsum("...lm,mlk->...km", am2, pm2.to(am2.dtype))
        q = 0.5 * (ftm + ftm_conj)
        u = (ftm - ftm_conj) / (2j)
        if q.dim() == 2:
            g1 = cuhpx_fft.healpix_irfft_class(q.contiguous(), self.mmax, self.nside).clone()
            g2 = cuhpx_fft.healpix_irfft_class(u.contiguous(), self.mmax, self.nside).clone()
            return g1, g2
        g1 = cuhpx_fft.healpix_irfft_batch(q.contiguous(), self.mmax, self.nside).clone()
        g2 = cuhpx_fft.healpix_irfft_batch(u.contiguous(), self.mmax, self.nside).clone()
        return g1, g2
