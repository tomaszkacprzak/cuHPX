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

"""Tests for CUDA-accelerated spin-2 SHT/iSHT implementations."""

import pytest
import torch
from conftest import get_roundtrip_tol

from cuhpx.hpx_sht_spin2 import SHTCUDA_spin2, iSHTCUDA_spin2


@pytest.mark.cuda
def test_shtcuda_spin2_ishtcuda_spin2_roundtrip(device, nside, lmax, mmax, complex_dtype):
    """Test SHTCUDA_spin2 + iSHTCUDA_spin2 round-trip on bandlimited spin-2 maps."""
    sht_cuda = SHTCUDA_spin2(nside, lmax=lmax, mmax=mmax, quad_weights="ring")
    isht_cuda = iSHTCUDA_spin2(nside, lmax=lmax, mmax=mmax)

    # Create strictly band-limited E/B coefficients. Spin-2 harmonics start at l=2,
    # and using half of lmax leaves a safety margin for the HEALPix quadrature.
    torch.manual_seed(42)
    E = torch.zeros((lmax, mmax), dtype=complex_dtype, device=device)
    B = torch.zeros((lmax, mmax), dtype=complex_dtype, device=device)
    max_l = min(lmax // 2, nside // 2)
    for l_idx in range(2, max_l):
        for m_idx in range(min(l_idx + 1, mmax)):
            if m_idx == 0:
                # Real-valued g1/g2 maps can only encode real m=0 coefficients;
                # the imaginary m=0 component has no conjugate negative-m partner
                # and is discarded by the real FFT inverse.
                E[l_idx, m_idx] = torch.randn((), dtype=E.real.dtype, device=device)
                B[l_idx, m_idx] = torch.randn((), dtype=B.real.dtype, device=device)
            else:
                E[l_idx, m_idx] = torch.randn((), dtype=complex_dtype, device=device)
                B[l_idx, m_idx] = torch.randn((), dtype=complex_dtype, device=device)

    valid_coeffs = torch.zeros((lmax, mmax), dtype=torch.bool, device=device)
    for l_idx in range(2, max_l):
        valid_coeffs[l_idx, : min(l_idx + 1, mmax)] = True

    g1, g2 = isht_cuda(E, B)
    E_back, B_back = sht_cuda(g1, g2)

    # HEALPix quadrature is approximate and can leave small out-of-band residuals
    # in coefficients that were intentionally zero in the band-limited input.
    # Do not feed those residual modes back into iSHT for the map round-trip check.
    E_back_bandlimited = torch.where(valid_coeffs, E_back, torch.zeros_like(E_back))
    B_back_bandlimited = torch.where(valid_coeffs, B_back, torch.zeros_like(B_back))
    g1_back, g2_back = isht_cuda(E_back_bandlimited, B_back_bandlimited)

    import numpy as np
    with np.printoptions(linewidth=200, threshold=10, precision=3, suppress=True, formatter={"float_kind": lambda x: f"{x: .6e}",  "complex_kind": lambda z: f"{z.real:+.6e}{z.imag:+.6e}j"}):

        E_np = E.detach().cpu().numpy().ravel()
        E_back_np = E_back.detach().cpu().numpy().ravel()
        B_np = B.detach().cpu().numpy().ravel()
        B_back_np = B_back.detach().cpu().numpy().ravel()
        g1_np = g1.detach().cpu().numpy().ravel()
        g2_np = g2.detach().cpu().numpy().ravel()
        g1_back_np = g1_back.detach().cpu().numpy().ravel()
        g2_back_np = g2_back.detach().cpu().numpy().ravel()

        print('E shape', E_np.shape)
        print('E_back shape', E_back_np.shape)
        print('B shape', B_np.shape)
        print('B_back shape', B_back_np.shape)
        print('g1 shape', g1_np.shape)
        print('g2 shape', g2_np.shape)
        print('g1_back shape', g1_back_np.shape)
        print('g2_back shape', g2_back_np.shape)

        nvals = 100
        print('g1     ', g1_np[:nvals])
        print('g2     ', g2_np[:nvals])
        print('g1_back', g1_back_np[:nvals])
        print('g2_back', g2_back_np[:nvals])

        print('E      ', E_np[E_np != 0][:nvals])
        print('E_back ', E_back_np[E_back_np != 0][:nvals])
        print('B      ', B_np[B_np != 0][:nvals])
        print('B_back ', B_back_np[B_back_np != 0][:nvals])

    rtol, atol = get_roundtrip_tol()
    assert torch.allclose(
        g1_back, g1, rtol=rtol, atol=atol
    ), f"Spin-2 CUDA g1 roundtrip failed: max diff = {(g1_back - g1).abs().max():.2e}"
    assert torch.allclose(
        g2_back, g2, rtol=rtol, atol=atol
    ), f"Spin-2 CUDA g2 roundtrip failed: max diff = {(g2_back - g2).abs().max():.2e}"
    assert torch.allclose(
        E_back, E, rtol=rtol, atol=atol
    ), f"Spin-2 CUDA E roundtrip failed: max diff = {(E_back - E).abs().max():.2e}"
    assert torch.allclose(
        B_back, B, rtol=rtol, atol=atol
    ), f"Spin-2 CUDA B roundtrip failed: max diff = {(B_back - B).abs().max():.2e}"


@pytest.mark.cuda
def test_ishtcuda_spin2_matches_healpy_alm2map_spin_from_synfast_eb_maps(device):
    """Compare cuhpx and healpy E/B-to-shear spin-2 synthesis on synfast maps."""
    hp = pytest.importorskip("healpy")
    import numpy as np

    nside = 16
    lmax = 3 * nside - 1
    mmax = lmax
    lmax_hp = lmax - 1
    mmax_hp = mmax - 1

    ell = np.arange(lmax, dtype=np.float64)
    cl = np.zeros(lmax, dtype=np.float64)
    cl[2:] = 1.0e-4 / (ell[2:] * (ell[2:] + 1.0))

    np.random.seed(42)
    e_map = hp.synfast(cl, nside=nside, lmax=lmax_hp, new=True, pol=False, verbose=False)
    b_map = hp.synfast(0.5 * cl, nside=nside, lmax=lmax_hp, new=True, pol=False, verbose=False)
    e_alm_hp = hp.map2alm(e_map, lmax=lmax_hp, mmax=mmax_hp, iter=0)
    b_alm_hp = hp.map2alm(b_map, lmax=lmax_hp, mmax=mmax_hp, iter=0)

    e_alm = np.zeros((lmax, mmax), dtype=np.complex128)
    b_alm = np.zeros((lmax, mmax), dtype=np.complex128)
    for ell_idx in range(lmax):
        for m_idx in range(min(ell_idx, mmax_hp) + 1):
            hp_idx = hp.Alm.getidx(lmax_hp, ell_idx, m_idx)
            e_alm[ell_idx, m_idx] = e_alm_hp[hp_idx]
            b_alm[ell_idx, m_idx] = b_alm_hp[hp_idx]

    isht_cuda = iSHTCUDA_spin2(nside, lmax=lmax, mmax=mmax)
    e_alm_cuda = torch.from_numpy(e_alm).to(device=device)
    b_alm_cuda = torch.from_numpy(b_alm).to(device=device)
    g1_cuhpx, g2_cuhpx = isht_cuda(e_alm_cuda, b_alm_cuda)

    g1_healpy, g2_healpy = hp.alm2map_spin([e_alm_hp, b_alm_hp], nside, spin=2, lmax=lmax_hp, mmax=mmax_hp)
    g1_healpy = torch.from_numpy(g1_healpy).to(device=device, dtype=g1_cuhpx.dtype)
    g2_healpy = torch.from_numpy(g2_healpy).to(device=device, dtype=g2_cuhpx.dtype)

    assert torch.allclose(g1_cuhpx, g1_healpy, rtol=2e-3, atol=2e-5), (
        f"cuhpx/healpy g1 spin-2 synthesis mismatch: max diff = {(g1_cuhpx - g1_healpy).abs().max():.2e}"
    )
    assert torch.allclose(g2_cuhpx, g2_healpy, rtol=2e-3, atol=2e-5), (
        f"cuhpx/healpy g2 spin-2 synthesis mismatch: max diff = {(g2_cuhpx - g2_healpy).abs().max():.2e}"
    )
