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
            E[l_idx, m_idx] = torch.randn((), dtype=complex_dtype, device=device)
            B[l_idx, m_idx] = torch.randn((), dtype=complex_dtype, device=device)

    g1, g2 = isht_cuda(E, B)
    E_back, B_back = sht_cuda(g1, g2)
    g1_back, g2_back = isht_cuda(E_back, B_back)

    rtol, atol = get_roundtrip_tol()
    assert torch.allclose(
        g1_back, g1, rtol=rtol, atol=atol
    ), f"Spin-2 CUDA g1 roundtrip failed: max diff = {(g1_back - g1).abs().max():.2e}"
    assert torch.allclose(
        g2_back, g2, rtol=rtol, atol=atol
    ), f"Spin-2 CUDA g2 roundtrip failed: max diff = {(g2_back - g2).abs().max():.2e}"
