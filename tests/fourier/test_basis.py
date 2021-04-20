import pathlib

import pytest
import torch
import torchvision

from fhmap.fourier import basis


class TestSpectrumToBasis:
    @pytest.fixture
    def sample_spectrum(self):
        H, W = 32, 17
        return torch.rand(H, W)

    def test__output_size(self, sample_spectrum):
        H, W = sample_spectrum.size()
        assert basis.spectrum_to_basis(sample_spectrum, False).size() == torch.Size(
            [H, H]
        )
        assert basis.spectrum_to_basis(sample_spectrum, True).size() == torch.Size(
            [H, H]
        )

    def test__norm(self, sample_spectrum):
        one = torch.ones(1, dtype=torch.float)
        assert not torch.allclose(
            basis.spectrum_to_basis(sample_spectrum, False).norm(dim=(-2, -1)), one
        )
        assert torch.allclose(
            basis.spectrum_to_basis(sample_spectrum, True).norm(dim=(-2, -1)), one
        )


class TestBasisSpectrum:
    @pytest.fixture
    def sample_sizes(self):
        H, W = 32, 17
        return H, W

    def test__output_size(self, sample_sizes):
        H, W = sample_sizes
        for b in basis.get_spectrum(H, W):
            assert b.size() == torch.Size([H, W])
