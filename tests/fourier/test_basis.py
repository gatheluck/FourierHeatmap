import pathlib

import pytest
import torch
import torchvision

from fhmap.fourier import basis


class TestSpectrumToBasis:
    @pytest.fixture
    def sample_spectrum(self):
        B, H, W = 16, 32, 17
        return torch.rand(B, H, W)

    def test__output_size(self, sample_spectrum):
        B, H, W = sample_spectrum.size()
        assert basis.spectrum_to_basis(sample_spectrum, False).size() == torch.Size(
            [B, H, H]
        )
        assert basis.spectrum_to_basis(sample_spectrum, True).size() == torch.Size(
            [B, H, H]
        )

    def test__norm(self, sample_spectrum):
        B, _, _ = sample_spectrum.size()
        ones = torch.ones(B, dtype=torch.float)
        assert not torch.allclose(
            basis.spectrum_to_basis(sample_spectrum, False).norm(dim=(-2, -1)), ones
        )
        assert torch.allclose(
            basis.spectrum_to_basis(sample_spectrum, True).norm(dim=(-2, -1)), ones
        )

    def test__consistency(self, sample_spectrum):
        """Check if FFT is applied to each spectrum independently."""
        B, _, _ = sample_spectrum.size()
        for b in range(B):
            assert basis.spectrum_to_basis(sample_spectrum)[b][None, :, :].allclose(
                basis.spectrum_to_basis(sample_spectrum[b][None, :, :])
            )


class TestBasisSpectrum:
    @pytest.fixture
    def sample_sizes(self):
        H, W = 32, 17
        return H, W

    def test__output_size(self, sample_sizes):
        H, W = sample_sizes
        assert basis.get_basis_spectrum(H, W).size() == torch.Size([H * W, H, W])

    def test__visualize_basis(self, logdir):
        logdir.mkdir(exist_ok=True, parents=True)

        # high center
        savepath = pathlib.Path(logdir) / "test_visualized_basis_high_center.png"
        if savepath.exists():
            savepath.unlink()

        H, W = 32, 17
        fourier_basis = basis.spectrum_to_basis(basis.get_basis_spectrum(H, W, False))
        torchvision.utils.save_image(fourier_basis[:, None, :, :], savepath, nrow=W)

        # low center
        savepath = pathlib.Path(logdir) / "test_visualized_basis_low_center.png"
        if savepath.exists():
            savepath.unlink()

        H, W = 32, 17
        fourier_basis = basis.spectrum_to_basis(basis.get_basis_spectrum(H, W, True))
        torchvision.utils.save_image(fourier_basis[:, None, :, :], savepath, nrow=W)
