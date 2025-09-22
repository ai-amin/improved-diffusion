"""Improved Diffusion package.

Includes original image diffusion utilities plus added diffraction dataset support.
"""

from .image_datasets import (
	load_data,
	DiffractionDataset,
	load_diffraction_data,
)

__all__ = [
	"load_data",
	"DiffractionDataset",
	"load_diffraction_data",
]
