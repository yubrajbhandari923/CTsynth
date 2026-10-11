"""Differentiable spectral CT forward model. Spec: docs/physics/forward-model-spec.md."""

from ctsynth.physics.forward import SpectralForwardModel
from ctsynth.physics.materials import MaterialBasis
from ctsynth.physics.projector import ParallelBeamProjector2D
from ctsynth.physics.spectrum import Spectrum

__all__ = ["MaterialBasis", "ParallelBeamProjector2D", "SpectralForwardModel", "Spectrum"]
