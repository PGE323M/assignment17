"""Wells in a two-dimensional, single-phase reservoir."""
import numpy as np
import scipy.sparse
from assignment15 import TwoDimReservoir as ReservoirBase


class TwoDimReservoir(ReservoirBase):
    """Extend the supplied reservoir with rate and bottomhole-pressure wells."""

    def compute_well_index_locations(self, well_type='rate'):
        """Return one x-fastest block index per well, preserving input order."""
        raise NotImplementedError("Implement compute_well_index_locations")

    def compute_productivity_index(self, well_type='bhp'):
        """Return unconverted isotropic Peaceman productivity indices."""
        raise NotImplementedError("Implement compute_productivity_index")

    def fill_matrices(self):
        """Add conservative well contributions to supplied sparse matrices."""
        raise NotImplementedError("Implement fill_matrices")
