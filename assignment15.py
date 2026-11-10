"""Two-dimensional, single-phase reservoir model on a Cartesian grid."""
import numpy as np
import scipy.sparse
from assignment13 import OneDimReservoir


class TwoDimReservoir(OneDimReservoir):
    """Reuse A13 initialization, time integration, and solution access."""

    def parse_inputs(self):
        """Provided: normalize cell properties and build x-fastest flat arrays."""
        numerical = self.inputs['numerical']
        grids = numerical['number of grids']
        for axis in ('x', 'y'):
            n = grids[axis]
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
                raise ValueError('grid counts must be positive integers')
        self.Nx, self.Ny = int(grids['x']), int(grids['y'])
        self.N = self.Nx * self.Ny
        rock, water = self.inputs['reservoir'], self.inputs['fluid']['water']
        self.viscosity = self.positive_scalar(water['viscosity'], 'viscosity')
        self.formation_volume_factor = self.positive_scalar(
            water['formation volume factor'], 'formation volume factor')
        self.compressibility = self.positive_scalar(water['compressibility'], 'compressibility')
        self.delta_t = self.positive_scalar(numerical['time step'], 'time step')
        self.conversion_factor = self.positive_scalar(
            self.inputs.get('conversion factor', 1.0), 'conversion factor')
        self.permeability = self.cell_values(rock['permeability'], 'permeability', allow_zero=True)
        self.porosity = self.cell_values(rock['porosity'], 'porosity')
        self.depth = self.cell_values(rock['depth'], 'depth')
        dx, dy = np.meshgrid(self.assign_delta_array('delta x'),
                             self.assign_delta_array('delta y'))
        self.delta_x, self.delta_y = dx.ravel(), dy.ravel()
        self.area = self.delta_x * self.delta_y  # plan-view area, not A13 face area

    def assign_delta_array(self, name='delta x'):
        """Provided: one positive width per column (x) or row (y)."""
        axis = name[-1]
        count = self.Nx if axis == 'x' else self.Ny
        extent = self.inputs['reservoir']['length' if axis == 'x' else 'height']
        values = self.inputs['numerical'].get(name, self.positive_scalar(extent, axis) / count)
        a = np.asarray(values)
        if a.ndim == 0:
            a = np.full(count, self.positive_scalar(values, name))
        if (a.shape != (count,) or a.dtype.kind not in 'iuf' or
                not np.all(np.isfinite(a)) or np.any(a <= 0)):
            raise ValueError(f'{name} must have {count} positive finite widths')
        return a.astype(float, copy=True)

    def cell_values(self, value, name, allow_zero=False):
        """Provided: scalar, flat length-N vector, or file containing N values."""
        if isinstance(value, str):
            value = np.loadtxt(value)
        a = np.asarray(value)
        if a.dtype.kind not in 'iuf' or a.ndim > 1:
            raise ValueError(f'{name} must be a scalar or flat length-N vector')
        if a.ndim == 0:
            a = np.full(self.N, float(a))
        if (a.shape != (self.N,) or not np.all(np.isfinite(a)) or
                np.any(a < 0 if allow_zero else a <= 0)):
            raise ValueError(f'invalid {name} values')
        return a.astype(float, copy=True)

    def boundary_transmissibility(self, i, axis):
        """Provided: unconverted cell-center-to-boundary-face conductance."""
        width = self.delta_x[i] if axis == 'x' else self.delta_y[i]
        face_area = self.depth[i] * (self.delta_y[i] if axis == 'x' else self.delta_x[i])
        return 2 * self.permeability[i] * face_area / (
            width * self.viscosity * self.formation_volume_factor)

    def compute_transmissibility(self, i, j):
        """Return unconverted conductance for face neighbors; i == j uses x."""
        ri, ci = divmod(i, self.Nx)
        rj, cj = divmod(j, self.Nx)
        if i == j or (ri == rj and abs(ci - cj) == 1):
            width, transverse = self.delta_x, self.delta_y
        elif ci == cj and abs(ri - rj) == 1:
            width, transverse = self.delta_y, self.delta_x
        else:
            raise ValueError('cells must share a face')
        if self.permeability[i] == 0 or self.permeability[j] == 0:
            return 0.0
        resistance = (width[i] / (2 * self.permeability[i] * transverse[i] * self.depth[i]) +
                      width[j] / (2 * self.permeability[j] * transverse[j] * self.depth[j]))
        return 1 / (self.viscosity * self.formation_volume_factor * resistance)

    def compute_accumulation(self, i):
        """Return cell storage without the transmissibility conversion factor."""
        volume = self.delta_x[i] * self.delta_y[i] * self.depth[i]
        return volume * self.porosity[i] * self.compressibility / self.formation_volume_factor

    def fill_matrices(self):
        """Assemble CSR T/B and source Q, counting each internal face once."""
        matrix = scipy.sparse.lil_matrix((self.N, self.N), dtype=float)
        source = np.zeros(self.N)
        factor = self.conversion_factor
        for i in range(self.N):
            row, col = divmod(i, self.Nx)
            neighbors = []
            if col + 1 < self.Nx:
                neighbors.append(i + 1)
            if row + 1 < self.Ny:
                neighbors.append(i + self.Nx)
            for j in neighbors:
                t = factor * self.compute_transmissibility(i, j)
                matrix[i, i] += t
                matrix[j, j] += t
                matrix[i, j] -= t
                matrix[j, i] -= t
            faces = []
            if col == 0:
                faces.append(('left', 'x'))
            if col == self.Nx - 1:
                faces.append(('right', 'x'))
            if row == 0:
                faces.append(('bottom', 'y'))
            if row == self.Ny - 1:
                faces.append(('top', 'y'))
            for side, axis in faces:
                bc = self.inputs['boundary conditions'][side]
                if bc['type'] == 'prescribed pressure':
                    t = factor * self.boundary_transmissibility(i, axis)
                    matrix[i, i] += t
                    source[i] += t * bc['value']
                elif bc['type'] != 'prescribed flux' or bc['value'] != 0:
                    raise ValueError('only prescribed pressure and zero prescribed flux are supported')
        self.T = matrix.tocsr()
        self.B = scipy.sparse.diags(
            [self.compute_accumulation(i) for i in range(self.N)], format='csr')
        self.Q = source
