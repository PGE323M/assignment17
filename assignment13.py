import numpy as np
import yaml
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

class OneDimReservoir():

    def __init__(self, inputs):
        '''
            Class for solving one-dimensional reservoir problems with
            a one-dimensional cell-centered finite-volume balance.
        '''

        #stores input dictionary as class attribute, either read from a yaml file
        #or directly from a Python dictonary
        if isinstance(inputs, str):
            with open(inputs) as f:
                self.inputs = yaml.safe_load(f)
        else:
            self.inputs = inputs

        self.parse_inputs()

        #applies the initial reservoir pressues to self.p
        self.apply_initial_conditions()

        #calls fill matrix method (must be completely implemented to work)
        self.fill_matrices()

        #create an empty list for storing data if plots are requested
        if 'plots' in self.inputs:
            self.p_plot = []

        return

    def parse_inputs(self):
        """Normalize block data once; supplied infrastructure."""
        numerical = self.inputs['numerical']
        n = numerical['number of grids']['x']
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError('number of grids must be an integer of at least two')
        self.N = self.Nx = int(n)
        rock, water = self.inputs['reservoir'], self.inputs['fluid']['water']
        self.permeability = self.block_values(rock['permeability'], 'permeability')
        self.area = self.block_values(rock['cross sectional area'], 'cross sectional area')
        self.porosity = self.block_values(rock['porosity'], 'porosity')
        dx = numerical.get('delta x', rock['length'] / self.N)
        self.delta_x = self.block_values(dx, 'delta x')
        self.viscosity = self.positive_scalar(water['viscosity'], 'viscosity')
        self.formation_volume_factor = self.positive_scalar(
            water['formation volume factor'], 'formation volume factor')
        self.compressibility = self.positive_scalar(water['compressibility'], 'compressibility')
        self.conversion_factor = self.positive_scalar(
            self.inputs.get('conversion factor', 1.0), 'conversion factor')
        self.delta_t = self.positive_scalar(numerical['time step'], 'time step')

    @staticmethod
    def positive_scalar(value, name):
        """Require a finite positive numeric scalar (not a boolean)."""
        a = np.asarray(value)
        if (a.ndim != 0 or a.dtype.kind not in 'iuf' or
                not np.isfinite(a) or a <= 0):
            raise ValueError(f'{name} must be a finite positive number')
        return float(a)

    def block_values(self, value, name):
        """Broadcast a scalar or copy a length-N numeric vector."""
        a = np.asarray(value)
        has_boolean = any(isinstance(v, (bool, np.bool_))
                          for v in np.asarray(value, dtype=object).flat)
        if has_boolean or a.dtype.kind not in 'iuf' or a.ndim > 1:
            raise ValueError(f'{name} must be a scalar or a length-N numeric vector')
        if a.ndim == 0:
            a = np.full(self.N, float(a))
        elif a.shape != (self.N,):
            raise ValueError(f'{name} must contain exactly N values')
        if not np.all(np.isfinite(a)) or np.any(a <= 0):
            raise ValueError(f'{name} values must be finite and positive')
        return np.array(a, dtype=float, copy=True)

    def compute_transmissibility(self, i, j):
        """Unconverted two-half-block conductance; i == j is a full block."""
        resistance = (self.delta_x[i] / (2 * self.permeability[i] * self.area[i]) +
                      self.delta_x[j] / (2 * self.permeability[j] * self.area[j]))
        return 1.0 / (self.viscosity * self.formation_volume_factor * resistance)

    def compute_accumulation(self, i):
        """Accumulation in block i, with no transmissibility conversion factor."""
        volume = self.area[i] * self.delta_x[i]
        return volume * self.porosity[i] * self.compressibility / self.formation_volume_factor

    def fill_matrices(self):
        """Assemble conservative face contributions and block storage (provided)."""
        n = self.N
        matrix = scipy.sparse.lil_matrix((n, n), dtype=float)
        self.Q = np.zeros(n)
        for i in range(n - 1):
            conductance = self.conversion_factor * self.compute_transmissibility(i, i + 1)
            matrix[i, i] += conductance
            matrix[i + 1, i + 1] += conductance
            matrix[i, i + 1] -= conductance
            matrix[i + 1, i] -= conductance
        self.apply_boundary_conditions(matrix, self.Q)
        self.T = matrix.tocsr()
        self.B = scipy.sparse.diags([self.compute_accumulation(i) for i in range(n)], format='csr')

    def apply_boundary_conditions(self, matrix, source):
        """Pressure at each end face, or zero outward flux (provided)."""
        for side, i in [('left', 0), ('right', self.N - 1)]:
            boundary = self.inputs['boundary conditions'][side]
            if boundary['type'] == 'prescribed pressure':
                face = 2 * self.conversion_factor * self.compute_transmissibility(i, i)
                matrix[i, i] += face
                source[i] += face * boundary['value']
            elif boundary['type'] != 'prescribed flux' or boundary['value'] != 0:
                raise ValueError('only prescribed pressure and zero prescribed flux are supported')

    def apply_initial_conditions(self):
        '''
            Applies initial pressures to self.p
        '''

        N = self.N

        self.p = np.ones(N) * self.inputs['initial conditions']['pressure']

        return


    def solve_one_step(self):
        '''
            Solve one step with explicit, implicit, or mixed time integration
        '''

        B = self.B
        T = self.T
        Q = self.Q

        dt = self.delta_t

        if self.inputs['numerical']['solver'] == 'explicit':
            self.p = self.p + dt * 1. / B.diagonal() * (Q - T.dot(self.p))
        elif self.inputs['numerical']['solver'] == 'implicit':
            next_p, info = scipy.sparse.linalg.cg(T + B / dt, B.dot(self.p) / dt + Q, atol=1.0e-8, rtol=0.0)
            if info != 0:
                raise RuntimeError(f'Conjugate gradient failed: info={info}')
            self.p = next_p
        elif isinstance(self.inputs['numerical']['solver'], dict):
            solver = self.inputs['numerical']['solver']
            if set(solver) != {'mixed method'} or not isinstance(solver['mixed method'], dict):
                raise ValueError('invalid mixed method configuration')
            settings = solver['mixed method']
            if set(settings) != {'theta'}:
                raise ValueError('mixed method requires theta')
            theta = settings['theta']
            if (isinstance(theta, (bool, np.bool_)) or
                    not isinstance(theta, (int, float, np.integer, np.floating)) or
                    not np.isfinite(theta) or not 0 <= theta <= 1):
                raise ValueError('theta must be a finite real number in [0, 1]')
            A = B / dt + (1 - theta) * T
            rhs = (B / dt - theta * T).dot(self.p) + Q
            next_p, info = scipy.sparse.linalg.cg(A, rhs, atol=1e-8, rtol=0.0)
            if info != 0:
                raise RuntimeError(f'Conjugate gradient failed: info={info}')
            self.p = next_p
        else:
            raise ValueError('solver must be explicit, implicit, or mixed method')

        return


    def solve(self):
        '''
            Solves until "number of time steps"
        '''

        for i in range(self.inputs['numerical']['number of time steps']):
            self.solve_one_step()

            if 'plots' in self.inputs and i % self.inputs['plots']['frequency'] == 0:
                self.p_plot.append(self.get_solution().copy())

        return

    def plot(self):
        '''
           Crude plotting function.  Plots pressure as a function of grid block #
        '''

        if hasattr(self, 'p_plot'):
            for i in range(len(self.p_plot)):
                plt.plot(self.p_plot[i])

        return

    def get_solution(self):
        '''
            Returns solution vector
        '''
        return self.p.copy()
