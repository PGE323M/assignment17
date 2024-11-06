#!/usr/bin/env python

# Copyright 2018-2020 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import nbconvert
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

with open("assignment17.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)


with open("assignment17.py", "w") as f:
    f.write(python_file)

from assignment17 import TwoDimReservoir

class TestSolution(unittest.TestCase):

    def setUp(self):

        self.inputs = {
            'conversion factor': 6.33e-3,
            'fluid': {
                'water': {
                    'compressibility': 1e-6, #psi^{-1}
                    'viscosity': 1, #cp
                    'formation volume factor': 1
                    },
            },
            'reservoir': {
                'permeability': 50, #mD
                'porosity': 0.2,
                'length': 10000, #ft
                'height': 200000, #ft
                'depth': 1 #ft
            },
            'initial conditions': {
                'pressure': 1000 #psi
            },
            'boundary conditions': {
                'left': {
                    'type': 'prescribed pressure',
                    'value': 2000 #psi
                },
                'right': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                },
                'top': {
                    'type': 'prescribed flux',
                    'value': 0 #psi
                },
                'bottom': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                }
            },
            'numerical': {
                'solver': 'implicit',
                'number of grids': {'x': 4, 'y': 1},
                'time step': 1, #day
                'number of time steps' : 3
            },
            'plots': {
                'frequency': 1
            }
        }

        return

    def test_compute_transmissibility(self):

        problem = TwoDimReservoir(self.inputs)

        np.testing.assert_allclose(problem.compute_transmissibility(0,0), 4000.0)

        return

    def test_compute_accumulation(self):

        problem = TwoDimReservoir(self.inputs)

        np.testing.assert_allclose(problem.compute_accumulation(0), 100.0)

        return

    def test_is_transmissiblity_matrix_sparse(self):

        problem = TwoDimReservoir(self.inputs)

        assert scipy.sparse.issparse(problem.T)

        return

    def test_implicit_solve_one_step(self):

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1295.1463, 1051.1036, 1008.8921, 1001.7998]),
                                   atol=0.5)
        return

    def test_explicit_solve_one_step(self):

        self.inputs['numerical']['solver'] = 'explicit'

        explicit = TwoDimReservoir(self.inputs)

        explicit.solve_one_step()

        np.testing.assert_allclose(explicit.get_solution(),
                               np.array([ 1506., 1000.,  1000.,  1000.004]),
                               atol=0.5)
        return

    def test_mixed_method_solve_one_step_implicit(self):

        self.inputs['numerical']['solver'] = {'mixed method': {'theta': 0.0}}

        mixed_implicit = TwoDimReservoir(self.inputs)

        mixed_implicit.solve_one_step()

        np.testing.assert_allclose(mixed_implicit.get_solution(),
                               np.array([1295.1463, 1051.1036, 1008.8921, 1001.7998]),
                               atol=0.5)
        return

    def test_mixed_method_solve_one_step_explicit(self):


        self.inputs['numerical']['solver'] = {'mixed method': {'theta': 1.0}}

        mixed_explicit = TwoDimReservoir(self.inputs)

        mixed_explicit.solve_one_step()

        np.testing.assert_allclose(mixed_explicit.get_solution(),
                               np.array([ 1506., 1000.,  1000.,  1000.004]),
                               atol=0.5)
        return

    def test_mixed_method_solve_one_step_crank_nicolson(self):

        self.inputs['numerical']['solver'] = {'mixed method': {'theta': 0.5}}

        mixed = TwoDimReservoir(self.inputs)

        mixed.solve_one_step()

        np.testing.assert_allclose(mixed.get_solution(),
                                   np.array([ 1370.4,  1037.8 ,  1003.8,  1000.4]),
                                   atol=0.5)
        return

    def test_implicit_solve(self):

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1582.9, 1184.8, 1051.5, 1015.9]),
                                   atol=0.5)
        return

    def test_implicit_solve_reverse_boundary_conditions(self):


        self.inputs['boundary conditions'] = {
                'right': {
                    'type': 'prescribed pressure',
                    'value': 2000 #psi
                },
                'left': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                },
                'top': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                },
                'bottom': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                }
            }

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1015.9, 1051.5, 1184.8, 1582.9]),
                                   atol=0.5)
        return

    def test_explicit_solve(self):

        self.inputs['numerical']['solver'] = 'explicit'

        explicit = TwoDimReservoir(self.inputs)

        explicit.solve()

        np.testing.assert_allclose(explicit.get_solution(),
                               np.array([1689.8, 1222.3, 1032.4, 1000.0]),
                               atol=0.5)
        return

    def test_mixed_method_solve_crank_nicolson(self):

        self.inputs['numerical']['solver'] = {'mixed method': {'theta': 0.5}}

        mixed = TwoDimReservoir(self.inputs)

        mixed.solve()

        np.testing.assert_allclose(mixed.get_solution(),
                                   np.array([ 1642.0,  1196.5,  1043.8,  1009.1]),
                                   atol=0.5)
        return

    def test_implicit_heterogeneous_permeability_solve_one_step(self):

        self.inputs['reservoir']['permeability'] = [10., 100., 50., 20]

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1085.3,  1005.8,  1001.3,  1000.1]),
                                   atol=0.5)
        return

    def test_implicit_heterogeneous_permeability_and_grid_size_solve_one_step(self):

        self.inputs['reservoir']['permeability'] = [10., 100., 50., 20]
        self.inputs['numerical']['delta x'] = [2000., 3000., 1500., 3500]

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1123.0,  1008.5,  1003.1,  1000.2]),
                                   atol=0.5)
        return


    def test_implicit_heterogeneous_permeability_and_grid_size_solve(self):

        self.inputs['reservoir']['permeability'] = [10., 100., 50., 20]
        self.inputs['numerical']['delta x'] = [2000., 3000., 1500., 3500]

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1295.6,  1039.1,  1019.9,  1002.5]),
                                   atol=0.5)
        return

    def test_two_dim_solve_one_step(self):


        self.inputs['numerical']['number of grids'] = {'x': 3, 'y': 3}
        self.inputs['reservoir']['height'] = 10000.
        self.inputs['reservoir']['depth'] = 20.
        self.inputs['boundary conditions'] = {
                'right': {
                    'type': 'prescribed pressure',
                    'value': 2000 #psi
                },
                'left': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                },
                'top': {
                    'type': 'prescribed flux',
                    'value': 0 #psi
                },
                'bottom': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                }
        }

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([ 1002.8,  1022.6,  1201.8,  1002.8, 1022.6,
                                              1201.8,  1002.8,  1022.6, 1201.8]), 
                                   atol=0.5)
        return


    def test_implicit_solve_one_step_with_wells_1(self):


        self.inputs['wells'] = {
                'rate': {
                    'locations': [(0.0, 1.0)],
                    'values': [1000],
                    'radii': [0.25]
                },
                'bhp': {
                    'locations': [(6250.0, 1.0)],
                    'values': [800],
                    'radii': [0.25]
                }
            }

        self.inputs['reservoir'] = {
                'permeability': 50, #mD
                'porosity': 0.2,
                'length': 10000, #ft
                'height': 2500, #ft
                'depth': 80 #ft
            }

        self.inputs['boundary conditions']['left']['type'] = 'prescribed flux'
        self.inputs['boundary conditions']['left']['value'] = 0.0
        self.inputs['boundary conditions']['right']['type'] = 'prescribed pressure'
        self.inputs['boundary conditions']['right']['value'] = 2000.0

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1008.9,  1004.7,  1019.1,  1290.5]),
                                   atol=0.5)
        return


    def test_implicit_solve_with_wells_1(self):


        self.inputs['wells'] = {
                'rate': {
                    'locations': [(0.0, 1.0)],
                    'values': [1000],
                    'radii': [0.25]
                },
                'bhp': {
                    'locations': [(6250.0, 1.0)],
                    'values': [800],
                    'radii': [0.25]
                }
            }

        self.inputs['reservoir'] = {
                'permeability': 50, #mD
                'porosity': 0.2,
                'length': 10000, #ft
                'height': 2500, #ft
                'depth': 80 #ft
            }


        self.inputs['boundary conditions']['left']['type'] = 'prescribed flux'
        self.inputs['boundary conditions']['left']['value'] = 0.0
        self.inputs['boundary conditions']['right']['type'] = 'prescribed pressure'
        self.inputs['boundary conditions']['right']['value'] = 2000.0

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1028.9,  1031.6,  1096.7,  1563.7]),
                                   atol=0.5)
        return


    def test_implicit_solve_one_step_with_wells_2(self):


        self.inputs['wells'] = {
                'bhp': {
                    'locations': [(9000,9000)],
                    'values': [800],
                    'radii': [0.25],
                },
                'rate': {
                    'locations': [(5000,5000)],
                    'values': [1000],
                    'radii': [0.25],
                }
            }

        self.inputs['reservoir'] = {
                'permeability': 50, #mD
                'porosity': 0.2,
                'length': 10000, #ft
                'height': 10000, #ft
                'depth': 20 #ft
            }

        self.inputs['numerical']['number of grids'] = {'x': 3, 'y': 3}

        self.inputs['boundary conditions']['left']['type'] = 'prescribed flux'
        self.inputs['boundary conditions']['left']['value'] = 0.0
        self.inputs['boundary conditions']['right']['type'] = 'prescribed pressure'
        self.inputs['boundary conditions']['right']['value'] = 2000.0

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1003.1, 1024.1, 1201.8, 1004.2, 1037.0,  1200.7,  1002.8, 1021.3, 1174.5]),
                                   atol=0.5)
        return

if __name__ == '__main__':
        unittest.main()
