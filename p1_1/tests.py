import unittest
import numpy as np
import scipy.sparse as sp
from scipy.special import erfc
from euler_methods import approx_wave


class TestWaveEquation(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.c = 1
        self.L = 1
        self.N = 400
        self.dx = self.L / self.N
        self.dt = 0.001
        self.x = np.arange(self.N + 1) * self.dx

        # Create the equations matrix
        self.D_2 = sp.diags(
            (np.ones(self.N - 2), -2 * np.ones(self.N - 1), np.ones(self.N - 2)),
            (-1, 0, 1)
        )
        self.initConditions = [
            np.sin(2 * np.pi * self.x),
            np.sin(5 * np.pi * self.x),
            np.sin(5 * np.pi * self.x)
        ]

        self.initConditions[2] *= (self.x > .2) & (self.x < .4)
        self.conditions_names =['sin(2πx)', 'sin(5πx)', 'windowed sin(5πx)']

    def test_RK(self):
        """Check RK method"""
        for i, condition in enumerate(self.initConditions):
            with self.subTest(initCondition=i, condition_name=self.conditions_names[i]):
                psi_expected = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='SV'
                )
                psi_actual = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='RK4'
                )

                self.assertLess(np.max(np.abs(psi_actual - psi_expected)), 1e-2)

    def test_EF(self):
        """Check EF method"""
        for i,condition in enumerate(self.initConditions):
            with self.subTest(initCondition=i, condition_name=self.conditions_names[i]):
                psi_expected = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='SV'
                )
                psi_actual = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='EF'
                )

                self.assertLess(np.max(np.abs(psi_actual - psi_expected)), 1e-2)

    def test_LP(self):
        """Check LP method"""
        for i,condition in enumerate(self.initConditions):
            with self.subTest(initCondition=i, condition_name=self.conditions_names[i]):
                psi_expected = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='SV'
                )
                psi_actual = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='LP'
                )

                self.assertLess(np.max(np.abs(psi_actual - psi_expected)), 1e-2)

    def test_SV(self):
        """Check SV method"""
        for i,condition in enumerate(self.initConditions):
            with self.subTest(initCondition=i, condition_name=self.conditions_names[i]):
                psi_expected = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='LP'
                )
                psi_actual = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='SV'
                )

                self.assertLess(np.max(np.abs(psi_actual - psi_expected)), 1e-2)

class TestHeatEquation(unittest.TestCase):
    def setUp(self):
        def analytical(y, t, D=1.0, n_terms=200):
            if t == 0:
                return np.zeros_like(y, dtype=float)
            c = np.zeros_like(y, dtype=float)
            sqrt2Dt = np.sqrt(2 * D * t)
            for i in range(n_terms):
                c += erfc((1 - y + 2 * i) / sqrt2Dt) - erfc((1 + y + 2 * i) / sqrt2Dt)
            return c

if __name__ == '__main__':
    unittest.main()
