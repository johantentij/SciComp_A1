import unittest
import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse as sp
from euler_methods import approx_wave
from main import integrate


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
                psi_expected = integrate(condition, T_steps=150)
                psi_actual = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='RK'
                )

                self.assertLess(np.max(np.abs(psi_actual - psi_expected)), 1e-2)

    def test_EF(self):
        """Check EF method"""
        for i,condition in enumerate(self.initConditions):
            with self.subTest(initCondition=i, condition_name=self.conditions_names[i]):
                psi_expected = integrate(condition, T_steps=150)
                psi_actual = approx_wave(
                    condition, time_steps=150,
                    mesh_points=self.x, equations_matrix=self.D_2,
                    dt=self.dt, c=self.c, dx=self.dx, N=self.N, method='EF'
                )

                self.assertLess(np.max(np.abs(psi_actual - psi_expected)), 1e-2)

if __name__ == '__main__':
    unittest.main()
