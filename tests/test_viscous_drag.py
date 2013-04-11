from unittest import TestCase
from whales.viscous_drag import ViscousDragModel
import numpy as np
import numpy.testing
from numpy.testing import assert_array_almost_equal_nulp

class MyTestCase(TestCase):
    def assertArraysEqual(self, a, b):
        numpy.testing.assert_array_equal(a, b)

class TaperedMemberTestCase(MyTestCase):
    """Test simple model with one member"""

    def setUp(self):
        config = {
            'drag coefficient': 0.7,
            'members': [{
                'end1': [0, 0, 0],
                'end2': [0, 0, -10],
                'diameter': [[0, 3, 5, 10], [2.2, 2.2, 5, 5]],
                'strip width': 0.5,
            }],
        }
        self.model = ViscousDragModel(config)

    def test_model_elements(self):
        """Check element strips setup correctly"""
        m = self.model
        self.assertArraysEqual(m.element_lengths, 0.5 * np.ones(20))

        # Centres of strips
        centres = np.zeros((20, 3))
        centres[:,2] = -np.arange(0.25, 10, 0.5)
        self.assertArraysEqual(m.element_centres, centres)

        # Element diameters
        self.assertEqual(m.element_diameters[ 0], 2.2)
        self.assertEqual(m.element_diameters[ 5], 2.2)
        self.assertEqual(m.element_diameters[10], 5.0)
        self.assertEqual(m.element_diameters[-1], 5.0)
        self.assertEqual(m.element_diameters[7], 2.2 + (5.0 - 2.2)*0.75/(5-3))

        # Element axes
        self.assertArraysEqual(m.element_axes, np.array([np.eye(3)] * 20))

class CylinderTestCase(MyTestCase):
    """Test simple model with vertical cylinder"""

    def setUp(self):
        config = {
            'drag coefficient': 0.7,
            'members': [{
                'end1': [0, 0, 0],
                'end2': [0, 0, -10],
                'diameter': 2.3,
                'strip width': 1,
            }],
        }
        self.model = ViscousDragModel(config)

    def test_model_elements(self):
        """Check element strips setup correctly"""
        m = self.model
        self.assertArraysEqual(m.element_lengths, np.ones(10))

        # Centres of strips
        centres = np.zeros((10, 3))
        centres[:,2] = -np.arange(0.5, 10, 1)
        self.assertArraysEqual(m.element_centres, centres)

        # Element diameters
        self.assertArraysEqual(m.element_diameters, 2.3)

        # Element axes
        self.assertArraysEqual(m.element_axes, np.array([np.eye(3)] * 10))

    def test_wave_velocity_transfer_func(self):
        """Test wave velocity transfer function"""
        w = np.array([1,2]) # frequencies to test
        H_uf = self.model.wave_velocity_transfer_function(w)

        # With waves in x-direction, sideways velocity should be zero
        self.assertArraysEqual(H_uf[:,:,1], 0)

        # Check variation in depth: exp(kz)
        iz1 = 3
        iz2 = 8
        z = self.model.element_centres[:,2]
        for i in range(2):
            assert_array_almost_equal_nulp(H_uf[i,iz1,:] / np.exp(w[i]**2/9.81*z[iz1]),
                                           H_uf[i,iz2,:] / np.exp(w[i]**2/9.81*z[iz2]))

        # Check all x velocities are in-phase and real, all z are imaginary
        self.assertTrue(np.isreal(     H_uf[:,:,0]).all())
        self.assertTrue(np.isreal(1j * H_uf[:,:,2]).all())

    def test_structural_velocity_transfer_func(self):
        """Test structural velocity with special cases"""
        w = np.array([1,2]) # frequencies to test

        # Case 1: pure surge motion
        H1 = np.zeros((2, 6)) # shape (freq, xyzXYZ)
        H1[:,0] = 1 # maximum surge at maximum datum wave height
        H_us = self.model.structural_velocity_transfer_function(w, H1)
        # all elements should have same surge velocity; all other velocities zero
        # at t=0, velocity is zero and becoming negative 90 deg later
        self.assertArraysEqual(H_us[1,:,0], 2j)
        self.assertArraysEqual(H_us[0,:,0], 1j)
        self.assertArraysEqual(H_us[:,:,1:], 0)

        # Case 2: pure roll motion
        H2 = np.zeros((2, 6)) # shape (freq, xyzXYZ)
        H2[:,3] = 1 # maximum roll at maximum datum wave height
        H_us = self.model.structural_velocity_transfer_function(w, H2)
        # x & z velocity should be zero
        self.assertArraysEqual(H_us[:,:,[0,2]], 0)
        # y velocity corresponding to rotation about origin (check bottom)
        # at t=0, ang. velocity is zero and becoming negative 90 deg later
        # Velocity of bottom element = 9.5 * ang vel
        self.assertArraysEqual(H_us[0,-1,1], 9.5 * 1j)
        self.assertArraysEqual(H_us[1,-1,1], 9.5 * 2j)


class ResolvingTestCase(MyTestCase):
    """Test differently-oriented members"""

    def setUp(self):
        config = {
            'drag coefficient': 0.7,
            'members': [{
            #     # Member in x-direction
            #     'end1': [0, 0, 0],
            #     'end2': [1, 0, 0],
            #     'diameter': 1,
            #     'strip width': 1,
            # }, {
                # Member in z-direction
                'end1': [0, 0, 0],
                'end2': [0, 0, 1],
                'diameter': 1,
                'strip width': 1,
            # }, {
            #     # Member in xy plane at 30deg from x axis
            #     'end1': [0, 0, 0],
            #     'end2': [np.cos(30*np.pi/180), np.sin(30*np.pi/180), 0],
            #     'diameter': 1,
            #     'strip width': 1,
            }],
        }
        self.model = ViscousDragModel(config)

    def test_resolve_to_local_coords(self):
        """Calculate locally-normal vector components"""
        resolve = self.model.resolve_perpendicular_to_elements

        # Test 1: velocity in x-direction
        v1 = [1, 0, 0]
        self.assertArraysEqual(resolve(v1), [
#            [0, 0], # member in x-direction, no normal velocity
            [1, 0], # member in z-direction
#            [np.nan, np.nan], # member 30deg from x-axis
        ])

        # Test 1: velocity in 3d
        v2 = [1, 1, 2]
        self.assertArraysEqual(resolve(v2), [
#            [0, 0], # member in x-direction, no normal velocity
            [1, 1], # member in z-direction
#            [np.nan, np.nan], # member 30deg from x-axis
        ])
