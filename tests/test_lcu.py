import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lcu_cirq.lcu import LCU
import cirq
import numpy as np
import unittest

class TestLCU(unittest.TestCase):

    def test_init(self):
        """Test the initialization and input validation."""
        coefficients = np.array([0.5, 0.5])
        unitaries = [np.eye(2), np.array([[0, 1], [1, 0]])]
        lcu = LCU(coefficients, unitaries)
        self.assertEqual(len(lcu.qubits), 2)

    def test_invalid_init(self):
        """Test invalid initialization."""
        # unitaries are in different shape
        coefficients = np.array([0.5, 0.5])
        unitaries = [np.eye(2), np.array([[0, 1], [1, 0], [0, 0]])]
        with self.assertRaises(AssertionError):
            lcu = LCU(coefficients, unitaries)
            
        # coefficients and unitaries have different length
        coefficients = np.array([0.5, 0.5])
        unitaries = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]])]
        with self.assertRaises(AssertionError):
            lcu = LCU(coefficients, unitaries)
            
        # the size of the input unitaries is not of power of 2
        coefficients = np.array([0.5, 0.5])
        unitaries = [np.eye(3), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
        with self.assertRaises(AssertionError):
            lcu = LCU(coefficients, unitaries)
   
    def test_lcu_simulation(self):
        """Test LCU, when the measurement of the first qubit yeild 0, if the unitaries are added """
        # case 1
        # Initialize coefficients and unitaries
#         coefficients = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
#         coefficients = np.array([1, 1])
        coefficients = np.array([0.2, 0.8])
        unitaries = [np.eye(2), np.array([[0, 1], [1, 0]])]
        
        # Initialize LCU class
        lcu = LCU(coefficients, unitaries)
        
        # Run the circuit on Cirq's simulator
        simulator = cirq.Simulator()
        results = simulator.simulate(lcu)

        # Get post-measurement state of the second qubit state and normalize the state
        output_on_0 = results.final_state_vector[0:2]
        
        coefficients_n = (coefficients)/sum(coefficients)
        
        np.testing.assert_allclose(output_on_0, coefficients_n, atol=1e-06)
        
        # case 2
        # Initialize coefficients and unitaries
#         coefficients = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
#         coefficients = np.array([1, 1])
        coefficients = np.array([0.2, 0.8])
        unitaries = [np.eye(4), np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])]

        
        # Initialize LCU class
        lcu = LCU(coefficients, unitaries)
        
        # Run the circuit on Cirq's simulator
        simulator = cirq.Simulator()
        results = simulator.simulate(lcu)

        # Get post-measurement state of the second qubit state and normalize the state
        output_on_0 = results.final_state_vector[0:2**2]
        
        expected_output=np.array([0.2, 0, 0, 0.8])
        
        np.testing.assert_allclose(output_on_0, expected_output, atol=1e-06)
        
        # case 3
        # Initialize coefficients and unitaries
#         coefficients = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
#         coefficients = np.array([1, 1])
        coefficients = np.array([0.2, 0.3, 0.5])
        unitaries = [np.eye(4), np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]), np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])]

        
        # Initialize LCU class
        lcu = LCU(coefficients, unitaries)
        
        # Run the circuit on Cirq's simulator
        simulator = cirq.Simulator()
        results = simulator.simulate(lcu)

        # Get post-measurement state of the second qubit state and normalize the state
        output_on_0 = results.final_state_vector[0:2**2]
        
        expected_output=np.array([0.7, 0, 0, 0.3])
        
        np.testing.assert_allclose(output_on_0, expected_output, atol=1e-06)



if __name__ == "__main__":
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestLCU))
