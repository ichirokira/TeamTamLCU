import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lcu_cirq.state_preparation import GroverRudolph
import cirq
import numpy as np
import unittest
class TestGroverRudolph(unittest.TestCase):
    
    def test_invalid_coefficient(self):
        # test when the input is not a non-empty np.ndarray, if the Error is raised
        
        # Test with integer(s)
        with self.assertRaises(AssertionError):
            grover_circuit = GroverRudolph(1)
        with self.assertRaises(AssertionError):
            grover_circuit = GroverRudolph(1, 2)
        
        # Test with string
        with self.assertRaises(AssertionError):
            grover_circuit = GroverRudolph(["a", "b"])
        with self.assertRaises(AssertionError):
            grover_circuit = GroverRudolph("invalid")
        
        # Test with dictionary
        with self.assertRaises(AssertionError):
            grover_circuit = GroverRudolph({"key": "value"})
        
        # Test with empty input
        with self.assertRaises(AssertionError):
            grover_circuit = GroverRudolph([])
        
        # Test with None 
        with self.assertRaises(AssertionError):
            grover_circuit = GroverRudolph(None)
            
        # Test with negative coefficient(s)
        with self.assertRaises(AssertionError):
            grover_circuit = GroverRudolph([-0.5, -0.5]) 
            
        with self.assertRaises(AssertionError):
            grover_circuit = GroverRudolph([-0.5, 0.5])  
    
    def test_qubit_and_padding(self):
        # Test when length of coefficients is a power of 2
        coefficients1 = np.array([1, 1, 1, 1])
        grover_circuit1 = GroverRudolph(coefficients1)
        self.assertEqual(grover_circuit1.num_qubits, 2)
        self.assertEqual(len(grover_circuit1.coefficients), 4)
        
        # Test when length of coefficients is not a power of 2
        coefficients2 = np.array([1, 1, 1])
        grover_circuit2 = GroverRudolph(coefficients2)
        self.assertEqual(grover_circuit2.num_qubits, 2)
        self.assertEqual(len(grover_circuit2.coefficients), 4)

        # Verify that the coefficients are padded with zeros
#         np.testing.assert_array_equal(grover_circuit2.coefficients, np.array([1/3, 1/3, 1/3, 0]))
        np.testing.assert_almost_equal(grover_circuit2.coefficients, np.array([1/3, 1/3, 1/3, 0]))
    
    def test_coefficient_normalization(self):
        # Test with some input coefficients that are not normalized and check if coefficients are normalized
        
        coefficients = np.array([2, 2, 2, 2])
        grover_circuit = GroverRudolph(coefficients)
        self.assertEqual(np.sum(grover_circuit.coefficients), 1)

        # another set
        coefficients = np.array([3, 1, 2])
        grover_circuit = GroverRudolph(coefficients)
        self.assertEqual(np.sum(grover_circuit.coefficients), 1)
        
        # another set
        coefficients = np.array([6])
        grover_circuit = GroverRudolph(coefficients)
        self.assertEqual(np.sum(grover_circuit.coefficients), 1)
        
    def test_compute_angle(self):
        # Initialize GroverRudolph object with known coefficients
        coefficients = np.array([0.2,0.3,0.3,0.2])
        grover_circuit = GroverRudolph(coefficients)

        # Test for step 0
        step = 0
        previous_sum = [1]  
        expected_angles_step_0 = [np.pi / 4]
        expected_current_sum_step_0 = [0.5, 0.5]
        
        angles_step_0, current_sum_step_0 = grover_circuit.compute_angle(step, previous_sum)
        
        np.testing.assert_almost_equal(angles_step_0[0], expected_angles_step_0)
        np.testing.assert_almost_equal(current_sum_step_0, expected_current_sum_step_0)
        
        # Test for step 1
        step = 1
        previous_sum = [0.5, 0.5] 
        expected_angles_step_1 = [np.arccos(np.sqrt(2/5)), np.arccos(np.sqrt(3/5))]
        expected_current_sum_step_1 = [0.2, 0.3, 0.3, 0.2]

        angles_step_1, current_sum_step_1 = grover_circuit.compute_angle(step, previous_sum)
        
        np.testing.assert_almost_equal(angles_step_1, expected_angles_step_1)
        np.testing.assert_almost_equal(current_sum_step_1, expected_current_sum_step_1)
        
    def test_circuit_output_state_vector(self):
        # test the GroverRudolph class to ensure it outputs the correct circuit for preparing the desired state
        
        # output the state preparation circuit corresponds to a certain input coefficients
        
        # test 1 with normalized coefficients whose length is power of 2
#         coefficients = np.array([0.25, 0.25, 0.25, 0.25])
        coefficients1 = np.array([0.2, 0.3, 0.3, 0.2])
        grover_circuit1 = GroverRudolph(coefficients1)
        
        # Simulate the circuit
        simulator = cirq.Simulator()
        result1 = simulator.simulate(grover_circuit1)
        
        # Get final state and normalize the state
        final_state1 = result1.final_state_vector
        final_state_normalized1 = np.abs(final_state1) / np.linalg.norm(final_state1)
        
        # Calculate expected state (amplitudes should be square root of coefficients)
        expected_state1 = np.sqrt(coefficients1)
        
        # Comparing the final and expected states
        np.testing.assert_almost_equal(final_state_normalized1, expected_state1, decimal=5)
        
        # test 2 with unnormalized coefficients whose length is power of 2
        coefficients2 = np.array([2, 3, 3, 2])
        grover_circuit2 = GroverRudolph(coefficients2)
        
        # Simulate the circuit
        simulator = cirq.Simulator()
        result2 = simulator.simulate(grover_circuit2)
        
        # Get final state and normalize the state
        final_state2 = result2.final_state_vector
        final_state_normalized2 = np.abs(final_state2) / np.linalg.norm(final_state2)
        
        # Calculate expected state (amplitudes should be square root of coefficients)
        n_coefficients2=(coefficients2)/sum(coefficients2)
        expected_state2 = np.sqrt(n_coefficients2)
        
        # Comparing the final and expected states
        np.testing.assert_almost_equal(final_state_normalized2, expected_state2, decimal=5)
        
        
        # test 3 with unnormalized coefficients whose length is not power of 2
        coefficients3 = np.array([2, 3, 3])
        grover_circuit3 = GroverRudolph(coefficients3)
        
        # Simulate the circuit
        simulator = cirq.Simulator()
        result3 = simulator.simulate(grover_circuit3)
        
        # 
        final_state3 = result3.final_state_vector
        final_state_normalized3 = np.abs(final_state3) / np.linalg.norm(final_state3)
        
        # Calculate expected state (amplitudes should be square root of coefficients, and we normalzied them)
        n_coefficients3=(coefficients3)/sum(coefficients3)
        expected_state3 = np.sqrt(n_coefficients3)
        
        # Comparing the final and expected states
        np.testing.assert_almost_equal(final_state_normalized3[0:len(coefficients3)], expected_state3, decimal=5)
        
    

if __name__ == '__main__':
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestGroverRudolph))