import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lcu_cirq.helpers import *
import cirq
import numpy as np
import unittest

class TestUnitary(unittest.TestCase):
    
    def test_unitary(self):
        """Test initialization with a unitary matrix,
        should work and _unitary_ should returns the unitary matrix of the gate."""
        unitary = np.array([[0, 1], [1, 0]])
        gate = Unitary(unitary)
        np.testing.assert_allclose(gate._unitary_(), unitary, atol=1e-06)
        

    def test_non_unitary(self):
        """Test initialization with a non-unitary matrix."""
        non_unitary = np.array([[1, 2], [3, 4]])
        with self.assertRaises(AssertionError):
            gate = Unitary(non_unitary)
            
    def test_non_square(self):
        """Test initialization with a non-square matrix."""
        non_square = np.array([[0, 1, 0], [1, 0, 0]])
        with self.assertRaises(ValueError):
            gate = Unitary(non_square)

    def test_num_qubits(self):
        """Test if _num_qubits_ return the correct number of qubit."""
        unitary = np.array([[0, 1], [1, 0]])
        gate = Unitary(unitary)
        self.assertEqual(gate._num_qubits_(), 1)
    
    def test_circuit_diagram_info(self):
        """Test if _circuit_diagram_info_ return the correct name."""
        unitary = np.array([[1, 0], [0, 1]])
        gate = Unitary(unitary, name="Test")
        self.assertEqual(gate._circuit_diagram_info_(None), "Test")
                        
        unitary_none = np.array([[1, 0], [0, 1]])
        gate = Unitary(unitary_none)
        self.assertEqual(gate._circuit_diagram_info_(None), "U")
        
    def test_non_ndarray_input(self):
        """Test initialization with a list instead of a numpy array."""
        # input a numerical list, this would still work, though we don't recommend
        unitary_list = [[0, 1], [1, 0]]
        gate = Unitary(unitary_list)
        self.assertTrue(np.allclose(gate._unitary_(), np.array(unitary_list)))
        
        # input a numerical list, this would not work because it can't be transformed into a square matrix
        unitary_list2 = [[0, 1, 1, 0]]
        with self.assertRaises(ValueError):
            gate2 = Unitary(unitary_list2)
        
        # input a non-numerical list
        unitary_list3 = [['a', 1], [1, 'b']]
        with self.assertRaises(ValueError):
            gate = Unitary(unitary_list3)
       
        

if __name__ == '__main__':
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestUnitary))
