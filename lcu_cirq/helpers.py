import cirq
import numpy as np
import math

def is_unitary(matrix: np.ndarray) -> bool:
    """
    is_unitary: checks if a given 2D array (matrix) is unitary or not. 
                A matrix is considered unitary if its conjugate transpose is also its inverse.
    Inputs: 
        matrix (Type: np.ndarray): A 2D numpy array representing the matrix to be checked.
        
    Outputs: 
        Returns True if the matrix is unitary, False otherwise (Type: bool).
        
    Error Handling: 
        Assumes that the input matrix is square and non-empty, matrix elements are numerical. 
        If the input matrix is not complex, change it to complex.
        No error handling for incorrect input dimensions.

    Description:: 
        The function returns a boolean value indicating whether the matrix is unitary or not.
        
    """
    # Ensure the matrix is complex
    matrix = matrix.astype(np.complex128)
    # if the matrix is non-square, raise an error
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The input matrix must be square.")
    unitary = True
    n = matrix.shape[0]
    error = np.linalg.norm(np.eye(n) - matrix.dot( matrix.transpose().conjugate()))

    if not(error < np.finfo(matrix.dtype).eps * 10.0 *n):
        unitary = False

    return unitary

class Unitary(cirq.Gate):
    """
    Unitary: This class extends the cirq.Gate class to define a custom quantum gate based on an input unitary matrix.
             It checks if the input matrix is unitary and provides methods for interacting with the Cirq library.
             
    Attributes: 
        array (np.ndarray): The unitary matrix defining the gate.
        name (str): The name of the custom gate, the default is "U"
        
    Methods: 
        __init__(self, array, name=None): Initializes the class attributes.
        _num_qubits_(self): Returns the number of qubits on which the gate acts.
        _unitary_(self): Returns the unitary matrix of the gate.
        _circuit_diagram_info_(self, args): Returns the name of the custom gate for circuit diagram representation.
    
    Error Handling: 
        Checks if the input array is a unitary matrix. Raises an AssertionError if it's not.
   
    """


    def __init__(self, array, name=None) -> None:
        """
        __init__: Initializes the class attributes.
        
        Args:
            - array (np.ndarray): A 2D array defining the unitary matrix for the custom gate.
            - name (str, Optional): The name for the custom gate.
            
        Raises:
            - AssertionError: if the input matrix is not unitary.
        
        """
        super(Unitary, self)

        # check array is an instance of array, if not convert to numpy array
        if not isinstance(array , np.ndarray):
            array = np.array(array, dtype=np.complex128)


        assert is_unitary(array) == True, "Invalid unitary"

        # define number of qubit that the gate acting on
        shape = array.shape
        self.num_qubits = int(np.log2(shape[0]))
        self.array = array

        if name:
            self.name = name
        else:
            self.name = "U"

    def _num_qubits_(self):
        """
        _num_qubits_: return the number of qubit that the gate acting on
        """
        return self.num_qubits

    def _unitary_(self):
        """
        _unitary_: Returns the unitary matrix of the gate.
        """
        return self.array

    def _circuit_diagram_info_(self, args):
        """
        _circuit_diagram_info_: Returns the name of the custom gate for circuit diagram representation
        """
        return self.name
