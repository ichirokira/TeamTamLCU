"""
Demonstrate Hamiltonian Simulation using LCU

Assuming H = a_1H_1 + a_2H_2 +...+ a_mH_m, with each H_i is efficiently simulable. 
In this example, we assume H_i is a PauliString.

The algorithm first truncates the exponential function of exp(iHt), according to a input 
variable t indicating the simulation time. Then the unitary U = exp(iHt) can be represented into
a linear combination of unitaries. Apply LCU circuit to simulate the unitary U.

====REFERENCE=====
Lectures on Quantum Algorithm by Andrew Child
https://www.cs.umd.edu/~amchilds/qa/qa.pdf

====EXPECTED OUTPUT====

================RESULT===============
[INFO] Quantum-run output state: [ 0.10899131+0.0000000e+00j  0.        -3.0104485e-01j
 -0.10095187+0.0000000e+00j  0.        +3.2430604e-01j
 -0.62535083+0.0000000e+00j  0.        -2.3676381e-09j
 -0.6253508 +0.0000000e+00j  0.        -2.3676381e-09j]
[INFO] Classical-run output state: [ 0.1089913 +0.j          0.        -0.30104481j -0.10095187+0.j
  0.        +0.32430599j -0.62535081+0.j          0.        +0.j
 -0.62535081+0.j          0.        +0.j        ]
Matched

"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cirq
import numpy as np
import math
from cirq.contrib.svg import SVGCircuit
from lcu_cirq.lcu import LCU

# Pauli Matrices
Paulis = {
    "I" : np.array([[1,0],[0,1]], dtype=np.complex128),
    "X" : np.array([[0,1],[1,0]], dtype=np.complex128),
    "Y" : np.array([[0,-1j],[1j,0]], dtype=np.complex128),
    "Z" : np.array([[1,0],[0,-1]], dtype=np.complex128)
}

def convert_to_base(number: int, base: int, length: int) -> list:
    """
    Conver arbitrary number into any base
    Args:
      number: `int` the convert number, non-negative
      base: `int` the base system, postive
      length: `int` reprensetation length, postive
    Return:
      list of bits represent `number` by `base`
    Raises:
        - AssertionError: if the number is negative 
        - AssertionError: if the base is not postive 
        - AssertionError: if the length is not postive 
        - ValueError: if the lengh of the result is larger than the length specified
    """
    
    assert number >=0, "number is non-negative"
    assert base >0, "number is postive"
    assert length >0, "length is postive"
    if number == 0:
        return [0 for i in range(length)]

    result = []
    while number > 0:
        remainder = number % base
        result.insert(0, remainder)
        number //= base
    if len(result) > length:
      raise ValueError("Representation is overflow")
    else:
      for i in range(length-len(result)):
        result.insert(0, 0)
    return result

def truncate_exp(H: dict, time: float, at_degree: int = None, error: float = 10e-5):
    """
    Truncate the Taylor series of the exponential function (to a certain degree) in form of
    linear combination of unitary.

    Args:
        H: a dictionary inf form of `{"PauliString": coefficients}`
        time: `float` the simulation time
        at_degree: `int` (Optional) the degree that the Taylor series is truncated.
                  If not specified, the degree is set by time + log(1/error)
        error: `float` (Default=10e-5) Simulation error

    Return:
        lcu_coefficients: `list(float)`, list of coefficients for LCU circuit
        lcu_unitaries: `list(np.array)`, list of unitaries for LCU circuit
    Raises:
        - ValueError: if H is empty
        - ValueError: if time H is negative
        - ValueError: if at_degree is negative
    """
    if not H:
        raise ValueError("Hamiltonian cannot be empty.")
    if time < 0:
        raise ValueError("Time cannot be negative.")
    if at_degree is not None and at_degree < 0:
        raise ValueError("at_degree cannot be negative.")
    if not (0 < error < 1):
        raise ValueError("Error must be between 0 and 1.")

    # Preprocess inputs
    former_coeffs = []
    former_unitaries = []
    for pauli_string, coef in H.items():
        former_coeffs.append(coef)
        u = Paulis[pauli_string[0]]
        for i in range(1, len(pauli_string)):
            u = np.kron(u, Paulis[pauli_string[i]])
        former_unitaries.append(u)

    size = former_unitaries[0].shape[0]
    # if time=0, return identity and lcu_coefficients is 1 (as a list)
    if time == 0:
        lcu_coefficients = [1]
        lcu_unitaries = [np.eye(size)]
    else:
        # Specify truncated point
        if not at_degree:
            K = math.ceil(time+np.log2(1/error))
        else:
            K = at_degree+1

        # Calculate lcu_coefficients and lcu_unitaries
        lcu_coefficients = []
        lcu_unitaries = []
        num_terms = len(former_coeffs) # number of terms in the former combination
        for k in range(K):
            if k == 0:
                lcu_coefficients.append(1.0)
                lcu_unitaries.append(np.eye(size))
            else:
                # For each power (a0+a1+...+am)^k, there are m^k terms
                for i in range(num_terms**k):
                    i_base_m = convert_to_base(i, base=num_terms, length=k)
                    coeff = (time)**k/math.factorial(k)
                    unitary = (1j)**k*np.eye(size)
                    for j in i_base_m:
                        coeff *= former_coeffs[j]
                        unitary = unitary@former_unitaries[j]

                    lcu_coefficients.append(coeff)
                    lcu_unitaries.append(unitary)
    return lcu_coefficients, lcu_unitaries

# Get Hamiltonian from dict
def get_Hamiltonian(H: dict) -> np.array:
    """
    get_Hamiltonian: convert a dictionary of Pauli strings and their coefficients into a Hamiltonian matrix
    Args:
        H: a dictionary inf form of `{"PauliString": coefficients}`

    Return:
        a numpy array representing the Hamiltonian matrix.
    
    Raises:
        - TypeError: if H is not a dictionary
        - ValueError: if H is empty
        
    """    
    if not isinstance(H, dict):
        raise TypeError("Input should be a dictionary.")
    if len(H) == 0:
        raise ValueError("Input should not be empty.")
    
    former_coeffs = []
    former_unitaries = []
    for pauli_string, coef in H.items():
        former_coeffs.append(coef)
        u = Paulis[pauli_string[0]]
        for i in range(1, len(pauli_string)):
            u = np.kron(u, Paulis[pauli_string[i]])
        former_unitaries.append(u)

    result = 0
    for c, u in zip(former_coeffs, former_unitaries):
        result += c*u
    return result

# Classical implement matrix exponential for testing (compare to quantum)
def exp_matrix(H: np.array, time: float, truncated_degree: int) -> np.ndarray:

  # Define the number of terms in the Taylor series
  matrix = 1j*H*time
  # Compute the Taylor series approximation
  taylor_series_approximation = np.eye(matrix.shape[0], dtype=np.complex128)  # Identity matrix of the same size as matrix
  for i in range(1, truncated_degree+1):
      A_power_i = np.linalg.matrix_power(matrix, i)
      factorial_i = np.math.factorial(i)
      term = A_power_i / factorial_i
      taylor_series_approximation += term

  return taylor_series_approximation

def classical_output(H: dict, time: float, truncated_degree: int, initial_state: np.ndarray = None) -> np.ndarray:
  """
  Calculate the energy of a Hamiltonian after evolving the system
  """
  # get the hamiltonian from input dict
  hamiltonian = get_Hamiltonian(H)

  # if initial_state is None, set to be |0>
  size = hamiltonian.shape[0]
  if initial_state is None:
    initial_state = np.eye(size)[0]

  # get unitary corresponds to the Hamiltonian on the time t
  U = exp_matrix(hamiltonian, time, truncated_degree)
  # Compute output_state
  output_state = U.dot(initial_state)
  output_state = output_state/np.sqrt(output_state.conjugate()@output_state)

  return output_state


def main():

    H = {"XYZ": 3, "YZZ": 3, "ZXX":3}
    simulation_time = 5
    truncated_degree = 5
    
    #Express the Hamiltonian dynamic in terms of Linear combination of unitaries
    lcu_coefficients, lcu_unitaries = truncate_exp(H, simulation_time, truncated_degree)

    # Specifies system qubits
    unitary_qubits = int(np.log2(lcu_unitaries[0].shape[0]))
    
    qubits = [cirq.NamedQubit('unit'+str(i)) for i in range(unitary_qubits)]

    #Contruct LCU circuit 
    lcu_circ = LCU(lcu_coefficients, lcu_unitaries, on_qubits=qubits)

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.simulate(lcu_circ, qubit_order=lcu_circ.qubits)

    # Read result
    final_state_vector = result.final_state_vector
    out_dimension = lcu_circ.unitary_qubits
    ## The output state is defined by the zeros state on coefficient qubits
    ## we only care about the amplitudes in "00..00 b0" to "00..00 b2**out_dimension"
    output_state = final_state_vector[0:2**out_dimension]
    ## the success prob get right output is scaled by sum of coefficient
    sum_coeff = sum(lcu_coefficients)
    output_state = sum_coeff*output_state
    output_state = output_state/np.sqrt(output_state.conjugate()@output_state)


    # Check if we output the correct state or not
    # We compare with classical calculation of exp(iHt) with the same truncated_degree
    classical_state = classical_output(H, time=simulation_time, truncated_degree=truncated_degree)
    print("================RESULT===============")
    print("[INFO] Quantum-run output state:",output_state)
    print("[INFO] Classical-run output state:", classical_state)
    try: 
        np.testing.assert_allclose(output_state, classical_state, atol=1e-6)
        print("Matched")
    except:
        print(ValueError("Mismatch between quantum output and classical output!"))

if __name__ == '__main__':
    main()