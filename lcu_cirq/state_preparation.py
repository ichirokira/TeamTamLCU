import cirq
import numpy as np
import math
from cirq.contrib.svg import SVGCircuit



class GroverRudolph(cirq.Circuit):
  """
    GroverRudolph: extends the cirq.Circuit class to implement Grover Rudolph state preparation based on the paper: 
    https://arxiv.org/abs/quant-ph/0208112. It is designed to prepare a quantum state whose amplitudes are square roots of given coefficients.

    Attributes:
        - num_qubits (int): The number of qubits needed for the state preparation.
        - coefficients (np.ndarray): The coefficients for the state preparation.
        - on_qubits (list of cirq.NamedQubit): The qubits on which the circuit will be applied.
                                               Fixed as None for this project.

    Methods:
        - __init__(self, coefficients, on_qubits=None): Initializes the class attributes.
        - compute_angle(self, step, previous_sum): Computes the rotation angles for Ry gates.
        - _build_circuit_(self): Builds the state preparation circuit.
        - visualise(self): Returns the SVG circuit for visualization.

    Error Handling:
        - Checks if the input coefficients are normalized to 1 using L1 norm, if not, normalize them.
        - If the length of the coefficients is not a power of 2, it pads with zeros to make it so.

  """ 
  def __init__(self, coefficients, on_qubits=None):
    """
    Specify the list of coefficients for the state and declare some paramaters
    for the construction.


    Args:
      coeffients(np.ndarray): list of coefficients for the state. If the coefficients are
                  not normalized, the coefficients will be normalized using l1 norm.
      on_qubits(cirq.NamedQubit, Optional): qubit or list of qubits that the circuit applies to. 
                                             Fixed as None for this project.
    Raises:
           - AssertionError: If coefficients are not a non-empty np.ndarray or not normalized or not non-negative.
    """
    super(GroverRudolph, self).__init__()

    # Ensure coefficients are of valid type: it is either np.ndarray or list
    assert isinstance(coefficients, (list, np.ndarray)), "Invalid coefficient type"

    if not isinstance(coefficients, np.ndarray):
      coefficients = np.array(coefficients)
    
    # Ensure not empty input
    assert len(coefficients) > 0, "Expect non-empty input"
    # Ensure all coefficients are numbers
    assert not np.issubdtype(coefficients.dtype, np.str_), "Invalid coefficient type"
    # Ensure all coefficients are non-negative
    assert np.all(coefficients >= 0), "All coefficients must be non-negative"


    # Delare number of qubits. Pad zero values if not length power of 2
    length = len(coefficients)
    self.num_qubits = math.ceil(np.log2(length))
    if not math.log2(length).is_integer():
      num_zeros_needed = 2**self.num_qubits - length

      coefficients = np.pad(coefficients, (0, num_zeros_needed), mode='constant')

    # normalizing coefficients in l1 norm
    self.coefficients = (coefficients)/sum(coefficients)
    assert np.absolute(np.sum(self.coefficients)-1) <= (10e-5), "Coefficients is not normalized"

    # Create qubits
    if on_qubits:
        self.qubits = on_qubits
        # create a test on number of qubits
        # TODO HERE
    else:
      self.qubits = [cirq.NamedQubit("i"+str(i)) for i in range(self.num_qubits)]

    # build circuit
    self._build_circuit_()

  def compute_angle(self, step, previous_sum):
    """
    compute_angle: Computes the rotation angles required for the Ry gates at a specific 
                       induction step based on the coefficients of the quantum state.

    Args:
        - step (int): The induction step, ranging from 0 to num_qubits - 1.
        - previous_sum (list of float): The sum of squared coefficients from the previous induction step.

    Returns:
        - angles (list of float): A list of rotation angles computed based on the coefficients and the induction step.
        - current_sum (list of float): The sum of squared coefficients for the current induction step.

    Description:
        The function computes rotation angles based on the previous sum of coefficients and the induction step.        
    """
    count = 0
    angles = []
    current_sum = []
    num_instances = 2**(self.num_qubits-step)//2
    for i in range(0, 2**(self.num_qubits), 2*num_instances):
      temp = sum(self.coefficients[i:i+num_instances])
      if temp == 0:
        break
      angle = np.arccos(np.around(np.sqrt(temp/previous_sum[count]), 10))
      angles.append(angle)
      current_sum.extend([temp, previous_sum[count]-temp])
      count += 1
    return angles, current_sum


  def _build_circuit_(self) -> None:
    """
    _build_circuit_: Constructs the state preparation circuit.

    Description:
        The function constructs the state preparation circuit based on the rotation angles computed.

    Returns:
        None (updates the circuit in-place).
    """

    # Perform induction steps
    sums = [1] # sum of coefficients is 1
    for step in range(len(self.qubits)):

      angles, sums = self.compute_angle(step, previous_sum=sums)

      if step == 0:
        self.append(cirq.Ry(rads=2*angles[0]).on(self.qubits[step]), strategy= cirq.InsertStrategy.NEW_THEN_INLINE)
      else:
        for j in range(len(angles)):
          self.append(cirq.Ry(rads=2*angles[j]).on(self.qubits[step]).controlled_by(*self.qubits[:step], \
                          control_values=[int(b) for b in format(j, "b").zfill(step)]), strategy= cirq.InsertStrategy.NEW_THEN_INLINE)

  def visualise(self):
    """
    visualise: Returns the SVG circuit for visualization.

    Returns:
        SVG representation of the circuit.
    """
    return SVGCircuit(self)


