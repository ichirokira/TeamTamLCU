import cirq
import numpy as np
import math
from cirq.contrib.svg import SVGCircuit


class GroverRudolph(cirq.Circuit):
  """
  Implement Grover Rudolph state preparation based on the paper: https://arxiv.org/abs/quant-ph/0208112
  Output the circuit prepare the state which amplitudes are square root of input coefficients.
  """
  def __init__(self, coefficients, on_qubits=None):
    """
    Specify the list of coefficients for the state and declare some paramaters
    for the construction.


    Args:
      coeffients: list of coefficients for the state. If the coefficients are
                  not normalized, the coefficients will be normalized using l1 norm.
      on_qubits: (Optional) qubit or list of qubits that the circuit applies to
    """
    super(GroverRudolph, self).__init__()


    if not isinstance(coefficients, np.ndarray):
      coefficients = np.array(coefficients)

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
    Compute the rotation angle for based on the induction step

    Args:
      step: 'int' induction step from 0 to num_qubits-1

    Returns:
      angle(s) corresponding to the induction step
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
    Construct the state preparation circuit

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
    """Get the SVG circuit (for visualization)"""
    return SVGCircuit(self)


