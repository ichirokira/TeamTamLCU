import cirq
import numpy as np
import math
from cirq.contrib.svg import SVGCircuit
from lcu_cirq.state_preparation import GroverRudolph
from lcu_cirq.helpers import *

class LCU(cirq.Circuit):
    """
      LCU: The LCU class extends cirq.Circuit to construct a quantum circuit for implementing the Linear Combination of Unitaries (LCU) algorithm. 
      The class allows you to input coefficients and corresponding unitary matrices to form the state preparation and unitary operations.

      Args:
          coefficients (np.ndarray): list of coefficients in the combination
          unitaries (list of np.ndarray): list of unitaries corresponding to the `coefficients`
                      in the combination.
          on_qubits(list of cirq.NamedQubit, Optional): list of qubits that the circuit
                    acting on. If it is None, a new qubit register is created
          prepare_orcacle(cirq.Circuit, Optional): circuit for state prepration oracle.
                          If it is not defined, the GroverRudolph technique will be used.

      Methods:
          prepare(): prepares the state preparation circuit based on input coefficients
          selection(): prepares the control-unitary gates based on position of the unitaries
          build_circuit(): build full LCU quantum circuit from the `prepare` and 'select' subroutines
    """
    def __init__(self, coefficients, unitaries, on_qubits=None, prepare_oracle=None):
        """
        Args:
            -coefficients (np.ndarray): list of coefficients in the combination
            -unitaries (list of np.ndarray): list of unitaries corresponding to the `coefficients`
                  in the combination.
            -on_qubits(list of cirq.NamedQubit, Optional): list of qubits that the circuit
                acting on. Defaults to None.
            -prepare_orcacle(cirq.Circuit, Optional): State preparation oracle. Defaults to None.

        Raises:
            - AssertionError: if the length of 'coefficients' and 'unitaries' are not the same.
            - AssertionError: if the unitaries are not all of the same size.
            - AssertionError: if the size of the unitaries is a power of 2.
        """
        super(LCU, self).__init__()
        # check inputs
        assert len(coefficients) == len(unitaries), "Expect `coefficients` and  `unitaries` has same length"
        assert np.all([(unitaries[i].shape[0]) == (unitaries[0].shape[0]) for i in range(len(unitaries))]), \
        "All unitary should be in same size"
        assert math.log2(unitaries[0].shape[0]).is_integer(), \
        "The input unitary should have size of power of 2"

        # Preprocessing inputs
        for i in range(len(coefficients)):
          # coefficients must by positive
          if coefficients[i] < 0:
            unitaries[i] = -unitaries[i]
            coefficients[i] = -coefficients[i]

        self.prepare_oracle = prepare_oracle


        self.unitary_qubits = int(np.log2(unitaries[0].shape[0]))
        self.coefficients_qubits = int(math.ceil(np.log2(len(coefficients))))
        self.coefficients = coefficients
        self.unitaries = unitaries

        # Create qubits
        if on_qubits:
            self.qubits = [cirq.NamedQubit('coeff'+str(i)) for i in range(self.coefficients_qubits)] + on_qubits
            # create a test on number of qubits
            # TODO HERE
        else:
            self.qubits = [cirq.NamedQubit('lcu'+str(i)) for i in range(self.coefficients_qubits+self.unitary_qubits)]
        # Build circuit
        self._build_circuit_()
    def prepare(self) -> cirq.Circuit:
        """
        prepare: Creates a cirq.Circuit that appends controlled-unitary gates based on the position of the unitaries in the list.
        """
        if self.prepare_oracle:
            return self.prepare_oracle
        else:
            return GroverRudolph(self.coefficients, on_qubits=self.qubits[:self.coefficients_qubits])

    def select(self) -> cirq.Circuit:
        """
        select: Creates a cirq.Circuit that appends controlled-unitary gates based on the position of the unitaries in the list.

        """
        circuit = cirq.Circuit()

        for i in range(len(self.coefficients)):
          # Get position of unitary
          binarized_i = format(i, "b").zfill(self.coefficients_qubits)
          # create cirq gate from input unitary
          U_gate = Unitary(self.unitaries[i], name = "U{}".format(i))
          # controlled the gate by the corresponding position
          circuit.append(U_gate.on(*self.qubits[self.coefficients_qubits:]).controlled_by(
              *self.qubits[:self.coefficients_qubits], \
                            control_values=[int(b) for b in binarized_i]
          ), strategy= cirq.InsertStrategy.NEW_THEN_INLINE)

        return circuit

    def _build_circuit_(self) -> None:
        """
        Builds the full LCU circuit by appending the circuits returned from "prepare" and "select"
        """

        prepare_circuit = self.prepare()
        select_circuit = self.select()
        # add state preparation circuit
        self.append(prepare_circuit, strategy= cirq.InsertStrategy.NEW_THEN_INLINE)
        # apply control-unitary gates
        self.append(select_circuit, strategy= cirq.InsertStrategy.NEW_THEN_INLINE)
        # apply inverse of state preparation circuit
        self.append(cirq.inverse(prepare_circuit), strategy= cirq.InsertStrategy.NEW_THEN_INLINE)

    def visualise(self):
        """Get the SVG circuit (for visualization)"""
        return SVGCircuit(self)