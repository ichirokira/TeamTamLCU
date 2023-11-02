# TeamTamLCU
The LCU_Cirq is the implementation of LCU technique using [Cirq](https://github.com/quantumlib/Cirq/).

## Set up


We recommend to follow the installation guides from [pyLIQTR](https://github.com/isi-usc-edu/pyLIQTR/tree/v0.3.0). 

**It requires Python Environment of 3.8**. We recommend to use virtual environment (i.e. conda environment); however it is optional.
The code general work with python 3.8 version (the other versions have not properly tested yet)
```
# Environment create (Optional)
conda create -n name_env python=3.8
conda activate name_env
# Install Package
pip install git+https://github.com/isi-usc-edu/pyLIQTR.git@v0.3.0
```
## Documentation
LCU_Cirq documentation is avalable at [docs](./docs/)

Usage:
```python
from lcu_cirq.lcu import LCU
import numpy as np
import cirq
from cirq.contrib.svg import SVGCircuit

# Define coefficients and unitaries
coefficients = [0.7, 0.5]
unitaries = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]])]

# Construct LCU circuit
circuit = LCU(coefficients=coefficients, unitaries=unitaries)
print(circuit)
```

Output:

```
lcu0: ───Ry(0.447π)───(0)───@────Ry(-0.447π)───
                      │     │
lcu1: ────────────────U0────U1─────────────────
```

## Examples
We present an example of using our LCU circuit for Hamiltonian Simulation:
```python
# Input
H = {"XYZ": 3, "YZZ": 3, "ZXX":3}
simulation_time = 5
truncated_degree = 5

# Output
================RESULT===============
[INFO] Quantum-run output state: [ 0.10899131+0.0000000e+00j  0.        -3.0104485e-01j
 -0.10095187+0.0000000e+00j  0.        +3.2430604e-01j
 -0.62535083+0.0000000e+00j  0.        -2.3676381e-09j
 -0.6253508 +0.0000000e+00j  0.        -2.3676381e-09j]
[INFO] Classical-run output state: [ 0.1089913 +0.j          0.        -0.30104481j -0.10095187+0.j
  0.        +0.32430599j -0.62535081+0.j          0.        +0.j
 -0.62535081+0.j          0.        +0.j        ]
Matched
```

Please feel free to change the input and run the following command:

(**Window**)
```bash
python .\samples\hamiltonian_simulation.py
```
(**Linux and Mac**)
```bash
python ./samples/hamiltonian_simulation.py
```

## Testing
We present different examples for our functionalities:

- `test_state_preparation.py`  
- `test_unitaries.py`
- `test_lcu.py` 

Please uses folowing command to run our tests:

(**Window**)
```bash
python -m unittest discover .\tests\
```
(**Linux and Mac**)
```bash
python -m unittest discover ./tests/
```

Expected Output
```
..............
----------------------------------------------------------------------
Ran 14 tests in 0.017s

OK
```

Feel free to modify the test cases in [tests](./tests/) 