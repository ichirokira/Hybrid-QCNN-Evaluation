from pennylane.operation import Operation, AnyWires
import torch.nn as nn
import torch.nn.functional as F
import torch
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

class StructureLayer(Operation):
    """
    This custom version is for only fixed number of 4 qubits
    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(self, weights, wires=None, do_queue=True, id=None):
        self.RY = qml.RY
        shape = qml.math.shape(weights)
        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")
        if shape[1] != len(wires):
            raise ValueError(f"Weights tensor must have second dimension of length {len(wires)}; got {shape[1]}")
        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):
        weights = self.parameters[0]
        repeat = qml.math.shape(weights)[0]
        with qml.tape.QuantumTape() as tape:
            for layer in range(repeat):
                for i in range(len(self.wires)):
                    self.RY(weights[layer][i], wires=self.wires[i: i + 1])

                qml.CNOT(wires=self.wires.subset([0, 1]))
                qml.CNOT(wires=self.wires.subset([2, 3]))
                qml.CNOT(wires=self.wires.subset([1, 2]))
        return tape

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns the shape of the weight tensor required for this template.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of qubits

        Returns:
            tuple[int]: shape
        """
        return n_layers, n_wires


class RQCNN(nn.Module):
    def __init__(self, kernel_size, depth, device, circuit_layers=1, method="random"):
        super().__init__()
        self.kernel_size = kernel_size
        self.depth = depth
        if self.kernel_size ** 2 >= self.depth:
            self.n_qubits = self.kernel_size ** 2
        else:
            self.n_qubits = self.depth
        self.circuit_layers = circuit_layers
        self.device = device

        dev = qml.device("default.qubit", wires=self.n_qubits)

        # rand_params = np.random.uniform(high=2 * np.pi, size=(self.circuit_layers, self.n_qubits))
        @qml.qnode(dev)
        def random_circuit(inputs, weights):
            for j in range(inputs.shape[0]):
                qml.RY(np.pi * inputs[j], wires=j)

            # Random quantum circuit
            RandomLayers(weights, wires=list(range(self.n_qubits)))

            # Measurement producing 4 classical output values
            return [qml.expval(qml.PauliZ(j)) for j in range(self.depth)]

        @qml.qnode(dev)
        def structure_circuit(inputs, weights):
            for j in range(inputs.shape[0]):
                qml.RY(np.pi * inputs[j], wires=j)

            StructureLayer(weights, wires=list(range(self.n_qubits)))

            # Measurement producing 4 classical output values
            return [qml.expval(qml.PauliZ(j)) for j in range(self.depth)]

        params = {'weights': (self.circuit_layers, self.n_qubits)}
        if method == "random":
            self.qlayer = qml.qnn.TorchLayer(random_circuit, params)
        elif method == "structure":
            self.qlayer = qml.qnn.TorchLayer(structure_circuit, params)

    def forward(self, x):
        q_out = torch.zeros(x.shape[0], (x.shape[3] - self.kernel_size + 1), (x.shape[3] - self.kernel_size + 1),
                            self.depth, device=self.device)

        for b in range(x.shape[0]):
            for row in range(x.shape[3] - self.kernel_size + 1):
                for col in range(x.shape[2] - self.kernel_size + 1):
                    for channel in range(x.shape[1]):
                        q_out[b, row, col] += self.qlayer(
                            self.flatten(x[b, channel, row:row + self.kernel_size, col:col + self.kernel_size]))
        return torch.reshape(q_out, (
        x.shape[0], self.depth, x.shape[3] - self.kernel_size + 1, x.shape[3] - self.kernel_size + 1))

    def flatten(self, t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t

