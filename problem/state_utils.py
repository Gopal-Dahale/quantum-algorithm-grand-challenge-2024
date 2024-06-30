import numpy as np
from quri_parts.circuit import X, RX, CNOT, RY, RZ, H, Y, Z, CZ
from qiskit import QuantumCircuit, transpile


def decompose_mcx(qubits):
    qiskit_to_quri_wire_map = {i: j for i, j in enumerate(qubits)}
    n = len(qubits)

    qc = QuantumCircuit(n)
    temp_wires = list(range(n))
    qc.mcx(temp_wires[:-1], temp_wires[-1])
    tqc = transpile(
        qc,
        basis_gates=["rx", "ry", "rz", "x", "y", "z", "h", "cx", "cz"],
        optimization_level=3,
    )

    qc_wires = [hash(q) for q in tqc.qubits]
    wire_map = dict(zip(qc_wires, range(len(qc_wires))))

    gates = []
    for instruction, qargs, _ in tqc.data:
        instr_name = getattr(instruction, "base_class", instruction.__class__).__name__
        operation_wires = [wire_map[hash(qubit)] for qubit in qargs]
        operation_wires = [qiskit_to_quri_wire_map[wire] for wire in operation_wires]

        if instr_name == "RZGate":
            gates.append(RZ(operation_wires[0], instruction.params[0]))

        elif instr_name == "RYGate":
            gates.append(RY(operation_wires[0], instruction.params[0]))

        elif instr_name == "RXGate":
            gates.append(RX(operation_wires[0], instruction.params[0]))

        elif instr_name == "CXGate":
            gates.append(CNOT(operation_wires[0], operation_wires[1]))

        elif instr_name == "CZGate":
            gates.append(CZ(operation_wires[0], operation_wires[1]))

        elif instr_name == "XGate":
            gates.append(X(operation_wires[0]))

        elif instr_name == "YGate":
            gates.append(Y(operation_wires[0]))

        elif instr_name == "ZGate":
            gates.append(Z(operation_wires[0]))

        elif instr_name == "HGate":
            gates.append(H(operation_wires[0]))

    return gates


def dict_to_3d_array(sparse_states):
    array_3d = []

    for key, value in sparse_states.items():
        point = [int(digit) for digit in key]
        array_3d.append([point, value])

    return array_3d


def custom_filter(func, iterable):
    return [item for item in iterable if func(item)]


def unequal_sets(t, n):

    best_qubit = None
    T_0 = []
    T_1 = []
    current_difference = float("-inf")

    for b in range(n):
        # Filter list based on boolean condition
        T_0 = custom_filter(lambda x, b=b: x[0][b] == 0, t)
        T_1 = custom_filter(lambda x, b=b: x[0][b] == 1, t)

        # Check if both sets are non-empty
        if len(T_0) != 0 and len(T_1) != 0:
            difference = abs(len(T_0) - len(T_1))
            # If new max difference
            if difference > current_difference:
                current_difference = difference
                best_qubit = b
                t_0 = T_0
                t_1 = T_1

    return best_qubit, t_0, t_1


def process_subsets(t, n, dif_qubits, dif_values):
    while len(t) > 1:
        b, T_0, T_1 = unequal_sets(t, n)
        dif_qubits.append(b)
        if len(T_0) < len(T_1):
            t = T_0
            dif_values.append(0)
        else:
            t = T_1
            dif_values.append(1)
    return dif_qubits, dif_values, t


def toggle_operations(index, n, x_x, gates, x_qubits, s):
    if x_x[0][index] != 1:  # Identical code
        gates += ["x"]
        x_qubits += [n - 1 - index]
        for x in s:
            x[0][index] = int(not x[0][index])


def conditional_toggle(gates, x_qubits, n, dif, b, s):
    gates += ["cx"]
    sx = [n - 1 - dif, n - 1 - b]
    x_qubits += [sx]
    for x in s:
        if x[0][dif] == 1:
            x[0][b] = int(not x[0][b])


def calc_theta_phi(x_1, x_2):
    # Know coefficients alpha, beta, return theta, phi
    alpha = x_2[1]
    beta = x_1[1]
    print(f"alpha: {alpha}")
    print(f"beta: {beta}")

    theta = np.arctan2(np.abs(alpha), np.abs(beta))
    phi = np.angle(alpha) - np.angle(beta)

    x_2[1] = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2) * np.exp(
        1j * np.angle(alpha)
    )

    print(f"theta: {theta}")
    print(f"phi: {phi}")

    return theta, phi


def my_unitary(theta, phi, target):
    return [
        RZ(target, phi),
        RY(target, theta),
        X(target),
        RY(target, -theta),
        RZ(target, -phi),
    ]


def my_controlled_unitary(theta, phi, ctrl, target):
    res = [RZ(target, phi), RY(target, theta)]
    res += decompose_mcx(list(set(ctrl)) + [target])
    res += [RY(target, -theta), RZ(target, -phi)]
    return res


def algorithm_1(
    s, n, gates, x_qubits, cx_qubits, cg_params, final_state, max_num_ctrls
):
    dif_qubits = []  # Where to operate
    dif_values = []  # What operation

    T = s

    dif_qubits, dif_values, t = process_subsets(T, n, dif_qubits, dif_values)

    dif = dif_qubits.pop()
    dif_values.pop()

    x_1 = t[0]
    t_prime = [
        x for x in s if all(x[0][q] == v for q, v in zip(dif_qubits, dif_values))
    ]
    t_prime.remove(x_1)

    dif_qubits, dif_values, t_prime = process_subsets(
        t_prime, n, dif_qubits, dif_values
    )

    x_2 = t_prime[0]

    toggle_operations(dif, n, x_1, gates, x_qubits, s)

    for b in range(n):
        if b != dif and x_1[0][b] != x_2[0][b]:
            conditional_toggle(gates, cx_qubits, n, dif, b, s)

    for b in dif_qubits:
        toggle_operations(b, n, x_2, gates, x_qubits, s)

    theta, phi = calc_theta_phi(x_1, x_2)

    gates += ["cg"]
    cg_param = [theta, phi, dif_qubits, dif]

    if len(dif_qubits) > 0:
        if len(dif_qubits) >= max_num_ctrls[0]:
            max_num_ctrls[0] = len(dif_qubits)
    else:
        cg_param.remove(dif_qubits)

    cg_params += [cg_param]
    s.remove(x_1)

    if len(s) > 1:
        algorithm_1(
            s, n, gates, x_qubits, cx_qubits, cg_params, final_state, max_num_ctrls
        )
    else:
        gates += ["end"]
        final_state += [x_2[0]]
