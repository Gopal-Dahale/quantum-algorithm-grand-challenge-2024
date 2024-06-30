from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pprint import pprint
import numpy as np
from state_utils import *
from quri_parts.circuit import X, RX, CNOT, RY, RZ, H, Y, Z, CZ
from quri_parts.circuit import QuantumCircuit as QuriQC
import pennylane as qml
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit


def get_initial_state(n_qubits, ham):
    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SGF, n_threads=8)
    driver.initialize_system(n_sites=n_qubits, n_elec=n_qubits // 2, heis_twos=1)
    b = driver.expr_builder()
    for key, val in ham.terms.items():
        key = np.array(key)
        idxs = key[:, 0]
        term_string = ""
        for i in key[:, 1]:
            if i == 1:
                term_string += "C"
            else:
                term_string += "D"
        b.add_term(term_string, idxs, val.real)
    mpo = driver.get_mpo(b.finalize(), iprint=0)
    ket = driver.get_random_mps(tag="GS")

    # execute DMRG by modifying the ket state in-place to minimize the energy
    driver.dmrg(
        mpo,
        ket,
        n_sweeps=30,
        bond_dims=[100, 200],
        noises=[1e-3, 1e-5],
        thrds=[1e-6, 1e-7],
        tol=1e-6,
    )

    # post-process the MPS to get an initial state
    dets, coeffs = driver.get_csf_coefficients(ket, iprint=0)
    solver = (dets, coeffs)

    wf_dict = qml.qchem.convert._dmrg_state(solver, tol=1e-1)
    pprint(wf_dict)
    return wf_dict


def get_initial_state_circuit(n_qubits, ham):
    wf_dict = get_initial_state(n_qubits, ham)

    sparse_states_dict = {}
    for (int_a, int_b), coeff in wf_dict.items():
        bin_a = bin(int_a)[2:][::-1]
        bin_a += "0" * (n_qubits - len(bin_a))
        bin_a = bin_a[::-1]
        sparse_states_dict[bin_a] = coeff

    norm = np.linalg.norm(np.array(list(sparse_states_dict.values())))
    sparse_states_dict = {i: j / norm for i, j in sparse_states_dict.items()}

    NUM_QUBITS = len(next(iter(sparse_states_dict)))

    sparse_states = dict_to_3d_array(sparse_states_dict)

    ops = []
    gates = []  # Stores operations
    x_qubits = []
    cx_qubits = []
    cg_params = []
    final_state = []
    max_num_ctrls = [0]
    anc = []

    if len(sparse_states) > 1:
        algorithm_1(
            sparse_states,
            NUM_QUBITS,
            gates,
            x_qubits,
            cx_qubits,
            cg_params,
            final_state,
            max_num_ctrls,
        )

        if max_num_ctrls[0] > 0:
            anc = [NUM_QUBITS + i for i in range(max_num_ctrls[0])]

        # Reversed order of operations from Alg. (1) in Gleinig & Hoefler paper
        for gate in gates[::-1]:
            if gate == "x":
                ops.append(X(x_qubits.pop()))
            elif gate == "cx":
                c, t = cx_qubits.pop()
                ops.append(CNOT(c, t))
            elif gate == "cg":
                cg = cg_params.pop()
                theta = cg[0]
                phi = cg[1]

                if len(cg) == 3:
                    # apply G to psi[dif]
                    ops += my_unitary(theta, phi, NUM_QUBITS - 1 - cg[2])
                else:
                    # apply G to psi[dif] controlled on qubits dif_qs
                    for i, d_q in enumerate(cg[2]):
                        ops.append(CNOT(NUM_QUBITS - 1 - d_q, anc[i]))

                    ops += my_controlled_unitary(
                        theta, phi, anc[0 : len(cg[2])], NUM_QUBITS - 1 - cg[3]
                    )

                    for i, d_q in enumerate(cg[2]):
                        ops.append(CNOT(NUM_QUBITS - 1 - d_q, anc[i]))

            # NOT any remaining non-zero gates
            elif gate == "end":
                for b in range(NUM_QUBITS):
                    if final_state[0][b] == 1:
                        ops.append(X(NUM_QUBITS - 1 - b))
    # NOT any non-zero gates
    else:
        for b in range(NUM_QUBITS):
            if sparse_states[0][0][b] == 1:
                ops.append(X(NUM_QUBITS - 1 - b))

    print("Number of ancilla qubits:", len(anc))

    circuit = LinearMappedUnboundParametricQuantumCircuit(NUM_QUBITS + len(anc))
    for gate in ops:
        circuit.add_gate(gate)
    return circuit, len(anc)
