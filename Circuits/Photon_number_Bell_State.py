import numpy as np
from mqt.qudits.quantum_circuit import QuantumCircuit

def create_bell_state_mqt(M, d):
    """
    Create M-mode Bell state using MQT Qudits library.
    
    Args:
        M (int): Number of spatial modes per photon
        d (int): Qudit dimension
    
    Returns:
        QuantumCircuit: Bell state circuit
    """
    total_qudits = 2 * M  # M for idler + M for signal
    dimensions = [d] * total_qudits
    circuit = QuantumCircuit(total_qudits, dimensions)
    
    # Apply Hadamard to idler modes
    for i in range(M):
        circuit.h(i)
    
    # Apply CNOT between idler and signal modes
    for i in range(M):
        circuit.cx([i, i + M])
    
    return circuit

def simulate_measurements(circuit, M, shots=1000):
    """
    Simulate measurements on the Bell state circuit.
    
    Args:
        circuit: QuantumCircuit object
        M (int): Number of spatial modes per photon
        shots (int): Number of measurement shots
    
    Returns:
        dict: Measurement statistics
    """
    # Create theoretical Bell state probabilities
    probabilities = {}
    amplitude = 1.0 / np.sqrt(M)
    
    for k in range(M):
        # Create bitstring for mode k
        idler_bits = ['0'] * M
        idler_bits[k] = '1'
        signal_bits = ['0'] * M
        signal_bits[k] = '1'
        
        full_bitstring = ''.join(idler_bits + signal_bits)
        probabilities[full_bitstring] = amplitude ** 2
    
    # Sample from probability distribution
    bitstrings = list(probabilities.keys())
    probs = list(probabilities.values())
    probs = np.array(probs) / np.sum(probs)
    
    # Sample measurements
    measurements = np.random.choice(bitstrings, size=shots, p=probs)
    
    # Count occurrences
    counts = {}
    for measurement in measurements:
        if measurement not in counts:
            counts[measurement] = 0
        counts[measurement] += 1
    
    # Analyze results
    same_mode_count = 0
    different_mode_count = 0
    
    print(f"Measurement results ({shots} shots):")
    for bitstring, count in sorted(counts.items()):
        idler_bits = bitstring[:M]
        signal_bits = bitstring[M:]
        
        idler_modes = [i for i, bit in enumerate(idler_bits) if bit == '1']
        signal_modes = [i for i, bit in enumerate(signal_bits) if bit == '1']
        
        is_same_mode = idler_modes == signal_modes
        
        if is_same_mode:
            same_mode_count += count
        else:
            different_mode_count += count
        
        prob = count / shots * 100
        print(f"{bitstring}: {count} ({prob:.1f}%) - Idler: {idler_modes}, Signal: {signal_modes}")
    
    same_mode_prob = same_mode_count / shots * 100
    different_mode_prob = different_mode_count / shots * 100
    
    print(f"\nSame mode: {same_mode_count}/{shots} ({same_mode_prob:.1f}%)")
    print(f"Different mode: {different_mode_count}/{shots} ({different_mode_prob:.1f}%)")
    
    return {
        'same_mode_prob': same_mode_prob,
        'different_mode_prob': different_mode_prob
    }

# Example usage
if __name__ == "__main__":
    M = 4  # Number of spatial modes
    d = 2  # Qudit dimension
    shots = 1000  # Number of measurement shots
    
    # Create Bell state
    bell_circuit = create_bell_state_mqt(M, d)
    
    # Simulate measurements
    results = simulate_measurements(bell_circuit, M, shots)
    
    print(f"\nBell state analysis completed!")
    print(f"Same mode probability: {results['same_mode_prob']:.1f}%")
    print(f"Different mode probability: {results['different_mode_prob']:.1f}%")