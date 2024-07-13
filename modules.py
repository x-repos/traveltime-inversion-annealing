import numpy as np # type: ignore
from dwave.system import DWaveSampler, EmbeddingComposite # type: ignore
"""
Dwave sampler is an object that represents the quantum annealer itself. It is responsible for directly interacting with the quantum hardware.
When you create an instance of DWaveSampler, you are preparing to submit your optimization problem to a D-Wave quantum annealer for solution.

EmbeddingComposite is a wrapper around samplers that automatically handles the problem of embedding. Embedding is the process of mapping your
problem from its logical representation to the physical representation.

"""

def dict_to_vector_auto(q_dict):
    """
    Convert a solution dictionary from a QUBO problem into a vector representation.
    
    Args:
        q_dict (dict): The solution dictionary where keys are variable indices and values are the binary values.
    
    Returns:
        np.ndarray: The vector representation of the solution.
    """
    # Determine the size of the vector (n) based on the highest index in q_dict
    n = max(q_dict.keys()) + 1
    
    # Initialize a vector of size n with zeros
    q_vector = np.zeros(n, dtype=int)
    
    # Populate the vector with the values from the solution dictionary
    for i, value in q_dict.items():
        q_vector[i] = value
    
    return q_vector


def qubo_to_matrix_auto(Q):
    """
    Convert a QUBO problem defined by a dictionary into a matrix representation,
    automatically determining the matrix size.
    
    Args:
        Q (dict): The QUBO problem coefficients, where keys are tuples representing interactions between variables,
                  and values are the weights of these interactions.
    
    Returns:
        np.ndarray: The matrix representation of the QUBO problem.
    """
    # Determine the size of the matrix (n) based on the highest index in Q
    n = max(max(pair) for pair in Q.keys()) + 1
    
    # Initialize an nxn matrix with zeros
    Q_matrix = np.zeros((n, n))
    
    # Populate the matrix with the values from the QUBO dictionary
    for (i, j), value in Q.items():
        Q_matrix[i, j] = value
        if i != j:  # Ensure the matrix is symmetric
            Q_matrix[j, i] = value
    
    return Q_matrix


def binary_least_squares_qubo(A, b):
    """
    Transforms the binary least squares problem into a QUBO problem.
    Args:
        A (np.ndarray): The matrix A in the binary least squares problem.
        b (np.ndarray): The vector b in the binary least squares problem.
    Returns:
        dict: The QUBO coefficients.
    """
    # Number of variables
    n = A.shape[1]

    # Compute A^T * A and A^T * b
    ATA = A.T @ A
    ATb = A.T @ b

    # Initialize the QUBO dictionary
    Q = {}

    # Fill the QUBO coefficients
    for i in range(n):
        for j in range(i, n):
            if i == j:  # Diagonal entries
                Q[(i, i)] = ATA[i, i] - 2 * ATb[i]
            else:  # Off-diagonal entries
                Q[(i, j)] = 2 * ATA[i, j]

    return Q
def calculate_objective_value(A, b, q):
    """
    Calculates the objective value ||Aq - b||^2 for a given binary vector q.

    Args:
        A (np.ndarray): The matrix A.
        b (np.ndarray): The vector b.
        q (np.ndarray): The binary vector q for which to calculate the objective value.

    Returns:
        float: The calculated objective value.
    """
    # Ensure q is a numpy array for matrix operations
    q = np.array(q)
    
    # Calculate the difference between Aq and b
    diff = np.dot(A, q) - b
    
    # Calculate the objective value as the square of the L2 norm of the difference
    objective_value = np.dot(diff, diff)
    
    return objective_value


def real2binary(x, n_bits, j0):
    """
    Discretize a real-valued vector x using b-bit fixed point approximation.
    
    Args:
    x (np.ndarray): The real-valued vector to be discretized.
    b (int): The number of bits for the fixed point binary representation.
    j0 (int): The position of the fixed point, with 0 being the least significant bit.
    
    Returns:
    np.ndarray: The binary vector representing the discretized version of x.
    """
    # Scale x by 2^(b-j0) to move the fixed point
    scaled_x = x * (2 ** (n_bits - j0))
    # Round the scaled values to the nearest integer
    int_x = np.rint(scaled_x).astype(int)
    # Ensure the integer values are within the allowed range for the b-bit representation
    max_int = 2 ** n_bits - 1
    int_x_clipped = np.clip(int_x, 0, max_int)
    # Convert the integer values to binary strings, remove the '0b' prefix, and zero-pad to b bits
    binary_x = [format(val, f'0{n_bits}b') for val in int_x_clipped]
    # Flatten the binary strings into a single binary vector
    binary_vector = np.array([int(bit) for bin_str in binary_x for bit in bin_str])
    
    return binary_vector

def binary2real(binary_vector, n_bits, j0):
    """
    Convert a binary vector to a real number using n-bit fixed point representation.

    Args:
        binary_vector (np.ndarray): The binary vector to be converted.
        n_bits (int): The number of bits used for the fixed point representation.
        j0 (int): The position of the fixed point.

    Returns:
        np.ndarray: The real number representation of the binary vector.
    """
    # Determine the number of elements in the real number vector

    num_elements = len(binary_vector) // n_bits
    
    # Initialize the real number vector
    real_vector = np.zeros(num_elements)
    
    # Extract the real number from the binary representation
    for i in range(num_elements):
        # Extract the binary representation of the current real number
        binary_repr = binary_vector[i * n_bits:(i + 1) * n_bits]
        
        # Convert binary to integer, taking into account the fixed point position
        int_value = 0
        for bit_pos, bit in enumerate(reversed(binary_repr)):
            int_value += bit * (2 ** (bit_pos - j0))
        
        # Add the integer value to the real number vector
        real_vector[i] = int_value
    
    return real_vector


# Function to construct the Ad matrix given A and n
def construct_Ad(A, n, j0):
    N, M = A.shape  # Number of rows and original number of columns in A
    Ad = np.zeros((N, M * n))  # Initialize Ad with the correct shape
    
    for i in range(M):
        for j in range(n):
            # Construct the columns of Ad based on A and the bit significance
            Ad[:, i * n + j] = A[:, i] * (2 ** (j0 - j))
    
    # The original code which follows the paper should return Ad.
    # However, after multiple tries, I come up with return Ad/2.
    return Ad/2