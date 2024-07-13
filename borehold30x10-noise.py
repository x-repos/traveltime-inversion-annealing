from dwave.system import DWaveSampler, EmbeddingComposite # type: ignore
from modules import construct_Ad, binary2real
from modules import binary_least_squares_qubo, dict_to_vector_auto
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import os
from tqdm import tqdm

# Functions
def velocity_generator(rows, cols):
    grid_size = (rows, cols)
    min_velocity = 3460
    max_velocity = 3530
    wedge_min_velocity = 3180
    wedge_max_velocity = 3220
    wedge_start_row = 10
    wedge_end_row = 20

    # Create a background velocity model with values ranging from 3460 to 3530
    velocity_model = np.linspace(min_velocity, max_velocity, rows).reshape(-1, 1)
    velocity_model = np.repeat(velocity_model, cols, axis=1)

    # Create a triangular wedge shape with vertices at (0, 10), (0, 20), and (10, 10)
    for i in range(wedge_start_row, wedge_end_row):
        row_velocity = np.linspace(wedge_min_velocity, wedge_max_velocity, wedge_end_row - wedge_start_row)[i - wedge_start_row]
        num_cols_in_wedge = wedge_end_row - i
        velocity_model[i, :num_cols_in_wedge] = row_velocity
    return velocity_model


def plot_velocity_model_with_sources_and_receivers(velocity_model, sources, receivers, cols, rows, showlines, linecolor, linewidth):
    plt.figure(figsize=(15, 7))  
    # plt.imshow(velocity_model, cmap='viridis', interpolation='nearest', origin='upper')
    plt.imshow(velocity_model, cmap='viridis', interpolation='nearest', extent=[0, cols, rows, 0], origin='upper', vmin=3175, vmax=3530)
    plt.colorbar(label='Velocity (m/s)')
    plt.title('Velocity Model')
    plt.xlabel('X')
    plt.ylabel('Y')

    # # Add numbers to the grid
    # rows, cols = velocity_model.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         plt.text(j + 0.5, i + 0.5, str(i * cols + j + 1), color='white', 
    #                  ha='center', va='center', fontsize=12, weight='bold')

    # # Plot the sources as stars with blue circles
    # for idx, source in enumerate(sources):
    #     plt.plot(source[0], source[1], marker='*', color='yellow', markersize=15)
    #     plt.text(source[0], source[1], f'{idx + 1}', color='blue', fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='circle'))

    # # Plot the receivers as dots
    # for idx, receiver in enumerate(receivers):
    #     plt.plot(receiver[0], receiver[1], marker='o', color='blue', markersize=10)
    #     plt.text(receiver[0], receiver[1], str(idx + 1), color='red', fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='circle'))
    
    # Plot lines between sources and receivers
    if showlines == True:
        for source in sources:
            for receiver in receivers:
                plt.plot([source[0], receiver[0]], [source[1], receiver[1]], color=linecolor, linestyle='--', linewidth=linewidth)

    # Plot the sources and receivers
    for idx, source in enumerate(sources):
        plt.plot(source[0], source[1], marker='o', color='red', markersize=5)

    for idx, receiver in enumerate(receivers):
        plt.plot(receiver[0], receiver[1], marker='o', color='blue', markersize=5)


def find_intersections(sources, receivers, grid_size):
    all_intersections = []
    for s_idx, source in enumerate(sources):
        for i, receiver in enumerate(receivers):
            x0, y0 = source
            x1, y1 = receiver
            dx = x1 - x0
            dy = y1 - y0

            for j in range(grid_size[0]):  # rows
                for k in range(grid_size[1]):  # columns
                    # Calculate intersections with grid lines
                    cell_x_min = k
                    cell_x_max = k + 1
                    cell_y_min = j
                    cell_y_max = j + 1
                    intersections = []

                    # Left boundary
                    if dx != 0:
                        t = (cell_x_min - x0) / dx
                        if 0 <= t <= 1:
                            y = y0 + t * dy
                            if cell_y_min <= y <= cell_y_max:
                                intersections.append((cell_x_min, y))

                    # Right boundary
                    if dx != 0:
                        t = (cell_x_max - x0) / dx
                        if 0 <= t <= 1:
                            y = y0 + t * dy
                            if cell_y_min <= y <= cell_y_max:
                                intersections.append((cell_x_max, y))

                    # Bottom boundary
                    if dy != 0:
                        t = (cell_y_min - y0) / dy
                        if 0 <= t <= 1:
                            x = x0 + t * dx
                            if cell_x_min <= x <= cell_x_max:
                                intersections.append((x, cell_y_min))

                    # Top boundary
                    if dy != 0:
                        t = (cell_y_max - y0) / dy
                        if 0 <= t <= 1:
                            x = x0 + t * dx
                            if cell_x_min <= x <= cell_x_max:
                                intersections.append((x, cell_y_max))

                    # Add entry and exit points
                    if (cell_x_min <= x0 <= cell_x_max) and (cell_y_min <= y0 <= cell_y_max):
                        intersections.append((x0, y0))
                    if (cell_x_min <= x1 <= cell_x_max) and (cell_y_min <= y1 <= cell_y_max):
                        intersections.append((x1, y1))

                    # Remove duplicate points
                    intersections = list(set(intersections))
                    # Sort intersections by distance from the source
                    intersections = sorted(intersections, key=lambda p: np.hypot(p[0] - x0, p[1] - y0))

                    # Add intersections to all_intersections
                    all_intersections.append(intersections)

    return all_intersections

# Remove duplicates from intersections
def remove_duplicate_intersections(intersections):
    seen = set()
    unique_intersections = []

    for point_pair in intersections:
        if tuple(point_pair) not in seen:
            unique_intersections.append(point_pair)
            seen.add(tuple(point_pair))
        else:
            unique_intersections.append([])

    return unique_intersections

# Calculate distances from unique intersections and store them in a NumPy array
def calculate_distances(unique_intersections, grid_size, sources, receivers):
    distances = np.zeros((grid_size[0], grid_size[1], len(receivers), len(sources)))
    idx = -1
    for s_idx, source in enumerate(sources):
        for i, receiver in enumerate(receivers):
            for j in range(grid_size[0]):
                for k in range(grid_size[1]):
                    idx = idx + 1
                    intersections = unique_intersections[idx]
                    if len(intersections) >= 2:
                        total_distance = 0
                        for idxx in range(len(intersections) - 1):
                            d = np.hypot(intersections[idxx + 1][0] - intersections[idxx][0], intersections[idxx + 1][1] - intersections[idxx][1])
                            total_distance += d
                        distances[j, k, i, s_idx] = total_distance
                    else:
                        distances[j, k, i, s_idx] = 0
    return distances

def rescale(arr, new_min, new_max):
    old_min = np.min(arr)
    old_max = np.max(arr)
    return (new_max - new_min) * (arr - old_min) / (old_max - old_min) + new_min

def device_location(n, rows, new_min, new_max, linear):
    if linear == True:
        z4 = np.linspace(new_min, new_max, n)
    else:
        n2 = int(np.ceil(n/2))
        rows2 = rows/2 - 1.3
        x = np.array([x for x in range(n2)])
        y = np.array([(i+3)**2 for i in x])
        z = (y - y[0]) * rows2 / (y[-1] - y[0])
        z1 = 15 - z + 15
        z2 = z1[::-1]
        z3 = np.concatenate((z, z2))
        z4 = rescale(z3, new_min, new_max) # Rescale array to range
    return z4


def noise_generator(size, noise_level):
    noise = (np.random.rand(size))-0.5
    noise = noise*noise_level
    return noise





# Defind the velocity model
rows, cols = 30, 10
grid_size = (rows, cols)
velocity_model = velocity_generator(rows, cols)


# Change z locations

for layer in tqdm(range(11,30), desc="Process", colour='green'):
    # -----------------------------------------------------------------
    # TODO: Part 1 - Create a part of velocity model, calculate travel time, ...
    velocity_01 = velocity_model[layer,:]
    velocity_01 = velocity_01.reshape(1, -1)

    rows_new, cols_new = 1, 10
    grid_size_new = (rows_new, cols_new)

    # location of z does not change so use the original values
    z = device_location(n=20, rows=rows, new_min=0.1, new_max=29.9, linear=False)
    z = z - layer
    sources = [(0, i) for i in z]
    receivers = [(cols, i ) for i in z]

    intersections = find_intersections(sources, receivers, grid_size=grid_size_new)
    unique_intersections = remove_duplicate_intersections(intersections)
    distances = calculate_distances(unique_intersections, grid_size=grid_size_new, sources=sources, receivers=receivers)

    # -----------------------------------------------------------------
    # TODO: Part 2 - Calculate D, T for running least square (paper)
    D = []
    T = []
    s1 = 1/velocity_01

    nreceiver = len(receivers)
    nsource = len(sources)

    for i in range(nsource):
        for j in range(nreceiver):
            D.append(distances[:,:,j,i].flatten())
            T.append(sum(sum(distances[:,:,j,i]*s1)))    
    D = np.array(D)
    Df = pd.DataFrame(D)
    s1 = s1.flatten()

    # -----------------------------------------------------------------
    # TODO: Part 3 - Create new T and new D
    T = np.array(T)
    indices_of_zero_np = np.where(T == 0)[0]

    # Add noise: it must be before remove 0 elements of T
    noise_percent = noise_generator(size=400, noise_level=0.015)
    T_noise = T + noise_percent*T

    T_new = np.delete(T_noise, indices_of_zero_np, axis=0)
    D_new = np.delete(D, indices_of_zero_np, axis=0)

    # -----------------------------------------------------------------
    # TODO: Part 4 - Create parameters, matrix for quantum annealing
    M = D_new.copy()
    I = np.ones(M.shape[1])  # Assuming M is a 2D numpy array
    R = 3
    t = T_new.copy()

    # -----------------------------------------------------------------
    # TODO: Part 5 - Annealing

    # s0: Initial guess
    # s1: Original data
    # s: iterative data
    s0 = np.ones(s1.shape)*1/3500 #--> Average velocity of the background
    A = construct_Ad(M, R, 1)
    L = max(abs(s1-s0)) + 0.05*max(abs(s1-s0))

    # fig, axs = plt.subplots(5, 2, figsize=(12, 15))
    directory = f'results30x10-noise/{layer}'
    os.makedirs(directory, exist_ok=True)

    for i in range(0, 10):
        b = (t + L*M@I - M@s0)/L

        Q = binary_least_squares_qubo(A, b)
        # Solve the QUBO using D-Wave's system
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample_qubo(Q, num_reads=100)
        # Print the best sample
        q = dict_to_vector_auto(sampleset.first.sample)
        x = binary2real(q, R, 2)
        
        # Update s
        s = s0 + L * (x - I)

        # NOTE: I do not use break condition because of the hardware limitation

        s0 = s
        L = L / 2
        filename = f'{directory}/s_{i}.txt'
        np.savetxt(filename, s)
    
    print(f'{layer}:',max(abs(1/s1-1/s)))