# Import necessary library
import numpy as np # type: ignore

# Functions

def velocity_generator(rows, cols):
    """
    Generate a 2D velocity model matrix with a background velocity gradient 
    and a triangular wedge region with a distinct velocity range.
    
    Parameters:
        rows (int): Number of rows in the velocity matrix.
        cols (int): Number of columns in the velocity matrix.
    
    Returns:
        np.ndarray: Generated velocity model.
    """
    grid_size = (rows, cols)
    min_velocity = 3460  # Minimum background velocity
    max_velocity = 3530  # Maximum background velocity
    wedge_min_velocity = 3180  # Minimum velocity within the wedge
    wedge_max_velocity = 3220  # Maximum velocity within the wedge
    wedge_start_row = 10  # Start row of the wedge
    wedge_end_row = 20  # End row of the wedge

    # Create a background velocity model with a linear gradient
    velocity_model = np.linspace(min_velocity, max_velocity, rows).reshape(-1, 1)
    velocity_model = np.repeat(velocity_model, cols, axis=1)

    # Add a triangular wedge shape to the velocity model
    for i in range(wedge_start_row, wedge_end_row):
        row_velocity = np.linspace(wedge_min_velocity, wedge_max_velocity, wedge_end_row - wedge_start_row)[i - wedge_start_row]
        num_cols_in_wedge = wedge_end_row - i
        velocity_model[i, :num_cols_in_wedge] = row_velocity

    return velocity_model

def find_intersections(sources, receivers, grid_size):
    """
    Find intersection points between source-receiver lines and grid cell boundaries.

    Parameters:
        sources (list): List of source coordinates [(x0, y0), ...].
        receivers (list): List of receiver coordinates [(x1, y1), ...].
        grid_size (tuple): Grid size as (rows, columns).

    Returns:
        list: List of intersection points for each source-receiver pair.
    """
    all_intersections = []
    for s_idx, source in enumerate(sources):
        for i, receiver in enumerate(receivers):
            x0, y0 = source
            x1, y1 = receiver
            dx = x1 - x0
            dy = y1 - y0

            for j in range(grid_size[0]):  # Loop over rows
                for k in range(grid_size[1]):  # Loop over columns
                    # Define grid cell boundaries
                    cell_x_min = k
                    cell_x_max = k + 1
                    cell_y_min = j
                    cell_y_max = j + 1
                    intersections = []

                    # Calculate intersection with the left boundary
                    if dx != 0:
                        t = (cell_x_min - x0) / dx
                        if 0 <= t <= 1:
                            y = y0 + t * dy
                            if cell_y_min <= y <= cell_y_max:
                                intersections.append((cell_x_min, y))

                    # Calculate intersection with the right boundary
                    if dx != 0:
                        t = (cell_x_max - x0) / dx
                        if 0 <= t <= 1:
                            y = y0 + t * dy
                            if cell_y_min <= y <= cell_y_max:
                                intersections.append((cell_x_max, y))

                    # Calculate intersection with the bottom boundary
                    if dy != 0:
                        t = (cell_y_min - y0) / dy
                        if 0 <= t <= 1:
                            x = x0 + t * dx
                            if cell_x_min <= x <= cell_x_max:
                                intersections.append((x, cell_y_min))

                    # Calculate intersection with the top boundary
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

# Remove duplicate intersection points
def remove_duplicate_intersections(intersections):
    """
    Remove duplicate intersection points.

    Parameters:
        intersections (list): List of intersection points.

    Returns:
        list: List of unique intersection points.
    """
    seen = set()
    unique_intersections = []

    for point_pair in intersections:
        if tuple(point_pair) not in seen:
            unique_intersections.append(point_pair)
            seen.add(tuple(point_pair))
        else:
            unique_intersections.append([])

    return unique_intersections

# Calculate distances based on intersection points
def calculate_distances(unique_intersections, grid_size, sources, receivers):
    """
    Calculate total distances along paths defined by unique intersection points.

    Parameters:
        unique_intersections (list): Unique intersection points.
        grid_size (tuple): Grid size as (rows, columns).
        sources (list): List of source coordinates.
        receivers (list): List of receiver coordinates.

    Returns:
        np.ndarray: 4D array of distances for each grid cell, receiver, and source.
    """
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
    """
    Rescale an array to a specified range.

    Parameters:
        arr (np.ndarray): Input array.
        new_min (float): New minimum value.
        new_max (float): New maximum value.

    Returns:
        np.ndarray: Rescaled array.
    """
    old_min = np.min(arr)
    old_max = np.max(arr)
    return (new_max - new_min) * (arr - old_min) / (old_max - old_min) + new_min

def device_location(n, rows, new_min, new_max, linear):
    """
    Generate device locations with linear or quadratic distribution.

    Parameters:
        n (int): Number of devices.
        rows (int): Number of rows in the grid.
        new_min (float): Minimum location value.
        new_max (float): Maximum location value.
        linear (bool): If True, generate linearly spaced locations; otherwise, use quadratic scaling.

    Returns:
        np.ndarray: Array of device locations.
    """
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
    """
    Generate random noise within a specified range.

    Parameters:
        size (int): Size of the noise array.
        noise_level (float): Magnitude of the noise.

    Returns:
        np.ndarray: Generated noise array.
    """
    noise = (np.random.rand(size)) - 0.5
    noise = noise * noise_level
    return noise
