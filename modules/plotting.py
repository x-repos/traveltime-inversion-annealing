from matplotlib import pyplot as plt

def plot_velocity_model_with_sources_and_receivers(velocity_model, sources, receivers, cols, rows, showlines, linecolor, linewidth):
    plt.figure(figsize=(15, 7))  # Set the size of the figure to 15x7 inches
    
    # Plot the velocity model using imshow with specified colormap, interpolation, and value limits
    # 'extent' sets the axes ranges, 'origin' ensures the correct orientation, and 'vmin'/'vmax' define the color scale limits
    plt.imshow(velocity_model, cmap='viridis', interpolation='nearest', extent=[0, cols, rows, 0], origin='upper', vmin=3175, vmax=3530)
    plt.colorbar(label='Velocity (m/s)')  # Add a color bar to show the velocity scale
    plt.title('Velocity Model')  # Set the title of the plot
    plt.xlabel('X')  # Label the x-axis
    plt.ylabel('Y')  # Label the y-axis

    # If showlines is True, draw dashed lines connecting each source to each receiver
    if showlines == True:
        for source in sources:  # Iterate over each source
            for receiver in receivers:  # Iterate over each receiver
                # Plot a dashed line between the source and receiver with the specified color and width
                plt.plot([source[0], receiver[0]], [source[1], receiver[1]], color=linecolor, linestyle='--', linewidth=linewidth)

    # Plot the sources as red circles with a specific size
    for idx, source in enumerate(sources):
        plt.plot(source[0], source[1], marker='o', color='red', markersize=5)

    # Plot the receivers as blue circles with a specific size
    for idx, receiver in enumerate(receivers):
        plt.plot(receiver[0], receiver[1], marker='o', color='blue', markersize=5)
