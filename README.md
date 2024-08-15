# Travel Time Inversion with Quantum Annealing

This project involves running seismic travel time inversion using quantum annealing techniques.

## Project Structure

- `modules.py`: Contains core code for QUBO (Quadratic Unconstrained Binary Optimization) optimization.
- `.ipynb` files: Jupyter notebooks for running the inversion process with quantum annealing.
  - `1-borehole30x10.ipynb`: Inversion without noise.
  - `2-borehole30x10-noise.ipynb`: Inversion with added noise levels.
  - Other notebooks follow similar processes with varying noise levels and configurations.

## Data and Results

- **Input Data**:
  - `D.npy` and `T.npy` represent the distance increment and travel, respectively.

- **Results**:
  - Output data is stored in the following directories:
    - `results30x10`
    - `results30x10-noise-1`
    - `results30x10-noise-01`
    - `results30x10-noise-001`
    - `results30x10-noise-2`
  - The noise levels in travel time for these results are 0%, 0.01%, and 2%.

## Plots

- `plots` directory and `plot.ipynb`: Contains output related to the inversion process.

## License

The project is licensed under MIT License.
