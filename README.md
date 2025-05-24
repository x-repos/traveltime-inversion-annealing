# Travel Time Inversion with Quantum Annealing

This project involves running seismic travel time inversion using quantum annealing techniques.

If you find this work useful, please cite it as:

Nguyen, H. A., & Tura, A. *Seismic Traveltime Inversion with Quantum Annealing*. *Sci. Rep.* **15**, 17984 (2025). [https://doi.org/10.1038/s41598-025-01188-8](https://doi.org/10.1038/s41598-025-01188-8)

## Project Structure

- `modules`: Contains core code for QUBO (Quadratic Unconstrained Binary Optimization) optimization, intial calculation and plotting
- `.ipynb` files: Jupyter notebooks for running the inversion process with quantum annealing.
  - `1-gendata-class-inv.ipynb`: Generate data for the classical inverison
  - `2-2-tikh-reg-inv.ipynb`: Inversion with added noise levels using Tikhonov scheme.
  - `3-borehole30x10.ipynb`: Inversion with quantum annealing, using small boundary L
  - `4-main-development.ipynb`: Major development for the inversion with quantum annealing, using small boundary L
  - `5-main-noise-constant-L.ipynb`: Main code the inversion with quantum annealing, using large and constant boundary L

## Data and Results

- **Input Data**:
  - In the `input` folder: `D.npy` and `T.npy` represent the distance increment and travel, respectively.

- **Results**:
  - Output data is stored in the `results` folders
  - The noise levels in travel time for these results are 1%, 2%, and 5%.

## Plots

- `plots` directory and `6-plots.ipynb`: Contains output related to the inversion process.

## License

The project is licensed under MIT License.
