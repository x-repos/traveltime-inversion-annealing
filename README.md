This code is used for runnung seismic traveltime with quantum annealing

For code for qubo optimization is in modules.py

The .ipynb files is for running the inversion process with quantum annealing

Input is stored in velocity_model.npy
and results are stored in folders `results30x10`, `results30x10-noise-1`, `results30x10-noise-01`, `results30x10-noise-001`, `results30x10-noise2` with noise level of traveltime being 0, 1%, 0.01% and 2%.

D and T are the distance increment and travel time for testing.