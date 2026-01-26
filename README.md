# Finite Temperature Tensor Network Simulations

This project implements finite temperature quantum simulations using the purification approach with imaginary time evolution (TEBD) within the TeNPy (Tensor Network Python) library.

## Overview

The project simulates two quantum lattice models at finite temperature:
1. **Kitaev Ladder Model** - A 1D ladder with Kitaev-type (XX, YY, ZZ) interactions and magnetic field
2. **J1J2 Triangular Lattice** - A 2D triangular lattice with nearest-neighbor (J1) and next-nearest-neighbor (J2) Heisenberg interactions

Using the purification method, we compute thermal density matrices and measure various observables as functions of temperature.

## Key Features

### Simulations
- **Dynamic data saving** - Data is saved after each temperature step to prevent loss on cluster job interruption
- **Automated site calculations** - Flux operator measurement sites are computed automatically based on system size
- **Command-line interface** - All simulation parameters can be set via CLI arguments
- **Comprehensive observables** - Measures magnetization (Sx, Sy, Sz), plaquette/flux operators (S1, S2, Wp), and specific heat (Cv)

### Analysis & Visualization
- **Contour plots** - Phase diagrams showing observables vs (h, T) or (h, J2, T)
- **Line cuts** - Temperature-dependent observables for fixed parameters
- **Multiple chi values** - Support for different bond dimensions with automatic file organization
- **Formatted output** - ASCII text files with detailed headers and organized data

## Project Structure

```
PurificationCodes/
├── models/                          # Model definitions
│   ├── __init__.py
│   ├── model_Kladder.py            # Kitaev Ladder model
│   └── model_J1J2.py               # J1J2 triangular lattice model
│
├── finite_T_Kladder.py             # Main simulation for Kitaev Ladder
├── finite_T_J1J2.py                # Main simulation for J1J2 model
│
├── mpo_fluxes_Kladder.py           # Utility for flux operator calculations
├── energy_dynamics_Kladder.py       # Energy dynamics analysis
├── scan_finite_T_Kladder.py        # Parameter scanning script
│
├── plotContours.py                 # Phase diagram contours (Cv, fluxes, magnetization)
├── plotCvCutsLadder.py            # Temperature cuts for Kladder at different h
├── plotCvCutsJ1J2.py              # Temperature cuts for J1J2 at different chi
│
├── clusterData/
│   ├── Kladder/
│   │   ├── chi100/
│   │   ├── chi200/
│   │   └── chi300/
│   └── J1J2/
│
├── Data_AFM_Kitaev_Ladder/         # Local simulation data
├── Data_FM_Kitaev_Ladder/
├── Data_J1J2/
└── ground_states/                  # Ground state MPS files
```

## Usage

### Running Simulations

#### Kitaev Ladder (1D)
```bash
python finite_T_Kladder.py \
  --L 23 \
  --beta_max 120.0 \
  --dt 0.025 \
  --h 0.5 \
  --chi_max 100
```

**Arguments:**
- `L`: Ladder length (number of rungs)
- `beta_max`: Maximum inverse temperature (β = 1/T)
- `dt`: Time step for imaginary time evolution
- `h`: Magnetic field strength
- `chi_max`: Maximum bond dimension for truncation

#### J1J2 Model (2D)
```bash
python finite_T_J1J2.py \
  --Lx 6 --Ly 6 \
  --beta_max 100.0 \
  --dt 0.025 \
  --J1 1.0 --J2 0.125 \
  --Fz 0.0 \
  --chi_max 50 \
  --conserve Sz
```

**Arguments:**
- `Lx`, `Ly`: Lattice dimensions
- `beta_max`: Maximum inverse temperature
- `dt`: Time step
- `J1`, `J2`: Coupling strengths
- `Fz`: Magnetic field strength
- `chi_max`: Maximum bond dimension
- `conserve`: Quantum number to conserve ('Sz' or 'None')

### Plotting

#### Phase Diagrams (Contours)
```bash
python plotContours.py
# Generates: contour_Cv.pdf, contour_fluxes.pdf, contour_magnetization.pdf
# Data range: h ∈ [0.05, 1.10], T ∈ [T_min, T_max]
```

**Outputs:**
1. **contour_Cv.pdf** - Cv and Cv/T as functions of (h, T)
2. **contour_fluxes.pdf** - Plaquette operators S1, S2, Wp as functions of (h, T)
3. **contour_magnetization.pdf** - Magnetization Sx, Sy, Sz as functions of (h, T)

#### Temperature Cuts (Ladder)
```bash
python plotCvCutsLadder.py
# Generates grid: Cv_cuts_grid.pdf
# Shows Cv and Cv/T vs T for each h value (4 columns × 5 rows)
```

#### Temperature Cuts (J1J2)
```bash
# Edit line 7 to set chi_max value
chi_max = 50  # or 100, 200, etc.

python plotCvCutsJ1J2.py
# Generates: Cv_vs_T_chi50.pdf, Sigmaz_vs_T_chi50.pdf
# Shows multiple beta curves on same plot
```

## Data Format

### Output Files
Data files are saved with comprehensive naming:
```
Data_AFM_Kitaev_Ladder/finite_T_data_L23_beta120.0_dt0.0250_chi100_h0.50.txt
```

### File Structure
```
# Finite Temperature Data for Kitaev Ladder
# chi_max = 100, h = 0.5000
# Columns: T | Sigmax | Sigmay | Sigmaz | S1 | S2 | Wp | Cv

2.000000e+01    -4.496946e-03    -4.496936e-03    -4.475882e-03    ...
1.000000e+01    -6.355888e-03    -6.355869e-03    -6.309857e-03    ...
...
```

## Observable Definitions

- **T**: Temperature (computed from β as T = 1/β)
- **Sigmax, Sigmay, Sigmaz**: Pauli operator expectation values
- **S1, S2**: Plaquette operators (four-site correlations)
- **Wp**: Hexagon operator (six-site correlation)
- **Cv**: Specific heat capacity = -β²(dE/dβ)

## Key Features

### Dynamic Data Saving
Data is saved automatically after each temperature step during simulation. This prevents data loss if the job is interrupted on a cluster.

### Automatic Site Calculations
For Kitaev Ladder:
- Number of flux operators: `num_flux_ops = (L-1)//2`
- Measurement sites: `[4*i for i in range(num_flux_ops)]`
- Normalization: `/num_flux_ops` and `/total_sites`

### Temperature Alignment
When creating contour plots from multiple h values, the code automatically:
1. Finds the common temperature range across all data
2. Trims datasets to ensure alignment
3. Creates consistent 2D grids for contour plotting

## Dependencies

- **TeNPy** (>= 0.9.0) - Tensor network library
- **NumPy** - Numerical computations
- **Matplotlib** - Plotting and visualization
- **SciPy** - Scientific computing utilities

## Installation

```bash
# Create conda environment
conda create -n ten python=3.9

# Activate environment
conda activate ten

# Install TeNPy and dependencies
pip install tenpy numpy matplotlib scipy
```

## Performance Notes

- **Bond dimension (chi)** - Higher chi values give better accuracy but increase computational cost
- **Time step (dt)** - Smaller dt is more accurate but requires more iterations
- **Temperature range** - Lower temperatures (higher β) require more iterations and longer runtimes
- **System size (L)** - Larger systems are computationally more expensive

## Typical Runtime
- L=23 ladder, β_max=120, chi=100: ~24 hours on CPU
- 6×6 J1J2, β_max=100, chi=50: ~12 hours on CPU
- Scales roughly as O(χ³ × L × β/dt)

## References

The implementation uses the purification approach for finite temperature simulations:
- TeNPy documentation: https://tenpy.readthedocs.io/
- TEBD algorithm for imaginary time evolution
- Density matrix purification method for thermal states

## Author Notes

This code was developed for studying phase diagrams and thermal properties of quantum lattice models. The modular structure allows easy modification of Hamiltonians and observables by editing the model files in the `models/` directory.

For cluster job submissions, use the dynamic saving feature and always set `beta_max` and `chi_max` as CLI arguments for reproducibility.
