# Simulation Outputs

Every YaXS simulation creates an output directory at the location specified using `-o/--output-path` (or defaults to `out`).
In this directory, it copies the input file, writes a file describing the simulated data (`meta.json`), and
writes data files (`data_<i>.npz`) where `i` is the chunk's index.

```
out
├── input-file.yml
├── data_0.npz
├── ...
├── data_<N>.npz
└── meta.json
```
Each data file contains the following simulation outputs and inputs. The first dimension
of each output array is the chunk size / XRD pattern number. The other dimensions depend on the output.

|  name  |  shape | description  |
| ------ | ------ | ------------- |
| intensities | [(chi, phi), n_steps ] | The rendered XRD patterns. In case of texture measurement, the shape may include chi and phi steps |
| mean_ds_nm | [n_phases, 6] | Mean domain size of all Phases in nanometers. In the isotropic case, the first of 6 parameters is the domain size. For anisotropic grain size, the values are the lower triangular half of the ellipsoid describing the domain size (order: 11, 12, 22, 31, 32, 33).
| ds_etas | [n_phases] | Gaussian-Lorentzian Mixing Parameter of the domain size for each phase |
| mustrain | [n_phases] | Microstrain used to compute peak broadening |
| mustrain_etas | [n_phases] | Gaussian-Lorentzian Mixing Parameter of microstrain for each phase |
| volume_fractions | [n_phases] | The phases' volume fractions |
| weight_fractions | [n_phases] | The phases' weight fractions |
| random_b_isos | [n_phases] | If temperature factors for a phase have been varied randomly, it's shown here |
| instrument_parameters | [6] | For ADXRD simulation. The instrument parameters U, V, W, X, Y, Z |
| sample_displacement_mu_m | [] | Sample displacement for each pattern |
| impurity_sum | [] | Sum of all impurity peak intensities added to the pattern | 
| impurity_max | [] | Maximum of all impurity peak intensities added to the pattern |
| background_parameters | [2 or coef + 1 ] | Only if a background is present. The background scale and slope in case of exponential background, or the scale and chebyshev polynomial coefficients for chebyshev polynomial background. |
