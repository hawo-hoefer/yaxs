# Input File Structure

Input files for YAXS are `yaml`-files consisting of three parts:
1. Simulation kind
2. Sample parameters
3. Simulation parameters

## Quick start
If you just want a configuration file to get started, you can just use [this](./examples/adxrd.yml)
for angle-dispersive X-ray diffraction, and [this](./examples/edxrd.yml) for energy-dispersive diffraction
as a starting point.

## Parameters

Some parameters may be specified as ranges or single values. Single value parameters
are fixed for all simulations, while parameters specified as ranges will be sampled uniformly.

| parameter                 | variability   |
|---------------------------|---------------|
| domain size               | both          |
| eta                       | both          |
| max structure strain      | value         |
| structure permuations     | value         |
| preferred orientation hkl | value         |
| preferred orientation r   | both          |
| emission line wavelength  | value         |
| emission line weight      | value         |
| caglioti parameters       | both          |
| background parameters     | both          |
| simulation parameters     | value         |
| strain                    | (see below)   |
| volume factions           | value         |
| sample displacement       | both          |

## 1. Simulation Kind
This section contains the type of the simulation (EnergyDispersive or AngleDispersive)
and the corresponding parameters.

### An example EDXRD configuration
```yaml
kind: !EnergyDispersive
    n_steps: 1024                            # number of energy steps
    energy_range_kev: [20, 160]              # energy range to simulate
    theta_deg: 3.965                         # beam reflection angle
    beamline:
      storage_ring_electron_energy_gev: 6    # storage ring electron kinetic energy in GeV
      storage_ring_current_a: 0.1            # storage ring current in Amps
      n_wiggler_magnets: 10                  # number of wiggler magnets
      distance_from_device_m: 100            # distance of EDXRD device from insertion device in m
```

### An example ADXRD configuration
Angle dispersive simulation simulates XRD patterns for the Bragg-Brentano diffractometer geometry.
```yaml
kind: !AngleDispersive
  emission_lines:                       # list of emission lines with wavelength and relative strength
  - wavelength_ams: 1.5406              # wavelength in Amstrong
    weight: 1.0                         # relative emission line strength in arbitrary units
  - wavelength_ams: 1.3923
    weight: 0.4121
  n_steps: 2048                         # number of steps for rastering two-theta
  two_theta_range: [10.0, 70.0]         # two-theta range
  goniometer_radius_mm: 180             # radius of the goniometer in bragg-brentano geometry
  sample_displacement_mu_m: [-250, 250] # sample displacement in micrometers (optional)
  noise: !Gaussian                      # ignored for now
    sigma_min: 0.0
    sigma_max: 1.0
  caglioti:                             # Optional caglioti parameters of the device
    kind: Raw                           # Mode for Caglioti parameters, Raw or GSAS
    u: 0.0                              # GSAS is a GSASII-Compatibility mode,  
    v: 0.0                              # as they multiply each parameter by 8 ln(2) / 10000
    w: 0.0
  background: !Chebyshev                # background type and parameters
    coefs:                              # chebyshev coefficient ranges
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    - [-1.0, 1.0]
    scale: 40.0
  # background: None             # alternate backgrounds
  # background: !Exponential
  #   slope: [-0.05, 0.025]
  #   scale: [200, 800]
```

### Caglioti Parameters
In ADXRD Simulation, Caglioti Parameters may be optionally specified using the 
`caglioti`-key. 
They can be used to specify the instrumental line broadening $\Delta$ using the Caglioti function
$$
    \Delta = u \tan^2\theta + v * \tan\theta + w.
$$
Additionally, the Caglioti kind may be specified as `Raw` or `GSAS` (defaults to `Raw`). 
This indicates to use the definition of caglioti parameters according to the formula
above (`Raw`) or do what GSASII does (`GSAS`). GSAS multiplies Caglioti parameters
($u$, $v$, and $w$) by $8 \ln 2 / 10000$ before passing them to the Caglioti function.
Should you want to compare yaxs-output patterns to GSASII simulations, either adjust 
the Caglioti kind or $u$, $v$ and $w$ values.

## 2. Sample Parameters
This section contains (physical) sample parameters, like the phase's domain sizes.
They are shared for energy- and angle dispersive XRD simulation.
```yaml
sample_parameters:
  mean_ds_nm: 100.0                                     # domain size
  eta: 0.5                                              # pseudo-voigt eta
  max_concentration_subset_dim: 4                       # (Optional) subsample the concentration space (all but max_concentration_subset_dim values are zeroed)
  impurities:                                           # (Optional) list of specifications of impurity peaks
    - d_hkl_ams: [2.5, 3.5]                                 # impurity d-hkl in amstrong
      intensity: 1e2                                        # integral over peak
      eta: 0.5                                              # pseudo-voigt eta
      mean_ds_nm: 10                                        # mean domain size for the peak to compute it's width
      n_peaks: 2                                            # (Optional) number of peaks at the given position
      probability:                                          # (Optional) probability of peak existance
  structure_permutations: 100                           # number of structure permutations to simulate
  structures:                                           # phases and their preferred orientation
                                                        # configuration using march-dollase model
    - path: phase-1.cif                                 # path to phase's cif, no preferred orientation here
    - path: phase-2-with-preferred-orientation.cif
      preferred_orientation:                            # march parameters for preferred orientation (optional)
        hkl: [1, 1, 1]                                  # axis of preferred orientation
        r: 0.9                                          # march parameter
      strain: !Maximum 0.01                             # optional strain specification
      volume_fraction: 0.5                              # fix volume fraction of phase 2 to 0.5
    - path: phase-3.cif
```
### Subsets
If `max_concentration_subset_dim` is set to a value, a random number of composition components will be set
to zero for each XRD pattern. The zeroed components respect fixed volume fractions. This allows sampling 
of concentration subspaces and is useful for adding XRD patterns with large volume fractions if many 
components are present in the dataset.
`max_concentration_subset_dim` may be at most the number of phases. The program will error if it encounters
larger values.

### Impurities
The optional field `impurities` contains a list of impurity peaks. They may be used to specify extra peaks which occur in experimental samples but don't have a particular association with the phases of interest.
- `d_hkl_ams` specifies the peak position using it's crystallographic plane distance in amstrong (range or value). 
- `intensity` specifies the integral over the peak's pseudo-voigt function (range or value)
- `eta` specifies the peak's pseudo-voigt mixing parameter eta (range or value)
- `mean_ds_nm` specifies the domain size in nanometers used for computing the peak's fwhm (range or value)
- `n_peaks` optionally specifies the number of peaks at the position indicated by `d_hkl_ams`. This only makes sense if some of the parameters are ranges, as the ranges will be sampled multiple times then.
- `probability` optionally specifies the probability (in the range [0, 1]) that the described peak exists. If `n_peaks` is larger than 0, each peak's existance is decided independently of the other's.

### Structure Definition
The `structures` field specifies a list of phases and their modifications. 
Preferred orientation and strain can be specified using fixed or range parameters,
and each phase's volume fraction may be fixed to specified values.

### Specifying Strain
Strain is specified for each structure separately, and may be omitted if no strain should be applied to the structures before simulation.

If the strain specification is not omitted, it's tag dictates it's behavior. 
| tag        | behavior                                                                               |
|------------|----------------------------------------------------------------------------------------|
| `!Maximum` | Space-Group respecting strain, uniformly sampled using maximum amplitude               |
| `!Ortho`   | Orthogonal strain, diagonal strain matrix sampled according to parameter specification |
| `!Full`    | Parameter specification for the full strain Matrix                                     |

Some of the methods may produce non-invertible strain tensors. In those cases, the
program will notify the user and exit.

#### Space-Group Respecting Strain using Maximum Amplitude
```yaml
strain: !Maximum 0.1
```
Maximum strain amplitude is 0.1 in each direction. The unit cell is stretched 
with a strain matrix preserving the original unit cell's symmetry.

#### Orthogonal Strain
```yaml
strain: !Ortho [1.0, [0.8, 1.2], 1.1]
```
Strain is applied using a diagonal strain matrix. The first diagonal component is 
set to 1.0, the second varies uniformly from 0.8 to 1.2, and the third component will be
fixed at 1.1.

#### Full Strain Matrix Specification
```yaml
strain: !Full [1.0, [-0.01, 0.01], 0.01, 1.1, 0.0, 1.0]
```
Parameter ranges may be used to specify sampling ranges, but note that this may produce
non-invertible strain tensors. The order of the parameters is 
`A[1, 1]`, `A[2, 1]`, `A[2, 2]`, `A[3, 1]`, `A[3, 2]`, `A[3, 3]`.


## 3. Simulation Parameters
Here, simulation specific parameters are set.
```yaml
simulation_parameters:
  normalize: true
  seed: 1234
  n_patterns: 10000
  abstol: 1e-2
  randomly_scale_peaks:
    scale: [0.5, 1.5]
    probability: 0.5
```

### Randomly Scale Peak Intensities
Use the key `randomly_scale_peaks` to randomly scale peaks after structure simulation.
Each for each structure, every peak will be scaled by the specified scale with 
the specified probability. This will produce peaks which are not physically accuracte,
but that may be beneficial for neural network training. This form of augmentation may
make training for quantification more robust, as the network should in theory be pushed
towards considering all peaks instead of only the largest few. Excessive (and non-balanced)
scaling may also cause problems, so use at your own risk.
