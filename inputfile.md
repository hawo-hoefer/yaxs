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

## 1. Simulation Kind
This section contains the type of the simulation (EnergyDisperse or AngleDisperse)
and the corresponding parameters.

### An example EDXRD configuration
```yaml
kind: !EnergyDisperse
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
```yaml
kind: !AngleDisperse
  emission_lines:                # list of emission lines with wavelength and relative strength
  - wavelength_ams: 1.5406       # wavelength in Amstrong
    weight: 1.0                  # relative emission line strength in arbitrary units
  - wavelength_ams: 1.3923
    weight: 0.4121
  n_steps: 2048                  # number of steps for rastering two-theta
  two_theta_range: [10.0, 70.0]  # two-theta range
  noise: !Gaussian               # ignored for now
    sigma_min: 0.0
    sigma_max: 1.0
  caglioti:                      # caglioti parameters of the device
    u: 0.0
    v: 0.0
    w: 0.0
  background: !Chebyshev         # background type and parameters
    coefs:                       # chebyshev coefficient ranges
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

## 2. Sample Parameters
This section contains (physical) sample parameters, like the phase's domain sizes.
They are shared for energy- and angle dispersive XRD simulation.
```yaml
sample_parameters:
  mean_ds_nm: 100.0                                     # domain size
  eta: 0.5                                              # pseudo-voigt eta
  sample_displacement_mu_m: 0.0                         # sample displacement
  structure_permutations: 100                           # number of structure permutations to simulate
  structures:                                           # phases and their preferred orientation
                                                        # configuration using march-dollase model
    - path: phase-1.cif                                 # path to phase's cif, no preferred orientation here
    - path: phase-2-with-preferred-orientation.cif
      preferred_orientation:                            # march-dollase parameters
        hkl: [1, 1, 1]                                  # axis of preferred orientation
        r: 0.9                                          # march parameter
      strain: 0.01                                      # optional strain specification
      volume_fraction: 0.5                              # fix volume fraction of phase 2 to 0.5
    - path: phase-3.cif
```

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
```
