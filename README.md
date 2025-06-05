# Yet another XRD Simulator (YaXS)

An (ED-)XRD simulation tool implemented in rust.

## Quickstart
To use this simulation tool, just clone this repository and build the project.
```bash
$ git clone gitlab.kit.edu/hawo.hoefer/yaxs.git
$ cd yaxs
$ cargo build --release
```

If you want to install the program, use
```bash
$ cargo install --path /path/to/yaxs/directory
```

Then, use 
```bash
$ /path/to/yaxs <path/to/configuration.yml>
```
to simulate a dataset from the input `yaml`-file describing the simulation parameters.
Further information on the input file structure can be found [here](./inputfile.md)

## Roadmap
- [x] Parse Structure from CIF
    - [x] implement parsing of CIF to HashMap and Vector of Tables
    - [x] Map HashMap / Vector of Tables to Structure
- [x] Volume and Wavelength Correction (Cullity 1976)
- [x] straining of unit cells
- [x] generating of XRD patterns instead of peak positions
    - [x] Chebyshev Backgrounds
    - [x] Caglioti Parameters
    - [x] Scherrer Broadening
    - [x] Reading Config from yaml
    - [x] writing to numpy `.npz`
    - [x] background height
- [x] EDXRD Simulation
- [x] Do we need to save peak info as d-spacing? then, we can use the same code for AD- and EDXRD
- [x] rendering info is computed only at render time then, less conversions needed -> better numerical accuracy?
- [ ] XRD pattern bonuses
    - [ ] Air Scattering
    - [x] Preferred Orientation
- [x] rendering using cuda backend
    - [x] peak positions 
    - [x] backgrounds
    - [x] normalization
- [x] IO
    - [x] output pattern-wise metadata to target arrays in data files
    - [x] output configuration-metadata to `meta.json`
    - [x] copy simulation input file to output directory
- [x] Add documentation for config YAML file
- [ ] feature: fixed strains for each structure as input (that way, we can simulate specific strain conditions more easily)
- [ ] feature: weights for the volume fractions of each phase or make them fixed
- [ ] feature: Add support for impurity peaks
- [ ] EDXRD: add fluorescence peaks?
- [ ] peak intensities
    - what about the computation of structure factors?
    - keep using pymatgen's version for now -> i really should understand what is going on there
- [ ] CPU backend multithreading
    - right now, the cpu backend is single threaded, which is really slow
    - `yaxs`'s use case is the generation of ML training data, so we don't really expect it to be used on devices without a GPU
    - therefore, this is low priority
- [ ] Implement support for noise 
    - Usually noise is added during augmentation anyway, so we may not need this
    - therefore, low priority
- [ ] Debye-Waller correction
    - not present in cif
    - in `pymatgen`, they are passed separately
    - figure out if I want to do that somehow
    - low priority
