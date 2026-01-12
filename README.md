# YAXS: an Accelerated XRD Simulator

A cuda-accelerated XRD simulation tool.

## Art Credit
The yak ASCII art was created by Joan Stark (Spunk).

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
or alternatively
```bash
$ cargo install --git https://github.com/hawo-hoefer/yaxs.git --tag=<version>
```
where `<version>` is the version you want to install.

If you don't have a CUDA-enabled GPU on your system, use the feature flag `no-gpu`. 
Note that compilation and installation requires a working C and NVCC compiler.

Then, use
```bash
$ /path/to/yaxs <path/to/configuration.yml> [options]
```
to simulate a dataset from the input `yaml`-file describing the simulation parameters.
Further information on the input file structure can be found [here](./inputfile.md)

## Roadmap
- **FEATURES**: 
    - [ ] add EDXRD fluorescence peaks?
    - [ ] add EDXRD radiation filters
    - [ ] mode for adapting existing config to given data, simulation based inference mode
    - [ ] surface roughness intensity correction Suortti (J. Appl. Cryst, 5,325-331, 1972)
    - [ ] Air Scattering
    - [ ] Peak Asymmetry
    - [ ] Revisit sample displacement for EDXRD, for now sample displacement is ignored
    - [ ] Add better controls for resource consumption
    - [ ] CPU backend multithreading
        - right now, the cpu backend is single threaded, which is really slow
        - `yaxs`'s use case is the generation of ML training data, so we don't really expect it to be used on devices without a GPU
        - therefore, this is low priority
- **BUGFIXES**
    - [ ] fix 'invalid resource handle' bug in CUDA when compiling with rustc 1.89
    - [ ] not setting chunk size for large amounts of XRD patterns causes crash with unhelpful error message
- [ ] DOCS: find some way to add example cifs into repo that does not infringe on someones copyright
- [ ] CI: improve CI speed using caching
- [x] Parse Structure from CIF
    - [x] implement parsing of CIF to HashMap and Vector of Tables
    - [x] Map HashMap / Vector of Tables to Structure
- [x] Preferred Orientation
    - [x] preferred orientation with direction relative to sample surface
    - [x] implement other directions
    - [x] also, how do we handle viewing angles orthogonal to the direction of the preferred orientation??
    - implemented via Bingham orientation distribution
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
- [x] rendering using cuda backend
    - [x] peak positions
    - [x] backgrounds
    - [x] normalization
- [x] IO
    - [x] output pattern-wise metadata to target arrays in data files
    - [x] output configuration-metadata to `meta.json`
    - [x] copy simulation input file to output directory
    - [x] custom parameter deserialization with range checking
- [x] Add documentation for config YAML file
- [x] feature: fixed strains for each structure as input (that way, we can simulate specific strain conditions more easily)
- [x] feature: weights for the volume fractions of each phase or make them fixed
    - volume fractions can be fixed optionally 
- [x] feature: bake git commit hash into executable
- [x] sample displacement for ADXRD simulation
- [x] feature: Add support for impurity peaks
- [x] Implement support for noise
    - Usually noise is added during augmentation anyway, so we may not need this
    - therefore, low priority
- [x] parallel front end / peak position simulation
- [x] parallel gpu job generation
- [x] CI
- [x] add mode for only outputting peak positions
- [x] peak intensities
    - what about the computation of structure factors?
    - keep using pymatgen's version for now -> i really should understand what is going on there
    - move to GSAS-implementation 
- [x] Debye-Waller correction
    - not present in cif
    - in `pymatgen`, they are passed separately
    - figure out if I want to do that somehow
    - low priority
- [x] finish up merge of main branch into bingham texture:
    - [x] complete all related TODOs
    - [x] writing data to files needs to be implemented properly (mustrain etc is missing)
    - [x] improve speed some more, i think there are easy gains to be had
