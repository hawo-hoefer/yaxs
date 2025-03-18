# Yet another XRD Simulator (YaXS)

An (ED-)XRD simulation tool implemented in rust.

## Roadmap
- [x] Parse Structure from CIF
    - [x] implement parsing of CIF to HashMap and Vector of Tables
    - [x] Map HashMap / Vector of Tables to Structure
- [ ] Debye-Waller correction
- [ ] Volume and Wavelength Correction (Cullity 1976)
- [ ] straining of unit cells
- [ ] generating of XRD patterns instead of peak positions
    - [ ] Chebyshev Backgrounds
    - [ ] Gaussian Noise
    - [ ] Caglioti Parameters
    - [ ] Scherrer Broadening
    - [ ] Air Scattering
    - [ ] Preferred Orientation
    - [ ] Reading Config from yaml
    - [ ] writing to HDF5
- [ ] EDXRD Simulation
