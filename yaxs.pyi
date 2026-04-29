from numpy.typing import NDArray
from typing import Any
import numpy as np

class Structure:
    """A crystallographic structure craeted from a CIF structure definition"""

    path: str
    max_strain: float
    domain_size_nm: tuple[float, float]
    domain_size_eta: tuple[float, float]

    def __init__(
        self,
        path: str,
        max_strain: float,
        domain_size_range: tuple[float, float],
        domain_size_eta: tuple[float, float],
    ) -> None:
        """Create a Structure

        Args:
            path: path to CIF structure definition
            max_strain: maximum macrostrain / deformation of the unit cell
            domain_size_range: range of the domain size in nanometers
            domain_size_eta: range for the domain_size pseudo-voigt mixing parameter.
                             upper and lower bound must be in [0, 1]
        """
        ...

class Sample:
    """Sample configuration containing ranges for sample parameters"""

    displacement_mu_m: tuple[float, float]
    """sample displacement range in micrometers"""

    def __init__(self, displacement_mu_m: tuple[float, float]) -> None:
        """Initialize PySample Structure

        Args:
            displacement_mu_m: range of sample displacement in micro meters
        """
        ...

class Parameter:
    """either range or fixed value"""

    @staticmethod
    def range(lo: float, hi: float) -> Parameter:
        """create a range parameter

        Args:
            lo: lower bound
            hi: upper bound

        Returns:
            Range Parameter
        Raises:
            ValueError: if the lower bound is larger than or equal to the upper bound
        """
        ...

    @staticmethod
    def fixed(v: float) -> Parameter:
        """create a range parameter

        Args:
            v: value

        Returns:
            Fixed Parameter
        """
        ...

class InstrumentParams:
    u: Parameter
    v: Parameter
    w: Parameter
    x: Parameter
    y: Parameter
    z: Parameter

    def __init__(
        self,
        u: Parameter,
        v: Parameter,
        w: Parameter,
        x: Parameter,
        y: Parameter,
        z: Parameter,
    ) -> None: ...

    """Create zero-initialized instrument parameters"""
    @staticmethod
    def zero() -> InstrumentParams: ...

class EmissionLine:
    """Emission Line for an XRD device"""

    wavelength: float
    """wavelength in angstrom"""
    weight: float
    """relative weight of the emission line"""

    """create a new emission line

    Args:
        wavelength: wavelength of the radiation in angstrom
        weight: relative weight compared to other emission lines in the spectrum
    """
    def __init__(self, wavelength: float, weight: float) -> None: ...

    """Create the Molybdenum K-Alpha emission lines"""
    @staticmethod
    def mo_ka() -> tuple[EmissionLine, EmissionLine]: ...

    """Create the Copper K-Alpha emission lines"""
    @staticmethod
    def cu_ka() -> tuple[EmissionLine, EmissionLine]: ...

class Instrument:
    """Instrument Configuration"""

    def __init__(
        self,
        instprms: InstrumentParams,
        monochromator_angle_deg: float,
        emission_lines: list[EmissionLine] | tuple[EmissionLine, ...],
        goniometer_radius: float,
        two_theta_range: tuple[float, float],
        n_steps: int,
    ) -> None:
        """Initialize an instrument configuration

        Args:
            instprms: instrument parameters for the XRD device
            monochromator_angle_deg: monochromator angle in degrees, usually 0.0
            emission_lines: list of emission lines the device uses
            goniometer_radius: goniometer radius in mm
            two_theta_range: two-theta range in degrees
            n_steps: number of steps of the XRD pattern in two-theta
        """
        ...

def simulate_adxrd(
    structures: list[Structure],
    i: Instrument,
    s: Sample,
    seed: int,
    n_samples: int,
    structure_permutations: int,
    cif_root: str,
) -> dict[str, list[NDArray[np.float32]]]:
    """simulate a set of adxrd patterns

    Args:
       structures: structures in the mixture
       i: instrument description
       s: sample description
       seed: simulation random seed
       n_samples: number of XRD patterns to simulate
       structure_permutations: how many permutations of each structure with random strain parameters
                               to compute
       cif_root: root directory for relative paths of cifs

    Returns: dict with simulated XRD patterns and metadata
    """
    ...
