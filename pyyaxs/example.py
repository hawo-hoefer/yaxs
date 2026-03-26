import os

import numpy as np
from matplotlib import pyplot as plt

import yaxs
from yaxs import Parameter

os.environ["LOG_LEVEL"] = "Error"

s = [
    yaxs.Structure("Na2CO3.cif", 0.0, (5, 70)),
    yaxs.Structure("CuO.cif", 0.0, (5, 70)),
]

two_theta_range = (5, 40)
n_steps = 2048

instprms = yaxs.InstrumentParams(
    u=Parameter.range(5.71, 15.71),
    v=Parameter.range(-9.14, 3.14),
    w=Parameter.range(2.61, 6.61),
    x=Parameter.range(0.0034, 0.0114),
    y=Parameter.range(-0.0723, -0.0273),
    z=Parameter.fixed(0),
)
emission_lines = yaxs.EmissionLine.mo_ka()

instrument = yaxs.Instrument(instprms, emission_lines, 180, two_theta_range, n_steps)
sample = yaxs.Sample((-500, 500))

result = yaxs.simulate_adxrd(
    s,
    instrument,
    sample,
    1238,
    100,
    os.path.join(os.path.dirname(__file__), "../examples/cif"),
)

for k, v in result.items():
    print(k, v[0].shape)

plt.plot(np.linspace(*two_theta_range, n_steps), result["intensities"][0][0])
plt.show()
