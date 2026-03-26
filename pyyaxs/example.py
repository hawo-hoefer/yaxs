import yaxs
import os

s = yaxs.PyStructure("Na2CO3.cif", 0.1, (5, 70))


instrument = yaxs.PyInstrument([0] * 6, [(1.5401, 1.0)], 180, (5, 40), 2048)
sample = yaxs.PySample((-500, 500))


result = yaxs.simulate_adxrd([s], instrument, sample, 1238, 100, os.path.join(os.path.dirname(__file__), "../examples/cif"))

for k, v in result.items():
    print(k, v[0].shape)
