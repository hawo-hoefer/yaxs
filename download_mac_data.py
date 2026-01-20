import os
import struct
from dataclasses import dataclass

import requests

U32_LE = "<L"
U8_LE = "<B"
F64_LE = "<d"


@dataclass
class MACData:
    energies: list[float]
    macs: list[float]
    z: int

    def to_bytes(self) -> bytes:
        # format: [
        #    atomic_number:              u8,
        #    num_entries:               u32,
        #    energies:    num_entries * f64,
        #    macs:        num_entries * f64,
        # ]

        data = []

        data.append(struct.pack(U8_LE, self.z))
        data.append(struct.pack(U32_LE, len(self.energies)))

        for energy in self.energies:
            data.append(struct.pack(F64_LE, energy))

        for mac in self.macs:
            data.append(struct.pack(F64_LE, mac))

        ret = bytes()
        return ret.join(data)


def download_mac_data(z: int) -> MACData:
    r = requests.get(
        f"https://physics.nist.gov/cgi-bin/ffast/ffast.pl?Z={z}&Formula=&gtype=3&lower=&upper=&density=&frames=no"
    )
    roi = r.text.split("graph of data")[1]
    _, roi = roi.split("cm<sup>2</sup> g<sup>-1</sup>\n")

    energies = []
    macs = []
    for line in roi.splitlines():
        line = line.strip()
        if line.startswith("</PRE>"):
            break

        if not line[0].isnumeric():
            raise ValueError("Something went wrong")

        energy, mac = map(float, line.split())
        energies.append(energy)
        macs.append(mac)

    return MACData(energies, macs, z)


zmax = 92
mac_data = []
for z in range(1, zmax + 1):
    print(
        f"Downloading mac data for z = {z} [{int((z - 1)/zmax * 100):>3}% done]",
        end="\r",
    )

    mac_data.append(download_mac_data(z))

cols = os.get_terminal_size().columns
print(f"{'Done downloading.':<{cols}}")

all_mac_data_as_bytes: list[bytes] = []
sizes: list[int] = []
for md in mac_data:
    all_mac_data_as_bytes.append(md.to_bytes())
    sizes.append(len(all_mac_data_as_bytes[-1]))

data_path = os.path.join(os.path.dirname(__file__), "src", "macdata.bin")
with open(data_path, "wb") as file:
    # file format
    # [   # Header
    #     number of elements:       u32 (num_el)
    #     offset for each element: [num_el * u32]
    #
    #     # Data section
    #     representation for each MACData element
    #     with format: [
    #        atomic_number:              u8,
    #        num_entries:               u32,
    #        energies:    num_entries * f64,
    #        macs:        num_entries * f64,
    #     ]
    # ]
    num_el = len(sizes)
    assert num_el < 256, "How can we have more than 256 elements? This should be 92"
    file.write(struct.pack(U8_LE, num_el))

    offset = 4 + num_el * 4

    file.write(struct.pack(U32_LE, offset))
    for size in sizes[:-1]:
        offset += size
        file.write(struct.pack(U32_LE, offset))

    for md in all_mac_data_as_bytes:
        file.write(md)
