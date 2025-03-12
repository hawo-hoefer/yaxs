use itertools::Itertools;
use nalgebra::{matrix, zero, Matrix3, Vector3};
use std::f64::consts::PI;

use sim_edxrd::element::Element;
use sim_edxrd::species::Species;

#[derive(Debug)]
struct Lattice {
    mat: Matrix3<f64>,
}
impl Lattice {
    fn recip_lattice(&self) -> Lattice {
        Self {
            mat: self.mat.try_inverse().unwrap().transpose() * 2.0 * PI,
        }
    }

    fn abc(&self) -> Vector3<f64> {
        let mut values = [0.0; 3];
        for (i, v) in self
            .mat
            .row_iter()
            .into_iter()
            .map(|x| x.iter().map(|a| a.powi(2)).sum::<f64>().sqrt())
            .enumerate()
        {
            values[i] = v;
        }
        Vector3::from([values[0], values[1], values[2]])
    }
}

impl std::fmt::Display for Lattice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lattice(\n")?;
        for row in self.mat.row_iter() {
            write!(f, "  {:5.2}, {:5.2}, {:5.2}\n", row[0], row[1], row[2])?;
        }
        write!(f, ")\n")
    }
}

// pub fn get_points_in_spheres(r: f64, lattice: Lattice) {
//     // pbc = True
//     // numerical_tol = 1e-8
//     // all_coords = center_coords= [[0, 0, 0]]
//     //
//     let gmin = Array1::from([-r, -r, -r]);
//     let gmax = Array1::from([r, r, r]);

//     // frac, dist, _, _ = self.get_points_in_sphere(
//     //     [[0, 0, 0]],    [0, 0, 0], max(lengths) * (1 + ltol), zip_results=False
//     //     ^ frac_points,  ^ center
//     // )

//     let recip_len: f64 = lattice.recip_lattice().abc();
//     let max_r = ((r + 0.15) * recip_len / (2.0 * PI)).ceil(); // TODO: ?????
//     // frac_coords = [0, 0, 0] | lattice.get_fractional_coords(center_coords)
//     // at center_coords = [0, 0, 0]
//     let center_frac_coords = Array1::new([0.0, 0.0, 0.0]);

// }
//         nmin_temp = np.floor(np.min(frac_coords, axis=0)) - maxr
//         nmax_temp = np.ceil(np.max(frac_coords, axis=0)) + maxr
//         nmin = np.zeros_like(nmin_temp)
//         nmin[_pbc] = nmin_temp[_pbc]
//         nmax = np.ones_like(nmax_temp)
//         nmax[_pbc] = nmax_temp[_pbc]
//         all_ranges = [np.arange(x, y, dtype="int64") for x, y in zip(nmin, nmax, strict=True)]
//         matrix = lattice.matrix

//         # Temporarily hold the fractional coordinates
//         image_offsets = lattice.get_fractional_coords(all_coords)
//         all_frac_coords = []

//         # Only wrap periodic boundary
//         for kk in range(3):
//             if _pbc[kk]:
//                 all_frac_coords.append(np.mod(image_offsets[:, kk : kk + 1], 1))
//             else:
//                 all_frac_coords.append(image_offsets[:, kk : kk + 1])
//         all_frac_coords = np.concatenate(all_frac_coords, axis=1)
//         image_offsets -= all_frac_coords
//         coords_in_cell = np.dot(all_frac_coords, matrix)

//         # Filter out those beyond max range
//         valid_coords = []
//         valid_images = []
//         valid_indices = []
//         for image in itertools.product(*all_ranges):
//             coords = np.dot(image, matrix) + coords_in_cell
//             valid_index_bool = np.all(
//                 np.bitwise_and(coords > global_min[None, :], coords < global_max[None, :]),
//                 axis=1,
//             )
//             ind = np.arange(len(all_coords))
//             if np.any(valid_index_bool):
//                 valid_coords.append(coords[valid_index_bool])
//                 valid_images.append(np.tile(image, [np.sum(valid_index_bool), 1]) - image_offsets[valid_index_bool])
//                 valid_indices.extend([k for k in ind if valid_index_bool[k]])
//         if not valid_coords:
//             return [[]] * len(center_coords)
//         valid_coords = np.concatenate(valid_coords, axis=0)
//         valid_images = np.concatenate(valid_images, axis=0)

//     else:
//         valid_coords = all_coords
//         valid_images = [[0, 0, 0]] * len(valid_coords)
//         valid_indices = np.arange(len(valid_coords))

//     # Divide the valid 3D space into cubes and compute the cube ids
//     all_cube_index = _compute_cube_index(valid_coords, global_min, r)
//     nx, ny, nz = _compute_cube_index(global_max, global_min, r) + 1
//     all_cube_index = _three_to_one(all_cube_index, ny, nz)
//     site_cube_index = _three_to_one(_compute_cube_index(center_coords, global_min, r), ny, nz)

//     # Create cube index to coordinates, images, and indices map
//     cube_to_coords: dict[int, list] = defaultdict(list)
//     cube_to_images: dict[int, list] = defaultdict(list)
//     cube_to_indices: dict[int, list] = defaultdict(list)
//     for ii, jj, kk, ll in zip(all_cube_index.ravel(), valid_coords, valid_images, valid_indices, strict=True):
//         cube_to_coords[ii].append(jj)
//         cube_to_images[ii].append(kk)
//         cube_to_indices[ii].append(ll)

//     # Find all neighboring cubes for each atom in the lattice cell
//     site_neighbors = find_neighbors(site_cube_index, nx, ny, nz)
//     neighbors: list[list[tuple[np.ndarray, float, int, np.ndarray]]] = []

//     for ii, jj in zip(center_coords, site_neighbors, strict=True):
//         l1 = np.array(_three_to_one(jj, ny, nz), dtype=np.int64).ravel()
//         # Use the cube index map to find the all the neighboring
//         # coords, images, and indices
//         ks = [k for k in l1 if k in cube_to_coords]
//         if not ks:
//             neighbors.append([])
//             continue
//         nn_coords = np.concatenate([cube_to_coords[k] for k in ks], axis=0)
//         nn_images = itertools.chain(*(cube_to_images[k] for k in ks))
//         nn_indices = itertools.chain(*(cube_to_indices[k] for k in ks))
//         distances = np.linalg.norm(nn_coords - ii[None, :], axis=1)
//         nns: list[tuple[np.ndarray, float, int, np.ndarray]] = []
//         for coord, index, image, dist in zip(nn_coords, nn_indices, nn_images, distances, strict=True):
//             # Filtering out all sites that are beyond the cutoff
//             # Here there is no filtering of overlapping sites
//             if dist < r + numerical_tol:
//                 if return_fcoords and (lattice is not None):
//                     coord = np.round(lattice.get_fractional_coords(coord), 10)
//                 nn = (coord, float(dist), int(index), image)
//                 nns.append(nn)
//         neighbors.append(nns)
//     return neighbors

const H_EV_S: f64 = 4.135_667_696e-15f64;
const C_M_S: f64 = 299_792_485.0f64;

pub fn e_kev_to_lambda_ams(e_kev: f64) -> f64 {
    // e = h * c / lambda
    // lambda = h * c / e
    // m      = ev * s * m / ev
    H_EV_S * C_M_S / e_kev * 1e7
}

struct Structure {
    lat: Lattice,
    sites: Vec<Site>,
}

struct Site {
    coords: Vector3<f64>,
    species: Species,
}

fn main() {
    let two_theta_range: (f64, f64) = (0.0, 15.0);
    // let wavelength_ams = 1.5405;
    // let wavelength_ams = e_kev_to_lambda_ams(8.04);
    let wavelength_ams = e_kev_to_lambda_ams(8.04);
    // let wavelength_ams = 0.69;

    let species = vec![
        Site {
            coords: Vector3::new(1.15473127, -0.85565, -2.75846457),
            species: "Cu2+".parse().unwrap(),
        },
        Site {
            coords: Vector3::new(1.15473127, 0.85565, -0.19406457),
            species: "Cu2+".parse().unwrap(),
        },
        Site {
            coords: Vector3::new(2.30946253, 0.27928416, -1.67032914),
            species: "O2-".parse().unwrap(),
        },
        Site {
            coords: Vector3::new(2.30946253, -0.27928416, -4.23472914),
            species: "O2-".parse().unwrap(),
        },
    ];

    let lattice = Lattice {
        #[rustfmt::skip]
        mat: Matrix3::new(
             2.16500590e-01,  2.16500590e-01, -3.27679720e-02,
             2.92175539e-01, -2.92175539e-01, 1.58116914e-17,
            -0.00000000e+00,  -0.00000000e+00, -1.94977383e-01
        ),
    };
    let min_r = (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0 - 1e-8;
    let max_r = (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0 + 1e-8;

    // println!("{min_r}, {max_r}");

    let recp_len = lattice.recip_lattice().abc();
    let r_max = ((max_r + 0.15) * recp_len / (2.0 * PI)).map(|x| x.ceil() as i32);
    // println!("{r_max}");

    let nmin = -r_max;
    let nmax = r_max;
    for (hkl, g_hkl) in (nmin[0]..nmax[0])
        .cartesian_product(nmin[1]..nmax[1])
        .cartesian_product(nmin[2]..nmax[2])
        .map(|((a, b), c)| {
            (
                Vector3::<f64>::new(a as f64, b as f64, c as f64),
                lattice.mat * Vector3::<f64>::new(a as f64, b as f64, c as f64),
            )
        })
        .map(|(hkl, pos)| (hkl, pos.magnitude()))
        .filter(|(hkl, dist)| (*dist <= max_r) && (*dist >= min_r))
    {
        if g_hkl == 0.0 {
            continue;
        }
        // bragg condition
        let theta = (wavelength_ams * g_hkl / 2.0).asin();
        println!("{theta}")
    }

    // 0, 0, 0  | 0.0
    // 0, 0, -1 | 0.19771170861976084
    // 0, 0, 1  | 0.19771170861976084
    // 0, -1, 0 | 0.36364687664399453
    // 0, 1, 0  | 0.36364687664399453
    // -1, 0, 0 | 0.3636468766439945
    // 1, 0, 0  | 0.3636468766439945
    //
    // frac_coords = lattice.get_fractional_coords(center_coords)
    // nmin_temp = np.floor(np.min(frac_coords, axis=0)) - maxr
    // nmax_temp = np.ceil(np.max(frac_coords, axis=0)) + maxr
}
