# Core Benchmark Tables

## Best-Seed Results

| Case | Primary metric | Field rel-L2 | Obs rel-L2 | Traction rel-L2 | mu err (%) | K err (%) | k0 err | eig err mean | axis err (deg) | Interface flux | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| forward_poisson_ellipsoid | 0.0023 | 0.0023 | - | - | - | - | - | - | - | - | 106.30 |
| forward_poisson_star | 0.2591 | 0.2591 | - | - | - | - | - | - | - | - | 229.36 |
| rabbit_poisson_schwarz | 0.0221 | 0.0221 | - | - | - | - | - | - | - | 0.0116 | 3.659e+03 |
| torus_inverse_original_atlas | 2.350e-07 | - | - | 2.350e-07 | 9.272e-06 | 2.289e-05 | - | - | - | - | 201.09 |
| torus_inverse_schwarz_displacement | 0.0017 | - | 0.0017 | 0.0096 | 4.083e-12 | 0.9993 | - | - | - | - | 38.63 |
| torus_inverse_schwarz_traction | 0.0058 | - | 0.0058 | 0.0058 | 0.3453 | 0.5412 | - | - | - | - | 76.89 |
| rabbit_inverse_elder | 0.0300 | - | - | - | - | - | 0.0300 | 0.0297 | 2.6968 | 0.0353 | 12.09 |

## Multi-seed Statistics (n >= 3)

| Case | n | Primary (mean ± std) | Field rel-L2 (mean ± std) | Obs rel-L2 (mean ± std) | Traction rel-L2 (mean ± std) | Interface flux (mean ± std) | Runtime (s, mean ± std) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rabbit_poisson_schwarz | 3 | 0.0600 ± 0.0513 | 0.0600 ± 0.0513 | - | - | 0.0812 ± 0.1049 | 1.357e+03 ± 1.636e+03 |
| torus_inverse_original_atlas | 3 | 1.393e-04 ± 9.834e-05 | - | - | 1.393e-04 ± 9.834e-05 | - | 128.74 ± 51.16 |
| torus_inverse_schwarz_displacement | 3 | 13.7164 ± 19.3948 | - | 13.7164 ± 19.3948 | 0.0145 ± 0.0049 | - | 31.59 ± 5.48 |
| torus_inverse_schwarz_traction | 3 | 0.0150 ± 0.0106 | - | 0.0150 ± 0.0106 | 0.0058 ± 0 | - | 43.41 ± 29.77 |
| rabbit_inverse_elder | 3 | 0.0300 ± 2.434e-08 | - | - | - | 0.0137 ± 0.0152 | 411.72 ± 283.72 |
