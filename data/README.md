# Data

These two files are redistributed from the [SeReMpy](https://github.com/dariograna/SeReMpy)
library (`Data/` folder), which is released under the MIT License.

    Copyright (c) 2020 Dario Grana

    Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley
    Grana and de Figueiredo, 2021, SeReMpy, GEOPHYSICS 86: F61-F69

The full MIT notice is reproduced in `THIRD_PARTY.md` at the repository root.

| File | Columns |
|---|---|
| `data5seis.dat` | time, near, mid, far (synthetic seismic traces) |
| `data5log.dat`  | phi, clay, sw, time, vp, vs, rho (reference well log) |

Only the `near` trace and the `vp`/`rho` columns are used by this project: the
inversion targets acoustic impedance `Z = vp * rho` at normal incidence.
