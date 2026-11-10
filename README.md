# Assignment 17 — Wells in a 2D reservoir

[![Tests](https://github.com/PGE323M/assignment17/actions/workflows/main.yml/badge.svg)](https://github.com/PGE323M/assignment17/actions)

Extend the supplied Assignment 15 simulator with constant-rate and constant-bottomhole-pressure (BHP) wells. First verify the small problem, then ask your agent to launch a larger simulation and schedule unattended completion checks that produce a pressure map.

## Reservoir and verification problem

`inputs.yml` specifies a 10,000 ft × 10,000 ft reservoir, 20 ft thick, permeability 50 mD, porosity 0.2, water viscosity 1 cp, formation volume factor 1, compressibility 10⁻⁶ psi⁻¹, and initial pressure 1,000 psi. The right boundary is held at 2,000 psi; the other faces have zero flux. The conversion factor is 0.00633. Use a 3×3 grid and implicit integration with three one-day steps. An injector at (5,000, 5,000) ft supplies 1,000 ft³/day; a BHP producer at (9,000, 9,000) ft is held at 800 psi. Both radii are 0.25 ft, with zero skin.

The separate `inputs-1d.yml` retains the older narrow-domain example for comparison. Do not substitute its well locations or geometry into the 3×3 problem.

## Provided code and student work

`assignment13.py` and `assignment15.py` are complete, protected base classes. In `assignment17.py`, implement only:

1. `compute_well_index_locations(well_type='rate')` — one flat block index **per well**, in input order, including repeated blocks.
2. `compute_productivity_index(well_type='bhp')` — one unconverted productivity index per well.
3. `fill_matrices()` — call the provided assembly and add well terms without replacing its boundary contributions.

Use x-fastest numbering: block `row*Nx + column`, with row zero at the bottom. Grid widths come from `assign_delta_array('delta x')` and `assign_delta_array('delta y')`. Coordinates lie in the half-open domain `[0,sum(dx)) × [0,sum(dy))`; a point exactly on an internal face belongs to the cell on its right/above. Reject nonfinite or out-of-domain coordinates. Support nonuniform widths, multiple wells, repeated blocks, and an absent/empty `wells` section.

Each well-control group has `locations` and `values` lists of equal length. BHP wells also require positive `radii`; optional `skin factor` is a finite scalar or one value per well. Rate wells do not require radii. Validate finite control values. Only `rate` and `bhp` controls are supported.

## Well equations and units

Positive rate means injection into the reservoir; negative means production. Rates are ft³/day in the same volume basis as the model balance. A BHP well's rate is **computed** from its specified pressure, not prescribed as a rate.

For isotropic permeability and a vertical well in a rectangular cell, use the course's Peaceman approximation:

$$
r_e = 0.14\sqrt{\Delta x_i^2+\Delta y_i^2},\qquad
J = \frac{2\pi k_i h_i}{\mu B_\alpha\,[\ln(r_e/r_w)+s]}.
$$

Here $h$ is thickness (ft), $k$ permeability (mD), $\mu$ viscosity (cp), $B_\alpha$ formation volume factor, $r_w$ well radius (ft), and $s$ dimensionless skin. Require $\ln(r_e/r_w)+s>0$. Skin is inside the same bracket as the logarithm. Zero permeability gives zero productivity.

Let $C$ denote the input's **`conversion factor`**, equal to $6.33\times10^{-3}$ for this problem's field units. It converts raw conductances computed from mD, ft, and cp to ft³/(day·psi). `compute_productivity_index` returns the raw $J$ **without** this factor; the converted BHP well conductance is $W=CJ$. The supplied base class already applies this factor to the flow transmissibilities in $\mathbf{T}$. Prescribed well rates are already in ft³/day and need no such conversion.

The supplied balance is

$$
\mathbf{B}\frac{d\mathbf{p}}{dt}+\mathbf{T}\mathbf{p}=\mathbf{Q}.
$$

For a rate well in cell $i$, add $q$ to $Q_i$. For a BHP well add $W$ to $T_{ii}$ and $Wp_w$ to $Q_i$, so its inflow is $W(p_w-p_i)$. **Sum**, rather than overwrite, contributions from wells sharing a cell. Do not multiply a prescribed rate by the conversion factor or apply that factor twice to a BHP conductance. Preserve sparse matrices and all no-well behavior. The inherited mixed method uses $\theta=0$ for implicit, $\theta=1$ for explicit, and $\theta=1/2$ for Crank–Nicolson.

## Small-grid checks

In the Codespace terminal, run:

```bash
python -m unittest -v test.py
```

All numerical checks must pass before running a large grid. Independently check one rate-well volume balance and one BHP diagonal/source contribution; review the relevant code diff, signs, and units with your agent. Do not change tests to make a result pass.

## Agent exercise

After the 3×3 numerical checks pass, give your agent this prompt:

> Run the simulation with the 1000 x 1000 grid. Set a watcher to check on the output every 30 seconds and once the simulation completes, create a pressure contour map png for inspection. After the plot has been created, remove the watcher.

Keep the Codespace and required agent session active until it finishes. Generated results and plots stay in ignored `.run/` and are not submitted.

## Submission

Submit only `assignment17.py`. Do not edit the supplied base classes, runner, tests, data, environment, instructions, or submission infrastructure. Ask your agent to **submit assignment 17** and approve the exact diff and bounded submission action. The supplied submission skill performs a dry run before an explicitly approved commit/push. No new written report is required.
