# PLAN 3 — VoF.jl

**Repo:** `pankgeorg/VoF.jl` (private).
**Depends on:** WaterLily upstream-hooks PR (PLAN 1) — needs both the
effective-density/viscosity hook *and* the scalar-transport helper.

## Scope

Two-phase incompressible flow on the existing Cartesian grid via a colour
function `α ∈ [0,1]` (1 = water, 0 = air). The smallest version covers:

- α advection by the divergence-free velocity field with a compressive
  limiter (MULES-style: anti-diffusion away from interface).
- Variable mixture density `ρ(α) = α ρ_w + (1-α) ρ_a` and dynamic
  viscosity `μ(α) = α μ_w + (1-α) μ_a`, fed into the existing momentum
  equation.
- Continuum surface force (Brackbill, Kothe, Zemach 1992): `f_st = σ κ ∇α`
  with curvature `κ = -∇·(∇α/|∇α|)`.
- Hydrostatic pressure decomposition: solve for `p_rgh = p - ρ g·h` to keep
  the Poisson source bounded near the interface.

## Non-goals

- **No** geometric/PLIC interface reconstruction (e.g., isoAdvector,
  VOF-PLIC). Algebraic VoF is much simpler and good enough for ship
  hydrostatics + wave-making. PLIC is a follow-on package if needed.
- **No** evaporation/condensation, no contact-angle modelling, no
  cavitation. All separable extensions.
- **No** three-phase / N-phase. Just water/air.
- **No** compressibility on either phase.

## API

```julia
using WaterLily, VoF

flow_ctor = (dims, uBC; kw...) -> VoF.two_phase_flow(
    Flow(dims, uBC; kw...);
    ρ_w = 1000.0, ρ_a = 1.0,
    μ_w = 1e-3,   μ_a = 1.8e-5,
    σ   = 0.072,    # surface tension N/m
    g   = (0, 0, -9.81),
    α₀  = (i,x) -> x[3] < z_water_line ? 1.0 : 0.0,
)
```

The returned `AbstractFlow` subtype carries an `α` field plus its
workspace. `mom_step!` now does:

1. Advect α with the *predictor* velocity using a compressive scheme.
2. Recompute `ρ[I]`, `μ[I]`, `f_st[I]` from α.
3. Run the existing WaterLily momentum predictor (with variable μ via
   the effective-viscosity hook) and projection (with α-weighted source).
4. Advect α again with the corrected velocity (operator splitting).

## Algorithms (primary references)

- α-advection with compressive flux: Weller & Tabor, *Comp. & Fluids* 1998;
  Berberović et al., *Phys. Rev. E* 2009 (interFoam description).
- MULES limiter: Marquez Damián, *Open Source CFD Library for Compressible
  Turbulent Flows*, 2013 ch. 5 (best public description).
- CSF surface tension: Brackbill, Kothe, Zemach, *J. Comp. Phys.* 100 (1992).
- Curvature from smoothed α: Cummins et al., *Comp. Struct.* 83 (2005).

**Implement from these papers, not from OpenFOAM source.**

## Validation

### Layer 1 — analytic / canonical (fast, per-PR CI)

- **Disk in solid-body rotation (Zalesak).** Pure α-advection, no momentum.
  Slotted disk rotates one revolution; measure shape error. Pass: L₁
  shape error < 0.05 at 100×100, < 0.025 at 200×200 (second-order ish).
- **2D dam break (analytic free-fall limit).** Water column collapses;
  initial-time front position vs `x_f(t) = 2t√(gH)` (Ritter 1892). Pass:
  within ±5% for `t < 0.2 √(H/g)`.
- **Rising bubble (Hysing benchmark, Test Case 1).** Standardised 2D
  bubble, σ = 24.5, μ_ratio = 10, ρ_ratio = 10. Pass: circularity,
  center-of-mass position, and rise velocity within the published
  envelope of converged codes (Hysing et al., *Int. J. Numer. Methods
  Fluids* 60, 2009).

### Layer 2 — OpenFOAM tutorial reproduction (nightly CI)

Target: `OpenFOAM/tutorials/incompressibleVoF/damBreak`.

The Martin–Moyce (1952) experiment is the standard validation. OpenFOAM
ships the tutorial with experimental data references. Procedure:

1. Run OpenFOAM `damBreak` in Docker, dump α and U at fixed sample points.
2. Run WaterLily/VoF.jl on the same domain with matched ρ, μ, σ, g.
3. Compare:
   - Water front position vs time
   - Water column height at the back wall vs time
   - Total kinetic + potential energy budget

Pass: front position within ±10% of OpenFOAM and within ±15% of the
Martin–Moyce experimental data.

### Layer 3 — release-blocking ship-scale check

Target: `OpenFOAM/tutorials/incompressibleVoF/DTCHullWave` — DTC hull in
regular waves, no propeller. This is the integration test that combines
VoF + a ship-shape body. Run only after Phase 3 milestone in MASTER_PLAN.

Pass: wave-resistance coefficient within ±15% of OpenFOAM at the same
Froude number; wave pattern (free-surface elevation along the hull)
qualitatively matches.

## Performance budget

| Component                       | Cost vs baseline WaterLily |
|---------------------------------|----------------------------|
| α advection (per step)          | ≈ 80% of one `conv_diff!`  |
| Variable ρ, μ updates           | ≈ 5%                       |
| Surface tension (CSF)           | ≈ 15%                      |
| Total VoF overhead              | ≤ 100% (i.e., ~2× slower)  |

Two-phase is genuinely more expensive than single-phase. If we land
under 2× on a 256³ damBreak, that's a win.

## Harness

- Layer 1 tests in `test/runtests.jl`, per-PR.
- Layer 2 + 3 in `test/openfoam/`, nightly via ShipFlow.jl harness.
- Reference data:
  - Martin–Moyce 1952 digitised values committed in-repo (small).
  - Hysing benchmark reference fields in the `cerulean-reference-data`
    LFS repo.
  - OpenFOAM tutorial outputs regenerated nightly (not committed).

## Risks & open questions

- **Mass conservation under operator splitting.** Algebraic VoF with
  predictor + corrector α-advection has a known small drift. Quantify
  on damBreak: total water mass should drift < 0.1% over 5 seconds.
  If worse, switch to a single-step α update per `mom_step!` (interFoam
  default).
- **Poisson solver behavior with `ρ`-jump.** WaterLily's multigrid
  smoother (`MultiLevelPoisson.jl`) assumes scalar-ish coefficients.
  ρ-jump factor 1000 may degrade convergence. Mitigation: use the
  rescaled pressure `p_rgh = p - ρ g·h`, which is C⁰ across the
  interface, and confirm multigrid still converges in ≤ 10 iterations.
- **CFL with surface tension.** The capillary timestep
  `Δt_σ < √(ρ Δx³ / (2π σ))` may dominate at small Δx. Detect and
  warn; add to `CFL(::Flow)`.
- **BDIM + VoF.** Bodies sitting at the air-water interface (a ship!) is
  the *whole point*. Verify with a static floating box: equilibrium
  draught matches Archimedes' principle within ±2%.

## Milestones

| # | Goal                                                  | Done when                                  |
|---|-------------------------------------------------------|--------------------------------------------|
| 1 | Pure α-advection passes Zalesak                       | Layer 1 disk test green                    |
| 2 | Coupled VoF+momentum on dry damBreak                  | Layer 1 dam break passes                   |
| 3 | Surface tension on rising bubble (Hysing)             | Layer 1 bubble test in published envelope  |
| 4 | damBreak vs OpenFOAM ±10%                             | Layer 2 passes nightly                     |
| 5 | DTC hull in waves vs OpenFOAM ±15%                    | Layer 3 — release-blocking                 |
